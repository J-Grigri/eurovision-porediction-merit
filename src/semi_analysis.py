"""Semi-final pivot: how much do semi-final results tell us about the final?

A. Entry-level correlations between semi and final scores (within-round z-scores), overall and by jury / public.
B. Final-prediction variants (Ridge, finals only), dev 2010-2024 rolling origin + hold-out 2025-2026:
   no_semi        history only (pre-contest inputs)
   semi_total     + entry's semi average + voter's semi score for the entry (current pre-final model)
   semi_rank      + entry's semi rank z-score (standardised within its semi)
   semi_split     + semi jury z and semi public z (available 2016-2022 and 2026 only, else flagged as missing)
   voter_split    + voter's semi jury and public points for the entry
   components     predict final jury points and public points separately (2016+) and add them up
C. Where the error sits: entries that came through a semi vs direct finalists (Big 5 + host).
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from features import CUTOFF_YEAR, FEATURES, PROC, load_tables
from models import DEV_YEARS, HOLDOUT_YEARS, fit_predict

z = lambda s: (s - s.mean()) / s.std()


def entry_semi_table(results):
    r = results[(results.disqualified == 0) & (results.year >= 2004)].copy()
    semi = r[r["round"] != "final"].copy()
    g = semi.groupby(["year", "round"])
    semi["semi_total_z"] = g.points_total.transform(z)
    semi["semi_jury_z"] = g.points_jury.transform(z)
    semi["semi_public_z"] = g.points_public.transform(z)
    semi["semi_rank_z"] = -g.place.transform(z)  # higher = better
    fin = r[r["round"] == "final"].copy()
    gf = fin.groupby("year")
    fin["final_total_z"] = gf.points_total.transform(z)
    fin["final_jury_z"] = gf.points_jury.transform(z)
    fin["final_public_z"] = gf.points_public.transform(z)
    d = fin.merge(semi[["year", "country", "semi_total_z", "semi_jury_z", "semi_public_z", "semi_rank_z"]], on=["year", "country"], how="left")
    return d


def correlations(d):
    d = d[(d.year <= CUTOFF_YEAR) & (d.year >= 2008)]
    out = []
    for a, b, label in [("semi_total_z", "final_total_z", "Semi total → final total (2008–2024)"),
                        ("semi_jury_z", "final_jury_z", "Semi jury → final jury"),
                        ("semi_public_z", "final_public_z", "Semi public → final public"),
                        ("semi_jury_z", "final_public_z", "Semi jury → final public"),
                        ("semi_public_z", "final_jury_z", "Semi public → final jury"),
                        ("semi_jury_z", "final_total_z", "Semi jury → final total"),
                        ("semi_public_z", "final_total_z", "Semi public → final total")]:
        x = d[[a, b]].dropna()
        r = pearsonr(x[a], x[b])
        out.append(dict(test=label, n=len(x), r=r.statistic, p=r.pvalue))
    # does the jury score add information beyond the total? partial correlation via residuals
    x = d[["semi_total_z", "semi_jury_z", "semi_public_z", "final_total_z"]].dropna()
    X = np.column_stack([np.ones(len(x)), x.semi_total_z]); yv = x.final_total_z.values
    res_y = yv - X @ np.linalg.lstsq(X, yv, rcond=None)[0]
    jd = x.semi_jury_z - x.semi_public_z
    res_j = jd - X @ np.linalg.lstsq(X, jd.values, rcond=None)[0]
    pr = pearsonr(res_j, res_y)
    out.append(dict(test="Jury-minus-public gap in semi → final, beyond semi total", n=len(x), r=pr.statistic, p=pr.pvalue))
    return pd.DataFrame(out)


def voter_semi_split(votes):
    s = votes[(votes["round"] != "final") & votes.kind.isin(["jury", "public"])]
    p = s.pivot_table(index=["year", "voter", "entry"], columns="kind", values="points", aggfunc="sum") / 12.0
    voted = s.groupby(["year", "kind"]).voter.apply(set)
    return p, voted


def build_final_frame(F, d, votes):
    f = F[F.is_final == 1].copy()
    sd = d.set_index(["year", "country"])
    for c in ["semi_total_z", "semi_jury_z", "semi_public_z", "semi_rank_z"]:
        f[c] = sd[c].reindex(list(zip(f.year, f.entry))).values
    f["has_split"] = f.semi_jury_z.notna().astype(int)
    p, voted = voter_semi_split(votes)
    k = list(zip(f.year, f.voter, f.entry))
    for kind in ["jury", "public"]:
        vals = p[kind].reindex(k).values if kind in p else np.full(len(f), np.nan)
        # a voter who voted in the entry's semi but gave nothing -> 0; not in that semi -> missing
        in_semi = f.semi_from_voter.notna().values
        f[f"semi_voter_{kind}"] = np.where(in_semi & (f.has_split == 1), np.nan_to_num(vals, nan=0.0), np.nan)
    # final-level jury and public targets (2016+), each as a share of 12
    fv = votes[(votes["round"] == "final") & votes.kind.isin(["jury", "public"])]
    fp = fv.pivot_table(index=["year", "voter", "contestant_id"], columns="kind", values="points", aggfunc="sum") / 12.0
    kk = list(zip(f.year, f.voter, f.contestant_id))
    for kind in ["jury", "public"]:
        f[f"y_{kind}"] = np.where(f.year >= 2016, np.nan_to_num(fp[kind].reindex(kk).values, nan=0.0), np.nan)
    return f


SEMI_TOTAL = FEATURES + ["has_semi", "semi_from_voter", "semi_y_mean"]
VARIANTS = {
    "no_semi": FEATURES,
    "semi_total": SEMI_TOTAL,
    "semi_rank": SEMI_TOTAL + ["semi_rank_z"],
    "semi_split": SEMI_TOTAL + ["has_split", "semi_jury_z", "semi_public_z"],
    "voter_split": SEMI_TOTAL + ["has_split", "semi_jury_z", "semi_public_z", "semi_voter_jury", "semi_voter_public"],
}


def evaluate(f, res):
    rows, entries = [], []
    known = f[f.year <= CUTOFF_YEAR]
    for T in DEV_YEARS + HOLDOUT_YEARS:
        tr = known if T > CUTOFF_YEAR else known[known.year < T]
        te = f[f.year == T].copy()
        for name, feats in VARIANTS.items():
            te[name], _ = fit_predict("ridge", tr, te, feats)
        # components model: separate jury and public models, trained on 2016+ finals only
        tr16 = tr[tr.year >= 2016]
        if T >= 2016 and tr16.year.nunique() >= 2:
            feats = VARIANTS["voter_split"]
            pj, _ = fit_predict("ridge", tr16.assign(y=tr16.y_jury), te, feats)
            pp, _ = fit_predict("ridge", tr16.assign(y=tr16.y_public), te, feats)
            te["components"] = (pj + pp) / 2  # both on a 12-point scale; total share is out of 24
        else:
            te["components"] = np.nan
        entries.append(te)
    P = pd.concat(entries, ignore_index=True)
    names = list(VARIANTS) + ["components"]
    E = P.groupby(["year", "contestant_id", "entry"], as_index=False).agg(
        place=("place", "first"), has_semi=("has_semi", "first"), actual=("y", "mean"), **{n: (n, "mean") for n in names})
    for n in names:
        for (year), g in E.groupby("year"):
            if g[n].isna().all():
                continue
            order = g.sort_values(n, ascending=False)
            act = g.sort_values("place")
            rows.append(dict(model=n, year=year, split="holdout" if year > CUTOFF_YEAR else "dev",
                             rho=spearmanr(g[n], -g.place).statistic,
                             rho_semi_only=spearmanr(g[g.has_semi == 1][n], -g[g.has_semi == 1].place).statistic,
                             rho_direct_only=spearmanr(g[g.has_semi == 0][n], -g[g.has_semi == 0].place).statistic if (g.has_semi == 0).sum() >= 4 else np.nan,
                             winner=int(order.contestant_id.iloc[0] == act.contestant_id.iloc[0]),
                             winner_rank=int(list(order.contestant_id).index(act.contestant_id.iloc[0]) + 1),
                             top5=len(set(order.contestant_id[:5]) & set(act.contestant_id[:5])),
                             pair_r=np.corrcoef(P.loc[(P.year == year), n], P.loc[(P.year == year), "y"])[0, 1]))
    return pd.DataFrame(rows), E


def main():
    votes, results, _ = load_tables()
    F = pd.read_parquet(PROC / "features.parquet")
    d = entry_semi_table(results)
    C = correlations(d)
    f = build_final_frame(F, d, votes)
    M, E = evaluate(f, results)
    C.to_parquet(PROC / "semi_correlations.parquet"); M.to_parquet(PROC / "semi_models.parquet")
    print(C.round(3).to_string())

    def summ(m, years=None):
        m = m if years is None else m[m.year.isin(years)]
        return m.groupby("model").agg(finals=("year", "count"), rho=("rho", "mean"), rho_semi=("rho_semi_only", "mean"),
                                      rho_direct=("rho_direct_only", "mean"), winners=("winner", "sum"),
                                      winner_rank_med=("winner_rank", "median"), top5=("top5", "mean"), pair_r=("pair_r", "mean"))
    order = list(VARIANTS) + ["components"]
    print("\nDEV 2010-2024\n", summ(M[M.split == "dev"]).reindex(order).round(3).to_string())
    print("\nDEV 2016-2022 (years with semi jury/public split)\n", summ(M[M.split == "dev"], [2016, 2017, 2018, 2019, 2021, 2022]).reindex(order).round(3).to_string())
    print("\nHOLD-OUT 2025-2026\n", summ(M[M.split == "holdout"]).reindex(order).round(3).to_string())
    print("\nPer-year hold-out\n", M[M.split == "holdout"].pivot(index="model", columns="year", values=["rho", "winner_rank"]).reindex(order).round(2).to_string())
    # bootstrap over years: gain of semi_total over no_semi, and of voter_split over semi_total
    dev = M[M.split == "dev"].pivot(index="year", columns="model", values="rho")
    rng = np.random.default_rng(0)
    for a, b in [("semi_total", "no_semi"), ("semi_split", "semi_total"), ("voter_split", "semi_total"), ("semi_rank", "semi_total")]:
        diff = (dev[a] - dev[b]).dropna()
        bs = [rng.choice(diff.values, len(diff)).mean() for _ in range(5000)]
        print(f"gain {a} vs {b}: {diff.mean():+.3f}  95% CI [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]  years={len(diff)}")
    json.dump({"note": "see parquet"}, open(PROC / "semi_meta.json", "w"))


if __name__ == "__main__":
    main()
