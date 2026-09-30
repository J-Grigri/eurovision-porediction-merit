"""Diaspora test: do migrant communities explain country bias, and is it the public (not the jury) that follows them?

Joins Eurostat migrant stock (data/external/migration.parquet) to voter -> country affinity.
  1. Correlation of log(migrant share) with public affinity vs jury affinity (2016-2024, same voters and songs)
  2. Regression: public affinity ~ log share + jury affinity. The jury affinity absorbs cultural/neighbour closeness
     that both juries and viewers share; what log share adds on top is the diaspora effect on the public.
  3. Prediction: does adding log migrant share to the Ridge pre-contest model improve held-out accuracy?
Only pairs whose voting country reports migrant data are used; pairs with no recorded migrants are dropped (unknown).
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from features import CUTOFF_YEAR, FEATURES, PROC, ROOT
from models import DEV_YEARS, HOLDOUT_YEARS, entry_table, fit_predict, rank_metrics

EXT = ROOT / "data" / "external"
RNG = np.random.default_rng(0)


def pair_share(mig, y0, y1):
    m = mig[(mig.year >= y0) & (mig.year <= y1) & (mig.share > 0)]
    s = m.groupby(["voter", "entry"]).share.mean()
    return np.log10(s).rename("log_share")


def boot(fn, d, n=2000):
    vals = [fn(d.sample(len(d), replace=True, random_state=i)) for i in range(n)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    mig = pd.read_parquet(EXT / "migration.parquet")
    jp = pd.read_parquet(PROC / "ev_jury_public_pairs.parquet")[["voter", "entry", "jury", "public"]]
    d = jp.merge(pair_share(mig, 2016, CUTOFF_YEAR).reset_index(), on=["voter", "entry"], how="inner")
    out = {"pairs_2016_2024": len(d), "voting_countries": int(d.voter.nunique())}
    rp, rj = pearsonr(d.log_share, d.public).statistic, pearsonr(d.log_share, d.jury).statistic
    out.update(r_public=rp, r_jury=rj, diff=rp - rj,
               diff_ci=boot(lambda x: pearsonr(x.log_share, x.public).statistic - pearsonr(x.log_share, x.jury).statistic, d))
    # regression with jury affinity as a control
    X = np.column_stack([np.ones(len(d)), d.log_share, d.jury]); yv = d.public.values
    beta = np.linalg.lstsq(X, yv, rcond=None)[0]
    bs = []
    for i in range(2000):
        s = d.sample(len(d), replace=True, random_state=i)
        Xs = np.column_stack([np.ones(len(s)), s.log_share, s.jury])
        bs.append(np.linalg.lstsq(Xs, s.public.values, rcond=None)[0][1])
    out.update(beta_log_share=beta[1], beta_ci=(float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))), beta_jury=beta[2],
               pts_per_10x=beta[1] * 12)
    # binned view for the chart
    d["band"] = pd.cut(d.log_share, [-9, -4, -3, -2.5, -2, 0], labels=["< 0.01%", "0.01–0.1%", "0.1–0.3%", "0.3–1%", "> 1%"])
    bands = d.groupby("band", observed=True).agg(public=("public", "mean"), jury=("jury", "mean"), n=("public", "count")).reset_index()
    bands.to_parquet(PROC / "diaspora_bands.parquet")
    top = d.assign(share_pct=10 ** d.log_share * 100).sort_values("log_share", ascending=False)
    top.to_parquet(PROC / "diaspora_pairs.parquet")

    # prediction test: add log migrant share to the pre-contest Ridge model
    F = pd.read_parquet(PROC / "features.parquet")
    res = pd.read_parquet(PROC / "results.parquet")
    F["qualified"] = res.set_index(["year", "round", "contestant_id"]).qualified.reindex(list(zip(F.year, F["round"], F.contestant_id))).values
    covered = set(mig.voter)
    lag = mig.assign(year=mig.year + 1)  # use last year's migrant figures: known before the contest
    ls = np.log10(lag[lag.share > 0].set_index(["year", "voter", "entry"]).share)
    F["log_share"] = ls.reindex(list(zip(F.year, F.voter, F.entry))).values
    F["mig_known"] = F.voter.isin(covered).astype(int)
    F["log_share"] = np.where(F.mig_known == 1, F.log_share.fillna(-6.0), np.nan)  # covered voter, nobody recorded -> tiny share
    F["has_mig"] = F.log_share.notna().astype(int)
    known = F[F.year <= CUTOFF_YEAR]
    P = []
    for T in DEV_YEARS + HOLDOUT_YEARS:
        tr = known if T > CUTOFF_YEAR else known[known.year < T]
        te = F[F.year == T].copy()
        te["split"] = "holdout" if T > CUTOFF_YEAR else "dev"
        te["base"], _ = fit_predict("ridge", tr, te, FEATURES)
        te["with_migration"], _ = fit_predict("ridge", tr, te, FEATURES + ["has_mig", "log_share"])
        P.append(te)
    P = pd.concat(P)
    pred = []
    for split, sp in P.groupby("split"):
        sub = sp[sp.has_mig == 1]  # the pairs where migration data exists
        for col in ["base", "with_migration"]:
            m = rank_metrics(entry_table(sp, col))
            pred.append(dict(split=split, model=col, pair_r_all=np.corrcoef(sp[col], sp.y)[0, 1],
                             pair_r_covered=np.corrcoef(sub[col], sub.y)[0, 1],
                             final_rank_corr=m[m["round"] == "final"].spearman.mean()))
    pred = pd.DataFrame(pred)
    pred.to_parquet(PROC / "diaspora_prediction.parquet")
    out["prediction"] = pred.round(4).to_dict("records")
    (PROC / "diaspora.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != "prediction"}, indent=2, default=lambda x: round(float(x), 4)))
    print(bands.round(3).to_string()); print(pred.round(4).to_string())
    print(top.head(12)[["voter", "entry", "share_pct", "jury", "public"]].round(3).to_string())


if __name__ == "__main__":
    main()
