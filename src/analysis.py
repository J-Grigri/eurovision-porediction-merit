"""Pattern analysis: voting affinity, blocs, structural effects, jury vs public."""
import json

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform

from features import CUTOFF_YEAR, PROC, load_tables, build_grids

MIN_YEAR = 2004
N_BLOCS = 7


def residuals(grids):
    g = grids[(grids.year >= MIN_YEAR) & (grids.year <= CUTOFF_YEAR)].copy()
    k = ["year", "round", "entry"]
    g["csum"] = g.groupby(k).y.transform("sum"); g["ccnt"] = g.groupby(k).y.transform("count")
    g = g[g.ccnt >= 5]
    g["resid"] = g.y - (g.csum - g.y) / (g.ccnt - 1)
    return g[["year", "round", "voter", "entry", "y", "resid"]]


def pair_affinity(R):
    a = R.groupby(["voter", "entry"]).resid.agg(mean="mean", sd="std", n="count").reset_index()
    a["t"] = a["mean"] / (a.sd / np.sqrt(a.n)).replace(0, np.nan)
    early = R[R.year < 2016].groupby(["voter", "entry"]).resid.mean().rename("mean_2004_2015")
    late = R[R.year >= 2016].groupby(["voter", "entry"]).resid.mean().rename("mean_2016_plus")
    return a.merge(early, on=["voter", "entry"], how="left").merge(late, on=["voter", "entry"], how="left")


def blocs(A, min_rounds=15, k=N_BLOCS):
    """Cluster countries by how similarly they give/receive affinity."""
    M = A.pivot(index="voter", columns="entry", values="mean")
    cnt = pd.concat([A.groupby("voter").n.sum(), A.groupby("entry").n.sum()], axis=1).min(axis=1)
    keep = [c for c in M.index if c in M.columns and cnt.get(c, 0) >= min_rounds * 10 and c != "WLD"]
    M = M.loc[keep, keep].fillna(0.0)
    S = (M.values + M.values.T) / 2  # symmetrised mutual affinity
    D = 1 - np.corrcoef(S)
    np.fill_diagonal(D, 0)
    Z = linkage(squareform(np.clip((D + D.T) / 2, 0, None), checks=False), "average")
    lab = fcluster(Z, k, "maxclust")
    order = leaves_list(Z)
    return pd.DataFrame({"country": np.array(keep)[order], "bloc": lab[order], "order": range(len(keep))}), S


def bloc_trend(R, bl):
    b = dict(zip(bl.country, bl.bloc))
    R = R.assign(vb=R.voter.map(b), eb=R.entry.map(b)).dropna(subset=["vb", "eb"])
    R["same"] = (R.vb == R.eb) & (R.voter != R.entry)
    return R.groupby(["year", "same"]).resid.mean().unstack().rename(columns={True: "within_bloc", False: "cross_bloc"}).reset_index()


def structural(F):
    """Model 5: OLS regression of an entry's within-year standardised final score on slot, host and form,
    with a bootstrap over years for the 95% intervals."""
    F = F[F.year <= CUTOFF_YEAR]
    fin = F[F.is_final == 1].groupby(["year", "contestant_id", "entry"], as_index=False).agg(
        y=("y", "mean"), running_pct=("running_pct", "first"), host=("entry_is_host", "first"), form=("form", "first"))
    z = lambda s: (s - s.mean()) / s.std()
    fin["y_z"] = fin.groupby("year").y.transform(z)
    fin["form_z"] = fin.groupby("year").form.transform(lambda s: z(s.fillna(s.mean())))
    fin["late_slot"] = (fin.running_pct > 0.66).astype(float); fin["early_slot"] = (fin.running_pct <= 0.33).astype(float)
    feats = ["early_slot", "late_slot", "host", "form_z"]
    d = fin.dropna(subset=feats + ["y_z"])
    X = np.column_stack([np.ones(len(d))] + [d[f].values for f in feats]); yv = d.y_z.values
    beta = np.linalg.lstsq(X, yv, rcond=None)[0]
    yrs = d.year.values; uy = np.unique(yrs); rng = np.random.default_rng(0); bs = []
    for _ in range(500):
        pick = rng.choice(uy, len(uy)); idx = np.concatenate([np.where(yrs == y)[0] for y in pick])
        bs.append(np.linalg.lstsq(X[idx], yv[idx], rcond=None)[0])
    bs = np.array(bs)
    tab = pd.DataFrame({"feature": ["intercept"] + feats, "effect_sd": beta, "lo95": np.percentile(bs, 2.5, 0), "hi95": np.percentile(bs, 97.5, 0)})
    slot = fin.assign(bin=pd.cut(fin.running_pct, [0, .2, .4, .6, .8, 1.0], labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"])) \
        .groupby("bin", observed=True).y_z.agg(["mean", "count"]).reset_index()
    return tab[tab.feature != "intercept"], slot


def jury_public(results):
    r = results[(results.year <= CUTOFF_YEAR) & (results["round"] == "final") & results.points_jury.notna() & (results.disqualified == 0)].copy()
    r["jury_pct"] = r.groupby("year").points_jury.rank(pct=True); r["public_pct"] = r.groupby("year").points_public.rank(pct=True)
    r["gap"] = r.public_pct - r.jury_pct
    per_year = r.groupby("year").apply(lambda g: pd.Series({"spearman": g.points_jury.corr(g.points_public, method="spearman"),
        "jury_winner": g.loc[g.points_jury.idxmax(), "country"], "public_winner": g.loc[g.points_public.idxmax(), "country"],
        "winner": g.loc[g.place.idxmin(), "country"]}), include_groups=False).reset_index()
    by_country = r.groupby("country").gap.agg(mean_gap="mean", n="count").reset_index()
    return per_year, by_country[by_country.n >= 3].sort_values("mean_gap")


def main():
    votes, results, contests = load_tables()
    grids = build_grids(votes, results)
    R = residuals(grids)
    A = pair_affinity(R); A.to_parquet(PROC / "pair_affinity.parquet")
    bl, S = blocs(A); bl.to_parquet(PROC / "blocs.parquet")
    bt = bloc_trend(R, bl); bt.to_parquet(PROC / "bloc_trend.parquet")
    F = pd.read_parquet(PROC / "features.parquet")
    tab, slot = structural(F); tab.to_parquet(PROC / "structural_effects.parquet"); slot.to_parquet(PROC / "slot_effect.parquet")
    py, bc = jury_public(results); py.to_parquet(PROC / "jury_public_year.parquet"); bc.to_parquet(PROC / "jury_public_country.parquet")
    print("blocs:"); print(bl.groupby("bloc").country.apply(list).to_string())
    print(bt.tail(8).round(3).to_string())
    print(tab.round(3).to_string()); print(slot.round(3).to_string())
    print(py.tail(6).to_string()); print(bc.head(4).to_string(), bc.tail(4).to_string())
    top = A[A.n >= 8].sort_values("mean", ascending=False).head(15)
    print(top.round(3).to_string())


if __name__ == "__main__":
    main()
