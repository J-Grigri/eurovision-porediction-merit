"""Leak-free features for the voter -> entry points model.

Every feature for year Y uses only information from years < Y, except the explicit `semi_*` features, which use the
same year's semi-final (it is broadcast before the final).

Target y = points a voter gave an entry / the maximum a voter can give (12, or 24 when jury and public are added).

Features (6 + 3):
  aff            how much more (or less) this voter has historically given this entry's country than other voters did,
                 i.e. voter-performer bias with song quality removed (decayed over years, shrunk when data is thin)
  form           the entry country's recent average score
  running_pct    position in the running order (0 = first, 1 = last)
  is_final, entry_is_host, voter_is_host
  semi_*         (finals only) the entry's semi-final average score, how this voter scored it in the semi, and a flag
"""
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"

LINEAGE = {"CS": "RS", "YU": "RS"}  # Yugoslavia / Serbia & Montenegro -> Serbia
H_PAIR = 10.0   # half-life (years) of affinity memory
H_FORM = 4.0    # half-life of country form memory
K_PAIR = 3.0    # shrinkage pseudo-count for affinity
MIN_YEAR_MODEL = 2004
CUTOFF_YEAR = 2024  # 2025-2026 are a hold-out: never used for fitting, history, calibration or analysis

FEATURES = ["aff", "form", "running_pct", "is_final", "entry_is_host", "voter_is_host"]
SEMI_FEATURES = ["has_semi", "semi_from_voter", "semi_y_mean"]


def norm(s):
    return s.replace(LINEAGE)


def load_tables():
    votes = pd.read_parquet(PROC / "votes.parquet")
    results = pd.read_parquet(PROC / "results.parquet")
    contests = pd.read_parquet(PROC / "contests.parquet")
    for df, cols in ((votes, ["voter", "entry"]), (results, ["country"]), (contests, ["host"])):
        for c in cols:
            df[c] = norm(df[c])
    return votes, results, contests


def build_grids(votes, results):
    """Full voter x entry grid per (year, round); missing points -> 0; own country excluded.
    y = points / max points a voter can give (24 if the voter had a jury+public split, else 12)."""
    tot = votes[votes.kind == "total"]
    jury_voters = set(map(tuple, votes.loc[votes.kind == "jury", ["year", "round", "voter"]].drop_duplicates().values))
    out = []
    res = results[(results.disqualified == 0)]
    for (year, rnd), rr in res.groupby(["year", "round"]):
        vt = tot[(tot.year == year) & (tot["round"] == rnd)]
        if vt.empty:
            continue
        pts = vt.pivot_table(index="voter", columns="contestant_id", values="points", aggfunc="sum")
        pts = pts.reindex(columns=rr.contestant_id.tolist()).fillna(0.0)
        cc = dict(zip(rr.contestant_id, rr.country))
        g = pts.stack().rename("points").reset_index()
        g["entry"] = g.contestant_id.map(cc)
        g = g[g.voter != g.entry].copy()
        g["year"], g["round"] = year, rnd
        g["denom"] = [24.0 if (year, rnd, v) in jury_voters else 12.0 for v in g.voter]
        g["y"] = g.points / g.denom
        out.append(g)
    return pd.concat(out, ignore_index=True)


def build_features(grids, results, contests):
    countries = sorted(set(grids.voter) | set(grids.entry))
    ix = {c: i for i, c in enumerate(countries)}
    C = len(countries)
    grids = grids.copy()
    grids["vi"] = grids.voter.map(ix)
    grids["ei"] = grids.entry.map(ix)
    host = dict(zip(contests.year, contests.host))
    res = results[results.disqualified == 0].set_index(["year", "round", "contestant_id"])
    n_perf = results[results.disqualified == 0].groupby(["year", "round"]).size()

    S = np.zeros((C, C)); N = np.zeros((C, C))
    decay = 0.5 ** (1.0 / H_PAIR)
    form_hist = {"final": defaultdict(list), "semi": defaultdict(list)}  # country -> [(year, mean y)]
    frames = []

    def form(kind, cty, year):
        """Decayed mean of the country's past entry-level mean y."""
        h_ = form_hist[kind][cty]
        if not h_:
            return np.nan
        w = np.array([0.5 ** ((year - y) / H_FORM) for y, _ in h_]); m = np.array([m for _, m in h_])
        return (w * m).sum() / w.sum()

    for year in sorted(grids.year.unique()):
        gy = grids[grids.year == year]
        A = np.where(N > 0, S / (N + K_PAIR), 0.0)
        if year >= MIN_YEAR_MODEL:
            f = gy.copy()
            f["aff"] = A[f.vi.values, f.ei.values]
            f["form"] = [form("final" if r == "final" else "semi", c, year) for c, r in zip(f.entry, f["round"])]
            f["entry_is_host"] = (f.entry == host.get(year)).astype(int)
            f["voter_is_host"] = (f.voter == host.get(year)).astype(int)
            keys = list(zip([year] * len(f), f["round"], f.contestant_id))
            rk = res.reindex(keys)
            f["running_pct"] = rk.running.values / n_perf.reindex(list(zip([year] * len(f), f["round"]))).values
            f["is_final"] = (f["round"] == "final").astype(int)
            f["place"] = rk.place.values
            frames.append(f)
        if year > CUTOFF_YEAR:
            continue  # hold-out years leave the history state frozen at CUTOFF_YEAR
        # ---- update history with this year's outcomes
        rounds = {}
        for rnd, gr in gy.groupby("round"):
            Y = np.full((C, C), np.nan); Y[gr.vi.values, gr.ei.values] = gr.y.values
            colsum = np.nansum(Y, 0); colcnt = (~np.isnan(Y)).sum(0)
            with np.errstate(invalid="ignore", divide="ignore"):
                loo = (colsum[None, :] - np.nan_to_num(Y)) / (colcnt[None, :] - 1)  # what the OTHER voters gave
            R = np.where((colcnt[None, :] >= 5) & ~np.isnan(Y), Y - loo, np.nan)
            rounds[rnd] = (Y, R, ~np.isnan(R))
        S *= decay; N *= decay
        for rnd, (Y, R, ok) in rounds.items():
            S += np.where(ok, R, 0.0); N += ok
            kind = "final" if rnd == "final" else "semi"
            for e in range(C):
                col = Y[:, e]
                if (~np.isnan(col)).sum() >= 5:
                    form_hist[kind][countries[e]].append((year, np.nanmean(col)))
    F = add_semi_features(pd.concat(frames, ignore_index=True), grids)
    return F, countries


def add_semi_features(F, grids):
    """Same-year semi-final information for final rows (semis are broadcast before the final)."""
    semi = grids[grids["round"].str.startswith("semi")]
    sv = semi.groupby(["year", "voter", "entry"]).y.first()
    sm = semi.groupby(["year", "entry"]).y.mean()
    fin = F["is_final"] == 1
    F["semi_from_voter"] = np.where(fin, sv.reindex(list(zip(F.year, F.voter, F.entry))).values, np.nan)
    F["semi_y_mean"] = np.where(fin, sm.reindex(list(zip(F.year, F.entry))).values, np.nan)
    F["has_semi"] = F.semi_y_mean.notna().astype(int)
    return F


def main():
    votes, results, contests = load_tables()
    grids = build_grids(votes, results)
    F, _ = build_features(grids, results, contests)
    F.to_parquet(PROC / "features.parquet")
    grids.to_parquet(PROC / "grids.parquet")
    print(F.shape, F.year.min(), F.year.max())
    print(F[FEATURES + SEMI_FEATURES].isna().mean().round(2).to_string())


if __name__ == "__main__":
    main()
