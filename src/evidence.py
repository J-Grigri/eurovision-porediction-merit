"""Defensible statistical evidence (2004-2024 only; 2025-2026 stay sealed).

Each test is designed so that it cannot be explained by song quality or by fitting and testing on the same data:
  1. Stability      affinity measured in 2004-2013 predicts affinity in 2014-2024 (split-half)
  2. Reciprocity    if A favours B, does B favour A? (permutation test)
  3. Bloc validity  blocs found in 2004-2013 still favour each other in 2014-2024 (out-of-sample)
  4. Jury vs public same voter, same songs, same year: is bias stronger in the televote than in the jury? (2016-2024)
  5. Friends in the semi   entries whose semi-final contains more historically friendly voters qualify more often
  6. Songwriter track record   writers whose earlier entries did well write entries that do well (quality proxy)
"""
import json

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr, spearmanr

from features import CUTOFF_YEAR, PROC, load_tables

RNG = np.random.default_rng(0)
N_PERM = 2000


def residual_grid(votes, results, kind, max_pts=12.0):
    """Full voter x entry grid for one kind of vote, with the leave-one-out residual (bias net of song quality)."""
    v = votes[(votes.kind == kind) & (votes.year >= 2004) & (votes.year <= CUTOFF_YEAR) & (votes.voter != "WLD")]
    res = results[results.disqualified == 0]
    out = []
    for (year, rnd), vt in v.groupby(["year", "round"]):
        rr = res[(res.year == year) & (res["round"] == rnd)]
        pts = vt.pivot_table(index="voter", columns="contestant_id", values="points", aggfunc="sum")
        pts = pts.reindex(columns=rr.contestant_id.tolist()).fillna(0.0)
        g = pts.stack().rename("points").reset_index()
        g["entry"] = g.contestant_id.map(dict(zip(rr.contestant_id, rr.country)))
        g = g[g.voter != g.entry].copy()
        g["year"], g["round"] = year, rnd
        out.append(g)
    g = pd.concat(out, ignore_index=True)
    g["y"] = g.points / (max_pts if kind != "total" else 1.0)
    if kind == "total":  # total = jury + public from 2016 (max 24), otherwise max 12
        g["y"] = g.points / np.where(g.year >= 2016, 24.0, 12.0)
    k = ["year", "round", "contestant_id"]
    s, n = g.groupby(k).y.transform("sum"), g.groupby(k).y.transform("count")
    g["resid"] = g.y - (s - g.y) / (n - 1)
    return g[n >= 5]


def pair_mean(g, min_n=1):
    a = g.groupby(["voter", "entry"]).resid.agg(["mean", "count"])
    return a[a["count"] >= min_n]["mean"]


def perm_p(observed, null):
    return (1 + np.sum(np.asarray(null) >= observed)) / (1 + len(null))


def test_stability(g):
    a = pair_mean(g[g.year <= 2013], 3).rename("early")
    b = pair_mean(g[g.year >= 2014], 3).rename("late")
    d = pd.concat([a, b], axis=1).dropna()
    r = pearsonr(d.early, d.late)
    return {"pairs": len(d), "r": r.statistic, "p": r.pvalue}, d.reset_index()


def test_reciprocity(g):
    a = pair_mean(g, 5)
    d = pd.DataFrame({"ab": a})
    d["ba"] = [a.get((e, v), np.nan) for v, e in d.index]
    d = d.dropna()
    d = d[[v < e for v, e in d.index]]  # each unordered pair once
    r = pearsonr(d.ab, d.ba).statistic
    null = [pearsonr(d.ab, RNG.permutation(d.ba.values)).statistic for _ in range(N_PERM)]
    return {"pairs": len(d), "r": r, "p_perm": perm_p(r, null)}, d.reset_index()


def blocs_from(g, k=7, min_obs=150):
    a = g.groupby(["voter", "entry"]).resid.mean().unstack()
    cnt = g.groupby("voter").size().add(g.groupby("entry").size(), fill_value=0)
    keep = [c for c in a.index if c in a.columns and cnt.get(c, 0) >= min_obs]
    M = a.loc[keep, keep].fillna(0.0).values
    S = (M + M.T) / 2
    D = np.clip(1 - np.corrcoef(S), 0, None); np.fill_diagonal(D, 0)
    lab = fcluster(linkage(squareform((D + D.T) / 2, checks=False), "average"), k, "maxclust")
    return dict(zip(keep, lab))


def test_bloc_out_of_sample(g):
    bl = blocs_from(g[g.year <= 2013])
    late = g[(g.year >= 2014) & g.voter.isin(bl) & g.entry.isin(bl)].copy()
    same = (late.voter.map(bl) == late.entry.map(bl)).values
    obs = late.resid.values[same].mean() - late.resid.values[~same].mean()
    # null: shuffle bloc labels across countries, keeping bloc sizes
    cs, labs = np.array(list(bl)), np.array(list(bl.values()))
    vi, ei = late.voter.values, late.entry.values
    null = []
    for _ in range(N_PERM):
        m = dict(zip(cs, RNG.permutation(labs)))
        s = np.array([m[v] == m[e] for v, e in zip(vi, ei)])
        null.append(late.resid.values[s].mean() - late.resid.values[~s].mean())
    return {"countries": len(bl), "within_minus_between": obs, "null_95pct": float(np.percentile(null, 95)), "p_perm": perm_p(obs, null),
            "within_pts12": obs * 12}, bl


def test_jury_public(votes, results):
    j = residual_grid(votes, results, "jury")
    p = residual_grid(votes, results, "public")
    j = j[j.year >= 2016]; p = p[p.year >= 2016]
    a = pd.concat([pair_mean(j, 3).rename("jury"), pair_mean(p, 3).rename("public")], axis=1).dropna()
    r = pearsonr(a.jury, a.public)
    # is bias stronger in the televote? compare the spread of pair affinity, with a bootstrap over pairs
    sd_j, sd_p = a.jury.std(), a.public.std()
    bs = [a.sample(len(a), replace=True, random_state=i) for i in range(1000)]
    ratio = np.array([b.public.std() / b.jury.std() for b in bs])
    top = a.assign(gap=a.public - a.jury).sort_values("gap", ascending=False)
    return {"pairs": len(a), "r_jury_public": r.statistic, "sd_jury": sd_j, "sd_public": sd_p, "ratio": sd_p / sd_j,
            "ratio_lo": float(np.percentile(ratio, 2.5)), "ratio_hi": float(np.percentile(ratio, 97.5))}, top.reset_index()


def test_friends_in_semi():
    """Uses the leak-free `aff` feature: affinity learned only from earlier years."""
    F = pd.read_parquet(PROC / "features.parquet")
    res = pd.read_parquet(PROC / "results.parquet")
    s = F[(F.is_final == 0) & (F.year <= CUTOFF_YEAR)]
    e = s.groupby(["year", "round", "contestant_id", "entry"], as_index=False).agg(
        friends=("aff", lambda x: (x > 0.15).sum()), support=("aff", "mean"), form=("form", "first"))
    q = res.set_index(["year", "round", "contestant_id"]).qualified
    e["qualified"] = q.reindex(list(zip(e.year, e["round"], e.contestant_id))).values
    e = e.dropna(subset=["qualified"])
    e["band"] = pd.cut(e.friends, [-1, 0, 2, 4, 100], labels=["0", "1-2", "3-4", "5+"])
    tab = e.groupby("band", observed=True).qualified.agg(rate="mean", n="count").reset_index()
    rho = spearmanr(e.friends, e.qualified)
    # does it add to the country's form? compare within form terciles
    e["form_t"] = pd.qcut(e.form.rank(method="first"), 3, labels=["low form", "mid form", "high form"])
    within = e.assign(friends_3plus=e.friends >= 3).groupby(["form_t", "friends_3plus"], observed=True).qualified.agg(["mean", "count"]).reset_index()
    return {"entries": len(e), "rho_friends_qualified": rho.statistic, "p": rho.pvalue}, tab, within


def test_songwriters(results):
    """Quality proxy: did the entry's writers/composers/artist have earlier entries, and how did those do?"""
    c = pd.read_parquet(PROC / "contestants_full.parquet")
    res = results[(results["round"] == "final") & (results.disqualified == 0) & results.points_total.notna()].copy()
    res["n"] = res.groupby("year").place.transform("count")
    res["score"] = 1 - (res.place - 1) / (res.n - 1)  # 1 = winner, 0 = last
    semi = results[(results["round"] != "final") & (results.disqualified == 0)]
    c = c.merge(res[["year", "contestant_id", "score"]].rename(columns={"contestant_id": "id"}), on=["year", "id"], how="left")
    dnq = set(zip(semi[semi.qualified == 0].year, semi[semi.qualified == 0].contestant_id))
    c.loc[[(y, i) in dnq for y, i in zip(c.year, c.id)], "score"] = 0.0  # failing to qualify counts as bottom
    split = lambda s: [x.strip() for x in s.split(";") if x.strip()] if isinstance(s, str) else []
    c["people"] = [set(split(w) + split(cp) + split(l)) for w, cp, l in zip(c.writers, c.composers, c.lyricists)]
    c["artist_key"] = c.artist.str.lower().str.strip()
    hist, art_hist, rows = {}, {}, []
    for year in sorted(c.year.unique()):
        cy = c[c.year == year]
        for _, r in cy.iterrows():
            past = [s for p in r.people for s in hist.get(p, [])]
            ap = art_hist.get(r.artist_key, [])
            rows.append({"year": year, "id": r.id, "score": r.score, "writer_prior_n": len(past),
                         "writer_prior_mean": np.mean(past) if past else np.nan, "artist_returning": int(bool(ap)),
                         "artist_prior_mean": np.mean(ap) if ap else np.nan})
        if year > CUTOFF_YEAR:
            continue  # hold-out results never enter the history
        for _, r in cy.dropna(subset=["score"]).iterrows():
            for p in r.people:
                hist.setdefault(p, []).append(r.score)
            art_hist.setdefault(r.artist_key, []).append(r.score)
    d = pd.DataFrame(rows)
    d[["year", "id", "writer_prior_n", "writer_prior_mean"]].rename(columns={"id": "contestant_id"}).to_parquet(PROC / "writer_history.parquet")
    d = d[(d.year >= 2004) & (d.year <= CUTOFF_YEAR)].dropna(subset=["score"])
    w = d.dropna(subset=["writer_prior_mean"])
    r_w = pearsonr(w.writer_prior_mean, w.score)
    a = d.dropna(subset=["artist_prior_mean"])
    r_a = pearsonr(a.artist_prior_mean, a.score)
    d["writer_band"] = np.select([d.writer_prior_n == 0, d.writer_prior_mean < 0.5, d.writer_prior_mean >= 0.5],
                                 ["New writers", "Writers with weak record", "Writers with strong record"], "New writers")
    tab = d.groupby("writer_band").score.agg(mean_score="mean", n="count").reset_index()
    return {"entries": len(d), "with_writer_history": len(w), "r_writer": r_w.statistic, "p_writer": r_w.pvalue,
            "returning_artists": len(a), "r_artist": r_a.statistic, "p_artist": r_a.pvalue}, tab


def test_host(results, contests):
    """Is the host penalty real, or regression to the mean / the no-semi-final route? Hosts are almost always last year's winner."""
    r = results[(results["round"] == "final") & (results.disqualified == 0)].copy()
    host = dict(zip(contests.year, contests.host))
    r["host"] = [int(host.get(y) == c) for y, c in zip(r.year, r.country)]
    r["n"] = r.groupby("year").place.transform("count")
    wins = r[r.place == 1].groupby("year").country.apply(set)
    back_to_back = [dict(years=f"{y - 1}–{y}", country=c, hosted=int(host.get(y) == c)) for y, cs in wins.items() for c in cs if y - 1 in wins.index and c in wins[y - 1]]
    eras = r[r.host == 1].assign(era=pd.cut(r.year, [1955, 1996, 2003, 2026], labels=["1956–1996 (juries)", "1997–2003 (televote begins)", "2004–2026 (semi-finals era)"]),
                                 pct=lambda d: d.place / d.n, won=lambda d: (d.place == 1).astype(int)) \
        .groupby("era", observed=True).agg(hosts=("year", "count"), host_wins=("won", "sum"), median_place_pct=("pct", "median")).reset_index()
    f = r[(r.year >= 2004) & (r.year <= CUTOFF_YEAR)].copy()
    f["z"] = f.groupby("year").points_total.transform(lambda s: (s - s.mean()) / s.std())
    semi = results[results["round"] != "final"]; via = set(zip(semi.year, semi.country))
    f["direct_not_host"] = [int((y, c) not in via and h == 0) for y, c, h in zip(f.year, f.country, f.host)]
    prev_year = lambda y: 2019 if y == 2021 else y - 1
    lag = {y: f[f.year == prev_year(y)].set_index("country") for y in f.year.unique()}
    f["lag_z"] = [lag[y].z.get(c, np.nan) for y, c in zip(f.year, f.country)]
    f["prev_place"] = [lag[y].place.get(c, np.nan) for y, c in zip(f.year, f.country)]
    d = f.dropna(subset=["lag_z"])
    rows = []
    # Big 5 (automatic qualifiers other than the host): same regression, coefficient on direct_not_host
    A = np.column_stack([np.ones(len(d)), d.host, d.lag_z, d.direct_not_host]); yv = d.z.values
    b = np.linalg.lstsq(A, yv, rcond=None)[0][3]; yrs = d.year.values; u = np.unique(yrs); bs = []
    for _ in range(1000):
        idx = np.concatenate([np.where(yrs == y)[0] for y in RNG.choice(u, len(u))])
        bs.append(np.linalg.lstsq(A[idx], yv[idx], rcond=None)[0][3])
    big5 = dict(effect=float(b), lo95=float(np.percentile(bs, 2.5)), hi95=float(np.percentile(bs, 97.5)), n=int(d.direct_not_host.sum()))
    for name, X in [("Host only", ["host"]), ("+ last year's score", ["host", "lag_z"]), ("+ last year's score + no-semi route", ["host", "lag_z", "direct_not_host"])]:
        A = np.column_stack([np.ones(len(d))] + [d[c] for c in X]); b = np.linalg.lstsq(A, d.z, rcond=None)[0]
        yrs = d.year.values; u = np.unique(yrs); bs = []
        for _ in range(1000):
            idx = np.concatenate([np.where(yrs == y)[0] for y in RNG.choice(u, len(u))])
            bs.append(np.linalg.lstsq(A[idx], d.z.values[idx], rcond=None)[0][1])
        rows.append(dict(model=name, host_effect=b[1], lo95=np.percentile(bs, 2.5), hi95=np.percentile(bs, 97.5)))
    drop = [dict(group=g, n=int(m.sum()), last_year_z=d.lag_z[m].mean(), this_year_z=d.z[m].mean(), hosts=int(d.host[m].sum()))
            for g, m in [("Won last year", d.prev_place == 1), ("2nd–3rd last year", d.prev_place.isin([2, 3])), ("4th–10th last year", d.prev_place.between(4, 10))]]
    info = {"big5": big5, "hosts_2004_2024": int(f.host.sum()), "host_is_last_winner": int(((d.host == 1) & (d.prev_place == 1)).sum()), "host_wins_since_2004": int(((r.year >= 2004) & (r.host == 1) & (r.place == 1)).sum())}
    return info, pd.DataFrame(back_to_back), eras, pd.DataFrame(rows), pd.DataFrame(drop)


def test_language(results):
    """Does singing in English help? Final score (z within year) of English-language vs other entries, 2004-2024,
    with the difference's 95% CI from a bootstrap over years, plus the share of winners sung in English."""
    ent = pd.read_parquet(PROC / "entries.parquet")
    r = results[(results["round"] == "final") & (results.disqualified == 0) & (results.year >= 2004) & (results.year <= CUTOFF_YEAR)].copy()
    r["z"] = r.groupby("year").points_total.transform(lambda s: (s - s.mean()) / s.std())
    r = r.merge(ent[["year", "contestant_id", "is_english"]], on=["year", "contestant_id"])
    diff = lambda d: d[d.is_english == 1].z.mean() - d[d.is_english == 0].z.mean()
    yrs = r.year.unique(); bs = []
    for _ in range(1000):
        pick = RNG.choice(yrs, len(yrs)); bs.append(diff(pd.concat([r[r.year == y] for y in pick])))
    w = r[r.place == 1]
    return dict(diff=float(diff(r)), lo95=float(np.percentile(bs, 2.5)), hi95=float(np.percentile(bs, 97.5)),
                share_finalists_english=float(r.is_english.mean()), share_winners_english=float(w.is_english.mean()), winners=len(w))


def main():
    votes, results, contests = load_tables()
    g = residual_grid(votes, results, "total")
    out = {}
    out["stability"], stab = test_stability(g)
    out["reciprocity"], rec = test_reciprocity(g)
    out["blocs"], _ = test_bloc_out_of_sample(g)
    out["jury_public"], jp = test_jury_public(votes, results)
    out["friends"], fr_tab, fr_within = test_friends_in_semi()
    out["songwriters"], sw_tab = test_songwriters(results)
    out["host"], b2b, host_eras, host_models, host_drop = test_host(results, contests)
    out["language"] = test_language(results)
    b2b.to_parquet(PROC / "ev_back_to_back.parquet"); host_eras.to_parquet(PROC / "ev_host_eras.parquet")
    host_models.to_parquet(PROC / "ev_host_models.parquet"); host_drop.to_parquet(PROC / "ev_host_drop.parquet")
    print(b2b.to_string()); print(host_eras.round(2).to_string()); print(host_models.round(2).to_string()); print(host_drop.round(2).to_string())
    stab.to_parquet(PROC / "ev_stability.parquet"); rec.to_parquet(PROC / "ev_reciprocity.parquet")
    jp.to_parquet(PROC / "ev_jury_public_pairs.parquet"); fr_tab.to_parquet(PROC / "ev_friends.parquet")
    fr_within.to_parquet(PROC / "ev_friends_within.parquet"); sw_tab.to_parquet(PROC / "ev_songwriters.parquet")
    (PROC / "evidence.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=2, default=lambda x: round(float(x), 4)))
    print(fr_tab.round(3).to_string()); print(fr_within.round(3).to_string()); print(sw_tab.round(3).to_string())
    print(jp.head(12).round(3).to_string()); print(jp.tail(6).round(3).to_string())


if __name__ == "__main__":
    main()
