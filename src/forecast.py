"""Forecast engine for the predictor app.

Two stages, both Ridge models on voter -> entry points share:
  early       history only: affinity, form, running order, host flags
  after_semis adds whether the entry came through a semi-final and its semi-final placing (a number a user can enter)

Evaluation (reported in the app) uses the leak-free features: dev years 2010-2024 rolling origin, hold-out 2025-2026 sealed.
The live forecast uses every contest up to 2026, since the evaluation is done.

Chances come from the backtest, not from assumptions: a logistic regression of past outcomes (win, top 5, top 10) on the
projected placing, fitted on the dev finals 2010-2024.
"""
import json
import pickle

import numpy as np
import pandas as pd

import features
from features import FEATURES, PROC, load_tables, build_grids
from models import DEV_YEARS, HOLDOUT_YEARS, fit_predict
from scipy.stats import spearmanr

STAGES = {"early": FEATURES, "after_semis": FEATURES + ["has_semi", "semi_pos"]}
BIG5 = {"FR", "DE", "IT", "ES", "GB"}
LIVE_CUTOFF = 2026


def semi_positions(results):
    s = results[(results["round"] != "final") & (results.disqualified == 0)].copy()
    s["n"] = s.groupby(["year", "round"]).place.transform("count")
    s["semi_pos"] = 1 - (s.place - 1) / (s.n - 1)  # 1 = won the semi, 0 = last
    return s.groupby(["year", "country"]).semi_pos.first()


def add_semi_pos(F, results):
    sp = semi_positions(results)
    F["semi_pos"] = np.where(F.is_final == 1, sp.reindex(list(zip(F.year, F.entry))).values, np.nan)
    return F


def train_rows(F, stage):
    return F[F.is_final == 1] if stage == "after_semis" else F


def voter_budget(voter):
    """Points a voter awards in a modern final: 1-12 to ten entries from the jury and again from the televote (58 + 58);
    the rest-of-world televote awards one set (58)."""
    return np.where(voter == "WLD", 58.0, 116.0)


def allocate(P, col, keys):
    """Turn predicted shares into points that respect each voter's fixed budget: within each voter (and show),
    points are proportional to the predicted share. Rankings in testing and in the live forecast both use this."""
    share = P[col].clip(lower=0)
    tot = share.groupby([P[k] for k in keys] + [P.voter]).transform("sum").replace(0, np.nan)
    return (share / tot * voter_budget(P.voter.values)).fillna(0.0)


def backtest(F, results):
    """Rolling-origin predictions for finals with both stage models, plus the pre-semi-final model on semi-finals
    (for qualification chances). Returns (final entries, semi-final entries)."""
    known = F[F.year <= features.CUTOFF_YEAR]; out, semis = [], []
    for T in DEV_YEARS + HOLDOUT_YEARS:
        tr = known if T > features.CUTOFF_YEAR else known[known.year < T]
        te = F[(F.year == T) & (F.is_final == 1)].copy()
        for stage, feats in STAGES.items():
            te[stage], _ = fit_predict("ridge", train_rows(tr, stage), te, feats)
            te[stage] = allocate(te, stage, ["year"])
        out.append(te)
        ts = F[(F.year == T) & (F.is_final == 0)].copy()
        if len(ts):
            ts["early"], _ = fit_predict("ridge", tr, ts, STAGES["early"])
            ts["early"] = allocate(ts, "early", ["year", "round"])
            semis.append(ts)
    P = pd.concat(out)
    E = P.groupby(["year", "contestant_id", "entry"], as_index=False).agg(place=("place", "first"), **{s: (s, "sum") for s in STAGES})
    for s in STAGES:
        E[f"{s}_rank"] = E.groupby("year")[s].rank(ascending=False, method="first")
    S = pd.concat(semis).groupby(["year", "round", "contestant_id", "entry"], as_index=False).early.sum()
    S["pct"] = S.groupby(["year", "round"]).early.rank(pct=True)
    q = results.set_index(["year", "round", "contestant_id"]).qualified
    S["qualified"] = q.reindex(list(zip(S.year, S["round"], S.contestant_id))).values
    return E, S.dropna(subset=["qualified"])


def calibration(E):
    """Probabilities and likely finishing range by projected placing, fitted on the dev finals (2010-2024).

    P(win), P(top 5), P(top 10): logistic regression of the outcome on log(projected placing), fitted on every entry
    at once, so the curve is smooth, falls steadily with placing and never reaches exactly 0 or 1.
    Likely range: 20th-80th percentile of actual placings for entries within +-2 of each projected placing,
    made monotone (a better projection never has a worse range)."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    dev = E[E.year <= features.CUTOFF_YEAR]; rows = []
    ranks = np.arange(1, 28)
    for stage in STAGES:
        r = dev[f"{stage}_rank"].values
        X = np.log(r)[:, None]
        probs = {}
        for col, k in (("p_win", 1), ("p_top5", 5), ("p_top10", 10)):
            m = LogisticRegression(C=1e6).fit(X, (dev.place <= k).astype(int))
            probs[col] = m.predict_proba(np.log(ranks)[:, None])[:, 1]
        for i, k in enumerate(ranks):
            g = dev[(r >= k - 2) & (r <= k + 2)]
            rows.append(dict(stage=stage, pred_rank=k, n=len(g), **{c: v[i] for c, v in probs.items()},
                             place_med=g.place.median(), place_lo=g.place.quantile(0.2), place_hi=g.place.quantile(0.8)))
    C = pd.DataFrame(rows)
    for stage, g in C.groupby("stage"):
        for col in ("place_med", "place_lo", "place_hi"):
            C.loc[g.index, col] = IsotonicRegression(increasing=True).fit_transform(g.pred_rank, g[col].ffill(), sample_weight=g.n.clip(lower=1))
    return C


def accuracy(E):
    out = {}
    for stage in STAGES:
        for split, g in (("dev", E[E.year <= features.CUTOFF_YEAR]), ("holdout", E[E.year > features.CUTOFF_YEAR])):
            per = []
            for y, gy in g.groupby("year"):
                w = gy[gy.place == 1]
                per.append(dict(rho=spearmanr(gy[stage], -gy.place).statistic, winner_rank=int(w[f"{stage}_rank"].iloc[0]),
                                top5=len(set(gy.nsmallest(5, f"{stage}_rank").entry) & set(gy.nsmallest(5, "place").entry)),
                                top10=len(set(gy.nsmallest(10, f"{stage}_rank").entry) & set(gy.nsmallest(10, "place").entry))))
            p = pd.DataFrame(per)
            out[f"{stage}_{split}"] = dict(finals=len(p), rank_corr=p.rho.mean(), winner_top3=int((p.winner_rank <= 3).sum()),
                                           winner_named=int((p.winner_rank == 1).sum()), top5=p.top5.mean(), top10=p.top10.mean(),
                                           winner_ranks=p.winner_rank.tolist())
    return out


def live_state(votes, results, contests):
    """Affinity and form as of after the last contest, for every recent country pair."""
    features.CUTOFF_YEAR = LIVE_CUTOFF
    grids = build_grids(votes, results)
    recent = results[results.year >= 2016]
    countries = sorted(set(recent.country.dropna()))
    voters = sorted(set(grids[grids.year >= 2016].voter))
    year = LIVE_CUTOFF + 1
    fake = pd.DataFrame([(v, e, 90000 + i, 0.0, "final", year, 12.0, np.nan) for i, e in enumerate(countries) for v in voters if v != e],
                        columns=["voter", "entry", "contestant_id", "points", "round", "year", "denom", "y"])
    c2 = pd.concat([contests, pd.DataFrame([{"year": year, "host": None}])], ignore_index=True)
    F, _ = features.build_features(pd.concat([grids, fake], ignore_index=True), results, c2)
    live = F[F.year == year]
    train = add_semi_pos(F[F.year <= LIVE_CUTOFF].copy(), results)
    features.CUTOFF_YEAR = 2024
    return live[["voter", "entry", "aff"]], live.groupby("entry").form.first(), train, countries, voters


def main():
    votes, results, contests = load_tables()
    # ---- evaluation on leak-free features (history frozen at 2024 for the hold-out)
    F = add_semi_pos(pd.read_parquet(PROC / "features.parquet"), results)
    E, S = backtest(F, results)
    cal, acc = calibration(E), accuracy(E)
    # qualification: logistic regression of qualifying on the entry's standing among its semi-final's entries (dev years)
    from sklearn.linear_model import LogisticRegression
    dev = S[S.year <= features.CUTOFF_YEAR]
    qm = LogisticRegression(C=1e6).fit(dev[["pct"]], dev.qualified.astype(int))
    hits = []
    for (y, r), g in S.groupby(["year", "round"]):
        k = int(g.qualified.sum()); hits.append((y, len(set(g.nlargest(k, "early").contestant_id) & set(g[g.qualified == 1].contestant_id)), k))
    h = pd.DataFrame(hits, columns=["year", "hit", "n"])
    acc["qualify_dev"] = dict(correct=int(h[h.year <= features.CUTOFF_YEAR].hit.sum()), total=int(h[h.year <= features.CUTOFF_YEAR].n.sum()),
                              base_rate=float(dev.qualified.mean()))
    acc["qualify_holdout"] = dict(correct=int(h[h.year > features.CUTOFF_YEAR].hit.sum()), total=int(h[h.year > features.CUTOFF_YEAR].n.sum()))
    (PROC / "forecast_qualify.json").write_text(json.dumps(dict(coef=float(qm.coef_[0][0]), intercept=float(qm.intercept_[0]))))
    S.to_parquet(PROC / "forecast_backtest_semis.parquet")
    cal.to_parquet(PROC / "forecast_calibration.parquet"); E.to_parquet(PROC / "forecast_backtest.parquet")
    # ---- live models on everything up to 2026
    aff, form, train, countries, voters = live_state(votes, results, contests)
    models = {}
    for stage, feats in STAGES.items():
        tr = train_rows(train, stage)
        mu = tr[feats].mean().fillna(0.0)
        _, m = fit_predict("ridge", tr, tr.head(2), feats)
        models[stage] = dict(feats=feats, mu=mu.to_dict(), model=m)
    pickle.dump(models, open(PROC / "forecast_models.pkl", "wb"))
    aff.to_parquet(PROC / "forecast_affinity.parquet"); form.rename("form").to_frame().to_parquet(PROC / "forecast_form.parquet")
    last = results[(results.year == LIVE_CUTOFF) & (results.disqualified == 0)]
    fin = last[last["round"] == "final"].sort_values("running")
    via = last[last["round"] != "final"].set_index("country")
    meta = dict(countries=countries, voters=voters, last_year=LIVE_CUTOFF,
                last_winner=fin.sort_values("place").country.iloc[0], last_host=contests.set_index("year").host.get(LIVE_CUTOFF),
                last_finalists=fin.country.tolist(), last_participants=sorted(last.country.unique()),
                last_semi_place={c: int(via.place.get(c)) for c in fin.country if c in via.index},
                accuracy=acc)
    (PROC / "forecast_meta.json").write_text(json.dumps(meta, indent=2, default=float))
    print(json.dumps(acc, indent=1, default=lambda x: round(float(x), 3)))
    print(cal[cal.pred_rank <= 6].round(2).to_string())


# ---------------------------------------------------------------- used by the app
def predict(finalists, host, stage, running=None, semi=None, voters=None):
    """voters: participating countries (each votes in the final); countries that withdraw neither perform nor vote."""
    """finalists: list of country codes. running: {country: slot 1..N}. semi: {country: semi place (1 = won)} for semi qualifiers,
    'direct' for Big 5 / host. Unknown values fall back to the training average. Returns (entries, pairs)."""
    models = pickle.load(open(PROC / "forecast_models.pkl", "rb"))
    A = pd.read_parquet(PROC / "forecast_affinity.parquet").set_index(["voter", "entry"]).aff
    form = pd.read_parquet(PROC / "forecast_form.parquet").form
    meta = json.loads((PROC / "forecast_meta.json").read_text())
    running, semi = running or {}, semi or {}
    # only participating countries vote (plus the rest-of-world televote); defaults to last year's participants
    voters = sorted(set(voters or meta["last_participants"] + ["WLD"]) | set(finalists))
    n = len(finalists)
    P = pd.DataFrame([dict(voter=v, entry=e) for e in finalists for v in voters if v != e])
    P["aff"] = A.reindex(list(zip(P.voter, P.entry))).fillna(0.0).values
    P["form"] = P.entry.map(form)
    P["running_pct"] = P.entry.map(lambda e: running[e] / n if running.get(e) else np.nan)
    P["is_final"] = 1
    P["entry_is_host"] = (P.entry == host).astype(int)
    P["voter_is_host"] = (P.voter == host).astype(int)
    direct = lambda e: e in BIG5 or e == host or semi.get(e) == "direct"
    P["has_semi"] = P.entry.map(lambda e: 0 if direct(e) else 1)
    # semi place -> position score, assuming a semi of about 16 entries
    P["semi_pos"] = P.entry.map(lambda e: np.nan if direct(e) or not isinstance(semi.get(e), (int, float)) else 1 - (semi[e] - 1) / 15)
    spec = models[stage]
    X = P[spec["feats"]].astype(float).fillna(pd.Series(spec["mu"]))
    P["pred"] = np.clip(spec["model"].predict(X), 0, None)
    P["exp_points"] = allocate(P, "pred", [])
    # contributions of each input, relative to the average finalist, in points
    sc, rg = spec["model"][0], spec["model"][-1]
    Z = (X - sc.mean_) / sc.scale_
    for f, c in zip(spec["feats"], rg.coef_):
        P[f"c_{f}"] = Z[f] * c
    # contributions: model units -> points, using each voter's own share-to-points factor
    factor = (P.exp_points / P.pred.replace(0, np.nan)).groupby(P.voter).transform("median").fillna(0.0)
    for f in spec["feats"]:
        P[f"c_{f}"] = P[f"c_{f}"] * factor
    E = P.groupby("entry").agg(points=("exp_points", "sum"), **{f"c_{f}": (f"c_{f}", "sum") for f in spec["feats"]})
    for f in spec["feats"]:
        E[f"c_{f}"] = E[f"c_{f}"] - E[f"c_{f}"].mean()  # relative to the average finalist
    E = E.sort_values("points", ascending=False)
    E["pred_rank"] = range(1, len(E) + 1)
    cal = pd.read_parquet(PROC / "forecast_calibration.parquet")
    cal = cal[cal.stage == stage].set_index("pred_rank")
    for c in ["p_win", "p_top5", "p_top10", "place_med", "place_lo", "place_hi"]:
        E[c] = E.pred_rank.map(cal[c])
    E[["p_win", "p_top5", "p_top10"]] = E[["p_win", "p_top5", "p_top10"]].fillna(0.0)
    # the field's probabilities must add up: one winner, five top-5 places, ten top-10 places
    E["p_win"] = E.p_win / E.p_win.sum()
    E["p_top5"] = (E.p_top5 * min(5, len(E)) / E.p_top5.sum()).clip(upper=1)
    E["p_top10"] = (E.p_top10 * min(10, len(E)) / E.p_top10.sum()).clip(upper=1)
    return E.reset_index(), P



def qualification(participants, host, voters=None):
    """Pre-semi-final chance of reaching the final for every participant that has to qualify.
    Each entry's projected standing among the countries competing in the semi-finals (percentile) is converted with the
    logistic regression fitted on past semi-finals. Big 5 and host qualify automatically (probability 1)."""
    q = json.loads((PROC / "forecast_qualify.json").read_text())
    semi = [c for c in participants if c not in BIG5 and c != host]
    if len(semi) < 2:
        return pd.Series(1.0, index=participants)
    E, _ = predict(semi, host, "early", voters=voters)
    pct = E.set_index("entry").points.rank(pct=True)
    p = 1 / (1 + np.exp(-(q["intercept"] + q["coef"] * pct)))
    p = (p * min(20, len(p)) / p.sum()).clip(upper=0.97)  # 20 qualifying places (10 per semi-final) since 2008
    out = pd.Series(1.0, index=participants); out[p.index] = p
    return out


if __name__ == "__main__":
    main()