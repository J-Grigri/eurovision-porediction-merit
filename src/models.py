"""Prediction models and their tests.

Models (predict points share y for each voter -> entry pair, then average over voters into an entry score):
  baseline   the entry country's recent form only (the bar to beat)
  ridge      Ridge regression on the features (interpretable: coefficients)
  lgbm       gradient boosting on the same features (does non-linearity help?)

Two settings: pre_contest (no same-year results) and pre_final (finals only; adds the same year's semi-final results).

Testing: development years 2010-2024 are predicted one at a time from earlier years only. Hold-out years 2025-2026
are predicted by models trained on 2004-2024 and are never used for fitting or tuning.
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import CUTOFF_YEAR, FEATURES, PROC, SEMI_FEATURES

DEV_YEARS = [y for y in range(2010, CUTOFF_YEAR + 1) if y != 2020]
HOLDOUT_YEARS = list(range(CUTOFF_YEAR + 1, 2027))

SETTINGS = {"pre_contest": FEATURES, "pre_final": FEATURES + SEMI_FEATURES}
MODELS = {
    "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
    "lgbm": lambda: LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=8, min_child_samples=100, subsample=0.8,
                                  subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, verbose=-1, random_state=0),
}


def fit_predict(kind, train, test, feats):
    mu = train[feats].mean().fillna(0.0)  # missing values -> training average (0 if a feature has no data yet)
    m = MODELS[kind]().fit(train[feats].fillna(mu), train["y"])
    return m.predict(test[feats].fillna(mu)), m


def entry_table(F, pred_col):
    """Average the pair-level predictions over voters -> one row per entry per round."""
    return F.groupby(["year", "round", "contestant_id", "entry"], as_index=False).agg(
        pred=(pred_col, "mean"), actual=("y", "mean"), place=("place", "first"), qualified=("qualified", "first"))


def rank_metrics(e):
    rows = []
    for (year, rnd), g in e.groupby(["year", "round"]):
        g = g.dropna(subset=["pred"])
        if len(g) < 5:
            continue
        by_pred = g.sort_values("pred", ascending=False)
        row = dict(year=year, round=rnd, spearman=spearmanr(g.pred, -g.place).correlation)
        if rnd == "final":
            act = g.sort_values("place")
            row.update(winner_hit=int(by_pred.contestant_id.iloc[0] == act.contestant_id.iloc[0]),
                       top5=len(set(by_pred.contestant_id[:5]) & set(act.contestant_id[:5])))
        else:
            nq = int(g.qualified.sum())
            row.update(qual_hit=len(set(by_pred.contestant_id[:nq]) & set(g.contestant_id[g.qualified == 1])), n_qual=nq)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    F = pd.read_parquet(PROC / "features.parquet")
    res = pd.read_parquet(PROC / "results.parquet"); ent = pd.read_parquet(PROC / "entries.parquet")
    q = res.set_index(["year", "round", "contestant_id"]).qualified
    F["qualified"] = q.reindex(list(zip(F.year, F["round"], F.contestant_id))).values
    F["baseline"] = F.form.fillna(F.groupby("is_final").y.transform("mean"))
    known = F[F.year <= CUTOFF_YEAR]

    # ---- predict every development and hold-out year
    preds = []
    for T in DEV_YEARS + HOLDOUT_YEARS:
        train = known if T > CUTOFF_YEAR else known[known.year < T]  # hold-out: everything up to the cutoff
        te = F[F.year == T].copy()
        te["split"] = "holdout" if T > CUTOFF_YEAR else "dev"
        for kind in MODELS:
            for setting, feats in SETTINGS.items():
                tr_, te_ = (train[train.is_final == 1], te[te.is_final == 1]) if setting == "pre_final" else (train, te)
                col = f"{kind}_{setting}"
                te[col] = np.nan
                te.loc[te_.index, col], _ = fit_predict(kind, tr_, te_, feats)
        # ablation: the ridge model without the affinity feature
        no_aff = [f for f in FEATURES if f != "aff"]
        te["ridge_no_affinity_pre_contest"], _ = fit_predict("ridge", train, te, no_aff)
        preds.append(te)
    P = pd.concat(preds, ignore_index=True)
    P.to_parquet(PROC / "backtest_pairs.parquet")

    # ---- metrics
    cols = ["baseline"] + [f"{k}_{s}" for k in MODELS for s in SETTINGS] + ["ridge_no_affinity_pre_contest"]
    metrics, pair = [], []
    for col in cols:
        sub = P[P[col].notna()]
        if col.endswith("pre_final"):
            sub = sub[sub.is_final == 1]
        m = rank_metrics(entry_table(sub, col)); m["model"] = col
        m["split"] = np.where(m.year > CUTOFF_YEAR, "holdout", "dev"); metrics.append(m)
        for split, sp in sub.groupby("split"):
            pair.append(dict(model=col, split=split, pair_corr=np.corrcoef(sp[col], sp.y)[0, 1],
                             pair_rmse=np.sqrt(((sp[col] - sp.y) ** 2).mean()), naive_rmse=sp.y.std()))
    M, PL = pd.concat(metrics, ignore_index=True), pd.DataFrame(pair)
    M.to_parquet(PROC / "backtest_metrics.parquet"); PL.to_parquet(PROC / "backtest_pair_level.parquet")

    # ---- ridge coefficients (trained on everything up to the cutoff): the explanation
    coefs = []
    for setting, feats in SETTINGS.items():
        tr = known[known.is_final == 1] if setting == "pre_final" else known
        _, m = fit_predict("ridge", tr, tr.head(5), feats)
        coefs += [dict(setting=setting, feature=f, coef=c) for f, c in zip(feats, m[-1].coef_)]
    pd.DataFrame(coefs).to_parquet(PROC / "ridge_coefficients.parquet")

    # ---- entry-level scoreboard for the dashboard
    out = []
    for col in cols[:-1]:
        sub = P[P[col].notna()]
        if col.endswith("pre_final"):
            sub = sub[sub.is_final == 1]
        e = entry_table(sub, col); e["model"] = col; out.append(e)
    E = pd.concat(out, ignore_index=True)
    E = E.merge(res[["year", "round", "contestant_id", "points_total"]], on=["year", "round", "contestant_id"], how="left")
    E = E.merge(ent[["year", "contestant_id", "artist", "song", "country"]], on=["year", "contestant_id"], how="left")
    g = ["model", "year", "round"]
    E["pred_points"] = E.pred * E.groupby(g).points_total.transform("sum") / E.groupby(g).actual.transform("sum")  # points budget is fixed by the rules
    E["pred_rank"] = E.groupby(g).pred.rank(ascending=False, method="first")
    E.to_parquet(PROC / "predictions.parquet")

    # ---- console summary
    for split in ("dev", "holdout"):
        s = M[M.split == split]
        fin = s[s["round"] == "final"].groupby("model").agg(finals=("year", "count"), rank_corr=("spearman", "mean"), winners=("winner_hit", "sum"), top5=("top5", "mean"))
        sm = s[s["round"] != "final"].groupby("model").agg(h=("qual_hit", "sum"), n=("n_qual", "sum"))
        fin["qual_acc"] = sm.h / sm.n
        print(f"\n{split.upper()}\n", fin.round(3).to_string())
    print("\nPAIR LEVEL\n", PL.round(3).to_string())
    print("\nRIDGE COEFFICIENTS\n", pd.DataFrame(coefs).round(4).to_string())


if __name__ == "__main__":
    main()
