import json

import pandas as pd
import streamlit as st

from src.app_utils import PROC, require_data, table
from ui import explain, hero, lead, method, stats, theme

require_data()
theme()
meta = json.loads((PROC / "forecast_meta.json").read_text())
acc = meta["accuracy"]
e, a = acc["early_dev"], acc["after_semis_dev"]
qd0 = acc["qualify_dev"]
hero("Methodology", "Accuracy", "Each past final was forecast using only data from earlier years, and the 2025–2026 contests were "
     "withheld entirely until final testing.")
stats([("Winner in top 3", f"{a['winner_top3']} of {a['finals']}", "Post-semi-final, 2010–2024"),
       ("Top 5 identified", f"{a['top5']:.1f} of 5", f"Post-semi-final (pre-semi-final: {e['top5']:.1f}; chance: ≈1)"),
       ("Qualifiers identified", f"{qd0['correct'] / qd0['total'] * 100:.0f}%", f"Pre-semi-final (random pick: {qd0['base_rate'] * 100:.0f}%)"),
       (f"{meta['last_year']} test", f"Winner ranked #{acc['after_semis_holdout']['winner_ranks'][-1]}",
        f"Post-semi-final projection of a withheld contest (pre-semi-final: #{acc['early_holdout']['winner_ranks'][-1]})"),
       ("Rank correlation", f"{a['rank_corr']:.2f}", f"Post-semi-final (pre-semi-final: {e['rank_corr']:.2f})")])

st.markdown("### Detailed results")
rows = []
for stage, label in (("early", "Pre-semi-final"), ("after_semis", "Post-semi-final")):
    d, h = acc[f"{stage}_dev"], acc[f"{stage}_holdout"]
    rows.append({"Stage": label, "Winner in predicted top 3": f"{d['winner_top3']} of {d['finals']}", "Top 5 identified": f"{d['top5']:.1f} of 5",
                 "Top 10 identified": f"{d['top10']:.1f} of 10", "Rank correlation": f"{d['rank_corr']:.2f}",
                 f"Winner's predicted rank {meta['last_year'] - 1} / {meta['last_year']}": " / ".join(str(r) for r in h["winner_ranks"])})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
qd, qh = acc["qualify_dev"], acc["qualify_holdout"]
st.dataframe(pd.DataFrame([{"Forecast": "Semi-final qualification (pre-semi-final)",
                            "Qualifiers identified, 2010–2024": f"{qd['correct']} of {qd['total']} ({qd['correct'] / qd['total'] * 100:.0f}%)",
                            "Random pick": f"{qd['base_rate'] * 100:.0f}%",
                            f"Withheld {meta['last_year'] - 1}–{meta['last_year']}": f"{qh['correct']} of {qh['total']} ({qh['correct'] / qh['total'] * 100:.0f}%)"}]),
             hide_index=True, width="stretch")
explain("Indicates the reliability of the forecast.",
        "Before the semi-finals the forecast identifies about twice as many of the actual top 5 as chance, but cannot single out the winner, as song "
        "quality is not yet observable. After the semi-finals the winner is usually among the top three projections. Note that pre-semi-final placing "
        "accuracy is measured on the actual finalists; in practice the finalists are not yet known, so it is slightly optimistic. Qualification is "
        "tested separately in each semi-final.", "Out-of-sample", "strong")

st.markdown("### Model comparison")
pl = table("backtest_pair_level").pivot(index="model", columns="split", values="pair_corr")
bm = table("backtest_metrics")
fr = bm[bm["round"] == "final"].groupby(["model", "split"]).agg(rho=("spearman", "mean"), top5=("top5", "mean")).unstack()
qa = bm[bm["round"] != "final"].groupby(["model", "split"]).apply(lambda g: g.qual_hit.sum() / g.n_qual.sum(), include_groups=False).unstack()
names = {"baseline": "Baseline: recent form only", "ridge_no_affinity_pre_contest": "Ridge, without voting affinity",
         "ridge_pre_contest": "Ridge, pre-semi-final (chosen)", "lgbm_pre_contest": "Gradient boosting (LightGBM), pre-semi-final",
         "ridge_pre_final": "Ridge, post-semi-final (chosen)", "lgbm_pre_final": "Gradient boosting (LightGBM), post-semi-final"}
cmp = pd.DataFrame({"Model": [names[m] for m in names],
                    "Pair-level r (2010–2024)": [pl.loc[m, "dev"] for m in names],
                    "Pair-level r (withheld)": [pl.loc[m, "holdout"] for m in names],
                    "Final rank correlation": [fr.loc[m, ("rho", "dev")] for m in names],
                    "Top 5 found (of 5)": [fr.loc[m, ("top5", "dev")] for m in names],
                    "Qualifiers found": [f"{qa.loc[m, 'dev'] * 100:.0f}%" if m in qa.index else "–" for m in names]})
st.dataframe(cmp.round(2), hide_index=True, width="stretch")
explain("Compares the chosen model with simpler and more complex alternatives on identical, unseen test years.",
        "Pair-level r: correlation between predicted and actual points for every voter→entry pair (the model's direct target). Removing voting affinity "
        f"drops it from 0.48 to 0.28, and the gain holds on the withheld years. Gradient boosting is not reliably better: slightly ahead after the semi-finals "
        f"on 2010–2024, equal on the withheld years, and behind before the semi-finals, while being harder to explain, so ridge was chosen. For "
        f"qualification the gain over the form baseline is small ({qa.loc['ridge_pre_contest', 'dev'] * 100:.0f}% vs {qa.loc['baseline', 'dev'] * 100:.0f}%; "
        f"random pick {acc['qualify_dev']['base_rate'] * 100:.0f}%), which is stated as a limitation.",
        "Out-of-sample", "strong")

st.markdown("### Method")
lead("<b>1.</b> Estimate, from voting since 2004, the additional points each country awards every other country.<br>"
     "<b>2.</b> Project each country's points to each finalist from these relationships, recent results, running order, host status and, "
     "where available, semi-final placing.<br>"
     "<b>3.</b> Share each voting country's 116 points (58 jury, 58 televote) in proportion to its predicted preferences, and sum them into a scoreboard.<br>"
     "<b>4.</b> Convert projected placings into probabilities with a logistic regression of past outcomes (win, top 5, top 10) on projected placing.")
explain("Documents how the forecast is produced.",
        "The model is a ridge regression. It performed as well as more complex methods (gradient boosting) in testing, and its contributions "
        "can be shown directly, as in the Forecast page's contributing-factors chart.")

st.markdown("### Limitations")
explain("Sets out what the forecast cannot capture.",
        "<b>Song quality</b> before any public vote. <b>Live performance</b> on the night. <b>One-off events</b>: wars have a measurable effect "
        "(see Jury and televote), but too few cases exist to adjust for them. <b>Political sentiment</b> is reflected only through past results and "
        "participation decisions.")
st.caption("Data: the unofficial Eurovision Song Contest dataset (2004–2026), Eurostat migrant stock, CEPII GeoDist, UCDP Armed Conflict Dataset.")

st.markdown("### Model specification")
method([
    ("Target", "Points each voting country awards each entry, as a share of the maximum (12, or 24 when jury and televote are combined)."),
    ("Model", "Ridge regression (L2 penalty α = 10) on standardised inputs, fitted separately for the pre-semi-final and post-semi-final stages. "
              "Gradient boosting (LightGBM) was tested and performed no better."),
    ("Inputs", "Voting affinity (leave-one-out residual, 10-year half-life, shrinkage pseudo-count 3); recent form (4-year half-life); position in "
               "the running order; host indicators; final indicator. Post-semi-final adds semi-final qualification and placing."),
    ("Aggregation", "Each voting country's predicted shares are scaled to its real points budget (116; rest-of-world televote 58) and summed "
                    "over all participating countries, giving projected points and placings. The same aggregation is used in testing."),
    ("Qualification", "Pre-semi-final model applied to the semi-final entries; each entry's projected standing (percentile) is converted to a "
                      "probability by logistic regression on past semi-finals, then scaled so the 20 qualifying places add up."),
    ("Probabilities", "Logistic regression of the outcome (win, top 5, top 10) on log(projected placing), fitted on 14 finals (2010–2024) and "
                      "normalised so the field's probabilities sum to 1, 5 and 10. Likely range: 20th–80th percentile of actual placings for "
                      "entries within ±2 places of the projection."),
    ("Validation", "Rolling-origin evaluation: each final 2010–2024 predicted from earlier years only. Hold-out: 2025 and 2026 were withheld from "
                   "fitting, feature history, tuning and calibration. Baseline: each country's recent form alone."),
])
