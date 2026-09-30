import plotly.express as px
import streamlit as st

from src.app_utils import table, require_data

st.set_page_config(page_title="Model", layout="wide")
require_data()
st.title("Prediction: how well does it work?")
st.markdown("""
**Task.** For every voter and every entry, predict the share of the maximum points the voter gives. Average over voters to get each entry's score, then rank.

| Model | Role |
|---|---|
| **Baseline** | The country's recent form only. The bar to beat. |
| **Ridge regression** | Simple linear model; its coefficients are the explanation. |
| **LightGBM** | Non-linear model on the same inputs: does extra flexibility help? |

**Inputs (6):** voter→country affinity, the country's recent form, running-order position, final vs semi, and whether the entry / voter is the host.
**Pre-final** adds 3 more from the same year's semi-final: the entry's semi score, how this voter scored it there, and a flag.

**Testing.** Development years (2010–2024) are predicted one at a time from earlier years only. Hold-out years (2025–2026) come from models trained on 2004–2024 and were never used for training, history or tuning.
""")
M, PL = table("backtest_metrics"), table("backtest_pair_level")
names = {"baseline": "Baseline", "ridge_pre_contest": "Ridge · pre-contest", "lgbm_pre_contest": "LightGBM · pre-contest",
         "ridge_pre_final": "Ridge · pre-final", "lgbm_pre_final": "LightGBM · pre-final", "ridge_no_affinity_pre_contest": "Ridge · pre-contest, no affinity"}


def summary(split):
    s = M[M.split == split]
    f = s[s["round"] == "final"].groupby("model").agg(finals=("year", "count"), rank_corr=("spearman", "mean"), winners_hit=("winner_hit", "sum"), top5_overlap=("top5", "mean"))
    q = s[s["round"] != "final"].groupby("model").agg(h=("qual_hit", "sum"), n=("n_qual", "sum"))
    f["qualifiers_correct"] = q.h / q.n
    return f.reindex(list(names)).rename(index=names)


st.subheader("Does affinity matter? Points from voter to entry")
p = PL.assign(model=PL.model.map(names)).pivot(index="model", columns="split", values="pair_corr").reindex(list(names.values())).round(3)
st.dataframe(p.rename(columns={"dev": "Development 2010–2024", "holdout": "Hold-out 2025–2026"}), width="stretch")
st.caption("Correlation between predicted and actual points share. Removing affinity drops the correlation from about 0.48 to 0.28–0.32, and the gain holds on unseen years. This is the project's most robust result.")

d1, d2 = st.columns(2)
d1.subheader("Final ranking: development"); d1.dataframe(summary("dev").round(3), width="stretch")
d2.subheader("Final ranking: hold-out"); d2.dataframe(summary("holdout").round(3), width="stretch")
st.caption("Rank corr = Spearman between predicted and actual final rank. Winners hit = winners named correctly (chance is about 1 in 25). Top-5 overlap is out of 5. "
           "Qualifiers correct = share of semi-final qualifiers named correctly. Pre-final models apply to finals only. Two hold-out finals is a tiny sample.")

st.subheader("What drives the Ridge predictions")
co = table("ridge_coefficients")
lab = {"aff": "Voter→country affinity", "form": "Country's recent form", "running_pct": "Later in running order", "is_final": "Final (vs semi)", "entry_is_host": "Entry is host",
       "voter_is_host": "Voter is host", "has_semi": "Went through a semi", "semi_from_voter": "Voter's semi-final score for entry", "semi_y_mean": "Entry's semi-final average"}
co = co.assign(feature=co.feature.map(lab))
st.plotly_chart(px.bar(co, x="coef", y="feature", color="setting", barmode="group", orientation="h",
                       labels={"coef": "Standardised coefficient", "feature": ""}), width="stretch")
st.caption("Coefficients are per one standard deviation of each input, so their sizes are comparable. Affinity is the largest pre-contest driver; after the semi-finals the semi-final score itself joins it.")

fin = M[M["round"] == "final"].query("model in ['baseline','ridge_pre_contest','ridge_pre_final']").assign(model=lambda d: d.model.map(names))
st.plotly_chart(px.line(fin, x="year", y="spearman", color="model", markers=True, title="Final ranking accuracy by year"), width="stretch")
st.warning("Voting relationships are learnable; song quality is not observable in this dataset. So before the contest, picking the winner stays close to chance. "
           "After the semi-finals, the final ranking is predicted much better.")
