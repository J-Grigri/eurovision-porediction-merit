import plotly.graph_objects as go
import streamlit as st

from src.app_utils import table, cname, require_data

st.set_page_config(page_title="Predictor", layout="wide")
require_data()
st.title("Scoreboard predictor")
E = table("predictions")
HOLDOUT = 2025
c = st.columns(4)
year = c[0].selectbox("Year", sorted(E.year.unique(), reverse=True), format_func=lambda y: f"{y} (hold-out)" if y >= HOLDOUT else str(y))
rounds = sorted(E[E.year == year]["round"].unique(), key=lambda r: r != "final")
rnd = c[1].selectbox("Round", rounds)
kind = c[2].selectbox("Model", ["ridge", "lgbm"], format_func={"ridge": "Ridge regression", "lgbm": "LightGBM"}.get)
setting = c[3].selectbox("Information", ["pre_contest", "pre_final"], format_func={"pre_contest": "Pre-contest (history only)", "pre_final": "Pre-final (adds semi-final results)"}.get)
g = E[(E.year == year) & (E["round"] == rnd) & (E.model == f"{kind}_{setting}")].sort_values("pred_rank").copy()
if g.empty:
    st.info("The pre-final model applies to finals only. Choose the final, or switch to pre-contest.")
    st.stop()
g["Country"] = g.country.map(cname)
if year >= HOLDOUT:
    st.success(f"{year} is a hold-out year: the model never saw it in training, and its history stops at 2024.")
else:
    st.caption("Model trained on years before the selected one only.")
fig = go.Figure()
fig.add_bar(x=g.Country, y=g.pred_points, name="Predicted points")
fig.add_scatter(x=g.Country, y=g.points_total, mode="markers", name="Actual points", marker=dict(size=9, color="crimson"))
fig.update_layout(yaxis_title="Points", height=420)
st.plotly_chart(fig, width="stretch")
cols = {"pred_rank": "Pred. rank", "Country": "Country", "artist": "Artist", "song": "Song", "pred_points": "Pred. points", "points_total": "Actual points", "place": "Actual place"}
if rnd != "final":
    cols["qualified"] = "Qualified"
st.dataframe(g[list(cols)].rename(columns=cols).round(1), hide_index=True, width="stretch")
