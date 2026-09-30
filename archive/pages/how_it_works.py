import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.app_utils import PROC, cname, require_data, table
from ui import HIGHLIGHT, JURY, PUBLIC, explain, hero, theme

require_data()
theme()
meta = json.loads((PROC / "forecast_meta.json").read_text())
ev = json.loads((PROC / "evidence.json").read_text())
dia = json.loads((PROC / "diaspora.json").read_text())
geo = json.loads((PROC / "geography.json").read_text())
sc = table("semi_correlations").set_index("test")

hero("Method", "How the forecast works",
     "For every pair of countries the model predicts how many points one gives the other, then adds them up into a scoreboard. "
     "It uses four signals. Each was tested on contests the model never saw, and anything that didn't improve results was left out.")

# ---------------------------------------------------------------- accuracy
acc = meta["accuracy"]
st.subheader("How accurate is it?")
rows = []
for stage, label in (("early", "Before the semi-finals"), ("after_semis", "After the semi-finals")):
    d, h = acc[f"{stage}_dev"], acc[f"{stage}_holdout"]
    rows.append({"Stage": label, "Winner in predicted top 3": f"{d['winner_top3']} of {d['finals']}", "Real top 5 found (of 5)": f"{d['top5']:.1f}",
                 "Real top 10 found (of 10)": f"{d['top10']:.1f}", "Ranking accuracy (ρ)": f"{d['rank_corr']:.2f}",
                 f"Unseen {meta['last_year'] - 1} / {meta['last_year']}: winner's predicted rank": " / ".join(str(r) for r in h["winner_ranks"])})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
explain("It tells you how much to trust the predictor. Every past final from 2010 to 2024 was predicted using only earlier years, "
        "and 2025–2026 were kept completely unseen until the end.",
        "Before the semi-finals the model finds about 2 of the real top 5 (chance: about 1). After them, the winner was in the top 3 in 11 of 14 finals "
        "and ranked 1st in 2026. Ranking accuracy is a rank correlation: 1 means a perfect order, 0 means random.",
        "Tested on unseen years", "strong")

st.divider()
st.subheader("The four signals")

a, b = st.columns(2)
with a:
    st.markdown("#### 1 · Voting friends")
    A = table("pair_affinity")
    t = A[A.n >= 10].nlargest(8, "mean")
    st.dataframe(pd.DataFrame({"Voter → performer": t.voter.map(cname) + " → " + t.entry.map(cname),
                               "Extra points (of 12)": (t["mean"] * 12).round(1), "Contests": t.n}), hide_index=True, width="stretch")
    explain("This is the model's most important input. Removing it cuts accuracy from 0.48 to 0.28 (the correlation between predicted and actual points).",
            f"Points above what every other country gave the same song, so song quality cancels out. The ties are stable (a decade predicts the next at "
            f"r = {ev['stability']['r']:.2f}) and mutual (r = {ev['reciprocity']['r']:.2f}). Blocs found in 2004–2013 still gave each other "
            f"+{ev['blocs']['within_pts12']:.1f} extra points in 2014–2024.", "Strong", "strong")
with b:
    st.markdown("#### 2 · Diaspora and neighbours")
    bd = table("diaspora_bands")
    fig = go.Figure()
    fig.add_bar(x=bd.band.astype(str), y=bd.jury * 12, name="Jury", marker_color=JURY)
    fig.add_bar(x=bd.band.astype(str), y=bd.public * 12, name="Public", marker_color=PUBLIC)
    fig.update_layout(barmode="group", height=280, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Extra points (of 12)",
                      xaxis_title="Migrants from the performing country (% of voting country)", legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, width="stretch")
    explain("It explains why the voting friends exist. The external data (Eurostat migrant stock, CEPII geography) tests the obvious explanations.",
            f"The public follows migrant communities (r = {dia['r_public']:.2f}); juries barely do (r = {dia['r_jury']:.2f}). With border, distance and language "
            f"held fixed, each 10× larger community still adds about 0.7 televote points. Geography explains {geo['total']['r2'] * 100:.0f}% of voting ties. "
            "Past voting already contains all of this, so the forecast doesn't need these datasets directly.", "Strong", "strong")

c, d = st.columns(2)
with c:
    st.markdown("#### 3 · The semi-final")
    E = table("forecast_backtest")
    per = E[E.year <= 2024].groupby("year").apply(lambda g: pd.Series({s: g[s].corr(-g.place, method="spearman") for s in ("early", "after_semis")}), include_groups=False).reset_index()
    fig = go.Figure()
    fig.add_scatter(x=per.year, y=per.early, name="Before semis", mode="lines+markers", line_color=JURY)
    fig.add_scatter(x=per.year, y=per.after_semis, name="After semis", mode="lines+markers", line_color=PUBLIC)
    fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Ranking accuracy (ρ)", legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, width="stretch")
    explain("It shows why the predictor asks for semi-final results: this is the only signal in the data about how good a song is.",
            f"A finalist's semi score correlates r = {sc.loc['Semi total → final total (2008–2024)', 'r']:.2f} with its final score, and adding it roughly doubles "
            f"ranking accuracy in almost every year. Juries and the public each stay consistent between semi and final, but differ from each other "
            f"(semi jury → final public: r = {sc.loc['Semi jury → final public', 'r']:.2f}). The Big 5 and the host skip the semis, so they stay harder to predict.",
            "Strong", "strong")
with d:
    st.markdown("#### 4 · Running order and hosting")
    s = table("structural_effects")
    s["label"] = s.feature.map({"early_slot": "First third of the show", "late_slot": "Last third of the show", "host": "Host country", "form_z": "Recent results (per SD)"})
    fig = go.Figure(go.Scatter(x=s.effect_sd, y=s.label, mode="markers", marker=dict(size=12, color=HIGHLIGHT),
                               error_x=dict(type="data", symmetric=False, array=s.hi95 - s.effect_sd, arrayminus=s.effect_sd - s.lo95, color=HIGHLIGHT)))
    fig.add_vline(x=0, line_dash="dash")
    fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="Effect on final score (standard deviations)")
    st.plotly_chart(fig, width="stretch")
    explain("These are the fixed parts of the show that affect every entry, and the predictor uses them as inputs.",
            "Dots are effects; lines are 95% intervals, and none cross zero. Late slots help and early slots hurt. Hosts have done worse since 2004, but the host is "
            "nearly always last year's winner, so the cause is unclear. Before 2004 hosts often won, and all five back-to-back winners were hosting.",
            "Strong (slot) · Moderate (host)", "moderate")

st.divider()
st.markdown("#### What the model can't see")
explain("It's better to know the limits before relying on a number.",
        "<b>How good a new song is</b>: before the semi-finals this is the biggest unknown, so pre-contest chances stay modest. "
        "<b>Politics and current events</b>: measurable, but they mostly change how votes split between jury and public, not the final result; see the Politics page. "
        "<b>Staging and the live performance</b> on the night.")
st.caption("Data: the unofficial Eurovision Song Contest dataset (2004–2026), Eurostat migrant stock, CEPII GeoDist, UCDP, UNHCR, GDELT.")
