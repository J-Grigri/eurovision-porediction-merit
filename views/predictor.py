import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.app_utils import PROC, cname, require_data
from forecast import BIG5, predict, qualification
from ui import HIGHLIGHT, JURY, NEGATIVE, POSITIVE, chart, explain, hero, stats, theme

require_data()
theme()
meta = json.loads((PROC / "forecast_meta.json").read_text())
NEXT = meta["last_year"] + 1
EXCLUDED = {"RU", "BY"}  # excluded by the EBU since 2022
countries = sorted([c for c in meta["countries"] if c not in EXCLUDED], key=cname)
ordinal = lambda n: f"{int(n)}{'th' if 11 <= int(n) % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(int(n) % 10, 'th')}"
pct = lambda p: "<1%" if p < 0.005 else f"{p * 100:.0f}%"

hero(f"Eurovision Song Contest {NEXT}", f"Eurovision {NEXT} Forecast",
     "Projected final placings and win probabilities, based on voting relationships since 2004, recent results, running order, "
     "host status and, once available, semi-final results.")

# ---------------------------------------------------------------- inputs
c1, c2, c3 = st.columns(3)
me = c1.selectbox("Country", countries, index=countries.index("PL") if "PL" in countries else 0, format_func=cname)
host = c2.selectbox("Host country", countries, index=countries.index(meta["last_winner"]), format_func=cname,
                    help=f"Normally the previous year's winner ({cname(meta['last_winner'])}).")
stage_label = c3.radio("Forecast stage", ["Pre-semi-final", "Post-semi-final"], horizontal=True,
                       help="Semi-final results roughly double forecast accuracy.")
stage = "early" if stage_label.startswith("Pre") else "after_semis"

direct = me in BIG5 or me == host
c4, c5, _ = st.columns(3)
field_n = len(set(meta["last_finalists"]) | {me})
slot = c4.selectbox("Final running order", ["Not yet drawn"] + list(range(1, field_n + 1)))
if stage == "after_semis" and not direct:
    opts = ["Unknown"] + [f"{ordinal(i)} in semi-final" for i in range(1, 11)]
    pick = c5.selectbox("Semi-final placing", opts, index=4)
    my_semi = None if pick == "Unknown" else opts.index(pick)
else:
    c5.selectbox("Semi-final placing", ["Qualified automatically (Big 5 / host)"] if direct else ["Not yet held"], disabled=True)
    my_semi = "direct" if direct else None

with st.expander("Participants, finalists and other results"):
    voters = st.multiselect("Participating countries", countries, default=[c for c in meta["last_participants"] if c not in EXCLUDED],
                            format_func=cname, help=f"All participants vote in the final. Defaults to the {meta['last_year']} line-up. "
                                                    "Remove countries that withdraw or boycott.")
    finalists = st.multiselect("Finalists", countries, default=[c for c in meta["last_finalists"] if c not in EXCLUDED],
                               format_func=cname, help=f"Defaults to the {meta['last_year']} final.")
    others = [c for c in finalists if c != me]
    ed = st.data_editor(pd.DataFrame({"Country": [cname(c) for c in others], "code": others,
                                      "Semi-final placing": [None] * len(others), "Running order": [None] * len(others)}),
                        hide_index=True, width="stretch", disabled=["Country"], column_config={
                            "code": None,
                            "Semi-final placing": st.column_config.NumberColumn(min_value=1, max_value=10, step=1),
                            "Running order": st.column_config.NumberColumn(min_value=1, max_value=30, step=1)})
    explain("Participation is where political decisions enter the forecast directly: a country that withdraws neither performs nor votes.",
            "Blank values are filled with typical values. Entering all semi-final placings gives the fairest post-semi-final comparison.")
finalists = list(dict.fromkeys([c for c in finalists if c in voters] + [me]))
running = {c: int(v) for c, v in zip(ed.code, ed["Running order"]) if pd.notna(v)}
semi = {c: int(v) for c, v in zip(ed.code, ed["Semi-final placing"]) if pd.notna(v)}
if slot != "Not yet drawn":
    running[me] = int(slot)
if my_semi is not None:
    semi[me] = my_semi

E, P = predict(finalists, host, stage, running=running, semi=semi, voters=voters + ["WLD"])
n_semi = sum(1 for c in finalists if c not in BIG5 and c != host)
entered = len([c for c in semi if c != me and semi[c] != "direct"])
if stage == "after_semis" and entered < n_semi - 1:
    st.warning(f"Semi-final placings entered for {entered} of {n_semi - (0 if direct else 1)} other qualifiers. "
               "Complete them under *Participants, finalists and other results* for a like-for-like comparison.")
row = E[E.entry == me].iloc[0]

# ---------------------------------------------------------------- summary
st.divider()
st.markdown(f"### {cname(me)}: forecast summary")
cards = [("Projected placing", ordinal(row.pred_rank), f"In the final; likely range {ordinal(round(row.place_lo))}–{ordinal(round(row.place_hi))}"),
         ("Win probability", pct(row.p_win), "Fitted on past finals at this projected placing"),
         ("Top-5 probability", pct(row.p_top5), "Fitted on past finals at this projected placing"),
         ("Projected points", f"{row.points:.0f}", "Average outcome, jury and televote combined")]
if stage == "early":
    q = qualification([c for c in voters if c != me] + [me], host, voters + ["WLD"])
    cards.insert(0, ("Qualification probability", "Automatic" if direct else pct(q[me]),
                     "Big 5 and host go straight to the final" if direct else "Chance of reaching the final from a semi-final"))
stats(cards)
leaders = E[E.p_win >= E.p_win.max() - 0.005]
lead_txt = (f"Leading contender: <b>{cname(leaders.entry.iloc[0])}</b> ({pct(leaders.p_win.iloc[0])})." if len(leaders) == 1 else
            f"Leading contenders, statistically tied at {pct(leaders.p_win.iloc[0])} each: <b>{', '.join(cname(e) for e in leaders.entry)}</b>.")
explain("Summarises the projected outcome for the selected country.",
        "Probabilities come from a logistic regression of past outcomes on projected placing, fitted on the 14 finals from 2010 to 2024, "
        "so they decline smoothly with placing and are never exactly zero. "
        + lead_txt + (" Placing and win probability assume the country reaches the final. Before the semi-finals no entry exceeds about 12%, "
                      "as song quality is not yet observable."
                      if stage == "early" else " After the semi-finals, the winner was within the top 3 projections in 11 of 14 past finals."),
        "Calibrated on past finals", "info")

left, right = st.columns([1.1, 1])
with left:
    st.markdown("### Win probability")
    top = E.head(10).iloc[::-1]
    fig = go.Figure(go.Bar(x=top.p_win * 100, y=[cname(e) for e in top.entry], orientation="h",
                           marker_color=[HIGHLIGHT if e == me else JURY for e in top.entry],
                           text=[pct(p) for p in top.p_win], textposition="outside", hovertemplate="%{y}: %{x:.0f}%<extra></extra>"))
    chart(fig, 380, xaxis_title="Win probability (%)", xaxis_range=[0, max(top.p_win.max() * 135, 5)])
    explain("Shows the strongest contenders and how open the contest is.",
            "Probabilities fall steadily with projected placing. Before the semi-finals the curve is flat (the leader has about 12%, the last "
            "finalist about 2%), because past winners often came from outside the projected top 10. The selected country is shown in gold.")
with right:
    st.markdown("### Contributing factors")
    lab = {"c_aff": "Voting relationships", "c_form": "Recent results", "c_running_pct": "Running order", "c_entry_is_host": "Host status",
           "c_has_semi": "Semi-final qualification", "c_semi_pos": "Semi-final placing"}
    if me != host:
        lab.pop("c_entry_is_host")
    if stage == "early" or direct:
        lab.pop("c_has_semi", None); lab.pop("c_semi_pos", None)
    contrib = pd.Series({lab[k]: row[k] for k in lab if k in row.index}).sort_values()
    names = [f"{k}  ({v:+.0f})" for k, v in contrib.items()]
    fig = go.Figure(go.Bar(x=contrib.values, y=names, orientation="h", marker_color=[NEGATIVE if v < 0 else POSITIVE for v in contrib.values],
                           hovertemplate="%{y}<extra></extra>"))
    fig.add_vline(x=0, line_color="#b9b2dd")
    chart(fig, 380, xaxis_title="Points relative to the average finalist", yaxis=dict(automargin=True))
    explain("Breaks the projection down by input.",
            "Points gained or lost relative to the average finalist. Voting relationships: support from countries that historically over-score this "
            "country (stable across decades, r = 0.72). Recent results: the last few contests, weighted towards the most recent. Running order: later "
            "performances score higher. Semi-final placing: the strongest single indicator once available.", "Strong evidence", "strong")

st.markdown(f"### Projected points by voting country: {cname(me)}")
sup = P[P.entry == me].nlargest(10, "exp_points")
fig = go.Figure(go.Bar(x=[cname(v) if v != "WLD" else "Rest of the world" for v in sup.voter], y=sup.exp_points, marker_color=JURY,
                       text=[f"{p:.1f}" for p in sup.exp_points], textposition="outside", hovertemplate="%{x}: %{y:.1f} points<extra></extra>"))
chart(fig, 320, yaxis_title="Projected points (max 24)", yaxis_range=[0, sup.exp_points.max() * 1.2])
explain("Identifies the country's main sources of support.",
        "Jury and televote combined, as an average across possible outcomes: each voting country's 116 points (58 jury, 58 televote) are shared "
        "in proportion to its predicted preferences. Averages are flatter than real scoreboards, where winners typically score 450–750 points, "
        "so use placings and probabilities for comparisons. The ranking follows historical voting relationships, which reflect geography, shared language and diaspora "
        "communities (see Voting factors).", "Strong", "strong")

with st.expander("Full forecast table"):
    show = E.assign(Country=E.entry.map(cname), Likely=[f"{ordinal(round(a))}–{ordinal(round(b))}" for a, b in zip(E.place_lo, E.place_hi)])
    st.dataframe(show[["pred_rank", "Country", "points", "p_win", "p_top5", "Likely"]].rename(columns={
        "pred_rank": "Projected placing", "points": "Projected points", "p_win": "Win probability", "p_top5": "Top-5 probability", "Likely": "Likely range"}),
        hide_index=True, width="stretch", column_config={"Win probability": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                                                         "Top-5 probability": st.column_config.NumberColumn(format="percent"),
                                                         "Projected points": st.column_config.NumberColumn(format="%.0f")})
    explain("Allows comparison across the full field.", "Likely range: the 20th–80th percentile of actual placings for entries at that projected placing.")
