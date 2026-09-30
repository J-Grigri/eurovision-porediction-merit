import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.app_utils import PROC, cname, require_data, table
from ui import HIGHLIGHT, JURY, NEGATIVE, POSITIVE, PUBLIC, chart, explain, hero, lead, method, stats, theme

require_data()
theme()
ev = json.loads((PROC / "evidence.json").read_text())
sc = table("semi_correlations").set_index("test")
se = table("structural_effects").set_index("feature")
hm = table("ev_host_models").set_index("model")
hero("Voting factors", "Contest format", "Four structural factors affect entries: running order, host status, automatic qualification (Big 5) and semi-final performance.")
stats([("Late running order", f"{se.loc['late_slot', 'effect_sd']:+.2f} SD", "Final third vs middle third"),
       ("Early running order", f"{se.loc['early_slot', 'effect_sd']:+.2f} SD", "First third vs middle third"),
       ("Host wins since 2004", f"0 of {ev['host']['hosts_2004_2024'] + 2}", "Since the introduction of semi-finals"),
       ("Big 5 effect", f"{ev['host']['big5']['effect']:+.2f} SD", "France, Germany, Italy, Spain, UK vs comparable entries"),
       ("Semi-final to final", f"r = {sc.loc['Semi total → final total (2008–2024)', 'r']:.2f}", "Correlation of scores")])
lead("SD: standard deviation of the year's scores. One SD corresponds to about 7 places in a 26-entry final, so +0.23 SD is about 1.5 places.")

# ---------------------------------------------------------------- running order
st.markdown("### Running order")
s = table("slot_effect")
fig = go.Figure(go.Bar(x=s.bin.astype(str), y=s["mean"], marker_color=[NEGATIVE if v < 0 else POSITIVE for v in s["mean"]],
                       text=[f"{v:+.2f}" for v in s["mean"]], textposition="outside", hovertemplate="%{x}: %{y:+.2f} SD<extra></extra>"))
chart(fig, 300, xaxis_title="Position in the running order", yaxis_title="Score relative to annual average (SD)")
explain("The running order is known before the final and is therefore a usable forecast input.",
        "Entries in the first fifth score clearly below average, and entries in the second half score above average. The effect is significant (95% confidence intervals "
        "exclude zero). It is an association: producers also determine the order.", "Strong", "strong")

# ---------------------------------------------------------------- hosting
st.markdown("### Host status")
c1, c2 = st.columns([1.2, 1])
with c1:
    he = table("ev_host_eras")
    st.dataframe(pd.DataFrame({"Era": he.era.astype(str), "Hosts": he.hosts, "Host wins": he.host_wins,
                               "Median host placing": [f"top {p * 100:.0f}%" for p in he.median_place_pct]}), hide_index=True, width="stretch")
with c2:
    b = table("ev_back_to_back")
    st.dataframe(pd.DataFrame({"Back-to-back winner": b.country.map(cname), "Years": b.years, "Host at second win": b.hosted.map({1: "Yes", 0: "No"})}),
                 hide_index=True, width="stretch")
explain("The host is known a year in advance, but its effect has changed over time.",
        "Before 2004 hosts performed well: all five consecutive wins were achieved by the host, including Ireland's three in a row. Since 2004, hosts have "
        f"finished mid-table on average and none has won. The host effect is {se.loc['host', 'effect_sd']:+.2f} SD controlling for recent form, and "
        f"{hm.loc['+ last year\'s score + no-semi route', 'host_effect']:+.2f} SD when also controlling for the previous year's score (regression to the mean) "
        "and automatic qualification. "
        f"As the host is almost always the previous winner ({ev['host']['host_is_last_winner']} of {ev['host']['hosts_2004_2024']}), the effects of hosting "
        "and defending the title cannot be separated.", "Moderate", "moderate")

# ---------------------------------------------------------------- semi-finals
st.markdown("### Big 5")
b5 = ev["host"]["big5"]
explain("France, Germany, Italy, Spain and the United Kingdom qualify automatically as the largest financial contributors. A common question is "
        "whether this helps them.",
        f"It does not: Big 5 entries score {b5['effect']:+.2f} SD below comparable entries (95% CI {b5['lo95']:+.2f} to {b5['hi95']:+.2f}; "
        f"{b5['n']} entries, 2005–2024), after controlling for the previous year's score. Possible explanations include less exposure, since they do "
        "not perform in a semi-final, and weaker voting relationships, but the data cannot separate them.", "Strong", "strong")

st.markdown("### Semi-final performance")
E = table("forecast_backtest")
per = E[E.year <= 2024].groupby("year").apply(lambda g: pd.Series({s: g[s].corr(-g.place, method="spearman") for s in ("early", "after_semis")}), include_groups=False).reset_index()
fig = go.Figure()
fig.add_scatter(x=per.year, y=per.early, name="Pre-semi-final forecast", mode="lines+markers", line_color=JURY)
fig.add_scatter(x=per.year, y=per.after_semis, name="Post-semi-final forecast", mode="lines+markers", line_color=PUBLIC)
chart(fig, 300, yaxis_title="Rank correlation (1 = perfect)", legend=dict(orientation="h", y=1.14))
e_rho, a_rho = per.early.mean(), per.after_semis.mean()
explain("Semi-finals provide the only scored evidence of song quality before the final.",
        f"Adding semi-final results improved the ranking accuracy in 13 of 14 finals, from {e_rho:.2f} to {a_rho:.2f} on average. The Big 5 (France, Germany, Italy, Spain, United Kingdom) and the "
        "host qualify automatically, so their entries remain harder to assess.", "Strong", "strong")

st.markdown("### Methods")
method([
    ("Data", "Finals 2004–2024: running order, host country and points; semi-final placings 2004–2024; host history since 1956."),
    ("Measure", "Entry score = mean share of maximum points across all voters, standardised within each final (z-score), so years with different "
                "field sizes and voting systems are comparable."),
    ("Running order and host", "Ordinary least squares regression of the entry score on first-third and last-third running-order indicators, a host "
                               "indicator and the country's recent form. 95% confidence intervals by cluster bootstrap over contest years (500 resamples)."),
    ("Host and Big 5", "Re-estimated with the previous year's score (regression to the mean) and an automatic-qualifier indicator as controls; the Big 5 effect is that indicator's coefficient "
                        "(1,000 bootstrap resamples); previous winners compared with previous 2nd–3rd and 4th–10th placed countries."),
    ("Semi-finals", "Ridge regression forecasts with and without semi-final placing, evaluated year by year (rolling origin: each final predicted "
                    "from earlier years only). Accuracy measured by Spearman rank correlation between projected and actual placings."),
    ("Limitations", "Running order is set by producers, so its effect cannot be separated from their choices. About 20 hosts since 2004 limits "
                    "the precision of the host estimate."),
])
