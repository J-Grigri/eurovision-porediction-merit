import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.app_utils import PROC, cname, require_data, table
from ui import JURY, PUBLIC, explain, hero, theme

require_data()
theme()
pol = json.loads((PROC / "politics.json").read_text())
war = pd.DataFrame(pol["war"]).set_index("measure")
ref, gd = pol["refugees"], pol["gdelt"]

hero("War, refugees and news", "Does politics move the vote?",
     "Three political signals, each tested against the jury as a control. Juries see the same songs in the same year, "
     "so whatever the public does differently points to something beyond the music. Data up to 2024 (jury/public split from 2016); 2025–2026 stay sealed.")

pmj, sur = war.loc["public_minus_jury"], war.loc["surprise"]
summary = pd.DataFrame([
    {"Signal": "War on home soil", "Measured with": "UCDP armed-conflict data", "Effect": f"Public +{pmj.difference:.1f} SD over the jury",
     "Evidence": f"Strong effect, very few cases (n = {int(pmj.at_war_n)}, p = {pmj.p_perm:.3f})", "Used in the forecast": "No: it shifts jury vs public, not the total"},
    {"Signal": "Refugees arriving", "Measured with": "UNHCR refugee statistics", "Effect": f"{ref['public_ctrl_jury']['slope'] * 12:+.2f} pts per 10× more refugees",
     "Evidence": "Not significant: juries moved the same way", "Used in the forecast": "No"},
    {"Signal": "News tone between countries", "Measured with": "GDELT news events, Jan–Apr before each contest",
     "Effect": f"{gd['tone_public']['slope'] * 12:+.3f} pts per tone point", "Evidence": "None", "Used in the forecast": "No"},
])
st.dataframe(summary, hide_index=True, width="stretch")
explain("The short answer for the whole page: which political signals leave a measurable trace in the votes, and whether the forecast should use them.",
        "Only war shows a clear effect, and it changes <i>who</i> votes for a country (the public far more than the jury) rather than reliably lifting its "
        "final result. Refugee arrivals and news tone don't move votes once each pair's usual relationship is accounted for.",
        "One signal, few cases", "moderate")

st.divider()
st.subheader("War on home soil (UCDP)")
cases = table("pol_war_cases").dropna(subset=["public_z"])
cases["label"] = cases.entry.map(cname) + " " + cases.year.astype(str)
fig = go.Figure()
fig.add_bar(x=cases.label, y=cases.jury_z, name="Jury", marker_color=JURY, text=[f"{v:+.1f}" for v in cases.jury_z], textposition="outside")
fig.add_bar(x=cases.label, y=cases.public_z, name="Public", marker_color=PUBLIC, text=[f"{v:+.1f}" for v in cases.public_z], textposition="outside")
fig.update_layout(barmode="group", height=340, margin=dict(l=0, r=0, t=20, b=0), yaxis_title="Score in the final (SD from that year's average)",
                  legend=dict(orientation="h", y=1.12))
st.plotly_chart(fig, width="stretch")
explain("UCDP records every armed conflict with at least 25 battle deaths a year. A country counts as at war if a war (1,000+ deaths) was fought on its "
        "territory in the year before the contest, or began that year before May (Russia's full invasion of Ukraine, 24 February 2022).",
        f"In every case the public ranked the country higher than the jury did. On average the gap was {pmj.difference:.1f} SD larger than for other "
        f"finalists; shuffling the labels 5,000 times gave a gap this large only {pmj.p_perm * 100:.1f}% of the time. But against what the model expected before "
        f"the contest, these entries did only {sur.difference:+.2f} SD better (p = {sur.p_perm:.2f}), which is not reliable: Ukraine won in 2016 and 2022, "
        "but fell short in 2023, and Azerbaijan fell short in 2021. With 6 cases, treat this as a strong pattern, not a law.",
        "Strong pattern · 6 cases", "moderate")
t = table("pol_war_cases").assign(Country=lambda d: d.entry.map(cname))
st.dataframe(t[["year", "Country", "place", "early_rank", "surprise", "public_minus_jury"]].rename(columns={
    "year": "Year", "place": "Final place", "early_rank": "Predicted rank (pre-contest)", "surprise": "Beat expectation (SD)",
    "public_minus_jury": "Public minus jury (SD)"}).round(2), hide_index=True, width="stretch")
explain("The individual cases behind the chart, so each one can be checked.",
        "Beat expectation: the actual score minus the pre-contest prediction, in standard deviations. 2014–2015 have no jury/public split, so their gap is blank.")

st.divider()
st.subheader("Refugees arriving (UNHCR)")
nat = table("pol_refugee_ukraine")
fig = go.Figure()
for kind, col in (("jury", JURY), ("public", PUBLIC)):
    for per, dash in (("2016–2021", "dot"), ("2022–2024", "solid")):
        d = nat[nat.period == per]
        fig.add_bar(x=d.band.astype(str), y=d[kind], name=f"{kind.title()} {per}", marker_color=col,
                    marker_pattern_shape="/" if per == "2016–2021" else "", opacity=0.6 if per == "2016–2021" else 1)
fig.update_layout(barmode="group", height=320, margin=dict(l=0, r=0, t=20, b=0), yaxis_title="Points to Ukraine above other voters (of 12)",
                  xaxis_title="Ukrainian refugees in the voting country (% of population)", legend=dict(orientation="h", y=1.15))
st.plotly_chart(fig, width="stretch")
explain("The 2022 invasion created a natural experiment: millions of Ukrainian refugees arrived in some countries and hardly any in others. "
        "If refugees vote like diaspora communities, those countries' televotes should shift toward Ukraine more than their juries.",
        f"Countries hosting the most refugees did become relatively warmer to Ukraine after 2022, but their juries moved just as much, so it isn't a televote "
        f"effect. It looks like neighbours' solidarity instead. Across all pairs and years, a 10× rise in refugees shifts the public by "
        f"{ref['public_ctrl_jury']['slope'] * 12:+.2f} points (95% CI {ref['public_ctrl_jury']['lo'] * 12:+.2f} to {ref['public_ctrl_jury']['hi'] * 12:+.2f}), "
        "which is not distinguishable from zero. Refugees are recent arrivals and many can't vote from abroad; settled migrant communities (Diaspora on the "
        "How-it-works page) show a clear effect.", "No reliable effect", "none")

st.divider()
st.subheader("News tone between countries (GDELT)")
gb = table("pol_gdelt_bands")
fig = go.Figure()
fig.add_bar(x=gb.tone_band.astype(str), y=gb.jury_w, name="Jury", marker_color=JURY)
fig.add_bar(x=gb.tone_band.astype(str), y=gb.public_w, name="Public", marker_color=PUBLIC)
fig.update_layout(barmode="group", height=300, margin=dict(l=0, r=0, t=20, b=0), yaxis_title="Change in points vs the pair's usual (of 12)",
                  xaxis_title="News tone between the two countries, compared with their usual", legend=dict(orientation="h", y=1.12))
st.plotly_chart(fig, width="stretch")
explain(f"GDELT codes world news into events between countries, with a tone and a cooperation/conflict score. We sampled every 8th day from January to April "
        f"before each contest and measured each pair's tone ({gd['pairs']} country pairs, {gd['pairs_years']:,} pair-years with enough coverage).",
        f"When coverage between two countries is warmer or colder than usual for that pair, the vote doesn't move: every bar is within ±0.15 points, and the "
        f"slopes' 95% intervals include zero for both the public and the jury. Pairs with warmer relations overall don't vote more warmly either "
        f"(r = {gd['between_pairs_r_public']:.2f}). Diplomatic news is a different thing from what viewers feel about a song.", "No effect", "none")

st.divider()
explain("How to read this page at a defence.",
        "Politics clearly exists in Eurovision, but most of it is already built into long-standing voting relationships (neighbours, diaspora), which the "
        "forecast uses. Short-term political shocks are measurable only for wars, where the public reacts much more than juries. The number of cases is small, "
        "so the forecast doesn't adjust for it automatically.")
