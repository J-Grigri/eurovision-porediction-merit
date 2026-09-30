import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.app_utils import cname, require_data, table
from ui import HIGHLIGHT, JURY, PUBLIC, chart, explain, hero, method, stats, theme

require_data()
theme()
res = table("results")
res = res[res.disqualified == 0]
fin = res[res["round"] == "final"]
semi = res[res["round"] != "final"]
last_year = int(res.year.max())

hero("History", "Country history", f"Every senior contest from 1956 to {last_year}: results by country and the all-time record.")

countries = sorted(set(fin.country.dropna()), key=cname)
me = st.selectbox("Country", countries, index=countries.index("PL") if "PL" in countries else 0, format_func=cname)

f = fin[fin.country == me].groupby("year").place.min().dropna()
field = fin.groupby("year").place.count()
dnq = semi[(semi.country == me) & (semi.qualified == 0)].groupby("year").place.min()
years = sorted(set(res[res.country == me].year))
wins = f[f == 1].index.tolist()
modern = [y for y in years if y >= 2004]
reached = sum(1 for y in modern if y in f.index)
recent = f[f.index >= last_year - 9]
stats([("Participations", len(years), f"First in {years[0]}"),
       ("Wins", len(wins), ", ".join(map(str, wins)) if wins else "No wins yet"),
       ("Best placing", f"{int(f.min())}" if len(f) else "–", f"Achieved {sum(f == f.min())}×" if len(f) else ""),
       ("Finals reached since 2004", f"{reached} of {len(modern)}", "Including automatic qualification"),
       ("Average final placing", f"{recent.mean():.1f}" if len(recent) else "–", f"Finals in {last_year - 9}–{last_year}")])

st.markdown(f"### {cname(me)}: placing by year")
fig = go.Figure()
fig.add_scatter(x=f.index, y=f.values, mode="lines+markers", name="Final placing", line=dict(color=JURY, width=2),
                marker=dict(size=[13 if p == 1 else 7 for p in f.values], color=[HIGHLIGHT if p == 1 else JURY for p in f.values]),
                customdata=field.reindex(f.index).values, hovertemplate="%{x}: %{y} of %{customdata}<extra></extra>")
if len(dnq):
    ymax = int(field.max()) + 2
    fig.add_scatter(x=dnq.index, y=[ymax] * len(dnq), mode="markers", name="Did not qualify", marker=dict(symbol="x", size=9, color=PUBLIC),
                    customdata=dnq.values, hovertemplate="%{x}: eliminated in semi-final (%{customdata:.0f}th)<extra></extra>")
fig.update_yaxes(autorange="reversed", title="Placing in the final (1 = winner)")
chart(fig, 360, legend=dict(orientation="h", y=1.12))
explain("Shows the country's long-term record, including semi-final eliminations since 2004.",
        "Gold markers are wins. Pink crosses at the bottom mark years the country was eliminated in a semi-final. Gaps are years it did not take part "
        "(the 2020 contest was cancelled). Field sizes vary, from 10 entries in 1957 to 25–27 finalists today.")

st.markdown("### All-time winners")
w = fin[fin.place == 1]
tab = w.groupby("country").year.agg(Wins="count", Years=lambda s: ", ".join(map(str, sorted(s)))).reset_index().sort_values(["Wins", "country"], ascending=[False, True])
tab.insert(0, "Country", tab.country.map(cname))
st.dataframe(tab.drop(columns="country").head(12), hide_index=True, width="stretch")
explain("Answers which countries have won most often.",
        f"Most wins: {' and '.join(tab[tab.Wins == tab.Wins.max()].Country)} with {int(tab.Wins.max())} each. The 1969 contest ended in a four-way tie, so each of its four winners "
        "is counted.")

st.markdown("### Methods")
method([
    ("Data", f"Unofficial Eurovision Song Contest dataset, all senior contests 1956–{last_year}: final placings, semi-final placings and "
             "qualification (semi-finals from 2004)."),
    ("Definitions", "Finals reached since 2004 counts semi-final qualification and automatic places (Big 5, host). Placings are not adjusted for "
                    "field size, which grew from 10 entries (1957) to 25–27 finalists. For 1956 only the winner was published."),
])
