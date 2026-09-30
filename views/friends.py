import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.app_utils import PROC, cname, require_data, table
from analysis import blocs
from ui import HIGHLIGHT, ISO3, JURY, PUBLIC, chart, explain, hero, lead, method, stats, theme

require_data()
theme()
ev = json.loads((PROC / "evidence.json").read_text())
hero("Voting factors", "Voting relationships", "Some countries consistently award each other more points than the song alone earns. "
     "This page identifies those relationships, their stability and their causes.")
lead("Each vote is compared with the points all <b>other</b> countries gave the same song in the same show. For example, if Greece awards Cyprus 12 points and "
     "the average elsewhere is 4, the difference of 8 reflects Greek preference rather than song quality. The average of this difference over "
     "2004–2024 defines the <b>voting relationship</b>.")
stats([("Stability", f"r = {ev['stability']['r']:.2f}", "2004–2013 relationships predict 2014–2024"),
       ("Reciprocity", f"r = {ev['reciprocity']['r']:.2f}", "Preference is usually mutual"),
       ("Bloc premium", f"+{ev['blocs']['within_pts12']:.1f} pts", "Additional points (of 12) within a voting bloc"),
       ("Forecast weight", "Primary input", "Removing it lowers pair-level correlation from 0.48 to 0.28")])

# ---------------------------------------------------------------- map
st.markdown("### Relationships by country")
A = table("pair_affinity")
A = A[A.n >= 5]
codes = sorted(set(A.voter) & set(A.entry), key=cname)
c1, c2 = st.columns([1, 1])
me = c1.selectbox("Country", codes, index=codes.index("GR"), format_func=cname)
direction = c2.radio("View", ["Points awarded", "Points received"], horizontal=True)
gives = direction == "Points awarded"
d = A[A.voter == me] if gives else A[A.entry == me]
d = d.assign(other=d.entry if gives else d.voter, pts=d["mean"] * 12)
d = d[d.other.isin(ISO3)]
on_map = d[~d.other.isin(["AU", "MA"])]
fig = go.Figure(go.Choropleth(locations=on_map.other.map(ISO3), z=on_map.pts, zmid=0, zmin=-3, zmax=6,
                              colorscale=[[0, JURY], [0.33, "#5a5488"], [1, PUBLIC]], marker_line_color="#0d0a26", marker_line_width=0.8,
                              text=on_map.other.map(cname), hovertemplate="%{text}: %{z:+.1f} points<extra></extra>",
                              colorbar=dict(title="Additional<br>points (of 12)", thickness=12, len=0.75)))
fig.add_trace(go.Choropleth(locations=[ISO3[me]], z=[1], colorscale=[[0, "#c98500"], [1, "#c98500"]], showscale=False,
                            marker_line_color="#ffffff", marker_line_width=1.5, hovertemplate=f"{cname(me)}<extra></extra>"))
fig.update_geos(scope="europe", projection_type="mercator", lataxis_range=[34, 71], lonaxis_range=[-25, 50], showframe=False,
                showcoastlines=False, bgcolor="rgba(0,0,0,0)", landcolor="#1a1442", showland=True, showcountries=False, showlakes=False)
chart(fig, 520, margin=dict(l=0, r=0, t=0, b=0))
top = d.nlargest(6, "pts")
st.markdown(f"**{'Strongest preferences of ' if gives else 'Strongest support for '}{cname(me)}**")
st.dataframe(pd.DataFrame({"Country": top.other.map(cname), "Additional points (of 12)": top.pts.round(1), "Contests": top.n}),
             hide_index=True, width="stretch")
explain("Summarises two decades of voting for the selected country.",
        f"{cname(me)} is shown in gold. Pink indicates more points than other countries awarded the same song; blue indicates fewer. "
        "Australia is not shown. These relationships are the forecast's primary input.", "Strong", "strong")

# ---------------------------------------------------------------- blocs
st.markdown("### Voting blocs")
bl, _ = blocs(table("pair_affinity"), k=7)
names = [("SE", "Nordic, Baltic, British Isles and others"), ("FR", "Western Europe"), ("RS", "Former Yugoslavia & Turkey"), ("UA", "Former Soviet states & Israel"),
         ("GR", "South-east Europe & Mediterranean")]
cols = st.columns(3)
i = 0
for b, g in bl.groupby("bloc"):
    if len(g) < 3:
        continue
    name = next((n for key, n in names if key in set(g.country)), f"Group {b}")
    cols[i % 3].markdown(f'<div class="esc-stat"><div class="k">{name}</div><div class="n" style="font-size:14px;color:#efeaff;margin-top:6px">'
                         f'{", ".join(cname(c) for c in g.country)}</div></div>', unsafe_allow_html=True)
    i += 1
st.markdown("")
explain("Identifies groups of countries that vote for one another.",
        f"Groups are derived from voting data alone (hierarchical clustering on 2004–2024). When blocs are identified from 2004–2013 only, they still exchanged "
        f"+{ev['blocs']['within_pts12']:.1f} additional points in 2014–2024, against a maximum of +{ev['blocs']['null_95pct'] * 12:.1f} for random groupings.", "Strong", "strong")

# ---------------------------------------------------------------- why
st.markdown("### Drivers of voting relationships")
g = table("geography_coefficients")
g = g[g.model == "Total affinity, 2004–2024"].copy()
g["label"] = g.feature.map({"contig": "Share a border", "comlang_ethno": "Share a language", "smctry": "Former common state",
                            "colony": "Colonial or imperial link", "log_dist": "Distance (per 10× increase)"})
g = g.sort_values("coef")
fig = go.Figure(go.Bar(x=g.coef * 12, y=g.label, orientation="h", marker_color=[JURY if v < 0 else HIGHLIGHT for v in g.coef],
                       error_x=dict(type="data", symmetric=False, array=(g.hi95 - g.coef) * 12, arrayminus=(g.coef - g.lo95) * 12, color="#b9b2dd"),
                       hovertemplate="%{y}: %{x:+.1f} points<extra></extra>"))
fig.add_vline(x=0, line_color="#b9b2dd")
chart(fig, 280, xaxis_title="Additional points (of 12)")
explain("Links voting relationships to geographic and cultural ties (CEPII GeoDist).",
        "Each bar shows the additional points associated with a tie, holding the others constant; lines show 95% confidence intervals, none of which "
        "include zero. Together these ties explain about a quarter of the variation in voting relationships; the remainder reflects other factors, "
        "such as diaspora communities and musical taste.", "Strong", "strong")

st.markdown("### Methods")
method([
    ("Data", "Unofficial Eurovision Song Contest dataset: 34,243 voter→entry observations from 56 shows (semi-finals and finals), 2004–2024. "
             "CEPII GeoDist for bilateral distance, borders, shared language (spoken by ≥9% of both populations), colonial links and former common states."),
    ("Measure", "<b>Affinity</b> = points a voter gave an entry minus the mean of all <i>other</i> voters for the same entry in the same show "
                "(leave-one-out residual), expressed as a share of the maximum a voter can award (12, or 24 when jury and televote are added). "
                "Averaged per voter→country pair (2,038 pairs with ≥5 shared shows). In the forecast, history is weighted with a 10-year half-life "
                "and shrunk towards zero for sparse pairs (pseudo-count 3)."),
    ("Models", "<b>Voting blocs</b>: agglomerative hierarchical clustering (average linkage; distance = 1 − Pearson correlation of the symmetrised "
               "affinity matrix; 46 countries; 7 clusters). <b>Drivers</b>: ordinary least squares regression of pair affinity on log distance and the "
               "four binary ties (2,034 pairs, R² = 0.25)."),
    ("Validation", "Stability: Pearson correlation of pair affinity, 2004–2013 vs 2014–2024 (1,594 pairs). Reciprocity: correlation of A→B with B→A, "
                   "permutation test (2,000 shuffles). Blocs: clusters fitted on 2004–2013, evaluated on 2014–2024 against 2,000 random label "
                   "permutations. Confidence intervals: bootstrap over voting countries (1,000 resamples). Forecast ablation: removing affinity lowers "
                   "out-of-sample pair-level correlation from 0.48 to 0.28."),
    ("Limitations", "Associations, not causes: shared history, language, diaspora and taste cannot be fully separated. Serbia & Montenegro and "
                    "Yugoslavia are merged into Serbia."),
])
