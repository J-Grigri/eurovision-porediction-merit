import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.app_utils import PROC, cname, require_data, table
from ui import JURY, PUBLIC, chart, explain, hero, lead, method, stats, theme

require_data()
theme()
py, bc = table("jury_public_year"), table("jury_public_country")
sc = table("semi_correlations").set_index("test")
war = pd.DataFrame(json.loads((PROC / "politics.json").read_text())["war"]).set_index("measure")

hero("Voting factors", "Jury and televote", "Since 2016 each country's points are split equally between a professional jury and a televote. "
     "The two frequently disagree, in consistent ways.")
res = table("results")
fin = res[(res["round"] == "final") & (res.disqualified == 0) & res.points_jury.notna() & res.year.isin(py.year)].copy()
fin["jury_rank"] = fin.groupby("year").points_jury.rank(ascending=False, method="min").astype(int)
fin["tele_rank"] = fin.groupby("year").points_public.rank(ascending=False, method="min").astype(int)
win = fin[fin.place == 1].set_index("year")
n_tele, n_jury = int((win.tele_rank == 1).sum()), int((win.jury_rank == 1).sum())
n_neither, worst = int(((win.tele_rank > 1) & (win.jury_rank > 1)).sum()), int(win[["jury_rank", "tele_rank"]].max(axis=1).max())
lead(f"Winning does not require both halves: the winner topped the televote in {n_tele} of {len(win)} finals and the jury in {n_jury}. "
     f"However, no winner finished outside the top {worst} of either half.")
stats([("Same favourite", f"{int((py.jury_winner == py.public_winner).sum())} of {len(py)}", "Finals where jury and televote ranked the same song first"),
       ("Rank agreement", f"r = {py.spearman.mean():.2f}", "Jury vs televote, average per final"),
       ("Jury consistency", f"r = {sc.loc['Semi jury → final jury', 'r']:.2f}", "Semi-final jury vs final jury"),
       ("Cross-prediction", f"r = {sc.loc['Semi jury → final public', 'r']:.2f}", "Semi-final jury vs final televote")])

st.markdown("### First place by jury and televote")
show = py.assign(Year=py.year, **{"Jury's favourite": py.jury_winner.map(cname), "Televote's favourite": py.public_winner.map(cname), "Overall winner": py.winner.map(cname)})
show["Winner's jury rank"] = show.year.map(win.jury_rank)
show["Winner's televote rank"] = show.year.map(win.tele_rank)
st.dataframe(show[["Year", "Jury's favourite", "Televote's favourite", "Overall winner", "Winner's jury rank", "Winner's televote rank"]],
             hide_index=True, width="stretch")
explain("Shows how the two halves combine to decide the winner.",
        f"The winner was the televote's favourite in {n_tele} finals and the jury's in {n_jury} (both in one final); in {n_neither} finals it topped "
        f"neither. In every final the winner placed in the top {worst} of both halves: first place in one half wins only when the other half also "
        "ranks the entry near the top.")

st.markdown("### Televote-favoured and jury-favoured countries")
b = pd.concat([bc.nsmallest(8, "mean_gap"), bc.nlargest(8, "mean_gap")]).drop_duplicates().sort_values("mean_gap")
fig = go.Figure(go.Bar(x=b.mean_gap * 100, y=b.country.map(cname), orientation="h", marker_color=[JURY if v < 0 else PUBLIC for v in b.mean_gap],
                       hovertemplate="%{y}: %{x:+.0f} percentile points<extra></extra>"))
fig.add_vline(x=0, line_color="#b9b2dd")
chart(fig, 480, xaxis_title="← ranked higher by jury · ranked higher by televote →  (percentile points)")
explain("Shows which half each country depends on.",
        "Average difference between televote and jury ranking in finals since 2016 (minimum three finals). "
        f"Televote-favoured countries (pink) include {', '.join(cname(c) for c in bc.nlargest(3, 'mean_gap').country)}; part of this may reflect "
        "diaspora voting (see Diaspora voting). "
        f"Jury-favoured countries (blue) include {', '.join(cname(c) for c in bc.nsmallest(3, 'mean_gap').country)}.", "Consistent pattern", "moderate")

st.markdown("### Countries at war")
cases = table("pol_war_cases").dropna(subset=["public_z"])
cases["label"] = cases.entry.map(cname) + " " + cases.year.astype(str)
fig = go.Figure()
fig.add_bar(x=cases.label, y=cases.jury_z, name="Jury", marker_color=JURY, text=[f"{v:+.1f}" for v in cases.jury_z], textposition="outside")
fig.add_bar(x=cases.label, y=cases.public_z, name="Televote", marker_color=PUBLIC, text=[f"{v:+.1f}" for v in cases.public_z], textposition="outside")
chart(fig, 340, barmode="group", yaxis_title="Score relative to annual average (SD)", legend=dict(orientation="h", y=1.12))
pmj, sur = war.loc["public_minus_jury"], war.loc["surprise"]
explain("Of the political signals tested, only war showed a measurable effect on voting. Source: UCDP Armed Conflict Dataset. A country is classified as at war "
        "if a war (1,000+ battle deaths) took place on its territory in the preceding year, or began before May of the contest year.",
        f"In every case the televote ranked the country above the jury, with a gap {pmj.difference:.1f} SD larger than for other finalists "
        f"(p = {pmj.p_perm:.3f}). The effect changes the composition of support rather than the final result: Ukraine won in 2016 and 2022 but "
        "finished below its projection in 2023. With six cases, this is a consistent pattern rather than an established rule.",
        "Consistent · 6 cases", "moderate")

st.markdown("### Methods")
method([
    ("Data", "Separate jury and televote points for finals 2016–2024 and semi-finals 2016–2022 (semi-finals in 2023–2025 used televote only). "
             "UCDP/PRIO Armed Conflict Dataset v26.1 (conflicts with ≥25 battle deaths per year; war = ≥1,000)."),
    ("Measures", "Agreement: Spearman rank correlation between jury and televote points in each final. Country preference: mean difference between "
                 "televote and jury percentile rank (countries with ≥3 finals). Semi-final consistency: Pearson correlation of scores standardised "
                 "within each show (120 finalists with split results)."),
    ("War test", "A country is classified as at war if a war was fought on its territory in the year before the contest, or began before May of the "
                 "contest year. Outcome: televote z-score minus jury z-score within the final. Compared with all other finalists using a permutation "
                 "test (5,000 shuffles). A second test compares the actual result with the pre-contest projection of the forecast model."),
    ("Limitations", "Only six war cases have split results, so the effect size is uncertain. Differences between jury and televote preferences "
                    "reflect several factors (diaspora, taste, staging) that cannot be fully separated."),
])
