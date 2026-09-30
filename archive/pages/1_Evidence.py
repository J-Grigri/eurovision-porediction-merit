import json

import pandas as pd
import plotly.express as px
import streamlit as st

from src.app_utils import PROC, table, cname, require_data

st.set_page_config(page_title="Evidence", layout="wide")
require_data()
st.title("Evidence: what holds up statistically")
st.caption("2004–2024 only; 2025–2026 stay sealed. Each test is built so song quality, or fitting and testing on the same data, cannot explain the result.")
ev = json.loads((PROC / "evidence.json").read_text())
dia = json.loads((PROC / "diaspora.json").read_text()) if (PROC / "diaspora.json").exists() else None
semi_c = table("semi_correlations").set_index("test")
geo = json.loads((PROC / "geography.json").read_text()) if (PROC / "geography.json").exists() else None
hm = table("ev_host_models").set_index("model")

rows = [
    ("Voting ties are stable over time", f"r = {ev['stability']['r']:.2f}", f"{ev['stability']['pairs']:,} country pairs; affinity 2004–2013 vs 2014–2024", "Strong"),
    ("Voting ties are mutual", f"r = {ev['reciprocity']['r']:.2f}", f"{ev['reciprocity']['pairs']:,} pairs; permutation p < 0.001", "Strong"),
    ("Blocs found in 2004–2013 still hold in 2014–2024", f"+{ev['blocs']['within_pts12']:.1f} pts of 12", f"Random blocs give at most +{ev['blocs']['null_95pct'] * 12:.1f}; p < 0.001", "Strong"),
    ("Country bias is stronger in the televote than the jury", f"{ev['jury_public']['ratio']:.2f}× the spread",
     f"95% CI {ev['jury_public']['ratio_lo']:.2f}–{ev['jury_public']['ratio_hi']:.2f}; same voters and songs, 2016–2024", "Strong"),
    ("A finalist's semi-final score predicts its final score", f"r = {semi_c.loc['Semi total → final total (2008–2024)', 'r']:.2f}",
     f"{int(semi_c.loc['Semi total → final total (2008–2024)', 'n'])} finalists, 2008–2024; standardised within each show", "Strong"),
    ("Jury and public are two separate, consistent tastes", f"r = {semi_c.loc['Semi jury → final jury', 'r']:.2f} / {semi_c.loc['Semi jury → final public', 'r']:.2f}",
     "Semi jury predicts final jury, but barely predicts final public (2016–2022)", "Strong"),
    ("Televote follows migrant communities; juries don't", f"r = {dia['r_public']:.2f} vs {dia['r_jury']:.2f}",
     f"{dia['pairs_2016_2024']} pairs, Eurostat migrant stock; difference 95% CI {dia['diff_ci'][0]:.2f}–{dia['diff_ci'][1]:.2f}", "Strong"),
    ("Geography and shared language shape voting", f"R² = {geo['total']['r2']:.2f}" if geo else "–",
     "CEPII GeoDist: border, distance, language, colonial ties, former same country; every factor's 95% CI excludes zero", "Strong"),
    ("Diaspora effect survives geography controls", "+0.7 pts per 10× migrants",
     "Public affinity; border, distance, language and history held fixed (95% CI 0.50–0.85 pts)", "Strong"),
    ("Hosts (= last year's winners) underperform since 2004",
     f"{hm.loc['+ last year\'s score + no-semi route', 'host_effect']:+.2f} SD",
     "0 wins in 22 modern hosts; not explained by regression to the mean or skipping the semi. Hosting and defending the title can't be separated.", "Moderate"),
    ("Songwriters with a good record write better-placed entries", f"r = {ev['songwriters']['r_writer']:.2f}",
     f"{ev['songwriters']['with_writer_history']} entries with traceable writers; p < 0.001", "Moderate"),
    ("More friendly voters in a semi-final → more likely to qualify", f"ρ = {ev['friends']['rho_friends_qualified']:.2f}",
     "p < 0.001, but largely overlaps with the country's recent form", "Weak"),
]
st.dataframe(pd.DataFrame(rows, columns=["Claim", "Effect", "Test", "Strength"]), hide_index=True, width="stretch")

t1, t2, t3, t6, t7, t8, t9, t4, t5 = st.tabs(["Stability", "Reciprocity", "Jury vs public", "Semi-finals", "Diaspora", "Geography", "Host effect", "Songwriters", "Friends in the semi"])
with t1:
    d = table("ev_stability")
    st.plotly_chart(px.scatter(d, x="early", y="late", opacity=0.35, hover_name=d.voter.map(cname) + " → " + d.entry.map(cname),
                               labels={"early": "Affinity 2004–2013", "late": "Affinity 2014–2024"}), width="stretch")
    st.caption("Each dot is one voter→country pair. If affinity were noise, the cloud would be flat. Instead a decade of voting predicts the next decade (r ≈ 0.72).")
with t2:
    d = table("ev_reciprocity")
    st.plotly_chart(px.scatter(d, x="ab", y="ba", opacity=0.35, hover_name=d.voter.map(cname) + " ↔ " + d.entry.map(cname),
                               labels={"ab": "A → B affinity", "ba": "B → A affinity"}), width="stretch")
    st.caption("When A favours B, B usually favours A. Shuffling the pairs never produced a correlation this high in 2,000 tries.")
with t3:
    d = table("ev_jury_public_pairs")
    d["pair"] = d.voter.map(cname) + " → " + d.entry.map(cname)
    st.markdown("Same voting country, same songs, same year: the televote shows **stronger** country bias than the professional jury. "
                "The pairs where the public is far more favourable than the jury are mostly countries with large emigrant communities in the voting country "
                "(for example Romanians in Italy, Albanians in Switzerland and Italy, Lithuanians in Ireland and the UK). That is **consistent with diaspora voting**, "
                "but this dataset alone cannot prove it; see *Missing data* below.")
    top = pd.concat([d.head(12), d.tail(6)])
    st.plotly_chart(px.bar(top.melt(id_vars="pair", value_vars=["jury", "public"]), x="value", y="pair", color="variable", barmode="group", orientation="h",
                           labels={"value": "Affinity (share of max points above other voters)", "pair": ""}, height=620).update_yaxes(autorange="reversed"), width="stretch")
with t6:
    st.markdown("How much does a finalist's **semi-final** score tell us about its **final** score? Scores are standardised within each show.")
    c = table("semi_correlations")
    st.dataframe(c.rename(columns={"test": "Semi → final", "n": "Finalists", "r": "Correlation", "p": "p-value"}).round(3), hide_index=True, width="stretch")
    st.caption("Jury and public each agree with themselves between semi and final (r ≈ 0.8) but not with each other. "
               "The semi jury/public split exists only for 2016–2022 and 2026; semis in 2023–2025 were decided by the public alone.")
    sm = table("semi_models")
    lab = {"no_semi": "History only", "semi_total": "+ semi score (current model)", "semi_rank": "+ semi rank", "semi_split": "+ semi jury/public split",
           "voter_split": "+ voter's semi jury/public points", "components": "Separate jury and public models"}
    summ = sm.groupby(["model", "split"]).agg(rank_corr=("rho", "mean"), winners=("winner", "sum"), finals=("year", "count"),
                                              winner_median_rank=("winner_rank", "median"), rank_corr_via_semi=("rho_semi_only", "mean")).reset_index()
    summ["model"] = summ.model.map(lab)
    st.subheader("Predicting the final after the semis (Ridge)")
    st.dataframe(summ.pivot(index="model", columns="split", values=["rank_corr", "winners", "winner_median_rank"]).reindex(list(lab.values())).round(2), width="stretch")
    st.caption("Adding the semi score raises final rank correlation by +0.31 (95% CI +0.20 to +0.42). Finer semi details add only +0.01 to +0.02. "
               "Winner median rank: where the model placed the actual winner. The weak spot is the Big 5 and host, which skip the semis.")
    yr = sm[sm.model.isin(["no_semi", "semi_total"])].assign(model=lambda x: x.model.map(lab))
    st.plotly_chart(px.line(yr, x="year", y="rho", color="model", markers=True, labels={"rho": "Final rank correlation"}), width="stretch")
with t7:
    if dia is None:
        st.info("Run `python src/fetch_migration.py` and `python src/diaspora.py` to add migrant data.")
    else:
        st.markdown(f"""**External data:** Eurostat migrant stock (people born in the performing country living in the voting country, or foreign citizens
where birth data is missing), divided by the voting country's population. It covers {dia['voting_countries']} voting countries with 2016–2024 jury and public data; non-EU/EFTA voters are not covered.

- Migrant share correlates with **public** affinity at r = {dia['r_public']:.2f}, but with **jury** affinity only at r = {dia['r_jury']:.2f}.
- Holding jury affinity fixed, each **10× larger** migrant community adds about **{dia['pts_per_10x']:.1f} televote points** (of 12), 95% CI {dia['beta_ci'][0] * 12:.2f}–{dia['beta_ci'][1] * 12:.2f}.
- Adding migrant share to the prediction model did **not** improve accuracy: past voting affinity already contains the diaspora effect. Migration explains *why* the bias exists; affinity already *measures* it.""")
        b = table("diaspora_bands")
        st.plotly_chart(px.bar(b.melt(id_vars=["band", "n"], value_vars=["jury", "public"]), x="band", y="value", color="variable", barmode="group",
                               hover_data=["n"], labels={"band": "Migrants from the performing country, as % of the voting country's population",
                                                         "value": "Affinity (share of max points above other voters)", "variable": ""}), width="stretch")
        d = table("diaspora_pairs").head(20)
        d = d.assign(pair=d.voter.map(cname) + " → " + d.entry.map(cname))
        st.dataframe(d[["pair", "share_pct", "public", "jury"]].rename(columns={"share_pct": "Migrant share %", "public": "Public affinity", "jury": "Jury affinity"}).round(3),
                     hide_index=True, width="stretch")
with t8:
    if geo is None:
        st.info("Run `python src/geography.py` to add CEPII GeoDist data.")
    else:
        st.markdown(f"""**External data:** CEPII GeoDist: distance between largest cities, shared border, a language spoken by at least 9% of both populations,
colonial ties, and whether two countries were once one state. Unlike the migration data, it covers **every** voting country.

- Geography alone explains **{geo['total']['r2'] * 100:.0f}%** of the variation in voting affinity ({geo['total']['pairs']:,} pairs, 2004–2024).
- It matters far more to the **public** (R² {geo['public']['r2']:.2f}) than to **juries** (R² {geo['jury']['r2']:.2f}).
- **Diaspora survives the controls:** with geography held fixed, migrant share still adds about 0.7 televote points per 10× larger community, and raises the explained share of public affinity from {geo['public_geo_only_same_pairs']['r2']:.2f} to {geo['public_with_migration']['r2']:.2f}. Shared language and history partly work *through* migration: their effects shrink once migration is included.
- **Prediction:** adding geography to the model changes pair-level accuracy by less than 0.01. Past voting already captures it. Without any voting history, geography alone recovers much of affinity's value (0.37 vs 0.48, against 0.28 with neither).""")
        C = table("geography_coefficients")
        sel = st.selectbox("Model", C.model.unique().tolist())
        c = C[C.model == sel].assign(pts=lambda x: x.coef * 12)
        fig = px.scatter(c, x="coef", y="label", error_x=c.hi95 - c.coef, error_x_minus=c.coef - c.lo95, labels={"coef": "Effect on affinity (share of max points; ×12 = points)", "label": ""})
        fig.add_vline(x=0, line_dash="dash")
        st.plotly_chart(fig, width="stretch")
        st.caption("95% intervals from a bootstrap over voting countries. Binary factors: the difference between having it and not. Distance and migrant share: per 10× increase.")
with t9:
    info = ev["host"]
    st.markdown(f"""**Is the host penalty trustworthy?** Partly. It's real in the modern era, but it has a confounder and it depends on the era.

- **Confounded:** the host is almost always **last year's winner** ({info['host_is_last_winner']} of {info['hosts_2004_2024']} hosts, 2004–2024). The data can't separate *hosting* from *defending a title*.
- **Beyond regression to the mean:** any country that did well last year tends to do a bit worse. But winners fall much further than runners-up (table below), and the penalty grows once last year's score is controlled for.
- **Not just skipping the semi:** hosts go straight to the final, like the Big 5, who also do worse (about −0.45 SD). The host penalty remains with that controlled.
- **Era-dependent:** before 2004 hosts did well, winning 6 times. Since the semi-finals began, no host has won ({info['host_wins_since_2004']} of 22).
- **Small sample:** about 20 hosts, so treat the size as rough and don't use it as a rule.""")
    c1, c2 = st.columns(2)
    c1.dataframe(table("ev_host_models").round(2).rename(columns={"model": "Model", "host_effect": "Host effect (SD)", "lo95": "95% low", "hi95": "95% high"}), hide_index=True, width="stretch")
    c2.dataframe(table("ev_host_drop").round(2).rename(columns={"group": "Group", "last_year_z": "Score last year (SD)", "this_year_z": "Score this year (SD)", "hosts": "Hosts"}), hide_index=True, width="stretch")
    c1.dataframe(table("ev_host_eras").round(2).rename(columns={"era": "Era", "hosts": "Hosts", "host_wins": "Host wins", "median_place_pct": "Median place (0 = top)"}), hide_index=True, width="stretch")
    b = table("ev_back_to_back").assign(country=lambda x: x.country.map(cname), hosted=lambda x: x.hosted.map({1: "Yes", 0: "No"}))
    c2.markdown("**Back-to-back winners**")
    c2.dataframe(b.rename(columns={"years": "Years", "country": "Country", "hosted": "Won while hosting"}), hide_index=True, width="stretch")
    st.caption("All five back-to-back wins happened while the winner was hosting, and all before 1995. Ireland won three in a row (1992–1994).")
with t4:
    d = table("ev_songwriters")
    st.plotly_chart(px.bar(d, x="writer_band", y="mean_score", text="n", labels={"writer_band": "", "mean_score": "Mean final position score (1 = winner, 0 = last or not qualified)"}), width="stretch")
    st.caption("The closest thing to a song-quality signal in the dataset. Adding it to the prediction model improved rank correlation by only about 0.02, "
               "because only about a third of entries have a traceable songwriter history, so it was left out.")
with t5:
    d = table("ev_friends")
    st.plotly_chart(px.bar(d, x="band", y="rate", text="n", labels={"band": "Voters in the semi with strong prior affinity to the entry", "rate": "Qualification rate"}), width="stretch")
    w = table("ev_friends_within")
    st.dataframe(w.rename(columns={"form_t": "Country's recent form", "friends_3plus": "3+ friendly voters", "mean": "Qualification rate", "count": "Entries"}).round(2),
                 hide_index=True, width="stretch")
    st.caption("Qualification rises with friendly voters overall, but within groups of similar recent form the effect is inconsistent. Friends and form overlap, so this is weak evidence.")

st.header("Missing data: what would make the model much stronger")
st.markdown("""
The dataset explains **who favours whom** well, but not **why**, and it can't see **song quality**. These external sources would fill those gaps.

| Priority | Data | What it would test or add | Where to get it |
|---|---|---|---|
| ✓ | **Bilateral migrant stock** (now connected: Eurostat, see the Diaspora tab) (people born in country A living in country B) | The diaspora hypothesis: does the televote favour countries with large emigrant communities? Directly tests the jury-vs-public pattern above. | UN DESA *International Migrant Stock* (free download, origin × destination, 5-year steps); Eurostat API datasets `migr_pop3ctb` / `migr_pop1ctz` (annual, EU/EFTA destinations); OECD International Migration Database |
| 2 | **Population of each voting country** | Turns migrant counts into shares (100k migrants mean more in Ireland than in Germany) | World Bank API, indicator `SP.POP.TOTL` |
| ✓ | **Geography and shared culture** (now connected: CEPII GeoDist, see the Geography tab) | Neighbour and language hypotheses; separates "neighbours" from "diaspora" | CEPII *GeoDist* dataset (free download) |
| 4 | **Pre-contest song popularity** (the missing "song quality") | The main lever for predicting winners | Google Trends (history available; `pytrends` is unofficial); YouTube Data API v3 and Spotify Web API give only *current* counts, so they must be snapshotted **before** each contest to avoid leakage |
| 5 | **Pre-contest betting odds** | Usually the strongest single predictor of the winner; aggregates public expectation | Historical odds archives (no official API; e.g. snapshots of odds-comparison sites) |

**Suggested first experiment:** join (1) and (2) to the pair-level affinity, and test whether *log(migrants from entry country in voter country ÷ voter population)* predicts **televote** affinity more than **jury** affinity. If it does, that is strong, defensible evidence for diaspora voting.
""")
