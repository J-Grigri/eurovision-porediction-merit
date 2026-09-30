import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.app_utils import PROC, cname, require_data, table
from ui import JURY, PUBLIC, chart, explain, hero, lead, method, stats, theme

require_data()
theme()
dia = json.loads((PROC / "diaspora.json").read_text())
hero("Voting factors", "Diaspora voting", "Emigrant communities continue to vote for their country of origin. The effect is a measurable driver "
     "of televote preferences and is largely absent from jury votes.")
lead("Voting data is matched with <b>Eurostat</b> migrant-stock figures (residents born in the performing country), and televote and jury "
     "results are compared for the same songs, 2016–2024.")
bands = table("diaspora_bands")
big = bands.iloc[-1]
stats([("Televote correlation", f"r = {dia['r_public']:.2f}", "Community size vs televote preference"),
       ("Jury correlation", f"r = {dia['r_jury']:.2f}", "Same voting countries and songs"),
       ("Communities above 1%", f"+{big.public * 12:.1f} pts", f"Televote premium (jury: +{big.jury * 12:.1f})"),
       ("Independent effect", "+0.7 pts", "Per 10× larger community, controlling for geography and language")])

st.markdown("### Televote premium by community size")
fig = go.Figure()
fig.add_bar(x=bands.band.astype(str), y=bands.jury * 12, name="Jury", marker_color=JURY, text=[f"{v:+.1f}" for v in bands.jury * 12], textposition="outside")
fig.add_bar(x=bands.band.astype(str), y=bands.public * 12, name="Televote", marker_color=PUBLIC, text=[f"{v:+.1f}" for v in bands.public * 12], textposition="outside")
chart(fig, 340, barmode="group", yaxis_title="Additional points (of 12)", xaxis_title="Residents born in the performing country (% of population)",
      legend=dict(orientation="h", y=1.12))
explain("Tests the diaspora hypothesis: if emigrants drive the preference, it should rise with community size and appear in the televote only.",
        "The televote premium rises sharply with community size, while the jury premium remains close to zero. "
        f"Eurostat data covers {dia['voting_countries']} voting countries, mainly in the EU and EFTA.", "Strong", "strong")

st.markdown("### Largest communities")
d = table("diaspora_pairs").head(12)
st.dataframe(pd.DataFrame({"Community": d.entry.map(cname) + " residents in " + d.voter.map(cname), "Share of population": (d.share_pct).round(1).astype(str) + "%",
                           "Televote premium (pts)": (d.public * 12).round(1), "Jury premium (pts)": (d.jury * 12).round(1)}), hide_index=True, width="stretch")
explain("Illustrates the effect with the largest communities.",
        "Examples include Russian communities in Latvia and Estonia, Polish residents in Iceland, and Italian and Portuguese residents in Switzerland. "
        "Historical voting already captures this effect, so the forecast does not use migration data directly.")

st.markdown("### Methods")
method([
    ("Data", "Eurostat REST API: <i>migr_pop3ctb</i> (population by country of birth); <i>migr_pop1ctz</i> (by citizenship) where birth data is "
             "unavailable (Germany, Portugal, Andorra, Moldova, Montenegro); <i>demo_pjan</i> (population). Jury and televote points, 2016–2024."),
    ("Measure", "Migrant share = residents born in the performing country ÷ population of the voting country, averaged over 2016–2024, log₁₀ scale. "
                f"Jury and televote affinity computed separately (leave-one-out residual per vote type, pairs with ≥3 shows): {dia['pairs_2016_2024']} pairs, "
                f"{dia['voting_countries']} voting countries."),
    ("Models", "Pearson correlation of log migrant share with televote and with jury affinity; 95% confidence interval for the difference by "
               "bootstrap over pairs (2,000 resamples). Ordinary least squares regression of televote affinity on log migrant share plus CEPII "
               "geography (distance, border, language, colonial link, former common state), bootstrap over voting countries."),
    ("Validation", "The jury acts as a control: same voting country, same songs, same year. A predictive test adding last year's migrant share to "
                   "the forecast model gave no gain, as historical affinity already captures the effect."),
    ("Limitations", "Coverage is limited to countries reporting to Eurostat (mainly EU/EFTA). Citizenship counts undercount naturalised residents. "
                    "The analysis is correlational."),
])
