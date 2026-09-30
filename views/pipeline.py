import html

import pandas as pd
import streamlit as st

from src.app_utils import require_data, table
from ui import explain, hero, lead, method, stats, theme

require_data()
theme()
votes, entries, results = table("votes"), table("entries"), table("results")

hero("Methodology", "Data pipeline", "How raw contest files and three open datasets become tested forecasts: the processing steps, the problems "
     "met along the way and how each was solved.")
stats([("Raw files parsed", "1,943", "Nested JSON: one per show and per entry"),
       ("Vote records", f"{len(votes):,}", "Country-to-country points, 1957–2026"),
       ("Entries", f"{len(entries):,}", "71 contests, 1956–2026"),
       ("External sources", "3", "Eurostat, CEPII GeoDist, UCDP")])

st.markdown("### Pipeline")
steps = [("1 · Ingest", "Contest JSON; Eurostat REST API; CEPII and UCDP downloads", "build_data.py · fetch_migration.py · fetch_conflict.py"),
         ("2 · Clean & join", "Tidy tables, country-code mapping, rule normalisation, integrity checks", "build_data.py"),
         ("3 · Features", "Leak-free: voting affinity, form, running order, host, semi-final placing", "features.py"),
         ("4 · Analysis", "Clustering, OLS with bootstrap, permutation tests", "analysis.py · evidence.py · diaspora.py · geography.py · politics.py"),
         ("5 · Models", "Ridge regression per voter→entry pair; baselines and LightGBM for comparison", "models.py · forecast.py"),
         ("6 · Calibration & test", "Logistic probabilities; rolling-origin test; sealed 2025–2026 hold-out", "forecast.py"),
         ("7 · Dashboard", "Streamlit app reading the processed tables", "app.py · views/")]
cols = st.columns(len(steps))
for c, (t, d, f) in zip(cols, steps):
    c.markdown(f'<div class="esc-stat" style="min-height:230px"><div class="k" style="color:#ffb3d9;font-weight:700">{html.escape(t)}</div>'
               f'<div class="n" style="font-size:13.5px;color:#efeaff;margin:6px 0 8px">{html.escape(d)}</div>'
               f'<div class="n" style="font-family:monospace;font-size:11.5px">{html.escape(f)}</div></div>', unsafe_allow_html=True)
explain("Shows the end-to-end flow from raw data to the dashboard.",
        "Every step is a script, and <code>python src/run_all.py</code> rebuilds all outputs from the raw files. Downloads of external data run "
        "separately, once per update.")

st.markdown("### Data processing challenges and solutions")
rows = [
    ("Raw format", "1,943 nested JSON files, one per show and per entry, with inconsistent optional fields.",
     "Parsed into four tidy tables (contests, entries, results, votes). Integrity check: every entry's country-by-country points sum exactly to its published total."),
    ("Changing voting rules", "One 12-point set until 2015; separate jury and televote sets since 2016; a rest-of-world televote since 2023; televote-only semi-finals 2023–2025.",
     "Points expressed as a share of each voter's maximum (12 or 24). Jury-vs-televote analyses use only years with split results."),
    ("Song quality hides bias", "A country may award 12 points simply because the song is good.",
     "Leave-one-out residual: each vote minus what all other voters gave the same song in the same show."),
    ("Changing countries", "Yugoslavia and Serbia & Montenegro became several states; Russia and Belarus were excluded from 2022; several countries withdrew in 2026.",
     "Lineage merged into Serbia; the forecast lets only actual participants vote."),
    ("Irregular editions", "1956 had two songs per country and published only the winner; 1969 ended in a four-way tie; 2020 was cancelled; one entry was disqualified in 2024.",
     "Handled explicitly: 1956 (no vote breakdown) is used only for results, ties are kept, 2020 is skipped and the disqualified entry is removed."),
    ("Five country-code systems", "ISO-2 (dataset), ISO-3 (CEPII), Eurostat codes (EL, UK), CEPII legacy codes (ROM, YUG) and UCDP's Gleditsch-Ward numbers.",
     "Explicit mapping tables in each fetch script; codes outside the contest's countries are dropped."),
    ("Gaps in external data", "Eurostat has no country-of-birth figures for Germany, Portugal and three others; UCDP counts wars by calendar year, not by the May contest.",
     "Citizenship counts used as a flagged fallback; a country is at war if fighting took place in the previous year or began before May."),
    ("Data leakage", "Using a year's own results to predict it would inflate accuracy.",
     "All features use earlier years only; each year is tested on a model trained without it; 2025–2026 were sealed for a final test."),
    ("Sparse history", "New country pairs and debutants have little or no voting history.",
     "Affinity shrunk towards zero (pseudo-count 3) with a 10-year half-life; missing inputs set to the training average."),
    ("Few winners to calibrate on", "Only 14 test finals, so step-wise probabilities gave some placings 0%.",
     "Smooth logistic calibration on projected placing, normalised so the field's probabilities add up."),
]
st.dataframe(pd.DataFrame(rows, columns=["Challenge", "Problem", "Solution"]), hide_index=True, width="stretch",
             column_config={"Challenge": st.column_config.TextColumn(width="small"), "Problem": st.column_config.TextColumn(width="large"),
                            "Solution": st.column_config.TextColumn(width="large")})
explain("Documents the practical problems in the data and the decision taken for each.",
        "Each solution is implemented in code and applied identically in testing and in the live forecast.")

st.markdown("### Justification and sustainability")
method([
    ("Reproducible", "One command rebuilds every table, model, test and chart from the raw files. Dependencies are pinned in requirements.txt."),
    ("Open data", "All inputs are public: the unofficial Eurovision dataset (GitHub), Eurostat (API), CEPII GeoDist and UCDP (downloads). No paid or private data."),
    ("Annual update", "Add the new contest's files and rerun. The feature history, models and calibration update automatically, and the new year becomes the next test case."),
    ("Honest testing", "Rolling-origin evaluation (each year predicted from earlier years only), comparison with simple baselines, and a withheld 2025–2026 test."),
    ("Interpretable", "Ridge regression was chosen over gradient boosting at equal accuracy, so every forecast can be broken down into its inputs."),
], title="Why the results can be trusted and maintained")
