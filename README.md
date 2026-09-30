# Eurovision Forecast

**Song or politics?** An evidence-based analysis of Eurovision Song Contest voting (1956–2026) and an explainable AI forecast of results, built on open data.

- **Dashboard** (Streamlit): forecast for next year's contest, key questions, country history, voting factors (voting relationships, diaspora, jury and televote, contest format), data pipeline and accuracy.
- **Pitch**: [`presentation/eurovision_pitch.mp4`](presentation/eurovision_pitch.mp4) (1:55), slides [`presentation/eurovision_pitch.pptx`](presentation/eurovision_pitch.pptx) with the narration as speaker notes, script [`presentation/script.md`](presentation/script.md).
- **Method notes**: [`METHODOLOGY.md`](METHODOLOGY.md): models, findings, test design and likely questions.

Patterns and models use 2004–2024; **2025–2026 were a sealed hold-out** for final testing (`CUTOFF_YEAR` in `src/features.py`).

## Run the dashboard

The processed tables needed by the dashboard are included, so no rebuild is needed:

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

## Rebuild everything from raw data

1. Download the **unofficial Eurovision Song Contest dataset** (https://eurovisionapi.runasp.net/) and copy its `data/senior` folder to `data/senior`. The raw dataset, including song lyrics, is not redistributed here.
2. Download external data (needs internet): `.venv/bin/python src/fetch_geography.py` (CEPII). Eurostat migration (`src/fetch_migration.py`) and UCDP conflict data (`src/fetch_conflict.py`) are already included as processed tables; rerun those scripts to refresh them.
3. `.venv/bin/python src/run_all.py`: parse → features → analysis → models → forecast.
4. Optional: `.venv/bin/python presentation/build_video.py` and `presentation/build_pptx.py` rebuild the pitch video and deck. The video narration uses Microsoft's neural text-to-speech via `edge-tts`, which sends the script text to that service.

## Repository layout

| Path | What it does |
|---|---|
| `src/build_data.py` | Raw JSON to tidy tables |
| `src/features.py` | Leak-free inputs: voting affinity, form, running order, host, semi-final results |
| `src/analysis.py` | Voting blocs (hierarchical clustering) and structural effects (OLS + bootstrap); jury vs televote |
| `src/models.py` | Baseline, ridge regression, LightGBM; rolling-origin test and hold-out |
| `src/evidence.py`, `semi_analysis.py`, `diaspora.py`, `geography.py`, `politics.py` | Statistical tests: stability, blocs, semi-finals, host and Big 5, language, Eurostat migration, CEPII geography, war (UCDP) |
| `src/fetch_migration.py`, `fetch_conflict.py`, `fetch_geography.py` | Download Eurostat, UCDP and CEPII data |
| `src/forecast.py` | Forecast engine: models trained up to 2026, probabilities calibrated on past finals, qualification chances |
| `app.py`, `views/`, `src/ui.py` | Dashboard pages, theme and explanation cards |
| `data/processed/`, `data/external/*.parquet` | Processed tables used by the dashboard |
| `presentation/` | Pitch video, deck, script, subtitles, slide images and their build scripts |
| `archive/` | Earlier pages and tests dropped for showing no effect (UNHCR refugees, GDELT news tone); not part of the app |

## Data sources and attribution

- Unofficial Eurovision Song Contest dataset: https://eurovisionapi.runasp.net/
- Eurostat, population by country of birth / citizenship (`migr_pop3ctb`, `migr_pop1ctz`, `demo_pjan`). © European Union, reuse authorised with attribution.
- CEPII GeoDist: Mayer, T. & Zignago, S. (2011), *Notes on CEPII's distances measures: the GeoDist Database*, CEPII Working Paper 2011-25.
- UCDP/PRIO Armed Conflict Dataset v26.1, Uppsala Conflict Data Program (CC BY 4.0).

## Licence

Code: MIT (see `LICENSE`). The licence covers this repository's code only; third-party data remains under its providers' terms (see above).
