# Eurovision Forecast · 2-minute hackathon pitch

**Slide 1 · Problem.** Every May, Eurovision sparks the same argument: is it the song, or is it politics? The problem: points mix song quality with voting bias. We separate the two, and forecast the result.

**Slide 2 · Data processing challenges and solutions.** The data was messy: nearly two thousand nested files, voting rules that changed twice, and five country-code systems. We normalised every vote to one scale, mapped every code, and checked that every score adds up.

**Slide 3 · Data processing pipeline.** Our pipeline runs end to end with one command: ingest, clean, build features from past years only, model, calibrate and test, then publish to the dashboard.

**Slide 4 · Core method (example).** The core idea: compare each vote with what every other country gave the same song. Lithuania gives Latvia over 9 points out of twelve, while everyone else gives about 2. That gap is bias, and it runs both ways.

**Slide 5 · ML models and methods.** Each question gets a fitting model: clustering for voting blocs, regression with bootstrap intervals for geography and diaspora, a ridge regression that predicts every country's points to every entry, and logistic calibration for probabilities.

**Slide 6 · Findings.** The findings: voting ties are stable across decades, the televote follows migrant communities while juries barely do, and hosts and the Big Five underperform.

**Slide 7 · Analysis of alternatives / model comparison.** We compared alternatives on unseen years. Voting affinity lifts pair-level accuracy from point 28 to point 48. Gradient boosting was no more accurate, so we kept the explainable ridge model.

**Slide 8 · Accuracy and performance metrics.** On performance: after the semi-finals, rank correlation doubles to point 66, the winner is in our top three in 11 of 14 finals, and we ranked the withheld 2026 winner first.

**Slide 9 · Justification and sustainability.** It is sustainable: open data, a one-command rebuild every year, and clear limits. Before the semi-finals, no one can judge a new song.

**Slide 10 · Close.** Eurovision isn't random. It's measurable.

Total: 115 s, 279 words. Voice: en-US-AndrewMultilingualNeural (+10%).
