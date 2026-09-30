# Eurovision voting patterns: method and defence notes

**Question.** Do geopolitical, cultural and structural patterns shape Eurovision voting, and how far can voting be predicted from this dataset alone?

**Data.** The unofficial Eurovision Song Contest dataset only (senior contest). 71 contests, 1,830 entries, about 83,500 voter→entry records. Patterns and models use 2004–2024 (20 contests; 2020 was cancelled). **2025–2026 are a sealed hold-out**: never used for training, history, tuning or analysis.

## The one idea: remove song quality to see bias

A country giving 12 points may simply like a good song. So for every vote we compute

> **affinity residual** = points the voter gave the entry − average points all *other* voters gave that same entry (both as a share of the maximum a voter can give)

Averaged over years for each voter→country pair, this measures bias between countries with song quality netted out. Recent years count more (half-life 10 years), and pairs seen only a few times are shrunk toward zero.

## The five models

| # | Model | Purpose | Key result |
|---|---|---|---|
| 1 | **Baseline** (country's recent average score) | The bar to beat | Pair-level correlation 0.24 (dev) |
| 2 | **Ridge regression** on 6 inputs (+3 after the semis) | Interpretable predictor | Correlation 0.48 dev, 0.48 hold-out |
| 3 | **LightGBM** on the same inputs | Does non-linearity help? | Same as Ridge (0.48 / 0.44), so no |
| 4 | **Hierarchical clustering** of countries by affinity | Finds blocs | Balkan, Caucasus, Nordic/Baltic, Mediterranean, Western European groups |
| 5 | **OLS regression + bootstrap over years** | Structural effects | Host −0.69 SD, first third −0.28 SD, last third +0.23 SD |

Everything else tried (song traits such as tempo, key, language and writers; entry-level models; a reverse-affinity input; similar-voter affinity; Monte Carlo win probabilities) added nothing measurable and was removed.

**Win probabilities in the dashboard** come from a logistic regression of past outcomes (win, top 5, top 10) on log(projected placing), fitted on the 14 development finals (2010–2024) and normalised so the field sums to 1, 5 and 10. This replaced an earlier step-wise estimate, which with only 14 winners made low placings look impossible (0%). Before the semi-finals the curve is flat (leader ≈ 12%, last finalist ≈ 2%), because 6 of 14 winners came from outside the projected top 10. After the semi-finals it is steep (projected 1st ≈ 31%).

## Findings

1. **Voting ties are real and large.** Greece→Cyprus (+0.74 of the maximum, over 22 contests), Bosnia→Croatia, North Macedonia→Albania, Moldova→Romania. Within-bloc affinity averages +0.16; across blocs −0.04.
2. **Affinity is the input that matters.** Removing it cuts pair-level correlation from 0.48 to 0.28 (dev) and 0.32 (hold-out).
3. **Structure matters.** Hosts do worse than peers (−0.69 SD); the first third of the running order does worse and the last third does better; a country's recent form helps (+0.28 SD per SD). All 95% intervals exclude zero.
4. **Jury and public disagree.** 2016–2024: average rank correlation 0.41, and they chose the same winner in 1 of 8 finals.
5. **Winner prediction is limited.** Voting relationships are predictable; song quality is not in the data. Before the contest, winner picks are near chance. After the semi-finals, final rank correlation rises from about 0.34 to about 0.64 (dev).

## Statistical evidence (`src/evidence.py`)

| Strength | Claim | Result |
|---|---|---|
| Strong | Ties are stable: affinity 2004–2013 predicts affinity 2014–2024 | r = 0.72, 1,594 pairs |
| Strong | Ties are mutual: A→B predicts B→A | r = 0.68, permutation p < 0.001 |
| Strong | Blocs found on 2004–2013 still hold in 2014–2024 | +1.8 of 12 points within bloc; random blocs ≤ +0.2 |
| Strong | Country bias is stronger in the televote than the jury (same voters, same songs, 2016–2024) | 1.24× the spread, 95% CI 1.17–1.31 |
| Strong | A finalist's semi-final score predicts its final score (within-show z-scores, 2008–2024) | r = 0.80, 319 finalists; adding it to the model: +0.31 rank corr (95% CI +0.20 to +0.42) |
| Strong | Jury and public are two separate, consistent tastes (2016–2022) | semi jury → final jury r = 0.80; semi jury → final public r = 0.12 |
| Strong | Televote follows migrant communities; juries don't (Eurostat migrant stock, 918 pairs, 2016–2024) | r = 0.40 public vs 0.10 jury (difference CI 0.23–0.38); +0.8 televote points per 10× larger community, jury affinity held fixed |
| Strong | Geography shapes voting (CEPII GeoDist: border, distance, shared language, colonial tie, former union) | R² = 0.25 over 2,034 pairs; public R² 0.21 vs jury 0.06 |
| Strong | Diaspora survives geography controls | +0.67 televote points per 10× migrants (95% CI 0.50–0.85) |
| Moderate | Hosts underperform since 2004 | −1.08 SD after controlling last year's score and the no-semi route; 0 host wins in 22. But hosts are nearly always last year's winner (can't separate), and before 2004 hosts won 6 times |
| Moderate | War on home soil (UCDP): the public rates the country far above the jury | +1.4 SD public-minus-jury gap vs other finalists, permutation p = 0.001, only 6 cases; overall result vs expectation +0.37 SD, p = 0.20 (not reliable) |
| Strong | Big 5 entries score below comparable entries | −0.45 SD (95% CI −0.65 to −0.24), 113 entries, controlling for last year's score |
| None | Singing in English helps | +0.06 SD (95% CI −0.16 to +0.23); 76% of finalists and 80% of winners sang in English since 2004 |
| Moderate | Songwriters with a good record place higher | r = 0.24, p < 0.001, 281 entries; only +0.02 rank correlation in the model, so left out |
| Weak | Friendly voters in a semi-final help qualification | ρ = 0.16, but overlaps with country form |

The pairs where the public is far warmer than the jury (Italy→Romania, Switzerland→Albania, Ireland→Lithuania, UK→Lithuania) match known emigrant communities. That is consistent with diaspora voting but can't be proven without migration data.

## Test design

- **Development:** each year 2010–2024 is predicted from earlier years only (rolling origin). No random splits, which would leak information across a contest.
- **Hold-out:** models trained on 2004–2024 predict 2025 and 2026 once. History (affinity, form) is frozen at 2024 for those years.
- **Metrics:** pair-level correlation and error (large sample, reliable); final rank correlation, winner hit, top-5 overlap, and share of semi-final qualifiers named correctly (small samples, noisy).

## Likely questions

**Why not a more complex model?** LightGBM was tested and matched Ridge; extra flexibility did not help, so the simpler, explainable model is preferred.

**How do you know affinity is bias, not just good songs?** The residual compares a voter with other voters *for the same song in the same show*, so song quality is subtracted out. And the ablation shows affinity predicts held-out votes: its removal costs about 0.2 in correlation.

**Isn't that circular / data leakage?** Affinity for year Y uses only years before Y. For the hold-out years, history stops at 2024. The only same-year information is the semi-final results, and only in the "pre-final" setting, because the semis are broadcast before the final.

**Why does winner prediction fail?** Nothing in the dataset measures how good a new song is. Ranking accuracy before the contest (about 0.34) is only slightly above the baseline (0.29). We report this rather than hide it.

**Only two hold-out finals. Is that enough?** No for winner or rank claims, and the write-up says so. The hold-out's real value is the pair-level result: about 2,900 hold-out voter→entry records, where the affinity gain holds (0.48 vs 0.32).

**Can you say *why* countries favour each other?** Partly. With Eurostat migrant stock added (`src/fetch_migration.py`, `src/diaspora.py`), the size of a migrant community predicts **televote** affinity but hardly **jury** affinity, which points to diaspora voting. Shared history, language or taste may explain the rest; juries capture those too, which is why jury affinity is the control.

**Why doesn't migration data improve the prediction model?** Because past voting affinity already contains the diaspora effect. Migration explains *why* the bias exists; affinity already *measures* it. It could still help for new situations, such as a fast-growing community.

**Is the host penalty real?** In 2004–2024, yes, and it's bigger than regression to the mean or skipping the semi explains. But 18 of 20 hosts were last year's winners, so "hosting" and "defending the title" can't be separated. Hosts did well before 2004, and all five back-to-back wins (Spain 1968–69, Luxembourg 1972–73, Israel 1978–79, Ireland 1992–93–94) came while hosting. So it's a modern-era pattern with an unclear cause, not a rule.

**Why did you merge Serbia and Montenegro / Yugoslavia?** So the affinity history carries across the country's successor states. Only Serbia & Montenegro (2004–2005) matters in the modelling years.

**What changed in 2016?** Jury and public points are reported separately from then. Points are normalised as a share of the maximum a voter can give (12, or 24 with jury + public), so eras are comparable, but the comparison isn't perfect.

**What would you do next?** External data on migration, language and geography to test *why* ties exist (excluded here by design), and more contests as they happen to grow the hold-out.

## Limits

- Associations, not causes.
- Song quality is unobserved.
- Small samples for winner-level results.
- Data on jury and public splits exist only from 2016.

## Dashboard audit fixes (forecast engine)

- **Points budget:** each voting country's predicted shares are scaled to its real budget (116 points; rest-of-world televote 58) before summing, in both testing and the live forecast. Previously projected totals did not add up (57–125 points per voter).
- **Participation:** only participating countries vote (previously all countries seen since 2016, including Russia, Belarus and 2026 withdrawals).
- **Qualification probability:** pre-semi-final model applied to semi-final entries; standing (percentile) converted by logistic regression on past semi-finals and scaled to the 20 qualifying places. 71% of qualifiers identified (random: 58%); withheld 2025–2026: 73%.
- **Caveat:** pre-semi-final placing accuracy is measured on the actual finalists, which are unknown before the semis, so it is slightly optimistic.

