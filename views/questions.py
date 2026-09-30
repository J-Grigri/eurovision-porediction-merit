import html
import json

import streamlit as st

from src.app_utils import PROC, cname, require_data, table
from forecast import predict
from ui import hero, theme

require_data()
theme()
ev = json.loads((PROC / "evidence.json").read_text())
dia = json.loads((PROC / "diaspora.json").read_text())
geo = json.loads((PROC / "geography.json").read_text())
meta = json.loads((PROC / "forecast_meta.json").read_text())
war = {r["measure"]: r for r in json.loads((PROC / "politics.json").read_text())["war"]}
se = table("structural_effects").set_index("feature")
py = table("jury_public_year")
acc = meta["accuracy"]
NEXT = meta["last_year"] + 1

hero("Overview", "Key questions", "Short answers to the questions most often asked about Eurovision results, with links to the evidence.")

E, _ = predict([c for c in meta["last_finalists"] if c not in ("RU", "BY")], meta["last_winner"], "early")
top3 = ", ".join(f"{cname(e)} ({p * 100:.0f}%)" for e, p in zip(E.entry[:3], E.p_win[:3]))
fin = table("results"); fin = fin[(fin["round"] == "final") & (fin.disqualified == 0)]
wins = fin[fin.place == 1].groupby("country").size()
most = " and ".join(cname(c) for c in wins[wins == wins.max()].index)
b5, lang = ev["host"]["big5"], ev["language"]

QA = [
    (f"Who is most likely to win in {NEXT}?",
     f"Before the semi-finals, based on the {meta['last_year']} line-up: {top3}. No entry exceeds about 12%, because song quality cannot be "
     "judged until the semi-finals. After them, the winner was among the top three projections in "
     f"{acc['after_semis_dev']['winner_top3']} of {acc['after_semis_dev']['finals']} past finals.", "views/predictor.py", "Forecast"),
    ("Will my country qualify for the final?",
     f"The Forecast page gives a qualification probability before the semi-finals. In testing it identified "
     f"{acc['qualify_dev']['correct']} of {acc['qualify_dev']['total']} qualifiers ({acc['qualify_dev']['correct'] / acc['qualify_dev']['total'] * 100:.0f}%, "
     f"against {acc['qualify_dev']['base_rate'] * 100:.0f}% for a random pick).", "views/predictor.py", "Forecast"),
    ("Which country has won the most?",
     f"{most}, with {int(wins.max())} wins each.", "views/history.py", "Country history"),
    ("Do neighbours vote for each other?",
     f"Yes. Countries sharing a border, a language or a common history give each other more points than the song alone earns, and these "
     f"relationships are stable across decades (r = {ev['stability']['r']:.2f}). Geography explains {geo['total']['r2'] * 100:.0f}% of them.",
     "views/friends.py", "Voting relationships"),
    ("Is Eurovision voting political?",
     f"Partly, and mainly through the televote. The public votes for countries with large communities living in its country (r = {dia['r_public']:.2f}; "
     f"juries r = {dia['r_jury']:.2f}), and for countries at war, whose televote exceeds their jury score by {war['public_minus_jury']['difference']:.1f} SD "
     "more than other finalists (6 cases).", "views/diaspora.py", "Diaspora voting"),
    ("Does the running order matter?",
     f"Yes. Performing in the last third is worth {se.loc['late_slot', 'effect_sd']:+.2f} SD and the first third {se.loc['early_slot', 'effect_sd']:+.2f} SD, "
     "about 1.5 to 2 places either way.", "views/show.py", "Contest format"),
    ("Do hosts or the Big 5 have an advantage?",
     f"No, the reverse. No host has won since 2004, and Big 5 entries score {b5['effect']:+.2f} SD below comparable entries.", "views/show.py", "Contest format"),
    ("Do the jury and the public agree?",
     f"Often not: they ranked the same song first in {int((py.jury_winner == py.public_winner).sum())} of {len(py)} finals (2016–2024). "
     "Every winner, however, finished in the top 5 of both.", "views/jury_public.py", "Jury and televote"),
    ("Does singing in English help?",
     f"No measurable effect: English-language entries score {lang['diff']:+.2f} SD relative to others (95% CI {lang['lo95']:+.2f} to {lang['hi95']:+.2f}). "
     f"{lang['share_finalists_english'] * 100:.0f}% of finalists and {lang['share_winners_english'] * 100:.0f}% of winners since 2004 sang in English.",
     None, None),
    ("How reliable are the forecasts?",
     f"Before the semi-finals, rank correlation {acc['early_dev']['rank_corr']:.2f}; after them, {acc['after_semis_dev']['rank_corr']:.2f} "
     "(1 = perfect). Every past final was forecast using only earlier years, and 2025–2026 were withheld for a final test.",
     "views/accuracy.py", "Accuracy"),
]

for q, a, page, label in QA:
    st.markdown(f'<div class="esc-card"><div class="row" style="font-weight:700;font-size:16px;color:#fff">{html.escape(q)}</div>'
                f'<div class="row">{a}</div></div>', unsafe_allow_html=True)
    if page:
        st.page_link(page, label=f"See: {label}", icon=":material/arrow_forward:")
