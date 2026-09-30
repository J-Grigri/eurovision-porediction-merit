"""Build the 2-minute hackathon pitch video: slides (PIL) + neural narration (edge-tts) -> presentation/eurovision_pitch.mp4

Structure follows the evaluation criteria: data processing challenges and solutions, ML model and pipeline,
accuracy and performance metrics, justification and sustainability, presentation.
Every number on the slides and in the narration is read from data/processed, so the video matches the dashboard.
Narration uses Microsoft's neural text-to-speech via the edge-tts package (the script text is sent to that service).
Run: .venv/bin/python presentation/build_video.py
"""
import json
import subprocess
import time
import wave
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
PROC, OUT = ROOT / "data" / "processed", ROOT / "presentation"
BUILD = OUT / "build"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
EDGE_TTS = ROOT / ".venv" / "bin" / "edge-tts"
W, H = 1920, 1080
VOICE, RATE = "en-US-AndrewMultilingualNeural", "+10%"  # alternatives: en-GB-RyanNeural, en-GB-SoniaNeural, en-US-AvaMultilingualNeural
BG, INK, MUTED, CARD, LINE = (13, 10, 38), (244, 241, 255), (185, 178, 221), (27, 21, 70), (60, 50, 120)
PINK, BLUE, GOLD, GREEN, RED = (224, 65, 143), (57, 135, 229), (201, 133, 0), (27, 175, 122), (230, 103, 103)
N_SLIDES = 10

# ---------------------------------------------------------------- data (same sources as the dashboard)
ev = json.loads((PROC / "evidence.json").read_text())
dia = json.loads((PROC / "diaspora.json").read_text())
geo = json.loads((PROC / "geography.json").read_text())
meta = json.loads((PROC / "forecast_meta.json").read_text())
war = {r["measure"]: r for r in json.loads((PROC / "politics.json").read_text())["war"]}
py = pd.read_parquet(PROC / "jury_public_year.parquet")
votes = len(pd.read_parquet(PROC / "votes.parquet"))
entries = len(pd.read_parquet(PROC / "entries.parquet"))
pl = pd.read_parquet(PROC / "backtest_pair_level.parquet").pivot(index="model", columns="split", values="pair_corr")
bm = pd.read_parquet(PROC / "backtest_metrics.parquet")
qa = bm[bm["round"] != "final"].groupby("model").apply(lambda g: g[g.split == "dev"].qual_hit.sum() / g[g.split == "dev"].n_qual.sum(), include_groups=False)
acc = meta["accuracy"]
a_dev, e_dev, q_dev = acc["after_semis_dev"], acc["early_dev"], acc["qualify_dev"]
same_fav = int((py.jury_winner == py.public_winner).sum())
last_rank = acc["after_semis_holdout"]["winner_ranks"][-1]
assert last_rank == 1, "narration says the withheld winner was ranked first"

# worked example: average points (of 12) received from the partner vs from all other voters in the same shows, 2004-2024
_g = pd.read_parquet(PROC / "grids.parquet")
_g = _g[(_g.year >= 2004) & (_g.year <= 2024)].assign(pts=lambda x: x.y * 12)


def pair_example(v, e):
    x = _g[_g.entry == e]
    d = pd.concat([x[x.voter == v].set_index(["year", "round"]).pts.rename("v"),
                   x[x.voter != v].groupby(["year", "round"]).pts.mean().rename("o")], axis=1).dropna()
    return d.v.mean(), d.o.mean(), len(d)


LT_LV, LV_LT = pair_example("LT", "LV"), pair_example("LV", "LT")


# ---------------------------------------------------------------- drawing helpers
def font(size, weight="Regular"):
    path = "/System/Library/Fonts/Avenir Next.ttc"
    for i in range(12):
        f = ImageFont.truetype(path, size, index=i)
        if f.getname()[1] == weight:
            return f
    return ImageFont.truetype(path, size)


def canvas():
    y, x = np.mgrid[0:H, 0:W]
    img = np.zeros((H, W, 3), float) + BG
    for (cx, cy, r, col, a) in ((W * 0.88, -80, 900, (90, 25, 130), 0.9), (-100, H * 0.25, 800, (20, 45, 120), 0.8), (W * 0.6, H * 1.1, 700, (120, 20, 80), 0.45)):
        d = np.clip(1 - np.hypot(x - cx, y - cy) / r, 0, 1) ** 2 * a
        img = img * (1 - d[..., None]) + np.array(col) * d[..., None]
    im = Image.fromarray(img.astype(np.uint8))
    dots = Image.new("L", (W, H), 0)
    dd = ImageDraw.Draw(dots)
    for yy in range(0, H, 34):
        for xx in range(0, W, 34):
            dd.ellipse([xx, yy, xx + 2, yy + 2], fill=38)
    im.paste(Image.new("RGB", (W, H), (255, 255, 255)), mask=dots)
    return im, ImageDraw.Draw(im)


def header(d, eyebrow, title, n):
    d.rounded_rectangle([110, 96, 118, 150], 4, fill=PINK)
    d.text((140, 92), eyebrow.upper(), font=font(26, "Demi Bold"), fill=(255, 190, 225))
    d.text((140, 126), title, font=font(70, "Heavy"), fill=INK)
    d.text((W - 150, 1010), f"{n} / {N_SLIDES}", font=font(24, "Medium"), fill=MUTED)
    d.text((110, 1010), "Eurovision Forecast · evidence-based voting analysis", font=font(24, "Medium"), fill=MUTED)


def wrap(d, text, xy, width, f, fill, lh):
    x, y = xy; line = ""
    for w in text.split():
        t = (line + " " + w).strip()
        if d.textlength(t, font=f) > width:
            d.text((x, y), line, font=f, fill=fill); y += lh; line = w
        else:
            line = t
    if line:
        d.text((x, y), line, font=f, fill=fill); y += lh
    return y


def card(d, box, label, value, note, accent=PINK, vsize=70):
    x0, y0, x1, y1 = box
    d.rounded_rectangle(box, 22, fill=CARD, outline=LINE, width=2)
    d.rounded_rectangle([x0, y0 + 26, x0 + 6, y1 - 26], 3, fill=accent)
    d.text((x0 + 34, y0 + 24), label, font=font(26, "Demi Bold"), fill=MUTED)
    d.text((x0 + 34, y0 + 62), value, font=font(vsize, "Heavy"), fill=INK)
    wrap(d, note, (x0 + 34, y0 + 72 + vsize + 12), x1 - x0 - 64, font(24, "Medium"), MUTED, 31)


def arrow(d, x, y, col=MUTED):
    d.polygon([(x, y - 14), (x + 22, y), (x, y + 14)], fill=col)


def row_table(d, rows, y, left_w, h=112, lsize=34, rsize=28):
    for lab, txt, col in rows:
        d.rounded_rectangle([140, y, 1780, y + h - 14], 16, fill=CARD, outline=LINE, width=2)
        d.rounded_rectangle([140, y + 18, 146, y + h - 32], 3, fill=col)
        wrap(d, lab, (175, y + 16), left_w - 60, font(lsize, "Demi Bold"), INK, int(lsize * 1.2))
        wrap(d, txt, (140 + left_w, y + 18), 1780 - 140 - left_w - 40, font(rsize, "Medium"), MUTED, int(rsize * 1.3))
        y += h


# ---------------------------------------------------------------- slides
def s1():
    im, d = canvas()
    d.text((140, 250), "EUROVISION SONG CONTEST · AI FORECAST", font=font(32, "Demi Bold"), fill=(255, 190, 225))
    d.text((132, 300), "Song", font=font(190, "Heavy"), fill=INK)
    d.text((132, 490), "or politics?", font=font(190, "Heavy"), fill=PINK)
    wrap(d, "The problem: points mix song quality with voting bias. We separate the two, and forecast the result.", (140, 740), 1300, font(44, "Medium"), MUTED, 58)
    return im


def s2():
    im, d = canvas(); header(d, "Data processing", "Challenges and solutions", 2)
    rows = [("1,943 nested raw files", f"Parsed into tidy tables: {votes:,} votes, {entries:,} entries. Every score verified to add up.", BLUE),
            ("Voting rules changed", "Jury and televote split in 2016, rest-of-world vote in 2023: every vote normalised to a share of its maximum.", GOLD),
            ("Song quality hides bias", "Leave-one-out residual: each vote minus what all other countries gave the same song.", PINK),
            ("Countries split, left, renamed", "Five country-code systems mapped; Yugoslav lineage merged; only real participants vote.", GREEN),
            ("Risk of data leakage", "Features use earlier years only; 2025–2026 sealed as a final test.", RED)]
    row_table(d, rows, 250, 620, h=146, lsize=34, rsize=29)
    return im


def s3():
    im, d = canvas(); header(d, "Data processing pipeline", "One command, end to end", 3)
    steps = [("Ingest", "Contest JSON, Eurostat API, CEPII, UCDP"), ("Clean & join", "Tidy tables, code mapping, integrity checks"),
             ("Features", "Only past years: affinity, form, running order, host, semis"), ("Model", "Ridge regression per voter→entry pair"),
             ("Validate", "Logistic probabilities, rolling-origin test, sealed hold-out"), ("Dashboard", "Streamlit forecast and evidence pages")]
    w = 250
    for i, (t, s) in enumerate(steps):
        x = 140 + i * 280
        d.rounded_rectangle([x, 270, x + w, 560], 20, fill=CARD, outline=PINK if i in (3, 4) else LINE, width=2)
        d.text((x + 22, 288), f"{i + 1}", font=font(44, "Heavy"), fill=PINK)
        d.text((x + 22, 350), t, font=font(30, "Heavy"), fill=INK)
        wrap(d, s.replace("→", " to "), (x + 22, 398), w - 40, font(22, "Medium"), MUTED, 29)
        if i < len(steps) - 1:
            arrow(d, x + w + 6, 415)
    card(d, (140, 620, 640, 900), "Contests", "71", "1956–2026, plus 3 open datasets", BLUE)
    card(d, (700, 620, 1200, 900), "Vote records", f"{votes:,}", "Country-to-country points", PINK)
    card(d, (1260, 620, 1780, 900), "Rebuild", "1 command", "python src/run_all.py", GREEN)
    return im


def s4():
    im, d = canvas(); header(d, "The idea", "Remove the song, measure the bias", 4)
    wrap(d, "Compare each vote with what all other countries gave the same song in the same show. The gap is voting bias with song quality removed.",
         (140, 250), 1640, font(38, "Medium"), INK, 52)
    d.rounded_rectangle([140, 420, 1780, 930], 26, fill=CARD, outline=LINE, width=2)
    d.text((190, 450), "Average points (of 12) per show, 2004–2024", font=font(30, "Demi Bold"), fill=MUTED)
    base, sc, y = 560, 60, 515
    for title, (v, o, n), vname in (("Lithuania to Latvia", LT_LV, "From Lithuania"), ("Latvia to Lithuania", LV_LT, "From Latvia")):
        d.text((190, y), title, font=font(32, "Heavy"), fill=INK)
        d.text((190 + d.textlength(title, font=font(32, "Heavy")) + 20, y + 8), f"{n} shows", font=font(24, "Medium"), fill=MUTED)
        for j, (lab, val, col) in enumerate(((vname, v, PINK), ("From all others", o, BLUE))):
            yy = y + 52 + j * 56
            d.text((190, yy + 6), lab, font=font(26, "Medium"), fill=MUTED)
            d.rounded_rectangle([base, yy, base + val * sc, yy + 44], 8, fill=col)
            d.text((base + val * sc + 16, yy + 4), f"{val:.1f}", font=font(30, "Demi Bold"), fill=INK)
        d.text((1500, y + 50), f"+{v - o:.1f}", font=font(76, "Heavy"), fill=PINK)
        d.text((1506, y + 140), "points of bias", font=font(24, "Demi Bold"), fill=MUTED)
        y += 200
    return im


def s5():
    im, d = canvas(); header(d, "ML models and methods", "The right model for each question", 5)
    rows = [("Voting bias", "Leave-one-out residual per vote, averaged per country pair (10-year half-life, shrinkage)", PINK),
            ("Voting blocs", "Hierarchical clustering (average linkage) of the affinity matrix", BLUE),
            ("Why ties exist", "OLS regression on geography and migrant share, bootstrap confidence intervals", GOLD),
            ("Points forecast", "Ridge regression: every voting country's points to every entry", GREEN),
            ("Probabilities", "Logistic calibration: win, top 5 and qualification chances", PINK),
            ("Validation", "Rolling-origin test 2010–2024; 2025–2026 withheld", RED)]
    row_table(d, rows, 250, 520, h=122, lsize=34, rsize=30)
    return im


def s6():
    im, d = canvas(); header(d, "Findings", "Bias is real, and measurable", 6)
    tiles = [("Stable ties", f"r = {ev['stability']['r']:.2f}", "A decade of voting predicts the next", PINK),
             ("Geography", f"{geo['total']['r2'] * 100:.0f}%", "Of voting ties explained by borders, language, history", BLUE),
             ("Diaspora", f"r = {dia['r_public']:.2f}", f"Televote follows migrants (jury r = {dia['r_jury']:.2f})", GOLD),
             ("Same favourite", f"{same_fav} of {len(py)}", "Finals where jury and televote agreed", PINK),
             ("Countries at war", f"+{war['public_minus_jury']['difference']:.1f} SD", "Televote over jury (6 cases)", RED),
             ("Hosts & Big 5", f"0 of {ev['host']['hosts_2004_2024'] + 2}", f"Host wins since 2004; Big 5 {ev['host']['big5']['effect']:+.2f} SD", GREEN)]
    for i, (lab, val, note, col) in enumerate(tiles):
        x = 140 + (i % 3) * 555; y = 250 + (i // 3) * 350
        card(d, (x, y, x + 525, y + 320), lab, val, note, col, vsize=64)
    return im


def s7():
    im, d = canvas(); header(d, "Analysis of alternatives", "Model comparison on unseen years", 7)
    models = [("Baseline: recent form", "baseline", MUTED), ("Ridge without affinity", "ridge_no_affinity_pre_contest", BLUE),
              ("Gradient boosting (LightGBM)", "lgbm_pre_contest", GOLD), ("Ridge regression (chosen)", "ridge_pre_contest", PINK)]
    d.text((140, 250), "Pair-level accuracy: correlation of predicted and actual points", font=font(30, "Demi Bold"), fill=MUTED)
    x0, sc = 720, 1700
    for i, (lab, key, col) in enumerate(models):
        y = 320 + i * 110
        d.text((140, y + 14), lab, font=font(32, "Demi Bold"), fill=INK)
        for j, (split, alpha) in enumerate((("dev", 1.0), ("holdout", 0.55))):
            v = pl.loc[key, split]; c = tuple(int(BG[k] + (col[k] - BG[k]) * alpha) for k in range(3))
            d.rounded_rectangle([x0, y + j * 42, x0 + v * sc, y + j * 42 + 34], 6, fill=c)
            d.text((x0 + v * sc + 14, y + j * 42), f"{v:.2f}", font=font(26, "Demi Bold"), fill=INK)
    d.text((x0, 770), "Upper bar: test years 2010–2024 · lower bar: withheld 2025–2026", font=font(24, "Medium"), fill=MUTED)
    wrap(d, "Also tested and dropped: song tempo, key and language; refugee flows; news tone. None improved accuracy.",
         (140, 840), 1640, font(30, "Medium"), INK, 40)
    return im


def s8():
    im, d = canvas(); header(d, "Accuracy and performance", "Tested like a real forecast", 8)
    card(d, (140, 250, 900, 560), "Rank correlation, before → after semi-finals".replace("→", "to"), f"{e_dev['rank_corr']:.2f} → {a_dev['rank_corr']:.2f}".replace("→", "to"),
         "1 = perfect order; semi-final results double accuracy", PINK)
    card(d, (960, 250, 1780, 560), "Winner in predicted top 3", f"{a_dev['winner_top3']} of {a_dev['finals']}", "Past finals after the semi-finals, each predicted from earlier years only", GREEN)
    card(d, (140, 600, 660, 920), "Withheld 2026", f"#{last_rank}", "Rank given to the actual winner", GOLD)
    card(d, (700, 600, 1220, 920), "Top 5 found", f"{a_dev['top5']:.1f} of 5", f"After the semis (before: {e_dev['top5']:.1f}; chance ≈ 1)", BLUE)
    card(d, (1260, 600, 1780, 920), "Qualifiers found", f"{q_dev['correct'] / q_dev['total'] * 100:.0f}%",
         f"Form baseline {qa['baseline'] * 100:.0f}%, random {q_dev['base_rate'] * 100:.0f}%", RED)
    return im


def s9():
    im, d = canvas(); header(d, "Justification and sustainability", "Trustworthy, and built to last", 9)
    d.text((140, 250), "Sustainable", font=font(40, "Heavy"), fill=GREEN)
    y = 320
    for t in ("Open data only: dataset, Eurostat, CEPII, UCDP", "One command rebuilds everything each year", "Explainable model: every forecast broken into its inputs",
              "Tested on unseen years, compared with baselines"):
        d.ellipse([140, y + 16, 154, y + 30], fill=GREEN); y = wrap(d, t, (176, y), 740, font(32, "Medium"), INK, 42) + 22
    d.text((1010, 250), "Limitations", font=font(40, "Heavy"), fill=PINK)
    y = 320
    for t in ("Song quality is unknown before the semi-finals: no entry above ~12%", "Associations, not causes",
              "Small samples: 6 war cases, ~20 hosts", "Qualification gain over the form baseline is small"):
        d.ellipse([1010, y + 16, 1024, y + 30], fill=PINK); y = wrap(d, t, (1046, y), 740, font(32, "Medium"), INK, 42) + 22
    d.rounded_rectangle([140, 800, 1780, 930], 22, fill=CARD, outline=LINE, width=2)
    wrap(d, "Next: add pre-contest streaming and betting signals to judge song quality before the semi-finals.", (180, 832), 1560, font(32, "Demi Bold"), INK, 42)
    return im


def s10():
    im, d = canvas()
    d.text((132, 330), "Eurovision isn't random.", font=font(120, "Heavy"), fill=INK)
    d.text((132, 480), "It's measurable.", font=font(120, "Heavy"), fill=PINK)
    wrap(d, "A reproducible pipeline and an explainable, tested AI forecast, built on open data.", (140, 680), 1300, font(40, "Medium"), MUTED, 54)
    d.text((140, 900), "Thank you", font=font(44, "Demi Bold"), fill=INK)
    return im


NARRATION = [
    "Every May, Eurovision sparks the same argument: is it the song, or is it politics? The problem: points mix song quality with voting bias. "
    "We separate the two, and forecast the result.",
    "The data was messy: nearly two thousand nested files, voting rules that changed twice, and five country-code systems. We normalised every "
    "vote to one scale, mapped every code, and checked that every score adds up.",
    "Our pipeline runs end to end with one command: ingest, clean, build features from past years only, model, calibrate and test, then publish to the dashboard.",
    f"The core idea: compare each vote with what every other country gave the same song. Lithuania gives Latvia over {int(LT_LV[0])} points out of twelve, "
    f"while everyone else gives about {round(LT_LV[1])}. That gap is bias, and it runs both ways.",
    "Each question gets a fitting model: clustering for voting blocs, regression with bootstrap intervals for geography and diaspora, a ridge "
    "regression that predicts every country's points to every entry, and logistic calibration for probabilities.",
    "The findings: voting ties are stable across decades, the televote follows migrant communities while juries barely do, and hosts and the "
    "Big Five underperform.",
    f"We compared alternatives on unseen years. Voting affinity lifts pair-level accuracy from point {int(round(pl.loc['ridge_no_affinity_pre_contest', 'dev'] * 100))} "
    f"to point {int(round(pl.loc['ridge_pre_contest', 'dev'] * 100))}. Gradient boosting was no more accurate, so we kept the explainable ridge model.",
    f"On performance: after the semi-finals, rank correlation doubles to point {int(round(a_dev['rank_corr'] * 100))}, the winner is in our top three in "
    f"{a_dev['winner_top3']} of {a_dev['finals']} finals, and we ranked the withheld 2026 winner first.",
    "It is sustainable: open data, a one-command rebuild every year, and clear limits. Before the semi-finals, no one can judge a new song.",
    "Eurovision isn't random. It's measurable.",
]
SLIDES = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10]
assert len(SLIDES) == len(NARRATION) == N_SLIDES


def duration(path):
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def tts(text, path):
    for attempt in range(4):  # the service occasionally drops a request
        r = subprocess.run([str(EDGE_TTS), "--voice", VOICE, f"--rate={RATE}", "--text", text, "--write-media", str(path)], capture_output=True, text=True)
        if r.returncode == 0 and path.exists() and path.stat().st_size > 1000:
            return
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"text-to-speech failed: {r.stderr[:300]}")


def main():
    BUILD.mkdir(parents=True, exist_ok=True)
    for old in BUILD.glob("*"):
        old.unlink()
    parts, t, srt = [], 0.0, []
    for i, (make, text) in enumerate(zip(SLIDES, NARRATION), 1):
        png, mp3, wav, mp4 = (BUILD / f"slide{i:02d}.png", BUILD / f"voice{i:02d}.mp3", BUILD / f"voice{i:02d}.wav", BUILD / f"part{i:02d}.mp4")
        make().save(png)
        tts(text, mp3)
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(mp3), "-ar", "44100", "-ac", "2", str(wav)], check=True)
        dur = duration(wav) + (0.55 if i < N_SLIDES else 1.4)
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-loop", "1", "-t", f"{dur:.2f}", "-i", str(png), "-i", str(wav),
                        "-filter_complex", f"[0:v]fade=t=in:st=0:d=0.3,fade=t=out:st={dur - 0.3:.2f}:d=0.3,format=yuv420p[v];[1:a]apad[a]",
                        "-map", "[v]", "-map", "[a]", "-t", f"{dur:.2f}", "-r", "30", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                        "-c:a", "aac", "-b:a", "192k", str(mp4)], check=True)
        stamp = lambda s: f"{int(s // 3600):02d}:{int(s % 3600 // 60):02d}:{int(s % 60):02d},{int(s * 1000 % 1000):03d}"
        srt.append(f"{i}\n{stamp(t + 0.15)} --> {stamp(t + dur - 0.4)}\n{text}\n")
        parts.append(mp4); t += dur
        print(f"slide {i}: {dur:.1f}s")
    (BUILD / "parts.txt").write_text("".join(f"file '{p}'\n" for p in parts))
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(BUILD / "parts.txt"), "-c", "copy",
                    str(OUT / "eurovision_pitch.mp4")], check=True)
    (OUT / "eurovision_pitch.srt").write_text("\n".join(srt))
    criteria = ["Problem", "Data processing challenges and solutions", "Data processing pipeline", "Core method (example)", "ML models and methods",
                "Findings", "Analysis of alternatives / model comparison", "Accuracy and performance metrics", "Justification and sustainability", "Close"]
    (OUT / "script.md").write_text("# Eurovision Forecast · 2-minute hackathon pitch\n\n" + "\n\n".join(
        f"**Slide {i} · {c}.** {n}" for i, (c, n) in enumerate(zip(criteria, NARRATION), 1))
        + f"\n\nTotal: {t:.0f} s, {sum(len(n.split()) for n in NARRATION)} words. Voice: {VOICE} ({RATE}).\n")
    print(f"total {t:.1f}s -> {OUT / 'eurovision_pitch.mp4'}")


if __name__ == "__main__":
    main()
