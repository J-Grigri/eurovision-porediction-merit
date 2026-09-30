"""Build a static, shareable HTML results page (report/eurovision_voting_evidence.html) from the processed outputs."""
import html
import json

import pandas as pd

from features import PROC, ROOT

COUNTRIES = json.loads((ROOT / "data" / "countries.json").read_text(encoding="utf-8"))
cn = lambda c: COUNTRIES.get(c, c)
esc = html.escape


def hbar_pairs(rows, w=640, row_h=34):
    """Grouped horizontal bars: jury (series-1) vs public (series-2), axis from -0.4 to 0.8."""
    lo, hi, left, right = -0.4, 0.8, 190, 16
    sx = lambda v: left + (v - lo) / (hi - lo) * (w - left - right)
    h = 28 + row_h * len(rows)
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Jury versus public affinity for selected country pairs">']
    for t in (-0.4, 0, 0.4, 0.8):
        x = sx(t)
        out.append(f'<line x1="{x:.1f}" y1="18" x2="{x:.1f}" y2="{h}" class="{"axis" if t == 0 else "grid"}"/>'
                   f'<text x="{x:.1f}" y="12" class="tick" text-anchor="middle">{t:+.1f}</text>')
    for i, r in enumerate(rows):
        y = 24 + i * row_h
        out.append(f'<text x="{left - 8}" y="{y + 15}" class="lab" text-anchor="end">{esc(r["pair"])}</text>')
        for j, (k, cls) in enumerate((("jury", "s1"), ("public", "s2"))):
            v = r[k]; x0, x1 = sorted((sx(0), sx(v)))
            out.append(f'<rect x="{x0:.1f}" y="{y + 3 + j * 13}" width="{max(x1 - x0, 1):.1f}" height="11" rx="2" class="{cls}">'
                       f'<title>{esc(r["pair"])} · {k}: {v:+.2f}</title></rect>')
    out.append("</svg>")
    return "".join(out)


def effects_chart(tab, w=640, row_h=40):
    lo, hi, left, right = -1.2, 0.6, 210, 20
    sx = lambda v: left + (v - lo) / (hi - lo) * (w - left - right)
    h = 30 + row_h * len(tab)
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Structural effects with 95% intervals">']
    for t in (-1.2, -0.8, -0.4, 0, 0.4):
        x = sx(t)
        out.append(f'<line x1="{x:.1f}" y1="18" x2="{x:.1f}" y2="{h}" class="{"axis" if t == 0 else "grid"}"/>'
                   f'<text x="{x:.1f}" y="12" class="tick" text-anchor="middle">{t:+.1f}</text>')
    for i, r in enumerate(tab.itertuples()):
        y = 34 + i * row_h
        out.append(f'<text x="{left - 10}" y="{y + 4}" class="lab" text-anchor="end">{esc(r.label)}</text>'
                   f'<line x1="{sx(r.lo95):.1f}" y1="{y}" x2="{sx(r.hi95):.1f}" y2="{y}" class="ci"/>'
                   f'<circle cx="{sx(r.effect_sd):.1f}" cy="{y}" r="6" class="s1"><title>{esc(r.label)}: {r.effect_sd:+.2f} SD (95% CI {r.lo95:+.2f} to {r.hi95:+.2f})</title></circle>'
                   f'<text x="{sx(r.effect_sd):.1f}" y="{y - 11}" class="val" text-anchor="middle">{r.effect_sd:+.2f}</text>')
    out.append("</svg>")
    return "".join(out)


def model_chart(rows, w=640, row_h=46):
    left, right = 250, 40
    sx = lambda v: left + v / 0.7 * (w - left - right)
    h = 30 + row_h * len(rows)
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Pair-level correlation by model, development and hold-out">']
    for t in (0, 0.2, 0.4, 0.6):
        x = sx(t)
        out.append(f'<line x1="{x:.1f}" y1="18" x2="{x:.1f}" y2="{h}" class="{"axis" if t == 0 else "grid"}"/>'
                   f'<text x="{x:.1f}" y="12" class="tick" text-anchor="middle">{t:.1f}</text>')
    for i, r in enumerate(rows):
        y = 26 + i * row_h
        out.append(f'<text x="{left - 10}" y="{y + 17}" class="lab" text-anchor="end">{esc(r["name"])}</text>')
        for j, (k, cls) in enumerate((("dev", "s1"), ("holdout", "s2"))):
            v = r[k]
            out.append(f'<rect x="{sx(0):.1f}" y="{y + 2 + j * 17}" width="{sx(v) - sx(0):.1f}" height="14" rx="2" class="{cls}">'
                       f'<title>{esc(r["name"])} · {"development" if k == "dev" else "hold-out"}: {v:.2f}</title></rect>'
                       f'<text x="{sx(v) + 5:.1f}" y="{y + 13 + j * 17}" class="val">{v:.2f}</text>')
    out.append("</svg>")
    return "".join(out)


def band_chart(b, w=640, h=260):
    """Grouped vertical bars by migrant-share band: jury (series-1) vs public (series-2)."""
    left, bottom, top_, lo, hi = 46, 40, 16, -0.1, 0.4
    sy = lambda v: top_ + (hi - v) / (hi - lo) * (h - top_ - bottom)
    cw = (w - left - 10) / len(b)
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Jury and public affinity by size of migrant community">']
    for t in (-0.1, 0, 0.1, 0.2, 0.3, 0.4):
        out.append(f'<line x1="{left}" y1="{sy(t):.1f}" x2="{w - 10}" y2="{sy(t):.1f}" class="{"axis" if t == 0 else "grid"}"/>'
                   f'<text x="{left - 6}" y="{sy(t) + 4:.1f}" class="tick" text-anchor="end">{t:+.1f}</text>')
    for i, r in enumerate(b.itertuples()):
        x = left + i * cw
        for j, (k, cls) in enumerate((("jury", "s1"), ("public", "s2"))):
            v = getattr(r, k); y0, y1 = sorted((sy(0), sy(v)))
            out.append(f'<rect x="{x + cw * 0.2 + j * cw * 0.31:.1f}" y="{y0:.1f}" width="{cw * 0.29:.1f}" height="{max(y1 - y0, 1):.1f}" rx="2" class="{cls}">'
                       f'<title>{esc(str(r.band))} · {k}: {v:+.2f} ({r.n} pairs)</title></rect>')
        out.append(f'<text x="{x + cw / 2:.1f}" y="{h - 22}" class="lab" text-anchor="middle">{esc(str(r.band))}</text>'
                   f'<text x="{x + cw / 2:.1f}" y="{h - 7}" class="tick" text-anchor="middle">{r.n} pairs</text>')
    out.append("</svg>")
    return "".join(out)


def main():
    ev = json.loads((PROC / "evidence.json").read_text())
    A = pd.read_parquet(PROC / "pair_affinity.parquet")
    top = A[A.n >= 10].nlargest(10, "mean")
    st = pd.read_parquet(PROC / "structural_effects.parquet")
    st["label"] = st.feature.map({"early_slot": "Performs in first third", "late_slot": "Performs in last third",
                                  "host": "Host country", "form_z": "Recent form (per SD)"})
    jp = pd.read_parquet(PROC / "ev_jury_public_pairs.parquet")
    jp["pair"] = jp.voter.map(cn) + " → " + jp.entry.map(cn)
    jp_rows = jp.head(8).to_dict("records")
    PL = pd.read_parquet(PROC / "backtest_pair_level.parquet")
    M = pd.read_parquet(PROC / "backtest_metrics.parquet")
    names = {"baseline": "Baseline (recent form)", "ridge_no_affinity_pre_contest": "Ridge without affinity",
             "ridge_pre_contest": "Ridge, pre-contest", "lgbm_pre_contest": "LightGBM, pre-contest",
             "ridge_pre_final": "Ridge, after semi-finals", "lgbm_pre_final": "LightGBM, after semi-finals"}
    pc = PL.pivot(index="model", columns="split", values="pair_corr")
    model_rows = [{"name": names[m], "dev": pc.loc[m, "dev"], "holdout": pc.loc[m, "holdout"]} for m in names]
    fin = M[M["round"] == "final"].groupby(["model", "split"]).agg(rank=("spearman", "mean"), win=("winner_hit", "sum"),
                                                                  n=("year", "count"), top5=("top5", "mean")).reset_index()
    fr = lambda m, s, c: fin[(fin.model == m) & (fin.split == s)][c].iloc[0]
    rank_rows = "".join(
        f"<tr><td>{names[m]}</td><td>{fr(m, 'dev', 'rank'):.2f}</td><td>{int(fr(m, 'dev', 'win'))} of {int(fr(m, 'dev', 'n'))}</td>"
        f"<td>{fr(m, 'dev', 'top5'):.1f}</td><td>{fr(m, 'holdout', 'rank'):.2f}</td><td>{int(fr(m, 'holdout', 'win'))} of 2</td></tr>"
        for m in ["baseline", "ridge_pre_contest", "lgbm_pre_contest", "ridge_pre_final", "lgbm_pre_final"])
    blocs = pd.read_parquet(PROC / "blocs.parquet")
    bloc_rows = "".join(f"<li>{esc(', '.join(cn(c) for c in g.country))}</li>" for _, g in blocs.groupby("bloc") if len(g) > 1)
    pyj = pd.read_parquet(PROC / "jury_public_year.parquet")
    dia = json.loads((PROC / "diaspora.json").read_text())
    bands = pd.read_parquet(PROC / "diaspora_bands.parquet")
    dpairs = pd.read_parquet(PROC / "diaspora_pairs.parquet").head(8)
    dia_rows = "".join(f"<tr><td>{cn(x.voter)} → {cn(x.entry)}</td><td class='num'>{x.share_pct:.1f}%</td><td class='num'>{x.public:+.2f}</td><td class='num'>{x.jury:+.2f}</td></tr>"
                       for x in dpairs.itertuples())
    geo = json.loads((PROC / "geography.json").read_text())
    gc = pd.read_parquet(PROC / "geography_coefficients.parquet")
    gcell = lambda m, f: gc[(gc.model == m) & (gc.feature == f)]
    geo_rows = "".join(
        f"<tr><td>{esc(lab)}</td>" + "".join(
            (lambda r: f"<td class='num'>{r.coef.iloc[0] * 12:+.2f}{'' if (r.lo95.iloc[0] > 0 or r.hi95.iloc[0] < 0) else ' <span class=muted>(n.s.)</span>'}</td>" if len(r) else "<td class='num muted'>–</td>")(gcell(m, f))
            for m in ["Jury affinity, 2016–2024", "Public affinity, 2016–2024", "Public affinity with migration, 2016–2024"]) + "</tr>"
        for f, lab in [("contig", "Shared border"), ("comlang_ethno", "Shared language"), ("colony", "Colonial tie"), ("smctry", "Were once the same country"),
                       ("log_dist", "Distance, per 10× further"), ("log_share", "Migrant share, per 10× larger")])
    hm = pd.read_parquet(PROC / "ev_host_models.parquet"); he = pd.read_parquet(PROC / "ev_host_eras.parquet")
    hd = pd.read_parquet(PROC / "ev_host_drop.parquet"); b2b = pd.read_parquet(PROC / "ev_back_to_back.parquet")
    host_model_rows = "".join(f"<tr><td>{esc(x.model)}</td><td class='num'>{x.host_effect:+.2f}</td><td class='num'>{x.lo95:+.2f} to {x.hi95:+.2f}</td></tr>" for x in hm.itertuples())
    host_era_rows = "".join(f"<tr><td>{esc(str(x.era))}</td><td class='num'>{x.hosts}</td><td class='num'>{x.host_wins}</td><td class='num'>{x.median_place_pct:.2f}</td></tr>" for x in he.itertuples())
    host_drop_rows = "".join(f"<tr><td>{esc(x.group)}</td><td class='num'>{x.n}</td><td class='num'>{x.last_year_z:+.2f}</td><td class='num'>{x.this_year_z:+.2f}</td></tr>" for x in hd.itertuples())
    b2b_text = ", ".join(f"{cn(x.country)} {x.years}" for x in b2b.itertuples())
    sc = pd.read_parquet(PROC / "semi_correlations.parquet")
    semi_rows = "".join(f"<tr><td>{esc(x.test)}</td><td class='num'>{x.r:.2f}</td><td class='num'>{x.n}</td></tr>" for x in sc.itertuples() if not x.test.startswith("Jury-minus"))
    sm = pd.read_parquet(PROC / "semi_models.parquet")
    sml = {"no_semi": "History only", "semi_total": "+ semi score (used)", "semi_rank": "+ semi rank", "semi_split": "+ semi jury/public split"}
    sms = sm[sm.model.isin(sml)].groupby(["model", "split"]).agg(rho=("rho", "mean"), win=("winner", "sum"), n=("year", "count"), wr=("winner_rank", "median")).reset_index()
    g_ = lambda m, sp, c: sms[(sms.model == m) & (sms.split == sp)][c].iloc[0]
    semi_model_rows = "".join(f"<tr><td>{lab}</td><td class='num'>{g_(m, 'dev', 'rho'):.2f}</td><td class='num'>{int(g_(m, 'dev', 'win'))} of {int(g_(m, 'dev', 'n'))}</td>"
                              f"<td class='num'>{g_(m, 'dev', 'wr'):.1f}</td><td class='num'>{g_(m, 'holdout', 'rho'):.2f}</td></tr>" for m, lab in sml.items())

    s, r, b, j, w, f = ev["stability"], ev["reciprocity"], ev["blocs"], ev["jury_public"], ev["songwriters"], ev["friends"]
    evidence = [
        ("Strong", "Voting ties are stable across decades", f"r = {s['r']:.2f}",
         f"Affinity measured in 2004–2013 predicts affinity in 2014–2024 across {s['pairs']:,} country pairs."),
        ("Strong", "Voting ties are mutual", f"r = {r['r']:.2f}",
         f"If A favours B, B tends to favour A ({r['pairs']:,} pairs). Shuffled pairs never came close in 2,000 tries (p &lt; 0.001)."),
        ("Strong", "Blocs hold up out of sample", f"+{b['within_pts12']:.1f} pts",
         f"Blocs found using 2004–2013 only still give each other {b['within_pts12']:.1f} extra points (of 12) in 2014–2024. Random groupings give at most +{b['null_95pct'] * 12:.1f}."),
        ("Strong", "Bias is stronger in the televote than in the jury", f"{j['ratio']:.2f}×",
         f"Same voting countries, same songs, 2016–2024: the spread of country bias is {j['ratio']:.2f}× larger for the public (95% CI {j['ratio_lo']:.2f}–{j['ratio_hi']:.2f})."),
        ("Strong", "A finalist's semi-final score predicts its final score", f"r = {sc[sc.test.str.startswith('Semi total')].r.iloc[0]:.2f}",
         "Standardised within each show, 319 finalists, 2008–2024. The semi score explains about two thirds of the variation in the final score."),
        ("Strong", "Jury and public are two separate, consistent tastes", f"r = {sc[sc.test == 'Semi jury → final jury'].r.iloc[0]:.2f}",
         f"The semi jury predicts the final jury, but barely predicts the final public (r = {sc[sc.test == 'Semi jury → final public'].r.iloc[0]:.2f}). 2016–2022."),
        ("Strong", "The televote follows migrant communities; juries don't", f"r = {dia['r_public']:.2f} vs {dia['r_jury']:.2f}",
         f"Eurostat migrant stock, {dia['pairs_2016_2024']} voter→country pairs. Each 10× larger community adds about {dia['pts_per_10x']:.1f} televote points of 12, with jury affinity held fixed."),
        ("Strong", "Geography and shared language shape voting", f"R² = {geo['total']['r2']:.2f}",
         f"CEPII GeoDist, {geo['total']['pairs']:,} pairs: border, distance, language, colonial ties and former union each matter. Geography matters far more to the public (R² {geo['public']['r2']:.2f}) than to juries ({geo['jury']['r2']:.2f})."),
        ("Moderate", "Hosts have underperformed since 2004", f"{hm.host_effect.iloc[-1]:+.2f} SD",
         "No host has won in 22 contests since 2004, even after controlling for last year's score and skipping the semi. But the host is nearly always last year's winner, and before 2004 hosts did well."),
        ("Moderate", "Proven songwriters place higher", f"r = {w['r_writer']:.2f}",
         f"Entries whose writers did well before also do well ({w['with_writer_history']} entries with traceable writers, p &lt; 0.001). Only about a third of entries have such a history."),
        ("Weak", "Friendly voters in the semi-final help qualification", f"ρ = {f['rho_friends_qualified']:.2f}",
         "Significant overall, but it largely overlaps with the country's recent form, so it is not independent evidence."),
    ]
    ev_rows = "".join(f'<tr><td><span class="pill {lvl.lower()}">{lvl}</span></td><td><strong>{c}</strong><br><span class="muted">{d}</span></td>'
                      f'<td class="num">{e}</td></tr>' for lvl, c, e, d in evidence)
    aff_rows = "".join(f"<tr><td>{cn(x.voter)} → {cn(x.entry)}</td><td class='num'>{x.mean:+.2f}</td><td class='num'>≈ {x.mean * 12:.1f}</td><td class='num'>{x.n}</td></tr>"
                       for x in top.itertuples())

    page = f"""<title>Eurovision Voting Evidence</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: single reading column, a results sheet: claim, number, test. */
:root {{
  --bg:#f7f7fb; --surface:#ffffff; --fg:#17162a; --muted:#5b5a6e; --line:#dddbe8; --accent:#3b2fa0;
  --s1:#2a78d6; --s2:#eb6834; --good:#0f7a3d; --warn:#9a6400; --weak:#8a3b3b; --chip:#eeedf7;
  --display:"Fraunces", Georgia, serif; --body:"IBM Plex Sans", system-ui, sans-serif; --mono:"IBM Plex Mono", ui-monospace, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg:#121120; --surface:#1b1a2c; --fg:#f1f0fa; --muted:#b3b1c8; --line:#34324d; --accent:#a99df5;
  --s1:#3987e5; --s2:#d95926; --good:#5fcf8c; --warn:#e2b04a; --weak:#e79a9a; --chip:#27253d; color-scheme:dark }} }}
:root[data-theme="dark"] {{
  --bg:#121120; --surface:#1b1a2c; --fg:#f1f0fa; --muted:#b3b1c8; --line:#34324d; --accent:#a99df5;
  --s1:#3987e5; --s2:#d95926; --good:#5fcf8c; --warn:#e2b04a; --weak:#e79a9a; --chip:#27253d; color-scheme:dark }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.6 var(--body); }}
main {{ max-width:760px; margin:0 auto; padding-inline:16px; padding-block:40px 64px; display:grid; gap:44px; }}
h1,h2 {{ font-family:var(--display); font-weight:600; text-wrap:balance; margin:0; line-height:1.15; }}
h1 {{ font-size:clamp(28px,5vw,40px); }} h2 {{ font-size:23px; }}
p {{ margin:0; max-width:65ch; }} section {{ display:grid; gap:14px; min-width:0; }}
.eyebrow {{ font:500 12px var(--mono); letter-spacing:.08em; text-transform:uppercase; color:var(--accent); }}
.muted {{ color:var(--muted); }} .small {{ font-size:13px; }}
.tbl {{ overflow-x:auto; border:1px solid var(--line); border-radius:8px; background:var(--surface); }}
table {{ border-collapse:collapse; width:100%; font-size:14px; }}
th,td {{ text-align:left; padding:10px 12px; border-bottom:1px solid var(--line); vertical-align:top; }}
tr:last-child td {{ border-bottom:0; }} th {{ font:500 12px var(--mono); letter-spacing:.04em; text-transform:uppercase; color:var(--muted); }}
.num {{ font-family:var(--mono); font-variant-numeric:tabular-nums; white-space:nowrap; }}
.pill {{ font:500 11px var(--mono); letter-spacing:.05em; text-transform:uppercase; padding:2px 8px; border-radius:99px; background:var(--chip); white-space:nowrap; }}
.pill.strong {{ color:var(--good); }} .pill.moderate {{ color:var(--warn); }} .pill.weak {{ color:var(--weak); }}
figure {{ margin:0; background:var(--surface); border:1px solid var(--line); border-radius:8px; padding:14px; overflow-x:auto; }}
svg {{ width:100%; min-width:520px; height:auto; display:block; font-family:var(--body); }}
.grid {{ stroke:var(--line); stroke-width:1; }} .axis {{ stroke:var(--muted); stroke-width:1; }}
.tick {{ fill:var(--muted); font-size:11px; font-family:var(--mono); }} .lab {{ fill:var(--fg); font-size:12.5px; }}
.val {{ fill:var(--muted); font-size:11px; font-family:var(--mono); }}
.s1 {{ fill:var(--s1); }} .s2 {{ fill:var(--s2); }} .ci {{ stroke:var(--muted); stroke-width:2; }}
.legend {{ display:flex; gap:18px; flex-wrap:wrap; font-size:13px; color:var(--muted); margin-bottom:8px; }}
.legend i {{ display:inline-block; width:12px; height:12px; border-radius:2px; margin-right:6px; vertical-align:-1px; }}
.stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; }}
.stat {{ background:var(--surface); border:1px solid var(--line); border-radius:8px; padding:14px; display:grid; gap:4px; }}
.stat b {{ font:600 26px var(--display); }} .stat span {{ font-size:13px; color:var(--muted); }}
ul {{ margin:0; padding-left:20px; display:grid; gap:6px; }}
</style>
<main>
<header style="display:grid;gap:12px">
  <div class="eyebrow">Eurovision Song Contest · 2004–2024 · hold-out 2025–2026 · with Eurostat migration and CEPII geography</div>
  <h1>Who votes for whom, and why it matters for prediction</h1>
  <p class="muted">Built only from the unofficial Eurovision dataset: {len(A):,} voter→country pairs from 20 contests. Every vote is compared with what all other countries gave the same song in the same show, so song quality cancels out and what remains is country bias. The 2025 and 2026 contests were never used for fitting and serve only as a final test.</p>
</header>

<div class="stats">
  <div class="stat"><b>{s['r']:.2f}</b><span>Decade-to-decade stability of country ties</span></div>
  <div class="stat"><b>+{b['within_pts12']:.1f}</b><span>Extra points (of 12) within a voting bloc</span></div>
  <div class="stat"><b>{j['ratio']:.2f}×</b><span>Stronger country bias in televote than jury</span></div>
  <div class="stat"><b>0.48</b><span>Pair-level prediction on unseen 2025–2026 votes (0.32 without affinity)</span></div>
</div>

<section>
  <div class="eyebrow">Findings</div>
  <h2>What holds up statistically</h2>
  <p class="muted small">Each test is designed so that song quality, or fitting and testing on the same years, can't explain it.</p>
  <div class="tbl"><table><thead><tr><th>Strength</th><th>Claim</th><th>Effect</th></tr></thead><tbody>{ev_rows}</tbody></table></div>
</section>

<section>
  <div class="eyebrow">Geopolitical</div>
  <h2>The strongest ties</h2>
  <p>Points given above what other countries gave the same song, as a share of the maximum, over at least 10 shared contests.</p>
  <div class="tbl"><table><thead><tr><th>Voter → performer</th><th>Share</th><th>Points of 12</th><th>Contests</th></tr></thead><tbody>{aff_rows}</tbody></table></div>
  <p class="small muted">Blocs found by clustering (Model 4):</p>
  <ul class="small">{bloc_rows}</ul>
</section>

<section>
  <div class="eyebrow">Cultural: diaspora signal</div>
  <h2>Where the public and the jury disagree most</h2>
  <p>Same voting country, same songs. The pairs where the televote is far warmer than the jury mostly involve large emigrant communities: Romanians in Italy, Albanians in Switzerland and Italy, Lithuanians in Ireland and the UK. That fits diaspora voting, but this dataset can't prove it (see the last section).</p>
  <figure><div class="legend"><span><i style="background:var(--s1)"></i>Jury</span><span><i style="background:var(--s2)"></i>Public</span></div>{hbar_pairs(jp_rows)}</figure>
  <p class="small muted">Jury and public picked the same winner in {int((pyj.jury_winner == pyj.public_winner).sum())} of {len(pyj)} finals from 2016 to 2024.</p>
</section>

<section>
  <div class="eyebrow">Structural</div>
  <h2>Running order, hosting and form</h2>
  <p>Effect on an entry's final score, in within-year standard deviations, with 95% intervals from a bootstrap over years (Model 5). None of the intervals cross zero. See the host section below for why the host figure needs care.</p>
  <figure>{effects_chart(st)}</figure>
</section>

<section>
  <div class="eyebrow">Prediction</div>
  <h2>How well the models predict</h2>
  <p>Correlation between predicted and actual points for every voter→entry pair. Removing affinity cuts accuracy by about a third, on past years and on the unseen hold-out alike. That's the strongest evidence that country bias is real and predictive.</p>
  <figure><div class="legend"><span><i style="background:var(--s1)"></i>Development 2010–2024</span><span><i style="background:var(--s2)"></i>Hold-out 2025–2026</span></div>{model_chart(model_rows)}</figure>
  <div class="tbl"><table><thead><tr><th>Final ranking</th><th>Rank corr (dev)</th><th>Winners (dev)</th><th>Top-5 overlap</th><th>Rank corr (hold-out)</th><th>Winners (hold-out)</th></tr></thead><tbody>{rank_rows}</tbody></table></div>
  <p class="small muted">Before the contest, the winner is close to unpredictable from this dataset: nothing in it measures how good a new song is. After the semi-finals, the final ranking is predicted far better. Two hold-out finals is too few for firm claims about winners.</p>
</section>

<section>
  <div class="eyebrow">Geography · CEPII GeoDist</div>
  <h2>Neighbours, language and history, and what's left for diaspora</h2>
  <p>Effect on affinity in points of 12 (2016–2024). Geography explains {geo['public']['r2'] * 100:.0f}% of the variation in public affinity but only {geo['jury']['r2'] * 100:.0f}% for juries. With migrant share added, the diaspora effect stays strong, while shared language and history shrink: part of their effect works through migration.</p>
  <div class="tbl"><table><thead><tr><th>Factor</th><th>Jury</th><th>Public</th><th>Public, with migration</th></tr></thead><tbody>{geo_rows}</tbody></table></div>
  <p class="small muted">n.s. means the 95% interval (bootstrap over voting countries) includes zero. The migration columns cover {geo['public_with_migration']['pairs']} pairs where Eurostat reports migrant data. Adding geography to the prediction model changed accuracy by less than 0.01, because past voting already contains it.</p>
</section>

<section>
  <div class="eyebrow">Structural · the host question</div>
  <h2>Do hosts really do worse?</h2>
  <p>Since 2004, no host has won. The penalty survives controls for last year's score (regression to the mean) and for skipping the semi-final. But the host is almost always last year's winner, so hosting and defending the title can't be told apart. And before 2004, hosts often won.</p>
  <div class="tbl"><table><thead><tr><th>Host effect on final score</th><th>SD</th><th>95% CI</th></tr></thead><tbody>{host_model_rows}</tbody></table></div>
  <div class="tbl"><table><thead><tr><th>Era</th><th>Hosts</th><th>Host wins</th><th>Median place (0 = top)</th></tr></thead><tbody>{host_era_rows}</tbody></table></div>
  <div class="tbl"><table><thead><tr><th>2004–2024</th><th>Countries</th><th>Score last year (SD)</th><th>Score this year (SD)</th></tr></thead><tbody>{host_drop_rows}</tbody></table></div>
  <p class="small muted">Back-to-back winners: {b2b_text}. Every one of them won while hosting, and all before 1995; Ireland won three in a row. Treat the modern host penalty as a real pattern with an unclear cause, not as a rule.</p>
</section>

<section>
  <div class="eyebrow">Semi-finals</div>
  <h2>The semi-final is the best predictor of the final</h2>
  <p>Scores standardised within each show, for every finalist that came through a semi-final. Jury and public each agree with themselves between semi and final, but not with each other.</p>
  <div class="tbl"><table><thead><tr><th>Semi → final</th><th>Correlation</th><th>Finalists</th></tr></thead><tbody>{semi_rows}</tbody></table></div>
  <div class="tbl"><table><thead><tr><th>Ridge model, final ranking</th><th>Rank corr (dev)</th><th>Winners (dev)</th><th>Winner's median predicted rank</th><th>Rank corr (hold-out)</th></tr></thead><tbody>{semi_model_rows}</tbody></table></div>
  <p class="small muted">Adding the semi score lifts final rank correlation by +0.31 (95% CI +0.20 to +0.42). Finer semi details add only +0.01 to +0.02. The jury/public split in semis exists only for 2016–2022 and 2026. The Big 5 and the host skip the semis, and they remain the weak spot.</p>
</section>

<section>
  <div class="eyebrow">Diaspora · Eurostat migrant stock</div>
  <h2>The public votes for its migrant communities</h2>
  <p>People born in the performing country and living in the voting country, as a share of the voting country's population (Eurostat; foreign citizens where birth data is missing). As communities grow, televote affinity rises steeply while jury affinity barely moves.</p>
  <figure><div class="legend"><span><i style="background:var(--s1)"></i>Jury</span><span><i style="background:var(--s2)"></i>Public</span></div>{band_chart(bands)}</figure>
  <div class="tbl"><table><thead><tr><th>Largest communities</th><th>Share of population</th><th>Public affinity</th><th>Jury affinity</th></tr></thead><tbody>{dia_rows}</tbody></table></div>
  <p class="small muted">Correlation with migrant share: public r = {dia['r_public']:.2f}, jury r = {dia['r_jury']:.2f} (difference 95% CI {dia['diff_ci'][0]:.2f}–{dia['diff_ci'][1]:.2f}). Adding migrant share to the prediction model did not improve accuracy, because past voting affinity already contains the diaspora effect. Migration explains why the bias exists. Coverage: {dia['voting_countries']} voting countries, mainly EU/EFTA.</p>
</section>

<section>
  <div class="eyebrow">Next step</div>
  <h2>Data that would make the model much stronger</h2>
  <div class="tbl"><table><thead><tr><th>Data</th><th>What it would test</th><th>Source</th></tr></thead><tbody>
  <tr><td><strong>Migrant stock outside the EU/EFTA</strong></td><td>Extends the diaspora test to voters Eurostat doesn't cover (Greece, Cyprus, Malta, Ukraine, Israel, Australia, the Caucasus, most of the Balkans)</td><td>UN DESA International Migrant Stock (global, 5-year steps)</td></tr>
  <tr><td><strong>Pre-contest popularity</strong></td><td>The missing song-quality signal, needed to predict winners</td><td>Google Trends (historical); YouTube Data API and Spotify Web API give current counts only, so snapshot them before each contest</td></tr>
  <tr><td><strong>Betting odds</strong></td><td>Usually the strongest single predictor of the winner</td><td>Historical odds archives (no official API)</td></tr>
  </tbody></table></div>
  <p class="small">For winner prediction, the gap is song quality, especially for the Big 5 and host, which skip the semis. Post-semi betting odds and streaming or search spikes are the most promising additions.</p>
</section>
</main>"""
    out = ROOT / "report" / "eurovision_voting_evidence.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
