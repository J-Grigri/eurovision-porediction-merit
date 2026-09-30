"""Eurovision look and shared explanation cards for the Streamlit pages."""
import html

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# colours validated for colour-blind separation and contrast on the chart surface (dataviz validator, dark mode)
JURY, PUBLIC, HIGHLIGHT = "#3987e5", "#e0418f", "#c98500"
POSITIVE, NEGATIVE = "#1baf7a", "#e66767"
SURFACE, INK, MUTED, GRID = "#15103a", "#f4f1ff", "#b9b2dd", "#2c2560"

pio.templates["esc"] = go.layout.Template(layout=dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=INK, family="Outfit, system-ui, sans-serif", size=13),
    colorway=[JURY, PUBLIC, HIGHLIGHT], xaxis=dict(gridcolor=GRID, zerolinecolor=MUTED, linecolor=GRID),
    yaxis=dict(gridcolor=GRID, zerolinecolor=MUTED, linecolor=GRID), legend=dict(bgcolor="rgba(0,0,0,0)"),
    hoverlabel=dict(bgcolor=SURFACE, font_color=INK, bordercolor=PUBLIC)))
pio.templates.default = "esc"

CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;800&display=swap" rel="stylesheet">
<style>
.stApp { font-family: "Outfit", system-ui, sans-serif; }  /* inherited only, so icon fonts keep their own family */
.stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4, [data-testid="stMetricValue"] { font-family: "Outfit", system-ui, sans-serif; }
.stApp { background: radial-gradient(1200px 600px at 85% -10%, #3a1464 0%, transparent 60%),
                     radial-gradient(900px 500px at -10% 10%, #0f2a6b 0%, transparent 55%), #0d0a26; }
[data-testid="stSidebar"] { background: #120d33; border-right: 1px solid #2c2560; }
h1, h2, h3 { font-weight: 800 !important; letter-spacing: -0.01em; }
.stApp h3 { font-size: 1.3rem !important; line-height: 1.3 !important; padding-top: .6rem; }
.esc-hero { position: relative; overflow: hidden; border-radius: 18px; padding: 26px 30px; margin-bottom: 18px;
  background: linear-gradient(120deg, #2b0f5c 0%, #5b1a7a 45%, #b0246f 100%); border: 1px solid #6b3aa0; }
.esc-hero::after { content: ""; position: absolute; inset: 0; pointer-events: none;
  background-image: radial-gradient(#ffffff55 1px, transparent 1.5px); background-size: 22px 22px; opacity: .25; }
.esc-hero .eyebrow { font-size: 12px; letter-spacing: .14em; text-transform: uppercase; color: #ffd1ea; font-weight: 600; }
.esc-hero .title { font-size: clamp(28px, 4vw, 44px); font-weight: 800; color: #fff; line-height: 1.05; margin: 6px 0 8px; }
.esc-hero .sub { color: #f3e6ff; max-width: 70ch; font-size: 15px; }
.esc-card { background: #15103acc; border: 1px solid #2c2560; border-left: 4px solid #e0418f; border-radius: 12px;
  padding: 12px 16px; margin: 4px 0 14px; font-size: 14px; color: #e6e1ff; }
.esc-card b.lab { color: #ffb3d9; font-weight: 600; letter-spacing: .02em; }
.esc-card .row { margin: 3px 0; }
.esc-pill { display: inline-block; font-size: 11px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
  padding: 2px 9px; border-radius: 99px; margin-left: 6px; vertical-align: 1px; }
.esc-pill.strong { background: #1baf7a26; color: #6fe3b7; } .esc-pill.moderate { background: #c9850026; color: #f2c265; }
.esc-pill.weak, .esc-pill.none { background: #e6676726; color: #ffa3a3; } .esc-pill.info { background: #3987e526; color: #9cc6ff; }
[data-testid="stMetric"] { background: #15103acc; border: 1px solid #2c2560; border-radius: 14px; padding: 14px 16px; }
[data-testid="stMetricValue"] { font-weight: 800; color: #ffffff; }
[data-testid="stExpander"] { background: #15103a99; border-radius: 12px; }
</style>
"""


def theme():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(eyebrow, title, sub):
    st.markdown(f'<div class="esc-hero"><div class="eyebrow">{html.escape(eyebrow)}</div><div class="title">{html.escape(title)}</div>'
                f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def explain(why, shows, strength=None, level="info"):
    """Explanation card under a chart or table: why it's shown, and what it tells you (with how strong the evidence is)."""
    pill = f'<span class="esc-pill {level}">{html.escape(strength)}</span>' if strength else ""
    st.markdown(f'<div class="esc-card"><div class="row"><b class="lab">Purpose</b> · {why}</div>'
                f'<div class="row"><b class="lab">Finding</b>{pill} · {shows}</div></div>', unsafe_allow_html=True)


STATS_CSS = """
<style>
.esc-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(135px, 1fr)); gap: 12px; margin: 6px 0 12px; }
.esc-stat { background: #15103acc; border: 1px solid #2c2560; border-radius: 14px; padding: 14px 16px; min-width: 0; }
.esc-stat .k { font-size: 13px; color: #cfc8f2; line-height: 1.3; }
.esc-stat .v { font-size: 28px; font-weight: 800; color: #fff; line-height: 1.15; margin: 4px 0 2px; overflow-wrap: anywhere; }
.esc-stat .n { font-size: 12.5px; color: #b9b2dd; line-height: 1.35; }
.esc-lead { font-size: 17px; color: #efeaff; max-width: 75ch; margin: 2px 0 14px; line-height: 1.5; }
</style>
"""


def stats(items):
    """Key-number cards that wrap instead of truncating. items: [(label, value, note), ...]"""
    cards = "".join(f'<div class="esc-stat"><div class="k">{html.escape(k)}</div><div class="v">{html.escape(str(v))}</div>'
                    f'<div class="n">{n}</div></div>' for k, v, n in items)
    st.markdown(STATS_CSS + f'<div class="esc-stats">{cards}</div>', unsafe_allow_html=True)


def lead(text):
    st.markdown(STATS_CSS + f'<div class="esc-lead">{text}</div>', unsafe_allow_html=True)


def chart(fig, height=320, **layout):
    """Consistent chart frame: room for labels, no toolbar clutter."""
    layout.setdefault("margin", dict(l=10, r=40, t=30, b=10))
    fig.update_layout(height=height, **layout)
    fig.update_traces(cliponaxis=False, selector=dict(type="bar"))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


ISO3 = {"AD": "AND", "AL": "ALB", "AM": "ARM", "AT": "AUT", "AU": "AUS", "AZ": "AZE", "BA": "BIH", "BE": "BEL", "BG": "BGR", "BY": "BLR",
        "CH": "CHE", "CY": "CYP", "CZ": "CZE", "DE": "DEU", "DK": "DNK", "EE": "EST", "ES": "ESP", "FI": "FIN", "FR": "FRA", "GB": "GBR",
        "GE": "GEO", "GR": "GRC", "HR": "HRV", "HU": "HUN", "IE": "IRL", "IL": "ISR", "IS": "ISL", "IT": "ITA", "LT": "LTU", "LU": "LUX",
        "LV": "LVA", "MA": "MAR", "MC": "MCO", "MD": "MDA", "ME": "MNE", "MK": "MKD", "MT": "MLT", "NL": "NLD", "NO": "NOR", "PL": "POL",
        "PT": "PRT", "RO": "ROU", "RS": "SRB", "RU": "RUS", "SE": "SWE", "SI": "SVN", "SK": "SVK", "SM": "SMR", "TR": "TUR", "UA": "UKR"}


METHOD_CSS = """
<style>
.esc-method { background: #120d33; border: 1px solid #3a2f7a; border-radius: 12px; padding: 14px 18px; margin: 8px 0 18px; }
.esc-method .h { font-size: 12px; letter-spacing: .12em; text-transform: uppercase; color: #9cc6ff; font-weight: 600; margin-bottom: 8px; }
.esc-method table { width: 100%; border-collapse: collapse; font-size: 13.5px; color: #e6e1ff; }
.esc-method td { padding: 6px 8px; vertical-align: top; border-top: 1px solid #2c2560; line-height: 1.45; }
.esc-method td:first-child { width: 150px; color: #cfc8f2; font-weight: 600; white-space: nowrap; }
@media (max-width: 640px) { .esc-method td:first-child { white-space: normal; width: 90px; } }
</style>
"""


def method(rows, title="Methods and models"):
    """Compact methods panel: [(label, description), ...]. Descriptions may contain simple HTML."""
    body = "".join(f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in rows)
    st.markdown(METHOD_CSS + f'<div class="esc-method"><div class="h">{html.escape(title)}</div><table>{body}</table></div>', unsafe_allow_html=True)
