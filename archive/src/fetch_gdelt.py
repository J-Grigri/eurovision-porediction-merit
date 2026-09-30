"""GDELT 1.0 daily event files -> country-pair tone and cooperation/conflict in the months before each contest.

Source: https://data.gdeltproject.org/events/YYYYMMDD.export.CSV.zip (free, no key). Each row is one event reported in
world news: who acted (Actor1 country) toward whom (Actor2 country), how cooperative or hostile it was (Goldstein scale,
-10 = war, +10 = strongest cooperation) and the average tone of the articles reporting it (AvgTone, roughly -10 to +10).

Sampling: every 8th day from 1 January to 30 April (the run-up to the May contest), 2016-2026. Each file is
summarised as it arrives and deleted, so only small pair-level tables are kept. Resumable: finished days are skipped.
Output: data/external/gdelt/pairs_<YYYYMMDD>.parquet and, combined, data/external/gdelt_pairs.parquet
  (year, a, b, events, mentions, tone, goldstein)  a/b in the Eurovision dataset's country codes
"""
import io
import ssl
import sys
import urllib.request
import zipfile
from datetime import date, timedelta
from pathlib import Path

import certifi
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "external" / "gdelt"
ISO3 = {"AND": "AD", "ALB": "AL", "ARM": "AM", "AUT": "AT", "AUS": "AU", "AZE": "AZ", "BIH": "BA", "BEL": "BE", "BGR": "BG", "BLR": "BY",
        "CHE": "CH", "CYP": "CY", "CZE": "CZ", "DEU": "DE", "DNK": "DK", "EST": "EE", "ESP": "ES", "FIN": "FI", "FRA": "FR", "GBR": "GB",
        "GEO": "GE", "GRC": "GR", "HRV": "HR", "HUN": "HU", "IRL": "IE", "ISR": "IL", "ISL": "IS", "ITA": "IT", "LTU": "LT", "LUX": "LU",
        "LVA": "LV", "MAR": "MA", "MCO": "MC", "MDA": "MD", "MNE": "ME", "MKD": "MK", "MLT": "MT", "NLD": "NL", "NOR": "NO", "POL": "PL",
        "PRT": "PT", "ROU": "RO", "SRB": "RS", "RUS": "RU", "SWE": "SE", "SVN": "SI", "SVK": "SK", "SMR": "SM", "TUR": "TR", "UKR": "UA"}
COLS = {7: "a", 17: "b", 30: "goldstein", 31: "mentions", 34: "tone"}
CTX = ssl.create_default_context(cafile=certifi.where())


def days(years=range(2016, 2027), step=8):
    for y in years:
        d = date(y, 1, 1)
        while d <= date(y, 4, 30):
            yield d
            d += timedelta(days=step)


def summarise(day):
    url = f"https://data.gdeltproject.org/events/{day:%Y%m%d}.export.CSV.zip"
    raw = urllib.request.urlopen(url, timeout=300, context=CTX).read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        df = pd.read_csv(z.open(z.namelist()[0]), sep="\t", header=None, usecols=list(COLS), dtype={7: str, 17: str}, quoting=3, on_bad_lines="skip")
    df = df.rename(columns=COLS)
    df = df[df.a.isin(ISO3) & df.b.isin(ISO3) & (df.a != df.b)]
    df["a"], df["b"] = df.a.map(ISO3), df.b.map(ISO3)
    df["w_tone"], df["w_gold"] = df.tone * df.mentions, df.goldstein * df.mentions
    g = df.groupby(["a", "b"]).agg(events=("tone", "size"), mentions=("mentions", "sum"), w_tone=("w_tone", "sum"), w_gold=("w_gold", "sum")).reset_index()
    g["year"] = day.year
    return g


def combine():
    parts = [pd.read_parquet(p) for p in sorted(OUT.glob("pairs_*.parquet"))]
    g = pd.concat(parts).groupby(["year", "a", "b"], as_index=False)[["events", "mentions", "w_tone", "w_gold"]].sum()
    g["tone"], g["goldstein"] = g.w_tone / g.mentions, g.w_gold / g.mentions  # mention-weighted means
    g.drop(columns=["w_tone", "w_gold"]).to_parquet(ROOT / "data" / "external" / "gdelt_pairs.parquet")
    print("combined", g.shape, "files:", len(parts))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [d for d in days() if not (OUT / f"pairs_{d:%Y%m%d}.parquet").exists() and d <= date.today()]
    print("days to fetch:", len(todo), flush=True)
    for i, d in enumerate(todo, 1):
        try:
            summarise(d).to_parquet(OUT / f"pairs_{d:%Y%m%d}.parquet")
            print(f"{i}/{len(todo)} {d}", flush=True)
        except Exception as e:  # missing day or network hiccup: skip, a rerun retries it
            print(f"{i}/{len(todo)} {d} failed: {e}", flush=True, file=sys.stderr)
    combine()


if __name__ == "__main__":
    main()
