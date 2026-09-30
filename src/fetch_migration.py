"""Fetch migrant stock (Eurostat) and population, and build a voter x entry diaspora table.

Source: Eurostat REST API (no key needed), JSON-stat 2.0.
  migr_pop3ctb  population on 1 January by country of birth  -> people born in the performing country living in the voting country
  migr_pop1ctz  population on 1 January by citizenship         -> fallback where country-of-birth data is missing (e.g. Germany,
                                                                   Portugal); undercounts people who have naturalised
  demo_pjan     population on 1 January                        -> voting country's population, to turn counts into shares
Coverage: destinations are EU/EFTA countries (plus a few others reporting to Eurostat); origins are all countries.
Output: data/external/migration.parquet  (year, voter, entry, migrants, voter_pop, share, source)
"""
import json
import ssl
import urllib.request
from pathlib import Path

import numpy as np
import certifi
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "data" / "external"
API = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{ds}?format=JSON&lang=EN&{q}"
TO_ESC = {"EL": "GR", "UK": "GB"}  # Eurostat -> Eurovision dataset country codes
YEARS = range(2004, 2025)


def fetch(ds, q):
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(API.format(ds=ds, q=q), timeout=300, context=ctx) as r:
        return json.load(r)


def jsonstat_to_frame(d):
    dims = d["id"]; sizes = d["size"]
    labels = [sorted(d["dimension"][k]["category"]["index"].items(), key=lambda kv: kv[1]) for k in dims]
    rows = []
    for flat, v in d["value"].items():
        i, idx = int(flat), []
        for s in reversed(sizes):
            idx.append(i % s); i //= s
        idx.reverse()
        rows.append([labels[j][idx[j]][0] for j in range(len(dims))] + [v])
    return pd.DataFrame(rows, columns=dims + ["value"])


def main():
    EXT.mkdir(parents=True, exist_ok=True)
    esc = set(json.loads((ROOT / "data" / "countries.json").read_text(encoding="utf-8")))
    esc_eurostat = sorted({k for k, v in TO_ESC.items()} | (esc - set(TO_ESC.values())))
    frames = []
    for y in YEARS:
        q = f"age=TOTAL&sex=T&unit=NR&time={y}&" + "&".join(f"c_birth={c}" for c in esc_eurostat)
        try:
            frames.append(jsonstat_to_frame(fetch("migr_pop3ctb", q)))
            print(y, len(frames[-1]))
        except Exception as e:  # a year with no data returns an error
            print(y, "no data:", e)
    m = pd.concat(frames, ignore_index=True).rename(columns={"c_birth": "entry", "geo": "voter", "time": "year", "value": "migrants"})
    pq = "age=TOTAL&sex=T&" + "&".join(f"time={y}" for y in YEARS)
    p = jsonstat_to_frame(fetch("demo_pjan", pq)).rename(columns={"geo": "voter", "time": "year", "value": "voter_pop"})
    m["voter"] = m.voter.replace(TO_ESC); m["year"] = m.year.astype(int)
    m["entry"] = m.entry.replace(TO_ESC)
    m = m[m.voter.isin(esc) & m.entry.isin(esc) & (m.voter != m.entry)][["year", "voter", "entry", "migrants"]]
    m["source"] = "country of birth"
    # fallback: citizenship for voting countries with no country-of-birth breakdown
    missing = sorted(esc - set(m.voter) - {"WLD"})
    to_eurostat = {v: k for k, v in TO_ESC.items()}
    geo_q = "&".join(f"geo={to_eurostat.get(c, c)}" for c in missing)
    cq = "age=TOTAL&sex=T&unit=NR&" + geo_q + "&" + "&".join(f"time={y}" for y in YEARS) + "&" + "&".join(f"citizen={c}" for c in esc_eurostat)
    c = jsonstat_to_frame(fetch("migr_pop1ctz", cq)).rename(columns={"citizen": "entry", "geo": "voter", "time": "year", "value": "migrants"})
    c["voter"] = c.voter.replace(TO_ESC); c["entry"] = c.entry.replace(TO_ESC); c["year"] = c.year.astype(int)
    c = c[c.voter.isin(esc) & c.entry.isin(esc) & (c.voter != c.entry)][["year", "voter", "entry", "migrants"]]
    c["source"] = "citizenship"
    print("citizenship fallback voters:", sorted(c.voter.unique()))
    m = pd.concat([m, c], ignore_index=True)
    p["voter"] = p.voter.replace(TO_ESC); p["year"] = p.year.astype(int)
    p = p[p.voter.isin(esc)][["year", "voter", "voter_pop"]]
    out = m.merge(p, on=["year", "voter"], how="left")
    out["share"] = out.migrants / out.voter_pop
    out.to_parquet(EXT / "migration.parquet")
    print(out.shape, "voters:", sorted(out.voter.unique()), "years:", out.year.min(), out.year.max())
    print(out.sort_values("share", ascending=False).drop_duplicates(["voter", "entry"]).head(15).to_string())


if __name__ == "__main__":
    main()
