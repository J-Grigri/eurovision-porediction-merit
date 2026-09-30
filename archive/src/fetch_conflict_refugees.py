"""Download UCDP armed-conflict data, UNHCR refugee stocks and World Bank population.

UCDP/PRIO Armed Conflict Dataset v26.1 (free download, https://ucdp.uu.se/downloads/): one row per conflict-year with
  location, the governments on each side, and intensity (1 = 25-999 battle deaths that year, 2 = war, 1,000+).
  -> data/external/conflict_country_year.parquet  (country, year, war_home, conflict_home, war_party)
     war_home      a war (intensity 2) was fought on the country's territory that year
     conflict_home any armed conflict (intensity 1 or 2) on its territory
     war_party     its government was a main side in a war (at home or abroad)
     war_new_before_may  a war on its territory whose current episode began that year before 1 May (e.g. 24 Feb 2022)
  -> data/external/conflict_dyads.parquet  (a, b, year, intensity): governments on opposite main sides of a conflict
UNHCR Refugee Data Finder API (https://api.unhcr.org/population/v1/): refugees + asylum seekers by origin and host, 2004-2025
World Bank API, indicator SP.POP.TOTL: population, to turn refugee counts into shares of the host country
  -> data/external/refugees.parquet  (year, voter = host, entry = origin, refugees, voter_pop, share)
"""
import io
import json
import ssl
import urllib.request
import zipfile
from pathlib import Path

import certifi
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "data" / "external"
CTX = ssl.create_default_context(cafile=certifi.where())
UCDP_URL = "https://ucdp.uu.se/downloads/ucdpprio/ucdp-prio-acd-261-csv.zip"
ISO3 = {"AND": "AD", "ALB": "AL", "ARM": "AM", "AUT": "AT", "AUS": "AU", "AZE": "AZ", "BIH": "BA", "BEL": "BE", "BGR": "BG", "BLR": "BY",
        "CHE": "CH", "CYP": "CY", "CZE": "CZ", "DEU": "DE", "DNK": "DK", "EST": "EE", "ESP": "ES", "FIN": "FI", "FRA": "FR", "GBR": "GB",
        "GEO": "GE", "GRC": "GR", "HRV": "HR", "HUN": "HU", "IRL": "IE", "ISR": "IL", "ISL": "IS", "ITA": "IT", "LTU": "LT", "LUX": "LU",
        "LVA": "LV", "MAR": "MA", "MCO": "MC", "MDA": "MD", "MNE": "ME", "MKD": "MK", "MLT": "MT", "NLD": "NL", "NOR": "NO", "POL": "PL",
        "PRT": "PT", "ROU": "RO", "SRB": "RS", "RUS": "RU", "SWE": "SE", "SVN": "SI", "SVK": "SK", "SMR": "SM", "TUR": "TR", "UKR": "UA"}
# Gleditsch-Ward state numbers used by UCDP -> dataset codes (345 = Yugoslavia / Serbia)
GW = {200: "GB", 205: "IE", 210: "NL", 211: "BE", 212: "LU", 220: "FR", 221: "MC", 225: "CH", 230: "ES", 232: "AD", 235: "PT", 255: "DE",
      260: "DE", 290: "PL", 305: "AT", 310: "HU", 316: "CZ", 317: "SK", 325: "IT", 331: "SM", 338: "MT", 339: "AL", 341: "ME", 343: "MK",
      344: "HR", 345: "RS", 346: "BA", 349: "SI", 350: "GR", 352: "CY", 355: "BG", 359: "MD", 360: "RO", 365: "RU", 366: "EE", 367: "LV",
      368: "LT", 369: "UA", 370: "BY", 371: "AM", 372: "GE", 373: "AZ", 375: "FI", 380: "SE", 385: "NO", 390: "DK", 395: "IS", 600: "MA",
      640: "TR", 666: "IL", 900: "AU"}


def get(url):
    return urllib.request.urlopen(url, timeout=300, context=CTX).read()


def gw_list(v):
    if pd.isna(v):
        return []
    return [GW[int(x)] for x in str(v).replace(" ", "").split(",") if x and int(x) in GW]


def ucdp():
    with zipfile.ZipFile(io.BytesIO(get(UCDP_URL))) as z:
        d = pd.read_csv(z.open([n for n in z.namelist() if n.endswith(".csv")][0]))
    d = d[d.year >= 2000]
    rows, dyads = [], []
    for r in d.itertuples():
        a, b, loc = gw_list(r.gwno_a), gw_list(r.gwno_b), gw_list(r.gwno_loc)
        ep = pd.to_datetime(r.start_date2, errors="coerce")
        new = int(r.intensity_level == 2 and pd.notna(ep) and ep.year == r.year and ep.month < 5)
        for c in loc:
            rows.append(dict(country=c, year=r.year, war_home=int(r.intensity_level == 2), conflict_home=1, war_party=0, war_new_before_may=new))
        for c in a + b:
            rows.append(dict(country=c, year=r.year, war_home=0, conflict_home=0, war_party=int(r.intensity_level == 2), war_new_before_may=0))
        for x in a:
            for y in b:
                dyads.append(dict(a=x, b=y, year=r.year, intensity=r.intensity_level))
    cy = pd.DataFrame(rows).groupby(["country", "year"], as_index=False).max()
    dy = pd.DataFrame(dyads)
    dy = pd.concat([dy, dy.rename(columns={"a": "b", "b": "a"})]).groupby(["a", "b", "year"], as_index=False).intensity.max()
    cy.to_parquet(EXT / "conflict_country_year.parquet"); dy.to_parquet(EXT / "conflict_dyads.parquet")
    print("UCDP years", d.year.min(), d.year.max(), "| country-years", len(cy), "| hostile dyad-years", len(dy))
    print(cy[cy.war_home == 1].groupby("country").year.apply(lambda s: sorted(s)[-4:]).to_string())
    print(dy.drop_duplicates(["a", "b"]).query("a < b").to_string())


def unhcr():
    rows = []
    for iso3 in ISO3:
        page = 1
        while True:
            u = (f"https://api.unhcr.org/population/v1/population/?limit=10000&page={page}&yearFrom=2004&yearTo=2025"
                 f"&coo_all=true&coa={iso3}&cf_type=ISO")
            j = json.loads(get(u))
            for it in j["items"]:
                if it.get("coo_iso") in ISO3:
                    num = lambda v: float(v) if str(v).replace(".", "").isdigit() else 0.0
                    rows.append(dict(year=it["year"], voter=ISO3[iso3], entry=ISO3[it["coo_iso"]],
                                     refugees=num(it["refugees"]) + num(it["asylum_seekers"])))
            if page >= j.get("maxPages", 1):
                break
            page += 1
    r = pd.DataFrame(rows).groupby(["year", "voter", "entry"], as_index=False).refugees.sum()
    wb = json.loads(get("https://api.worldbank.org/v2/country/" + ";".join(ISO3) + "/indicator/SP.POP.TOTL?format=json&date=2004:2025&per_page=5000"))[1]
    pop = pd.DataFrame([dict(voter=ISO3.get(x["countryiso3code"]), year=int(x["date"]), voter_pop=x["value"]) for x in wb if x["value"]])
    pop = pop.dropna()
    last = pop.sort_values("year").groupby("voter").tail(1)  # fill years the World Bank hasn't published yet
    for y in range(pop.year.max() + 1, 2026):
        pop = pd.concat([pop, last.assign(year=y)])
    r = r.merge(pop, on=["year", "voter"], how="left")
    r["share"] = r.refugees / r.voter_pop
    r.to_parquet(EXT / "refugees.parquet")
    print("UNHCR rows", len(r), "years", r.year.min(), r.year.max(), "| hosts", r.voter.nunique(), "| missing pop", r.voter_pop.isna().sum())
    print(r.sort_values("share", ascending=False).drop_duplicates(["voter", "entry"]).head(10).to_string())


if __name__ == "__main__":
    EXT.mkdir(parents=True, exist_ok=True)
    ucdp()
    unhcr()
