"""War on home soil (UCDP) vs Eurovision voting (up to 2024; 2025-2026 stay sealed).

Do entries from countries at war get more from the public than from the jury, and do they beat the result the
pre-contest model expected? The jury is the control: same songs, same year, professionals.
A country counts as at war if a war (1,000+ battle deaths) was fought on its territory in the year before the contest,
or began that year before May (Russia's full invasion of Ukraine, 24 February 2022).
"""
import json

import numpy as np
import pandas as pd

from features import PROC, ROOT, load_tables

EXT = ROOT / "data" / "external"
Y1 = 2024
RNG = np.random.default_rng(0)


def war_entries(results):
    cy = pd.read_parquet(EXT / "conflict_country_year.parquet")
    war = set(zip(cy[cy.war_home == 1].country, cy[cy.war_home == 1].year))
    new = set(zip(cy[cy.war_new_before_may == 1].country, cy[cy.war_new_before_may == 1].year))
    at_war = lambda c, y: (c, y - 1) in war or (c, y) in new
    E = pd.read_parquet(PROC / "forecast_backtest.parquet")
    r = results[(results["round"] == "final") & (results.disqualified == 0)]
    E = E.merge(r[["year", "country", "points_jury", "points_public", "points_total"]].rename(columns={"country": "entry"}), on=["year", "entry"])
    E = E[(E.year >= 2010) & (E.year <= Y1)].copy()
    z = lambda s: (s - s.mean()) / s.std()
    E["surprise"] = E.groupby("year").points_total.transform(z) - E.groupby("year").early.transform(z)  # beat (+) / missed (-) expectation
    E["public_z"] = E.groupby("year").points_public.transform(z); E["jury_z"] = E.groupby("year").points_jury.transform(z)
    E["public_minus_jury"] = E.public_z - E.jury_z
    E["at_war"] = [int(at_war(c, y)) for c, y in zip(E.entry, E.year)]
    rows = []
    for col in ("surprise", "public_minus_jury"):
        d = E.dropna(subset=[col]); lab = d.at_war.values; v = d[col].values
        obs = v[lab == 1].mean() - v[lab == 0].mean(); null = []
        for _ in range(5000):
            p = RNG.permutation(lab); null.append(v[p == 1].mean() - v[p == 0].mean())
        rows.append(dict(measure=col, at_war_n=int(lab.sum()), others_n=int((lab == 0).sum()), difference=obs,
                         p_perm=(1 + np.sum(np.array(null) >= obs)) / 5001))
    cases = E[E.at_war == 1][["year", "entry", "place", "early_rank", "surprise", "public_z", "jury_z", "public_minus_jury"]].sort_values("year")
    return pd.DataFrame(rows), cases


def main():
    _, results, _ = load_tables()
    war, cases = war_entries(results)
    war.to_parquet(PROC / "pol_war.parquet"); cases.to_parquet(PROC / "pol_war_cases.parquet")
    (PROC / "politics.json").write_text(json.dumps(dict(war=war.to_dict("records")), indent=2, default=float))
    print(war.round(3).to_string()); print(cases.round(2).to_string())


if __name__ == "__main__":
    main()
