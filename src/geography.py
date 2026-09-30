"""Geography and shared culture (CEPII GeoDist) vs voting affinity.

Source: CEPII GeoDist, dist_cepii.xls (free download: https://www.cepii.fr/distance/dist_cepii.zip), saved in
data/external/cepii/. Bilateral variables used:
  dist            distance between the two countries' largest cities (km) -> log10
  contig          shares a land border
  comlang_ethno   a language spoken by at least 9% of both populations
  colony          ever in a colonial relationship
  smctry          were ever the same country (e.g. Czechia/Slovakia, former USSR and Yugoslavia)
The file predates the Serbia/Montenegro split: both map to "YUG", and the Serbia<->Montenegro pair is set by hand
(border, shared language, same country, ~280 km Belgrade-Podgorica).

Tests (2004-2024; 2025-2026 sealed):
  1. Total affinity ~ geography, every voter (geography covers all countries, unlike the migration data)
  2. The same for jury and public separately (2016-2024): which ties are shared by experts and viewers?
  3. Diaspora vs neighbours: public affinity ~ log migrant share + geography. Does diaspora survive the controls?
  4. Prediction: does geography improve the Ridge model, overall and for pairs with little voting history?
"""
import json

import numpy as np
import pandas as pd

from features import CUTOFF_YEAR, FEATURES, PROC, ROOT
from models import DEV_YEARS, HOLDOUT_YEARS, fit_predict

EXT = ROOT / "data" / "external"
ISO3 = {"AD": "AND", "AL": "ALB", "AM": "ARM", "AT": "AUT", "AU": "AUS", "AZ": "AZE", "BA": "BIH", "BE": "BEL", "BG": "BGR", "BY": "BLR",
        "CH": "CHE", "CY": "CYP", "CZ": "CZE", "DE": "DEU", "DK": "DNK", "EE": "EST", "ES": "ESP", "FI": "FIN", "FR": "FRA", "GB": "GBR",
        "GE": "GEO", "GR": "GRC", "HR": "HRV", "HU": "HUN", "IE": "IRL", "IL": "ISR", "IS": "ISL", "IT": "ITA", "LT": "LTU", "LU": "LUX",
        "LV": "LVA", "MA": "MAR", "MC": "MCO", "MD": "MDA", "ME": "YUG", "MK": "MKD", "MT": "MLT", "NL": "NLD", "NO": "NOR", "PL": "POL",
        "PT": "PRT", "RO": "ROM", "RS": "YUG", "RU": "RUS", "SE": "SWE", "SI": "SVN", "SK": "SVK", "SM": "SMR", "TR": "TUR", "UA": "UKR"}
GEO = ["log_dist", "contig", "comlang_ethno", "colony", "smctry"]
LABELS = {"log_dist": "Distance (per 10× further)", "contig": "Shared border", "comlang_ethno": "Shared language",
          "colony": "Colonial tie", "smctry": "Were once the same country", "log_share": "Migrant share (per 10× larger)"}


def load_geo():
    d = pd.read_excel(next((EXT / "cepii").glob("dist_cepii.xls")))
    inv = {}
    for k, v in ISO3.items():
        inv.setdefault(v, []).append(k)
    d = d[d.iso_o.isin(inv) & d.iso_d.isin(inv)]
    rows = []
    for r in d.itertuples():
        for a in inv[r.iso_o]:
            for b in inv[r.iso_d]:
                if a != b:
                    rows.append(dict(voter=a, entry=b, dist=r.dist, contig=r.contig, comlang_ethno=r.comlang_ethno, colony=r.colony, smctry=r.smctry))
    g = pd.DataFrame(rows)
    fix = dict(dist=280.0, contig=1, comlang_ethno=1, colony=0, smctry=1)  # Serbia <-> Montenegro
    for a, b in (("RS", "ME"), ("ME", "RS")):
        g.loc[(g.voter == a) & (g.entry == b), list(fix)] = list(fix.values())
    g["log_dist"] = np.log10(g.dist)
    return g


def ols(d, y, X, B=1000):
    """OLS with a bootstrap over voting countries (pairs sharing a voter aren't independent)."""
    A = np.column_stack([np.ones(len(d))] + [d[c].values for c in X]); b = np.linalg.lstsq(A, d[y].values, rcond=None)[0]
    r2 = 1 - ((d[y].values - A @ b) ** 2).sum() / ((d[y] - d[y].mean()) ** 2).sum()
    voters = d.voter.unique(); rng = np.random.default_rng(0); groups = {v: np.where(d.voter.values == v)[0] for v in voters}; bs = []
    for _ in range(B):
        idx = np.concatenate([groups[v] for v in rng.choice(voters, len(voters))])
        bs.append(np.linalg.lstsq(A[idx], d[y].values[idx], rcond=None)[0])
    bs = np.array(bs)
    return pd.DataFrame({"feature": ["intercept"] + X, "coef": b, "lo95": np.percentile(bs, 2.5, 0), "hi95": np.percentile(bs, 97.5, 0)}).iloc[1:], r2


def main():
    g = load_geo()
    g.to_parquet(PROC / "geography_pairs.parquet")
    out, coefs = {}, []
    A = pd.read_parquet(PROC / "pair_affinity.parquet")
    d1 = A[A.n >= 5].merge(g, on=["voter", "entry"])
    t, r2 = ols(d1, "mean", GEO); t["model"] = "Total affinity, 2004–2024"; coefs.append(t)
    out["total"] = {"pairs": len(d1), "r2": r2}
    jp = pd.read_parquet(PROC / "ev_jury_public_pairs.parquet").merge(g, on=["voter", "entry"])
    for k in ("jury", "public"):
        t, r2 = ols(jp, k, GEO); t["model"] = f"{k.title()} affinity, 2016–2024"; coefs.append(t); out[k] = {"pairs": len(jp), "r2": r2}
    dia = pd.read_parquet(PROC / "diaspora_pairs.parquet").merge(g, on=["voter", "entry"])
    for k in ("public", "jury"):
        t, r2 = ols(dia, k, ["log_share"] + GEO); t["model"] = f"{k.title()} affinity with migration, 2016–2024"; coefs.append(t)
        out[f"{k}_with_migration"] = {"pairs": len(dia), "r2": r2}
        t0, r20 = ols(dia, k, GEO); out[f"{k}_geo_only_same_pairs"] = {"r2": r20}
    C = pd.concat(coefs, ignore_index=True); C["label"] = C.feature.map(LABELS)
    C.to_parquet(PROC / "geography_coefficients.parquet")

    # prediction: add geography to the Ridge pre-contest model
    F = pd.read_parquet(PROC / "features.parquet").merge(g[["voter", "entry"] + GEO], on=["voter", "entry"], how="left")
    F["aff_n"] = F.groupby(["voter", "entry"]).cumcount()  # rough count of earlier modelled meetings, for the "little history" subset
    known = F[F.year <= CUTOFF_YEAR]; P = []
    for T in DEV_YEARS + HOLDOUT_YEARS:
        tr = known if T > CUTOFF_YEAR else known[known.year < T]
        te = F[F.year == T].copy(); te["split"] = "holdout" if T > CUTOFF_YEAR else "dev"
        te["base"], _ = fit_predict("ridge", tr, te, FEATURES)
        te["with_geo"], _ = fit_predict("ridge", tr, te, FEATURES + GEO)
        te["geo_no_affinity"], _ = fit_predict("ridge", tr, te, [f for f in FEATURES if f != "aff"] + GEO)
        P.append(te)
    P = pd.concat(P); pr = []
    for split, sp in P.groupby("split"):
        new = sp[sp.aff_n <= 2]
        for col in ("base", "with_geo", "geo_no_affinity"):
            pr.append(dict(split=split, model=col, pair_r=np.corrcoef(sp[col], sp.y)[0, 1], pair_r_little_history=np.corrcoef(new[col], new.y)[0, 1], n_little=len(new)))
    pr = pd.DataFrame(pr); pr.to_parquet(PROC / "geography_prediction.parquet")
    out["prediction"] = pr.round(4).to_dict("records")
    (PROC / "geography.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != "prediction"}, indent=2, default=lambda x: round(float(x), 3)))
    print(C[["model", "feature", "coef", "lo95", "hi95"]].round(3).to_string()); print(pr.round(3).to_string())


if __name__ == "__main__":
    main()
