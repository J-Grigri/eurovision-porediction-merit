"""Parse the raw Eurovision JSON dataset into tidy tables (data/processed/*.parquet)."""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "senior"
OUT = ROOT / "data" / "processed"

ROUND_ORDER = {"semifinal": 0, "semifinal1": 0, "semifinal2": 0, "final": 1}


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def lyrics_language(cdir):
    """Original-lyrics languages from filenames like o_english.txt / o_italian,spanish.txt."""
    langs = []
    for f in sorted((cdir / "lyrics").glob("o_*.txt")) if (cdir / "lyrics").exists() else []:
        name = f.stem[2:]
        name = re.sub(r"_\d+$", "", name)
        first = name.split("_")[0]  # drop script/variant suffix (greek_ellinika)
        for lg in first.split(","):
            if lg and lg not in langs:
                langs.append(lg)
    return langs


def flat(d):
    """Flatten a JSON object for display: lists become '; '-joined strings."""
    return {k: ("; ".join(map(str, v)) if isinstance(v, list) else v) for k, v in d.items()}


def original_lyrics(cdir):
    fs = sorted((cdir / "lyrics").glob("o_*.txt")) if (cdir / "lyrics").exists() else []
    return "\n\n".join(f.read_text(encoding="utf-8") for f in fs[:1]) or None, [f.name for f in (cdir / "lyrics").glob("*.txt")] if (cdir / "lyrics").exists() else []


def parse_key(tone):
    if not tone or not isinstance(tone, str):
        return None, None
    parts = tone.split()
    return parts[0], (parts[1].lower() if len(parts) > 1 else None)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    contests, entries, results, votes = [], [], [], []
    contests_full, contestants_full, rounds_meta = [], [], []
    for ydir in sorted(RAW.iterdir()):
        if not ydir.is_dir():
            continue
        year = int(ydir.name)
        c = load(ydir / "contest.json")
        contests.append({"year": year, "host": c.get("country"), "city": c.get("city")})
        contests_full.append({"year": year, **flat(c)})
        id2cc = {}
        for cdir in sorted((ydir / "contestants").iterdir()):
            if not cdir.is_dir():
                continue
            d = load(cdir / "contestant.json")
            cid = d.get("id")
            if cid is None:
                cid = int(cdir.name.split("_")[0])
            id2cc[cid] = d["country"]
            langs = lyrics_language(cdir)
            text, lfiles = original_lyrics(cdir)
            contestants_full.append({"year": year, **flat(d), "lyrics_original": text, "lyrics_available": "; ".join(sorted(lfiles))})
            key, mode = parse_key(d.get("tone"))
            n = lambda k: len(d.get(k) or [])
            entries.append({
                "year": year, "contestant_id": cid, "country": d["country"],
                "artist": d["artist"], "song": d["song"],
                "bpm": d.get("bpm") or None, "key": key, "mode": mode,
                "language": langs[0] if langs else None,
                "n_languages": len(langs), "is_english": int("english" in langs),
                "n_artist_people": n("artistPeople"), "n_writers": max(n("writers"), n("composers") + n("lyricists")),
                "n_backings": n("backings"), "n_dancers": n("dancers"),
                "has_stage_director": int(bool(d.get("stageDirector"))),
                "broadcaster": d.get("broadcaster"),
            })
        for rfile in sorted((ydir / "rounds").glob("*.json")):
            rname = rfile.stem
            r = load(rfile)
            dq = set(r.get("disqualifieds") or [])
            rounds_meta.append({"year": year, "round": rname, "date": r.get("date"), "time": r.get("time"),
                                "disqualified_ids": "; ".join(map(str, sorted(dq))) or None})
            for p in r.get("performances") or []:
                cid = p["contestantId"]
                row = {"year": year, "round": rname, "contestant_id": cid,
                       "country": id2cc.get(cid), "running": p.get("running"),
                       "place": p.get("place"), "disqualified": int(cid in dq)}
                for s in p.get("scores") or []:
                    nm = "public" if s["name"] in ("public", "tele") else s["name"]
                    row[f"points_{nm}"] = s["points"]
                    for voter, pts in (s.get("votes") or {}).items():
                        votes.append((year, rname, nm, voter, id2cc.get(cid), cid, pts))
                results.append(row)
    pd.DataFrame(contests).to_parquet(OUT / "contests.parquet")
    pd.DataFrame(contests_full).to_parquet(OUT / "contests_full.parquet")
    pd.DataFrame(contestants_full).to_parquet(OUT / "contestants_full.parquet")
    pd.DataFrame(rounds_meta).to_parquet(OUT / "rounds_meta.parquet")
    pd.DataFrame(entries).to_parquet(OUT / "entries.parquet")
    res = pd.DataFrame(results)
    res["stage"] = res["round"].map(ROUND_ORDER)
    # qualification: a semifinal entry qualified iff it also appears in that year's final
    fin = set(zip(res.loc[res["round"] == "final", "year"], res.loc[res["round"] == "final", "contestant_id"]))
    res["qualified"] = [int((y, c) in fin) if r != "final" else None
                        for y, r, c in zip(res["year"], res["round"], res["contestant_id"])]
    res.to_parquet(OUT / "results.parquet")
    v = pd.DataFrame(votes, columns=["year", "round", "kind", "voter", "entry", "contestant_id", "points"])
    v.to_parquet(OUT / "votes.parquet")
    print(f"contests={len(contests)} entries={len(entries)} results={len(res)} votes={len(v)}")


if __name__ == "__main__":
    main()
