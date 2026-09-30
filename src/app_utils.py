import sys
from pathlib import Path

import pandas as pd
import streamlit as st

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
PROC = SRC.parent / "data" / "processed"
COUNTRIES = __import__("json").load(open(SRC.parent / "data" / "countries.json", encoding="utf-8"))
COUNTRIES.update({"WLD": "Rest of the World"})


@st.cache_data
def _read(name, mtime):
    return pd.read_parquet(PROC / f"{name}.parquet")


def table(name):
    """Cached read; the file's mtime is part of the cache key so a pipeline rebuild is picked up without a restart."""
    return _read(name, (PROC / f"{name}.parquet").stat().st_mtime)


def cname(code):
    return COUNTRIES.get(code, code)


def require_data():
    if not (PROC / "predictions.parquet").exists():
        st.error("Processed data missing. Run `python src/run_all.py` first.")
        st.stop()
