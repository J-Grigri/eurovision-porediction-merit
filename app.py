import importlib
import sys
from pathlib import Path

import streamlit as st

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

# Streamlit doesn't always notice edits to helper modules in src/, so a long-running server can keep a stale copy
# (e.g. "cannot import name 'chart' from 'ui'"). Reload any project module whose file is newer than the loaded copy.
for name, mod in list(sys.modules.items()):
    path = Path(getattr(mod, "__file__", None) or "")
    if path.parent == SRC and path.exists():
        mtime = path.stat().st_mtime
        if getattr(mod, "_loaded_mtime", None) != mtime:
            if hasattr(mod, "_loaded_mtime") or name in ("ui", "src.ui", "app_utils", "src.app_utils", "forecast"):
                mod = importlib.reload(mod)
            mod._loaded_mtime = mtime

st.set_page_config(page_title="Eurovision Forecast", page_icon="🎤", layout="wide")
st.navigation({
    "Forecast": [st.Page("views/predictor.py", title="Forecast", icon=":material/emoji_events:", default=True),
                 st.Page("views/questions.py", title="Key questions", icon=":material/help:")],
    "History": [st.Page("views/history.py", title="Country history", icon=":material/history:")],
    "Voting factors": [st.Page("views/friends.py", title="Voting relationships", icon=":material/diversity_3:"),
                              st.Page("views/diaspora.py", title="Diaspora voting", icon=":material/flight:"),
                              st.Page("views/jury_public.py", title="Jury and televote", icon=":material/how_to_vote:"),
                              st.Page("views/show.py", title="Contest format", icon=":material/theater_comedy:")],
    "Methodology": [st.Page("views/pipeline.py", title="Data pipeline", icon=":material/account_tree:"),
                    st.Page("views/accuracy.py", title="Accuracy", icon=":material/verified:")],
}).run()
