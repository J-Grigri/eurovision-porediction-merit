import streamlit as st

from src.app_utils import table, cname, require_data, COUNTRIES

st.set_page_config(page_title="Data", layout="wide")
require_data()
st.title("Dataset explorer")
st.caption("Everything in the senior-contest part of the dataset, 1956–2026, as parsed from the source files. This page shows raw data; the models and "
           "patterns elsewhere use 2004–2024 only. Junior Eurovision is not included. Lyrics translations are not loaded (only the original lyrics text and the list of available versions).")

contests, contestants = table("contests_full"), table("contestants_full")
rounds, results, votes = table("rounds_meta"), table("results"), table("votes")

c = st.columns(5)
c[0].metric("Contests", len(contests)); c[1].metric("Contestants", f"{len(contestants):,}")
c[2].metric("Round results", f"{len(results):,}"); c[3].metric("Vote records", f"{len(votes):,}"); c[4].metric("Countries", len(COUNTRIES) - 1)

years = sorted(contests.year.unique())
yr = st.select_slider("Years", options=years, value=(years[0], years[-1]))
in_years = lambda d: d[(d.year >= yr[0]) & (d.year <= yr[1])]


def show(df, name, height=520):
    st.caption(f"{len(df):,} rows")
    st.dataframe(df, hide_index=True, width="stretch", height=height)
    st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), f"{name}.csv", "text/csv", key=f"dl_{name}")


tabs = st.tabs(["Contests", "Contestants", "Rounds & results", "Votes", "Country codes"])

with tabs[0]:
    d = in_years(contests).assign(country=lambda x: x.country.map(cname), intendedCountry=lambda x: x.intendedCountry.map(lambda v: cname(v) if v else v))
    show(d.rename(columns={"country": "host", "intendedCountry": "intended host"}), "contests")

with tabs[1]:
    d = in_years(contestants)
    codes = sorted(d.country.unique(), key=cname)
    sel = st.multiselect("Countries", codes, format_func=cname, key="cc")
    q = st.text_input("Search artist or song")
    if sel:
        d = d[d.country.isin(sel)]
    if q:
        d = d[d.artist.str.contains(q, case=False, na=False) | d.song.str.contains(q, case=False, na=False)]
    d = d.assign(country=d.country.map(cname))
    show(d.drop(columns=["lyrics_original"]), "contestants")
    with st.expander("Original lyrics"):
        opts = d[d.lyrics_original.notna()]
        if opts.empty:
            st.write("No lyrics for the current selection.")
        else:
            pick = st.selectbox("Song", opts.index, format_func=lambda i: f"{opts.year[i]} · {opts.country[i]} · {opts.artist[i]} – {opts.song[i]}")
            st.text(opts.lyrics_original[pick])

with tabs[2]:
    r = in_years(results).merge(contestants[["year", "id", "artist", "song"]].rename(columns={"id": "contestant_id"}), on=["year", "contestant_id"], how="left")
    rnds = sorted(r["round"].unique())
    sel_r = st.multiselect("Rounds", rnds, default=["final"])
    if sel_r:
        r = r[r["round"].isin(sel_r)]
    r = r.assign(country=r.country.map(cname)).sort_values(["year", "round", "place"])
    cols = ["year", "round", "place", "country", "artist", "song", "running", "points_total", "points_jury", "points_public", "disqualified"]
    show(r[cols], "results")
    with st.expander("Round dates"):
        st.dataframe(in_years(rounds), hide_index=True, width="stretch")

with tabs[3]:
    st.markdown("One row per voter → entry record. From 2016, `kind` splits into `jury`, `public` (televote) and `total`; earlier years have `total` only.")
    v = in_years(votes)
    l, m, rr, k = st.columns(4)
    vsel = l.multiselect("Voter", sorted(v.voter.unique(), key=cname), format_func=cname)
    esel = m.multiselect("Entry country", sorted(v.entry.dropna().unique(), key=cname), format_func=cname)
    rsel = rr.multiselect("Round", sorted(v["round"].unique()))
    ksel = k.multiselect("Kind", sorted(v.kind.unique()), default=["total"])
    if vsel:
        v = v[v.voter.isin(vsel)]
    if esel:
        v = v[v.entry.isin(esel)]
    if rsel:
        v = v[v["round"].isin(rsel)]
    if ksel:
        v = v[v.kind.isin(ksel)]
    v = v.assign(voter=v.voter.map(cname), entry=v.entry.map(cname))[["year", "round", "kind", "voter", "entry", "points"]]
    show(v.head(50000), "votes")
    if len(v) > 50000:
        st.warning(f"Showing the first 50,000 of {len(v):,} rows. Narrow the filters to see the rest.")

with tabs[4]:
    import pandas as pd
    st.dataframe(pd.DataFrame({"code": list(COUNTRIES), "name": list(COUNTRIES.values())}), hide_index=True, width="stretch")
