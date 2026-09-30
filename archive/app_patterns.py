import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.app_utils import table, cname, require_data

st.set_page_config(page_title="Eurovision Merit", page_icon="🎤", layout="wide")
require_data()
st.title("Eurovision: voting patterns")
st.caption("Built only from the unofficial Eurovision dataset. Patterns use 2004–2024 contests; 2025–2026 are kept aside to test the models.")

A = table("pair_affinity")
top = A[A.n >= 10].nlargest(1, "mean").iloc[0]
py0 = table("jury_public_year")
c = st.columns(3)
c[0].metric("Contests analysed (2004–2024)", table("results").query("2004 <= year <= 2024 and `round` == 'final' and points_total.notna()").year.nunique())
c[1].metric("Strongest tie", f"{cname(top.voter)} → {cname(top.entry)}", f"{top['mean']:.2f} above other voters")
c[2].metric("Finals where jury and public agreed on the winner", f"{int((py0.jury_winner == py0.public_winner).sum())} of {len(py0)}")
st.markdown("**Affinity** = points a voter gave an entry minus what all *other* countries gave that same entry, as a share of the maximum "
            "a voter can give. It removes song quality and leaves voter–performer bias.")

tab1, tab2, tab3 = st.tabs(["Affinity & blocs", "Structural effects", "Jury vs public"])

with tab1:
    sub1, sub2, sub3 = st.tabs(["Country profile", "Strongest pairs", "Blocs"])
    with sub1:
        codes = sorted(set(A.voter) | set(A.entry), key=cname)
        cc = st.selectbox("Country", codes, index=codes.index("GR") if "GR" in codes else 0, format_func=cname)
        min_n = st.slider("Minimum contests observed", 3, 20, 8)
        l, r = st.columns(2)
        gives = A[(A.voter == cc) & (A.n >= min_n)].nlargest(10, "mean")
        gets = A[(A.entry == cc) & (A.n >= min_n)].nlargest(10, "mean")
        l.subheader(f"{cname(cc)} gives most to")
        l.plotly_chart(px.bar(gives.assign(name=gives.entry.map(cname)), x="mean", y="name", orientation="h", hover_data=["n"]).update_yaxes(autorange="reversed", title=""), width="stretch")
        r.subheader(f"{cname(cc)} receives most from")
        r.plotly_chart(px.bar(gets.assign(name=gets.voter.map(cname)), x="mean", y="name", orientation="h", hover_data=["n"]).update_yaxes(autorange="reversed", title=""), width="stretch")
    with sub2:
        n = st.slider("Minimum contests observed together", 5, 20, 10, key="n2")
        t = A[A.n >= n].nlargest(25, "mean").copy()
        t["pair"] = t.voter.map(cname) + " → " + t.entry.map(cname)
        st.dataframe(t[["pair", "mean", "n", "t", "mean_2004_2015", "mean_2016_plus"]].round(3), hide_index=True, width="stretch")
        st.caption("t ≈ mean/(sd/√n); values this large are not luck. The last two columns show whether a tie weakened after the 2016 jury/public split.")
    with sub3:
        from analysis import blocs
        k = st.slider("Number of blocs", 3, 12, 7)
        bl, S = blocs(A, k=k)
        st.dataframe(bl.groupby("bloc").country.apply(lambda s: ", ".join(cname(x) for x in s)).reset_index(), hide_index=True, width="stretch")
        names = [cname(x) for x in bl.country]
        fig = go.Figure(go.Heatmap(z=S, x=names, y=names, colorscale="RdBu", zmid=0, zmin=-.5, zmax=.5))
        fig.update_layout(height=650, title="Mutual affinity, countries ordered by cluster")
        st.plotly_chart(fig, width="stretch")
        bt = table("bloc_trend")
        st.plotly_chart(px.line(bt, x="year", y=["within_bloc", "cross_bloc"], title="Average affinity within vs between blocs (7-bloc clustering)"), width="stretch")

with tab2:
    st.markdown("Effect on an entry's average final score, in within-year standard deviations. Bars are 95% intervals from a bootstrap over years.")
    tab = table("structural_effects")
    labels = {"early_slot": "Performs in first third", "late_slot": "Performs in last third", "host": "Host country", "form_z": "Country's recent form (per SD)"}
    tab = tab.assign(label=tab.feature.map(labels))
    fig = px.scatter(tab, x="effect_sd", y="label", error_x=tab.hi95 - tab.effect_sd, error_x_minus=tab.effect_sd - tab.lo95)
    fig.add_vline(x=0, line_dash="dash")
    fig.update_yaxes(title="", categoryorder="array", categoryarray=tab.label[::-1]); fig.update_xaxes(title="Effect (SD)")
    st.plotly_chart(fig, width="stretch")
    s = table("slot_effect")
    st.plotly_chart(px.bar(s, x="bin", y="mean", labels={"bin": "Position in running order", "mean": "Mean standardised score"}, title="Running order"), width="stretch")
    st.caption("Intervals that cross zero cannot be told apart from no effect. Effects are associations: they can't separate a slot's effect from who the broadcaster put there.")

with tab3:
    py, bc = table("jury_public_year"), table("jury_public_country")
    py = py.assign(same=py.jury_winner == py.public_winner)
    c = st.columns(3)
    c[0].metric("Finals with split scores", len(py)); c[1].metric("Same jury and public winner", int(py.same.sum()))
    c[2].metric("Average rank correlation", f"{py.spearman.mean():.2f}")
    st.plotly_chart(px.bar(py, x="year", y="spearman", title="Rank agreement between jury and public (Spearman)"), width="stretch")
    show = py.assign(jury_winner=py.jury_winner.map(cname), public_winner=py.public_winner.map(cname), winner=py.winner.map(cname))
    st.dataframe(show[["year", "jury_winner", "public_winner", "winner", "spearman"]].round(2), hide_index=True, width="stretch")
    bc = bc.assign(name=bc.country.map(cname))
    st.plotly_chart(px.bar(bc, x="mean_gap", y="name", orientation="h", title="Public rank percentile minus jury rank percentile (min. 3 finals)").update_yaxes(title=""), width="stretch")
    st.caption("Negative: the jury rated the country's entries higher than the public did. Positive: the public did.")
