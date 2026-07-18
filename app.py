"""Esports Soccer Tracker — pro scene stats, charts, and match predictions.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import core

# ---------------------------------------------------------------- palette ----
# Validated light-mode palette (see dataviz reference palette).
SERIES = ["#2a78d6", "#008300", "#e87ba4", "#eda100"]  # fixed order, max 4
ACCENT_B = "#eb6834"       # player B in head-to-head charts
NEUTRAL = "#898781"        # draws / muted ink
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "matches.csv")

st.set_page_config(page_title="Esports Soccer Tracker", page_icon="⚽",
                   layout="wide")


def _layout(fig: go.Figure, title: str, x_title: str = "", y_title: str = "",
            showlegend: bool = True) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(color=INK, size=16)),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif",
                  color=INK_2, size=13),
        margin=dict(l=10, r=10, t=48, b=10),
        showlegend=showlegend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0,
                    xanchor="right", x=1.0),
        hovermode="closest",
    )
    fig.update_xaxes(title_text=x_title, gridcolor=GRID, zerolinecolor=GRID,
                     linecolor="#c3c2b7", tickfont=dict(color=NEUTRAL))
    fig.update_yaxes(title_text=y_title, gridcolor=GRID, zerolinecolor=GRID,
                     linecolor="#c3c2b7", tickfont=dict(color=NEUTRAL))
    return fig


@st.cache_data
def _load(path: str, mtime: float) -> pd.DataFrame:
    return core.load_matches(path)


def load_data() -> pd.DataFrame:
    mtime = os.path.getmtime(DATA_PATH) if os.path.exists(DATA_PATH) else 0.0
    return _load(DATA_PATH, mtime)


# ---------------------------------------------------------------- sidebar ----
st.sidebar.title("⚽ Esports Soccer Tracker")
st.sidebar.caption("EA SPORTS FC pro scene — stats, form, and match models.")

df_all = load_data()

uploaded = st.sidebar.file_uploader(
    "Import a matches CSV", type="csv",
    help="Columns: date, stage, player_a, player_b, goals_a, goals_b, source",
)
if uploaded is not None:
    try:
        df_all = core.load_matches(uploaded)
        st.sidebar.success(f"Loaded {len(df_all)} matches from upload.")
    except ValueError as e:
        st.sidebar.error(str(e))

include_sim = st.sidebar.checkbox(
    "Include simulated demo matches", value=True,
    help="The bundled dataset mixes 5 real EWC 2026 play-in results with a "
         "clearly-labeled simulated history so the charts and model have "
         "something to chew on. Untick to keep only real results.",
)
if not include_sim:
    df_all = df_all[~df_all["source"].str.contains("Simulated", case=False,
                                                   na=False)]

stages = sorted(df_all["stage"].dropna().unique().tolist())
stage_sel = st.sidebar.multiselect("Stages", stages, default=stages)
df = df_all[df_all["stage"].isin(stage_sel)] if stage_sel else df_all

st.sidebar.download_button(
    "Download current data as CSV",
    df.drop(columns=["total_goals"]).to_csv(index=False).encode(),
    file_name="matches_export.csv", mime="text/csv",
)

if df.empty:
    st.warning("No matches in the current selection — adjust the filters or "
               "import a CSV.")
    st.stop()

lf = core.long_form(df)
ptable = core.player_table(lf)
players = ptable.index.tolist()

tab_overview, tab_players, tab_matches, tab_predict = st.tabs(
    ["Overview", "Players", "Matches", "Predictions"]
)

# --------------------------------------------------------------- overview ----
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Matches tracked", f"{len(df)}")
    c2.metric("Players", f"{len(players)}")
    c3.metric("Avg total goals", f"{df.total_goals.mean():.1f}")
    line_hist = st.session_state.get("line", 16.5)
    c4.metric(f"Over {line_hist} hit rate",
              f"{(df.total_goals > line_hist).mean():.0%}")

    counts = df.total_goals.value_counts().sort_index()
    fig = go.Figure(go.Bar(
        x=counts.index, y=counts.values, name="Matches",
        marker=dict(color=SERIES[0], line=dict(color=SURFACE, width=2)),
        hovertemplate="%{y} matches finished with %{x} total goals"
                      "<extra></extra>",
    ))
    fig.add_vline(x=line_hist, line_dash="dash", line_color=INK_2,
                  annotation_text=f"line {line_hist}",
                  annotation_font_color=INK_2)
    _layout(fig, "Total goals per match", "Total goals (both players)",
            "Matches", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    monthly = (df.set_index("date")["total_goals"]
                 .resample("W").mean().dropna())
    fig2 = go.Figure(go.Scatter(
        x=monthly.index, y=monthly.values, mode="lines",
        line=dict(color=SERIES[0], width=2), name="Weekly avg",
        hovertemplate="Week of %{x|%d %b}: %{y:.1f} avg total goals"
                      "<extra></extra>",
    ))
    _layout(fig2, "Average total goals by week", "", "Avg total goals",
            showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------- players ----
with tab_players:
    show = ptable.copy()
    show["record"] = (show.wins.astype(str) + "-" + show.draws.astype(str)
                      + "-" + show.losses.astype(str))
    top = show.sort_values("win_rate")
    fig = go.Figure(go.Bar(
        x=top.win_rate, y=top.index, orientation="h", name="Win rate",
        marker=dict(color=SERIES[0], line=dict(color=SURFACE, width=2)),
        text=[f"{v:.0%}" for v in top.win_rate], textposition="outside",
        textfont=dict(color=INK_2),
        hovertemplate="%{y}: %{x:.0%} win rate<extra></extra>",
    ))
    _layout(fig, "Win rate by player", "Win rate", "", showlegend=False)
    fig.update_xaxes(range=[0, 1.05], tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_bar(x=show.index, y=show.gf_pm, name="Goals for / match",
                 marker=dict(color=SERIES[0],
                             line=dict(color=SURFACE, width=2)),
                 hovertemplate="%{x}: %{y:.1f} scored per match"
                               "<extra></extra>")
    fig2.add_bar(x=show.index, y=show.ga_pm, name="Goals against / match",
                 marker=dict(color=SERIES[1],
                             line=dict(color=SURFACE, width=2)),
                 hovertemplate="%{x}: %{y:.1f} conceded per match"
                               "<extra></extra>")
    fig2.update_layout(barmode="group")
    _layout(fig2, "Attack vs defense", "", "Goals per match")
    st.plotly_chart(fig2, use_container_width=True)

    sel = st.multiselect("Form chart — pick up to 4 players",
                         players, default=players[:2], max_selections=4)
    if sel:
        fig3 = go.Figure()
        for i, p in enumerate(sel):
            rows = lf[lf.player == p].sort_values("date")
            fig3.add_scatter(
                x=rows.date, y=rows.gd.cumsum(), mode="lines", name=p,
                line=dict(color=SERIES[i], width=2),
                hovertemplate=p + " — %{x|%d %b}: cumulative GD %{y}"
                              "<extra></extra>",
            )
        _layout(fig3, "Form: cumulative goal difference", "",
                "Cumulative goal difference")
        st.plotly_chart(fig3, use_container_width=True)

    st.dataframe(
        show[["matches", "record", "gf", "ga", "win_rate", "gf_pm",
              "ga_pm", "gd_pm"]].rename(columns={
            "matches": "P", "record": "W-D-L", "gf": "GF", "ga": "GA",
            "win_rate": "Win %", "gf_pm": "GF/m", "ga_pm": "GA/m",
            "gd_pm": "GD/m"}).style.format({
            "Win %": "{:.0%}", "GF/m": "{:.1f}", "GA/m": "{:.1f}",
            "GD/m": "{:+.1f}"}),
        use_container_width=True,
    )

# ---------------------------------------------------------------- matches ----
with tab_matches:
    st.dataframe(
        df.sort_values("date", ascending=False)
          .assign(date=lambda d: d.date.dt.date)
          .rename(columns={"player_a": "Player A", "player_b": "Player B",
                           "goals_a": "A", "goals_b": "B",
                           "total_goals": "Total"}),
        use_container_width=True, hide_index=True,
    )

    with st.form("add_match", clear_on_submit=True):
        st.markdown("**Log a match**")
        c1, c2, c3 = st.columns(3)
        date = c1.date_input("Date")
        stage = c2.text_input("Stage / competition", "EWC 2026")
        source = c3.text_input("Source label", "Manual entry")
        c4, c5, c6, c7 = st.columns(4)
        pa = c4.text_input("Player A")
        ga = c5.number_input("Goals A", 0, 99, 0)
        pb = c6.text_input("Player B")
        gb = c7.number_input("Goals B", 0, 99, 0)
        if st.form_submit_button("Add match"):
            if pa.strip() and pb.strip() and pa.strip() != pb.strip():
                row = pd.DataFrame([{
                    "date": date, "stage": stage, "player_a": pa.strip(),
                    "player_b": pb.strip(), "goals_a": int(ga),
                    "goals_b": int(gb), "source": source,
                }])
                base = pd.read_csv(DATA_PATH)
                pd.concat([base, row], ignore_index=True).to_csv(
                    DATA_PATH, index=False)
                st.success(f"Saved {pa} {ga}–{gb} {pb}.")
                st.rerun()
            else:
                st.error("Enter two different player names.")

# ------------------------------------------------------------- predictions ----
with tab_predict:
    st.info(
        "**These are model estimates, not betting advice.** Probabilities come "
        "from a Poisson model over a small (partly simulated, unless you "
        "untick it) match sample. Esports results are volatile — patches, "
        "formats, and day-to-day form move outcomes far more than this model "
        "can see. If you bet, treat these as one input among many, expect the "
        "model to be wrong often, and only stake what you can afford to lose."
    )

    c1, c2, c3 = st.columns([2, 2, 1])
    pa = c1.selectbox("Player A", players, index=0)
    pb = c2.selectbox("Player B", [p for p in players if p != pa])
    line = c3.number_input("Over/under line", 0.5, 60.5, 16.5, step=1.0,
                           key="line")

    try:
        pred = core.predict(df, pa, pb, line=line)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    st.caption(f"Sample: {pred.n_a} matches for {pa}, {pred.n_b} for {pb}. "
               "Small samples = wide error bars the model doesn't show.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"{pa} wins", f"{pred.p_win_a:.0%}",
              f"fair odds {pred.fair_odds(pred.p_win_a):.2f}",
              delta_color="off")
    m2.metric("Draw", f"{pred.p_draw:.0%}",
              f"fair odds {pred.fair_odds(pred.p_draw):.2f}",
              delta_color="off")
    m3.metric(f"{pb} wins", f"{pred.p_win_b:.0%}",
              f"fair odds {pred.fair_odds(pred.p_win_b):.2f}",
              delta_color="off")
    m4.metric("Expected total goals", f"{pred.expected_total:.1f}")

    o1, o2 = st.columns(2)
    o1.metric(f"Over {line}", f"{pred.p_over:.0%}",
              f"fair odds {pred.fair_odds(pred.p_over):.2f}",
              delta_color="off")
    o2.metric(f"Under {line}", f"{pred.p_under:.0%}",
              f"fair odds {pred.fair_odds(pred.p_under):.2f}",
              delta_color="off")

    pmf = core.total_goals_pmf(pred)
    over_mask = pmf.total_goals > line
    fig = go.Figure()
    fig.add_bar(x=pmf.total_goals[~over_mask],
                y=pmf.probability[~over_mask], name=f"Under {line}",
                marker=dict(color=SERIES[0],
                            line=dict(color=SURFACE, width=2)),
                hovertemplate="P(total = %{x}) = %{y:.1%}<extra>Under</extra>")
    fig.add_bar(x=pmf.total_goals[over_mask],
                y=pmf.probability[over_mask], name=f"Over {line}",
                marker=dict(color=ACCENT_B,
                            line=dict(color=SURFACE, width=2)),
                hovertemplate="P(total = %{x}) = %{y:.1%}<extra>Over</extra>")
    fig.update_layout(barmode="overlay")
    _layout(fig, f"Modelled total-goals distribution — {pa} vs {pb}",
            "Total goals", "Probability")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    h2h = df[((df.player_a == pa) & (df.player_b == pb)) |
             ((df.player_a == pb) & (df.player_b == pa))]
    st.markdown(f"**Head-to-head on record: {len(h2h)} match(es)**")
    if len(h2h):
        st.dataframe(h2h.assign(date=lambda d: d.date.dt.date),
                     use_container_width=True, hide_index=True)

    with st.expander("How the model works"):
        st.markdown(
            "- Each player gets an **attack rate** (goals scored per match) "
            "and **defense rate** (conceded per match), shrunk toward the "
            "league average so tiny samples don't produce wild numbers.\n"
            "- Expected goals for A vs B: `λ_A = attack_A × defense_B ÷ "
            "league average` (and symmetrically for B).\n"
            "- Scores are modelled as independent Poissons; win/draw "
            "probabilities come from the joint score grid, and total goals "
            "follow `Poisson(λ_A + λ_B)`.\n"
            "- **Fair odds** are simply `1 ÷ probability` — the break-even "
            "decimal odds if the model were exactly right (it isn't). A "
            "bookmaker price above fair odds is only 'value' if you trust "
            "the model more than the market, which you usually shouldn't."
        )
