"""
NHL Player Ratings Dashboard — Streamlit Edition.

Comprehensive player evaluation system with WAR, Win Shares,
contract analysis, and free agent projections.

Usage: streamlit run dashboard/streamlit_app.py
"""

import os
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from unicodedata import normalize as _ucnorm

st.set_page_config(page_title="NHL Player Ratings", layout="wide", page_icon="🏒")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(BASE, "output")
CONTRACTS = os.path.join(BASE, "contracts")

COLORS = {
    "EV_O": "#1f77b4", "EV_D": "#d62728", "PP": "#ff7f0e",
    "PK": "#9467bd", "PEN": "#2ca02c",
    "GV": "#17becf", "OOI": "#bcbd22", "RAPM": "#e377c2",
    "composite": "#1a1a2e", "PV": "#ff9896", "IV": "#aec7e8",
}
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def season_label(s):
    return f"{int(s)}-{str(int(s)+1)[-2:]}"


# ── Load data (cached) ──────────────────────────────────────────────────────

@st.cache_data
def load_data():
    # Prefer trimmed deploy files if available (for Streamlit Cloud deployment)
    _daily_path = os.path.join(OUTPUT, "v5_daily_ratings_deploy.csv")
    if not os.path.exists(_daily_path):
        _daily_path = os.path.join(OUTPUT, "v5_daily_ratings.csv")
    daily = pd.read_csv(_daily_path)
    daily["game_date"] = pd.to_datetime(daily["game_date"])
    daily["total_gar"] = daily["EV_O_gar"] + daily["EV_D_gar"] + daily["PP_gar"] + daily["PK_gar"] + daily["PEN_gar"]

    # Build game matchup lookup from skaters_by_game
    _sbg_files = [os.path.join(BASE, "data", f) for f in
                  ["skaters_by_game_deploy.csv", "skaters_by_game.csv",
                   "skaters_by_game2025_deploy.csv", "skaters_by_game2025.csv"]
                  if os.path.exists(os.path.join(BASE, "data", f))]
    _matchup_parts = []
    for _f in _sbg_files:
        if os.path.exists(_f):
            _s = pd.read_csv(_f, usecols=["playerId", "gameId", "playerTeam", "opposingTeam", "situation"],
                             low_memory=False)
            _s = _s[_s["situation"] == "all"].drop_duplicates(["playerId", "gameId"])
            _matchup_parts.append(_s[["playerId", "gameId", "playerTeam", "opposingTeam"]])
    if _matchup_parts:
        _matchups = pd.concat(_matchup_parts, ignore_index=True)
        daily = daily.merge(_matchups, left_on=["player_id", "game_id"],
                            right_on=["playerId", "gameId"], how="left")
        daily["matchup"] = daily["playerTeam"].fillna("") + " vs " + daily["opposingTeam"].fillna("")
        daily.drop(columns=["playerId", "gameId", "playerTeam", "opposingTeam"], inplace=True, errors="ignore")
    else:
        daily["matchup"] = ""

    daily_war = pd.read_csv(os.path.join(OUTPUT, "v5_daily_war.csv"))
    bpm = pd.read_csv(os.path.join(OUTPUT, "v4_bpm_player_seasons.csv"))
    composite = pd.read_csv(os.path.join(OUTPUT, "v5_composite_player_seasons.csv"))
    composite_war = pd.read_csv(os.path.join(OUTPUT, "v5_season_war.csv"))
    rapm_latest = pd.read_csv(os.path.join(OUTPUT, "v3_rolling_rapm_latest.csv"))
    # Box score stats (goals, assists, shots, etc.)
    skater_box = pd.read_csv(os.path.join(OUTPUT, "dashboard_skater_war.csv"))

    # Win Shares
    ws_path = os.path.join(OUTPUT, "win_shares_by_season.csv")
    win_shares = pd.read_csv(ws_path) if os.path.exists(ws_path) else None

    # Contracts: season-aware lookup table
    active_contracts_path = os.path.join(CONTRACTS, "active_contracts_by_season.csv")
    if os.path.exists(active_contracts_path):
        active_contracts = pd.read_csv(active_contracts_path)
    else:
        active_contracts = None

    # Surplus (v2 multi-metric model if available, else v1)
    surplus_v2_path = os.path.join(CONTRACTS, "surplus_values_v2.csv")
    surplus_v1_path = os.path.join(CONTRACTS, "surplus_values.csv")
    if os.path.exists(surplus_v2_path):
        surplus = pd.read_csv(surplus_v2_path)
    elif os.path.exists(surplus_v1_path):
        surplus = pd.read_csv(surplus_v1_path)
    else:
        surplus = None

    career_v2_path = os.path.join(CONTRACTS, "career_surplus_v2.csv")
    career_v1_path = os.path.join(CONTRACTS, "career_surplus.csv")
    if os.path.exists(career_v2_path):
        career_surplus = pd.read_csv(career_v2_path)
    elif os.path.exists(career_v1_path):
        career_surplus = pd.read_csv(career_v1_path)
    else:
        career_surplus = None

    # FA projections
    fa_path = os.path.join(CONTRACTS, "fa_projections_2026.csv")
    fa_proj = pd.read_csv(fa_path) if os.path.exists(fa_path) else None

    # Goalie WAR
    goalie_path = os.path.join(OUTPUT, "dashboard_goalie_war.csv")
    goalie_war = pd.read_csv(goalie_path) if os.path.exists(goalie_path) else None

    # Player projections
    proj_path = os.path.join(CONTRACTS, "player_projections_2026.csv")
    player_proj = pd.read_csv(proj_path) if os.path.exists(proj_path) else None

    # Draft pick value
    dpv_path = os.path.join(CONTRACTS, "draft_pick_value_chart.csv")
    draft_pick_value = pd.read_csv(dpv_path) if os.path.exists(dpv_path) else None
    dpv_detail_path = os.path.join(CONTRACTS, "draft_pick_value_detail.csv")
    draft_pick_detail = pd.read_csv(dpv_detail_path) if os.path.exists(dpv_detail_path) else None
    draft_picks_path = os.path.join(BASE, "data", "nhl_draft_picks.csv")
    draft_picks_raw = pd.read_csv(draft_picks_path) if os.path.exists(draft_picks_path) else None

    # Build unified season WAR
    unified = daily_war.copy()
    # Total TOI for rate calculations
    unified["toi_total"] = unified["toi_5v5"] + unified["toi_pp"] + unified["toi_pk"]

    # Build box score stats from ALL sources (dashboard_skater_war + raw skaters_by_game)
    # This ensures every player has stats, even those with accent-name mismatches
    _sbg_files = [os.path.join(BASE, "data", f) for f in
                  ["skaters_by_game_deploy.csv", "skaters_by_game.csv",
                   "skaters_by_game2025_deploy.csv", "skaters_by_game2025.csv"]
                  if os.path.exists(os.path.join(BASE, "data", f))]
    _sbg_parts = []
    for _f in _sbg_files:
        if os.path.exists(_f):
            _s = pd.read_csv(_f, usecols=["playerId", "season", "situation",
                "I_F_goals", "I_F_primaryAssists", "I_F_secondaryAssists",
                "I_F_points", "I_F_shotsOnGoal", "I_F_hits",
                "I_F_blockedShotAttempts", "icetime"], low_memory=False)
            _s = _s[_s["situation"] == "all"]
            _sbg_parts.append(_s)

    if _sbg_parts:
        _sbg_all = pd.concat(_sbg_parts, ignore_index=True)
        _sbg_box = _sbg_all.groupby(["playerId", "season"]).agg(
            goals=("I_F_goals", "sum"),
            assists_1=("I_F_primaryAssists", "sum"),
            assists_2=("I_F_secondaryAssists", "sum"),
            points=("I_F_points", "sum"),
            shots=("I_F_shotsOnGoal", "sum"),
            hits=("I_F_hits", "sum"),
            blocks=("I_F_blockedShotAttempts", "sum"),
            toi_all=("icetime", "sum"),
        ).reset_index().rename(columns={"playerId": "player_id"})
        unified = unified.merge(_sbg_box, on=["player_id", "season"], how="left")
    else:
        # Fallback to dashboard_skater_war only
        box_cols = ["player_id", "season", "goals", "assists_1", "assists_2", "points",
                    "shots", "hits", "blocks", "toi_all"]
        box_merge = skater_box[[c for c in box_cols if c in skater_box.columns]].drop_duplicates(["player_id", "season"])
        unified = unified.merge(box_merge, on=["player_id", "season"], how="left")

    comp_cols = ["composite_O", "composite_D", "PV_O", "PV_D", "IV_O", "IV_D"]
    comp_merge = composite[["player_id", "season"] + [c for c in comp_cols if c in composite.columns]]
    unified = unified.merge(comp_merge, on=["player_id", "season"], how="left")

    bpm_cols = ["GV_O", "GV_D", "OOI_O", "OOI_D", "RAPM_O", "RAPM_D", "PP_O", "PK_D", "ozPct"]
    bpm_merge = bpm[["player_id", "season"] + [c for c in bpm_cols if c in bpm.columns]]
    unified = unified.merge(bpm_merge, on=["player_id", "season"], how="left")

    rapm_cols = ["player_id", "BPR_O", "BPR_D", "BPR", "BPR_se"]
    rapm_merge = rapm_latest[[c for c in rapm_cols if c in rapm_latest.columns]]
    unified = unified.merge(rapm_merge, on="player_id", how="left")

    return {
        "daily": daily, "unified": unified, "skater_box": skater_box,
        "win_shares": win_shares, "surplus": surplus,
        "career_surplus": career_surplus, "fa_proj": fa_proj,
        "player_proj": player_proj, "goalie_war": goalie_war,
        "active_contracts": active_contracts,
        "draft_pick_value": draft_pick_value, "draft_pick_detail": draft_pick_detail,
        "draft_picks_raw": draft_picks_raw,
    }


data = load_data()
unified = data["unified"]
daily = data["daily"]
win_shares = data["win_shares"]


def get_contract_lookup(season):
    """Get {player_name: contract_row} for a specific season."""
    ac = data["active_contracts"]
    if ac is None:
        return {}
    szn = ac[ac["season"] == season]
    return {row["player_name"]: row for _, row in szn.iterrows()}

ALL_SEASONS = sorted(unified["season"].unique())
LATEST_SEASON = max(ALL_SEASONS)


def pos_filter(df, pos):
    if pos == "F":
        return df[df["position"].isin(["C", "L", "R"])]
    elif pos == "D":
        return df[df["position"] == "D"]
    return df


def style_fig(fig):
    fig.update_layout(
        plot_bgcolor="white", hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig.update_yaxes(gridcolor="#eee", zeroline=True, zerolinecolor="gray")
    fig.update_xaxes(gridcolor="#eee")
    return fig


# ── Navigation ───────────────────────────────────────────────────────────────

st.markdown(
    """<style>
    [data-testid="stSidebar"] { display: none; }
    div[data-testid="stTabs"] button { font-size: 16px; font-weight: 600; }
    </style>""",
    unsafe_allow_html=True,
)

st.markdown("## NHL Player Ratings")

_tabs = st.tabs([
    "Leaderboard", "Player Profile", "Compare",
    "Win Shares", "Goalies", "Team View", "Contracts & Value",
    "Free Agent Projections", "Player Projections", "Draft Pick Value",
    "Trade Evaluator", "Research",
])
_tab_names = [
    "Leaderboard", "Player Profile", "Compare",
    "Win Shares", "Goalies", "Team View", "Contracts & Value",
    "Free Agent Projections", "Player Projections", "Draft Pick Value",
    "Trade Evaluator", "Research",
]


# ═══════════════════════════════════════════════════════════════════════════════
# LEADERBOARD
# ═══════════════════════════════════════════════════════════════════════════════

with _tabs[0]:
    st.header("Season Leaderboard")

    c1, c2, c3, c4, c5, c6 = st.columns([1.5, 1.2, 1, 1.5, 1, 1.2])
    season = c1.selectbox("Season", ALL_SEASONS, index=len(ALL_SEASONS)-1,
                          format_func=season_label, key="lb_season")
    pos = c2.selectbox("Position", ["ALL", "F", "D"], key="lb_pos")
    min_gp = c3.number_input("Min GP", 1, 82, 20, key="lb_mingp")
    sort_col = c4.selectbox("Sort By", ["WAR", "WAR_82", "WAR_O", "WAR_D",
                            "EV_O_GAR", "EV_D_GAR", "PP_GAR", "PK_GAR"],
                            key="lb_sort")
    top_n = c5.number_input("Top N", 5, 200, 30, key="lb_topn")
    rate_mode = c6.toggle("Per 60 min", key="lb_rate")

    df = unified[unified["season"] == season].copy()
    df = pos_filter(df, pos)
    df = df[df["GP"] >= min_gp]
    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=False).head(top_n)
    df = df.reset_index(drop=True)
    df["rank"] = df.index + 1

    # Contract info (season-aware)
    cl = get_contract_lookup(season)
    df["Cap Hit"] = df["player_name"].map(
        lambda n: f"${cl[n]['cap_hit']:,.0f}" if n in cl else None)

    if rate_mode:
        # Convert counting stats to per-60 rates
        toi_hrs = (df["toi_total"] / 60).clip(lower=0.1)
        rate_cols = {}
        for c in ["EV_O_GAR", "EV_D_GAR", "PP_GAR", "PK_GAR", "PEN_GAR",
                   "WAR_O", "WAR_D", "WAR"]:
            if c in df.columns:
                df[c] = df[c] / toi_hrs
                rate_cols[c] = f"{c}/60"
        if "points" in df.columns:
            df["points"] = df["points"] / toi_hrs
            rate_cols["points"] = "Pts/60"
        if "goals" in df.columns:
            df["goals"] = df["goals"] / toi_hrs
            rate_cols["goals"] = "G/60"

        # Build table
        table_cols = ["rank", "player_name", "position", "GP", "toi_total", "Cap Hit"]
        optional = ["goals", "points",
                    "EV_O_GAR", "EV_D_GAR", "PP_GAR", "PK_GAR", "PEN_GAR",
                    "WAR_O", "WAR_D", "WAR"]
        for c in optional:
            if c in df.columns and not df[c].isna().all():
                table_cols.append(c)
        # Rename columns for display
        df = df.rename(columns={**rate_cols, "toi_total": "TOI"})
        table_cols = [rate_cols.get(c, c) if c != "toi_total" else "TOI" for c in table_cols]
    else:
        # Raw counting stats
        table_cols = ["rank", "player_name", "position", "GP", "Cap Hit"]
        if "goals" in df.columns:
            table_cols += ["goals", "points"]
        optional = ["EV_O_GAR", "EV_D_GAR", "PP_GAR", "PK_GAR", "PEN_GAR",
                    "WAR_O", "WAR_D", "WAR", "WAR_82"]
        for c in optional:
            if c in df.columns and not df[c].isna().all():
                table_cols.append(c)

    display = df[table_cols].round(3).copy()
    # Make player names clickable links to the Player Profile tab
    display["player_name"] = display["player_name"].apply(
        lambda n: f'<a href="?player={n.replace(" ", "+")}" target="_self">{n}</a>'
    )
    st.markdown(
        display.to_html(escape=False, index=False),
        unsafe_allow_html=True,
    )
    # Style the table
    st.markdown("""<style>
    .stMarkdown table { width: 100%; border-collapse: collapse; font-size: 13px; }
    .stMarkdown table th { background: rgba(255,255,255,0.1); padding: 6px 10px; text-align: center;
                           font-weight: 700; border-bottom: 2px solid rgba(255,255,255,0.2); }
    .stMarkdown table td { padding: 5px 10px; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.05); }
    .stMarkdown table tr:hover { background: rgba(255,255,255,0.05); }
    .stMarkdown table a { color: #4a9eff; text-decoration: none; font-weight: 600; }
    .stMarkdown table a:hover { text-decoration: underline; }
    </style>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PLAYER PROFILE
# ═══════════════════════════════════════════════════════════════════════════════

with _tabs[1]:
    st.header("Player Profile")

    all_players = sorted(unified["player_name"].unique())

    # Check query params for player link from leaderboard
    _qp = st.query_params
    _linked_player = _qp.get("player", "").replace("+", " ")
    if _linked_player in all_players:
        _default_idx = all_players.index(_linked_player)
    elif "Connor McDavid" in all_players:
        _default_idx = all_players.index("Connor McDavid")
    else:
        _default_idx = 0

    c1, c2 = st.columns([3, 1])
    player_name = c1.selectbox("Player", all_players, index=_default_idx)
    season = c2.selectbox("Season", ALL_SEASONS, index=len(ALL_SEASONS)-1, format_func=season_label)

    ph = unified[unified["player_name"] == player_name].sort_values("season")
    pw = unified[(unified["player_name"] == player_name) & (unified["season"] == season)]

    # Box score for this player-season
    skater_box = data["skater_box"]
    box_row = skater_box[(skater_box["player_name"] == player_name) & (skater_box["season"] == season)]

    # Surplus for this player-season (v2 uses pred_cap_pct / surplus_pct)
    surplus_data = data["surplus"]
    surplus_row = None
    if surplus_data is not None:
        _sr = surplus_data[(surplus_data["player_name"] == player_name) & (surplus_data["season"] == season)]
        if len(_sr):
            surplus_row = _sr.iloc[0]

    if len(pw):
        r = pw.iloc[0]
        cl = get_contract_lookup(season)
        cinfo = cl.get(player_name)
        b = box_row.iloc[0] if len(box_row) else None

        # ── Player header ──
        st.markdown(f"### {player_name}")
        st.caption(f"{r['position']} — {int(r['GP'])} GP — {season_label(season)}"
                   + (f" — {b['team']}" if b is not None and "team" in box_row.columns else ""))

        # ── Row 1: Basic Stats ──
        st.markdown("**Traditional Stats**")
        if b is not None:
            gp = int(b["GP"])
            m1 = st.columns(10)
            m1[0].metric("GP", gp)
            m1[1].metric("Goals", int(b["goals"]))
            m1[2].metric("Assists", int(b.get("assists_1", 0) + b.get("assists_2", 0)))
            m1[3].metric("Points", int(b["points"]))
            m1[4].metric("Shots", int(b["shots"]))
            m1[5].metric("Hits", int(b["hits"]))
            m1[6].metric("Blocks", int(b["blocks"]))
            toi_pg = b["toi_all"] / max(gp, 1)
            m1[7].metric("TOI/GP", f"{toi_pg:.1f}")
            pp_pg = b["toi_pp"] / max(gp, 1)
            m1[8].metric("PP/GP", f"{pp_pg:.1f}")
            pk_pg = b["toi_pk"] / max(gp, 1)
            m1[9].metric("PK/GP", f"{pk_pg:.1f}")
        else:
            st.caption("No box score data for this season.")

        st.divider()

        # ── Row 2: Advanced Stats ──
        st.markdown("**Advanced Metrics**")
        m2 = st.columns(9)
        m2[0].metric("WAR", f"{r.get('WAR', 0):+.2f}")
        m2[1].metric("WAR/82", f"{r.get('WAR_82', 0):+.2f}")
        m2[2].metric("WAR_O", f"{r.get('WAR_O', 0):+.2f}")
        m2[3].metric("WAR_D", f"{r.get('WAR_D', 0):+.2f}")
        m2[4].metric("EV_O GAR", f"{r.get('EV_O_GAR', 0):+.2f}")
        m2[5].metric("EV_D GAR", f"{r.get('EV_D_GAR', 0):+.2f}")
        m2[6].metric("PP GAR", f"{r.get('PP_GAR', 0):+.2f}")
        m2[7].metric("PK GAR", f"{r.get('PK_GAR', 0):+.2f}")
        m2[8].metric("PEN GAR", f"{r.get('PEN_GAR', 0):+.2f}")

        st.divider()

        # ── Row 3: Contract ──
        st.markdown("**Contract & Value**")
        m3 = st.columns(6)
        if cinfo is not None:
            m3[0].metric("Cap Hit", f"${cinfo['cap_hit']:,.0f}")
            m3[1].metric("Cap %", f"{cinfo.get('cap_pct', 0):.1f}%")
            m3[2].metric("Type", str(cinfo.get("contract_type", "—")))
            m3[3].metric("Status", str(cinfo.get("sign_status", "—")))
        else:
            m3[0].metric("Cap Hit", "—")
            m3[1].metric("Cap %", "—")
            m3[2].metric("Term", "—")
            m3[3].metric("Type", "—")
        if surplus_row is not None:
            mkt_col = "pred_market_value" if "pred_market_value" in surplus_row.index else "market_value"
            m3[4].metric("Market Value", f"${surplus_row[mkt_col]:,.0f}")
            sv = surplus_row["surplus_value"]
            m3[5].metric("Surplus", f"${sv:+,.0f}", delta=f"{'Bargain' if sv > 0 else 'Overpaid'}")
        else:
            m3[4].metric("Market Value", "—")
            m3[5].metric("Surplus", "—")

        st.divider()

        # ── Chart 1: GAR & WAR by Season (grouped bar) ──
        if len(ph) > 1:
            ph_chart = ph.copy()
            ph_chart["GAR"] = ph_chart["GAR_O"] + ph_chart["GAR_D"]
            slabels = [season_label(s) for s in ph_chart["season"]]

            fig1 = go.Figure()
            fig1.add_trace(go.Bar(
                x=slabels, y=ph_chart["GAR"], name="GAR",
                marker_color="#4a90d9", opacity=0.85,
                hovertemplate="%{x}<br>GAR: %{y:.2f}<br>"
                              "GAR_O: %{customdata[0]:.2f}<br>"
                              "GAR_D: %{customdata[1]:.2f}<extra></extra>",
                customdata=ph_chart[["GAR_O", "GAR_D"]].values,
            ))
            fig1.add_trace(go.Bar(
                x=slabels, y=ph_chart["WAR"], name="WAR",
                marker_color="#2d2d2d", opacity=0.85,
                hovertemplate="%{x}<br>WAR: %{y:.2f}<br>"
                              "WAR_O: %{customdata[0]:.2f}<br>"
                              "WAR_D: %{customdata[1]:.2f}<extra></extra>",
                customdata=ph_chart[["WAR_O", "WAR_D"]].values,
            ))
            fig1.update_layout(
                barmode="group", title="GAR & WAR by Season", height=400,
                yaxis_title="Value",
            )
            style_fig(fig1)
            st.plotly_chart(fig1, use_container_width=True)

        # ── Chart 2: Daily Rating Timeline (DARKO-style, selected season) ──
        pg = daily[(daily["player_name"] == player_name) & (daily["season"] == season)].copy()
        pg = pg.sort_values("game_date")

        if len(pg):
            pg["Total"] = pg["EV_O"] + pg["EV_D"] + pg["PP"] + pg["PK"] + pg["PEN"]

            _daily_metric_options = {
                "Total": ("#2d2d2d", "Total"),
                "EV_O": (COLORS["EV_O"], "EV Offense"),
                "EV_D": (COLORS["EV_D"], "EV Defense"),
                "PP": (COLORS["PP"], "Power Play"),
                "PK": (COLORS["PK"], "Penalty Kill"),
                "PEN": (COLORS["PEN"], "Penalties"),
            }

            selected_metrics = st.multiselect(
                "Metrics (max 3)", list(_daily_metric_options.keys()),
                default=["Total"], max_selections=3, key="daily_metrics",
            )

            if selected_metrics:
                fig_daily = go.Figure()

                for metric_key in selected_metrics:
                    color, label = _daily_metric_options[metric_key]
                    fig_daily.add_trace(go.Scatter(
                        x=pg["game_date"], y=pg[metric_key],
                        name=label, mode="lines+markers",
                        line=dict(color=color, width=3),
                        marker=dict(color=color, size=5),
                        customdata=pg["matchup"].values,
                        hovertemplate="%{customdata}<br>%{x|%b %d, %Y}<br>" + label + ": %{y:.3f}<extra></extra>",
                    ))

                fig_daily.add_hline(y=0, line_color="#999", line_width=1)

                fig_daily.update_layout(
                    title=f"Daily Rating — {season_label(season)} (Bayesian-Smoothed, per 60)",
                    height=450,
                    yaxis_title="Rating (per 60, above average)",
                    xaxis=dict(tickformat="%b %d"),
                )
                style_fig(fig_daily)
                st.plotly_chart(fig_daily, use_container_width=True)
    else:
        st.info("No data for this player/season.")


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARE
# ═══════════════════════════════════════════════════════════════════════════════

with _tabs[2]:
    st.header("Player Comparison")

    all_players = sorted(unified["player_name"].unique())
    defaults = ["Connor McDavid", "Leon Draisaitl", "Nikita Kucherov", "Nathan MacKinnon", "Kirill Kaprizov"]
    defaults = [p for p in defaults if p in all_players]

    players = st.multiselect("Players (up to 6)", all_players, default=defaults[:5], max_selections=6)
    c1, c2 = st.columns(2)
    season = c1.selectbox("Season", ALL_SEASONS, index=len(ALL_SEASONS)-1, format_func=season_label, key="cmp_szn")
    metric = c2.selectbox("Chart Metric", ["total_gar", "EV_O_gar", "PP_gar", "PK_gar", "EV_O", "PP"], key="cmp_met")

    if players:
        is_cum = metric in ["total_gar", "PP_gar", "EV_O_gar", "PK_gar"]

        # Time series
        fig = go.Figure()
        for i, name in enumerate(players[:6]):
            pg = daily[(daily["player_name"] == name) & (daily["season"] == season)].copy()
            if len(pg) == 0:
                continue
            y = pg[metric].cumsum() if is_cum else pg[metric]
            fig.add_trace(go.Scatter(x=pg["game_date"], y=y, name=name,
                                     line=dict(color=PALETTE[i % len(PALETTE)], width=2.5)))
        mlabel = metric.replace("_gar", " GAR").replace("_", " ").upper()
        fig.update_layout(title=f"{'Cumulative ' if is_cum else ''}{mlabel} — {season_label(season)}",
                          height=450, xaxis=dict(tickformat="%b %d"))
        style_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

        # Radar
        gar_cats = ["EV_O_GAR", "EV_D_GAR", "PP_GAR", "PK_GAR", "PEN_GAR"]
        fig_radar = go.Figure()
        for i, name in enumerate(players[:6]):
            pw = unified[(unified["player_name"] == name) & (unified["season"] == season)]
            if len(pw) == 0:
                continue
            r = pw.iloc[0]
            vals = [r.get(c, 0) for c in gar_cats] + [r.get(gar_cats[0], 0)]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=["EV Off", "EV Def", "PP", "PK", "PEN", "EV Off"],
                name=f"{name} ({r.get('WAR', 0):+.1f})",
                line=dict(color=PALETTE[i % len(PALETTE)], width=2), fill="toself", opacity=0.25))
        fig_radar.update_layout(title=f"GAR Component Radar — {season_label(season)}",
                                height=420,
                                polar=dict(radialaxis=dict(visible=True, gridcolor="#ddd")))
        st.plotly_chart(fig_radar, use_container_width=True)

        # Table
        rows = []
        cl = get_contract_lookup(season)
        for name in players:
            pw = unified[(unified["player_name"] == name) & (unified["season"] == season)]
            if len(pw) == 0:
                continue
            r = pw.iloc[0]
            cinfo = cl.get(name)
            row = {"Player": name, "Pos": r["position"], "GP": int(r["GP"]),
                   "Cap Hit": f"${cinfo['cap_hit']:,.0f}" if cinfo is not None else None,
                   "WAR": round(r.get("WAR", 0), 2), "WAR/82": round(r.get("WAR_82", 0), 2)}
            for c in ["EV_O_GAR", "EV_D_GAR", "PP_GAR", "PK_GAR", "PEN_GAR"]:
                row[c] = round(r.get(c, 0), 2)
            rows.append(row)
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# WIN SHARES
# ═══════════════════════════════════════════════════════════════════════════════

with _tabs[3]:
    st.header("Win Shares")
    st.caption("Actual team wins allocated to individual players based on GAR contributions")

    if win_shares is not None:
        ws_seasons = sorted(win_shares["season"].unique())
        c1, c2, c3 = st.columns([1.5, 1.2, 1])
        season = c1.selectbox("Season", ws_seasons, index=len(ws_seasons)-1,
                              format_func=season_label, key="ws_season")
        pos = c2.selectbox("Position", ["ALL", "F", "D"], key="ws_pos")
        min_gp = c3.number_input("Min GP", 1, 82, 20, key="ws_mingp")

        ws = win_shares[win_shares["season"] == season].copy()
        ws = pos_filter(ws, pos)
        ws = ws[ws["GP"] >= min_gp].sort_values("WS", ascending=False).head(40)

        # Chart
        fig = go.Figure()
        fig.add_trace(go.Bar(y=ws["player_name"], x=ws["OWS"], name="Offensive WS",
                             orientation="h", marker_color=COLORS["EV_O"], opacity=0.8))
        fig.add_trace(go.Bar(y=ws["player_name"], x=ws["DWS"], name="Defensive WS",
                             orientation="h", marker_color=COLORS["EV_D"], opacity=0.8))
        fig.update_layout(barmode="stack",
                          yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
                          xaxis_title="Win Shares",
                          title=f"Win Shares — {season_label(season)}",
                          height=max(400, len(ws) * 22))
        style_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

        # Table
        st.dataframe(ws[["player_name", "team", "position", "GP", "OWS", "DWS", "WS", "WS_82"]].round(2),
                     use_container_width=True, hide_index=True)

        # Team decomposition
        st.subheader("Team Win Decomposition")
        team_ws = win_shares[win_shares["season"] == season].groupby("team").agg(
            total_WS=("WS", "sum"), players=("player_name", "count"),
            top_player=("WS", "max")
        ).reset_index().sort_values("total_WS", ascending=False)
        st.dataframe(team_ws.round(1), use_container_width=True, hide_index=True)
    else:
        st.warning("Win shares data not found. Run `python bpr/win_shares.py` first.")


# ═══════════════════════════════════════════════════════════════════════════════
# GOALIES
# ═══════════════════════════════════════════════════════════════════════════════

with _tabs[4]:
    st.header("Goalies")

    gw = data["goalie_war"]
    if gw is not None:
        gw_seasons = sorted(gw["season"].unique())

        c1, c2, c3 = st.columns([1.5, 1, 1])
        g_season = c1.selectbox("Season", gw_seasons, index=len(gw_seasons)-1,
                                format_func=season_label, key="g_season")
        g_min_shots = c2.number_input("Min Shots", 100, 3000, 500, key="g_min_shots")
        g_sort = c3.selectbox("Sort By", ["GOALIE_WAR", "GSAx_adj", "sv_pct", "shots_faced"],
                              key="g_sort")

        gw_szn = gw[gw["season"] == g_season].copy()
        gw_szn = gw_szn[gw_szn["shots_faced"] >= g_min_shots]
        gw_szn = gw_szn.sort_values(g_sort, ascending=False).reset_index(drop=True)
        gw_szn["rank"] = gw_szn.index + 1

        # Contract info
        cl = get_contract_lookup(g_season)
        gw_szn["Cap Hit"] = gw_szn["goalie_name"].map(
            lambda n: f"${cl[n]['cap_hit']:,.0f}" if n in cl else None)

        # Leaderboard table
        st.subheader("Goalie Leaderboard")
        show_cols = ["rank", "goalie_name", "Cap Hit", "shots_faced",
                     "sv_pct", "GSAx_adj", "GOALIE_GAR", "GOALIE_WAR"]
        show_cols = [c for c in show_cols if c in gw_szn.columns]
        st.dataframe(gw_szn[show_cols].round(3), use_container_width=True, hide_index=True)

        # GSAx chart
        st.subheader("Goals Saved Above Expected")
        top_g = gw_szn.head(20)
        fig_g = go.Figure()
        colors_g = [COLORS["EV_O"] if v > 0 else COLORS["EV_D"] for v in top_g["GSAx_adj"]]
        fig_g.add_trace(go.Bar(
            y=top_g["goalie_name"], x=top_g["GSAx_adj"], orientation="h",
            marker_color=colors_g, opacity=0.85,
            hovertemplate="%{y}<br>GSAx: %{x:.1f}<br>WAR: %{customdata:.2f}<extra></extra>",
            customdata=top_g["GOALIE_WAR"],
        ))
        fig_g.update_layout(
            yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
            xaxis_title="Goals Saved Above Expected (adjusted)",
            title=f"GSAx — {season_label(g_season)}",
            height=max(400, len(top_g) * 24), showlegend=False,
        )
        fig_g.add_vline(x=0, line_color="#999", line_width=1)
        style_fig(fig_g)
        st.plotly_chart(fig_g, use_container_width=True)

        # Goalie profile
        st.divider()
        st.subheader("Goalie Profile")
        all_goalies = sorted(gw["goalie_name"].unique())
        g_select = st.selectbox("Select Goalie", all_goalies,
                                index=all_goalies.index("Connor Hellebuyck") if "Connor Hellebuyck" in all_goalies else 0,
                                key="g_profile")

        g_hist = gw[gw["goalie_name"] == g_select].sort_values("season")
        if len(g_hist):
            # Summary
            latest = g_hist.iloc[-1]
            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            mc1.metric("WAR", f"{latest['GOALIE_WAR']:.2f}")
            mc2.metric("GSAx", f"{latest['GSAx_adj']:.1f}")
            mc3.metric("Sv%", f"{latest['sv_pct']:.3f}")
            mc4.metric("Shots", f"{int(latest['shots_faced']):,}")
            cinfo = cl.get(g_select)
            mc5.metric("Cap Hit", f"${cinfo['cap_hit']:,.0f}" if cinfo else "—")

            if len(g_hist) > 1:
                # WAR by season
                fig_gh = go.Figure()
                fig_gh.add_trace(go.Bar(
                    x=[season_label(s) for s in g_hist["season"]],
                    y=g_hist["GOALIE_WAR"],
                    marker_color=COLORS["EV_O"], opacity=0.85,
                    hovertemplate="%{x}<br>WAR: %{y:.2f}<br>GSAx: %{customdata:.1f}<extra></extra>",
                    customdata=g_hist["GSAx_adj"],
                ))
                fig_gh.update_layout(title=f"{g_select} — WAR by Season", height=350,
                                     yaxis_title="Goalie WAR")
                style_fig(fig_gh)
                st.plotly_chart(fig_gh, use_container_width=True)

                # Sv% trend
                fig_sv = go.Figure()
                fig_sv.add_trace(go.Scatter(
                    x=[season_label(s) for s in g_hist["season"]],
                    y=g_hist["sv_pct"], mode="lines+markers",
                    line=dict(color=COLORS["EV_O"], width=3),
                    marker=dict(size=7),
                ))
                fig_sv.update_layout(title=f"{g_select} — Save Percentage", height=300,
                                     yaxis_title="Sv%", yaxis_tickformat=".3f")
                style_fig(fig_sv)
                st.plotly_chart(fig_sv, use_container_width=True)
    else:
        st.warning("Goalie WAR data not found.")


# ═══════════════════════════════════════════════════════════════════════════════
# TEAM VIEW
# ═══════════════════════════════════════════════════════════════════════════════

with _tabs[5]:
    st.header("Team View — 2025-26")

    ws_data = data["win_shares"]
    surplus_data = data["surplus"]
    skater_box = data["skater_box"]

    if ws_data is not None:
        ws25 = ws_data[ws_data["season"] == LATEST_SEASON].copy()
        all_teams = sorted(ws25["team"].unique())

        team_sel = st.selectbox("Team", all_teams,
                                index=all_teams.index("EDM") if "EDM" in all_teams else 0,
                                key="tv_team")

        # Get roster from win shares (has team assignment)
        roster = ws25[ws25["team"] == team_sel].copy()
        player_ids = set(roster["player_id"])

        # Merge box score stats
        box25 = skater_box[skater_box["season"] == LATEST_SEASON].copy()
        roster = roster.merge(
            box25[["player_id", "goals", "assists_1", "assists_2", "points", "shots",
                   "hits", "blocks", "toi_all"]],
            on="player_id", how="left"
        )

        # Merge surplus
        if surplus_data is not None:
            sv25 = surplus_data[surplus_data["season"] == LATEST_SEASON].copy()
            surplus_cols = ["player_id", "cap_hit", "surplus_value"]
            mkt_col = "pred_market_value" if "pred_market_value" in sv25.columns else "market_value"
            if mkt_col in sv25.columns:
                surplus_cols.append(mkt_col)
            roster = roster.merge(sv25[surplus_cols], on="player_id", how="left")

        # Contract info (season-aware)
        cl = get_contract_lookup(LATEST_SEASON)
        roster["Cap Hit"] = roster["player_name"].map(
            lambda n: f"${cl[n]['cap_hit']:,.0f}" if n in cl else None)

        roster = roster.sort_values("WS", ascending=False)

        # ── Team summary metrics ──
        # Gini coefficient on WS_82
        def gini(values):
            v = np.sort(np.abs(values))
            n = len(v)
            if n == 0 or v.sum() == 0:
                return 0
            index = np.arange(1, n + 1)
            return (2 * np.sum(index * v) - (n + 1) * np.sum(v)) / (n * np.sum(v))

        ws82_gini = gini(roster["WS_82"].clip(lower=0).values)

        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        c1.metric("Players", len(roster))
        c2.metric("Total WAR", f"{roster['GAR_O'].sum() + roster['GAR_D'].sum():.1f} GAR")
        c3.metric("Total WS", f"{roster['WS'].sum():.1f}")
        total_cap = roster["cap_hit"].sum() if "cap_hit" in roster.columns else 0
        c4.metric("Roster Cap", f"${total_cap/1e6:.1f}M" if total_cap > 0 else "—")
        total_surplus = roster["surplus_value"].sum() if "surplus_value" in roster.columns else 0
        c5.metric("Total Surplus", f"${total_surplus/1e6:+.1f}M" if total_surplus != 0 else "—")
        avg_ws = roster["WS"].mean()
        c6.metric("Avg WS", f"{avg_ws:.2f}")
        c7.metric("WS Gini", f"{ws82_gini:.3f}",
                  help="0 = perfectly equal distribution, 1 = one player has all win shares")

        st.divider()

        # ── Roster table ──
        st.subheader("Roster")
        display_cols = ["player_name", "position", "GP", "Cap Hit"]

        # Basic stats
        if "goals" in roster.columns:
            roster["assists"] = roster.get("assists_1", 0) + roster.get("assists_2", 0)
            display_cols += ["goals", "assists", "points", "shots"]

        # Advanced
        display_cols += ["OWS", "DWS", "WS", "WS_82"]

        # Surplus
        if "surplus_value" in roster.columns:
            roster["Surplus"] = roster["surplus_value"].apply(
                lambda v: f"${v/1e6:+.1f}M" if pd.notna(v) else "—")
            display_cols.append("Surplus")

        display_cols = [c for c in display_cols if c in roster.columns]
        st.dataframe(roster[display_cols].round(2), use_container_width=True, hide_index=True)

        # ── Win Shares breakdown chart ──
        st.subheader("Win Shares by Player")
        top_roster = roster.head(20)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=top_roster["player_name"], x=top_roster["OWS"], name="Offensive WS",
            orientation="h", marker_color=COLORS["EV_O"], opacity=0.85,
        ))
        fig.add_trace(go.Bar(
            y=top_roster["player_name"], x=top_roster["DWS"], name="Defensive WS",
            orientation="h", marker_color=COLORS["EV_D"], opacity=0.85,
        ))
        fig.update_layout(barmode="stack",
                          yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
                          xaxis_title="Win Shares", height=max(400, len(top_roster) * 24),
                          title=f"{team_sel} Win Shares Breakdown")
        style_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

        # ── Cap allocation chart ──
        if "cap_hit" in roster.columns and roster["cap_hit"].notna().any():
            st.subheader("Cap Allocation")
            cap_roster = roster[roster["cap_hit"].notna()].sort_values("cap_hit", ascending=False).head(20)
            fig_cap = go.Figure()
            fig_cap.add_trace(go.Bar(
                y=cap_roster["player_name"], x=cap_roster["cap_hit"] / 1e6,
                orientation="h", marker_color="#4a90d9", opacity=0.85,
                hovertemplate="%{y}: $%{x:.1f}M<extra></extra>",
            ))
            fig_cap.update_layout(
                yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
                xaxis_title="Cap Hit ($M)", height=max(400, len(cap_roster) * 24),
                title=f"{team_sel} Cap Allocation", showlegend=False,
            )
            style_fig(fig_cap)
            st.plotly_chart(fig_cap, use_container_width=True)
    else:
        st.warning("Win shares data not found. Run `python bpr/win_shares.py` first.")


# ═══════════════════════════════════════════════════════════════════════════════
# CONTRACTS & VALUE
# ═══════════════════════════════════════════════════════════════════════════════

with _tabs[6]:
    st.header("Contracts & Surplus Value")

    tab1, tab2 = st.tabs(["Surplus Value", "Cap Hit vs Performance"])

    with tab1:
        surplus = data["surplus"]
        if surplus is not None:
            st.subheader("Season Surplus Value")
            st.caption("Market value (based on stats) minus actual cap hit. Positive = bargain.")

            c1, c2 = st.columns([1.5, 1.5])
            szn = c1.selectbox("Season", sorted(surplus["season"].unique()),
                               index=len(surplus["season"].unique())-1,
                               format_func=season_label, key="sv_season")
            view = c2.selectbox("View", ["Best Values", "Worst Values", "All"], key="sv_view")

            sv = surplus[surplus["season"] == szn].copy()
            if view == "Best Values":
                sv = sv.nlargest(30, "surplus_value")
            elif view == "Worst Values":
                sv = sv.nsmallest(30, "surplus_value")
            else:
                sv = sv.sort_values("surplus_value", ascending=False)

            fig = go.Figure()
            colors = ["#2d6a4f" if v > 0 else "#d32f2f" for v in sv["surplus_value"]]
            fig.add_trace(go.Bar(
                y=sv["player_name"], x=sv["surplus_value"] / 1e6,
                orientation="h", marker_color=colors,
                hovertemplate="%{y}: $%{x:.1f}M<extra></extra>",
            ))
            fig.update_layout(
                yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
                xaxis_title="Surplus Value ($M)",
                title=f"Surplus Value — {season_label(szn)}",
                height=max(400, len(sv) * 20),
                showlegend=False,
            )
            style_fig(fig)
            st.plotly_chart(fig, use_container_width=True)

            show_cols = ["player_name", "position", "GP", "WAR", "cap_hit",
                         "surplus_value", "contract_type", "sign_status"]
            mkt_col = "pred_market_value" if "pred_market_value" in sv.columns else "market_value"
            if mkt_col in sv.columns:
                show_cols.insert(5, mkt_col)
            display = sv[[c for c in show_cols if c in sv.columns]].copy()
            for c in ["cap_hit", mkt_col, "surplus_value"]:
                if c in display:
                    display[c] = display[c].apply(lambda v: f"${v:,.0f}")
            st.dataframe(display, use_container_width=True, hide_index=True)
        else:
            st.warning("Surplus data not found. Run `python contracts/surplus_model_v2.py` first.")

    with tab2:
        st.subheader("Cap Hit vs Performance")

        surplus = data["surplus"]
        ws = data["win_shares"]
        if surplus is not None:
            szn = st.selectbox("Season", sorted(surplus["season"].unique()),
                               index=len(surplus["season"].unique())-1,
                               format_func=season_label, key="ce_season")
            sv = surplus[(surplus["season"] == szn) & (surplus["GP"] >= 30)].copy()
            sv["cap_hit_m"] = sv["cap_hit"] / 1e6
            # Expected performance / actual contract as %
            mkt_col = "pred_market_value" if "pred_market_value" in sv.columns else "market_value"
            if mkt_col in sv.columns:
                sv["value_ratio"] = (sv[mkt_col] / sv["cap_hit"].clip(lower=1)) * 100
            else:
                sv["value_ratio"] = np.nan

            # Join win shares
            if ws is not None:
                ws_szn = ws[ws["season"] == szn][["player_id", "WS"]].copy()
                sv = sv.merge(ws_szn, on="player_id", how="left")
            else:
                sv["WS"] = np.nan

            metrics = [
                ("WAR", "WAR", "Cap Hit vs WAR"),
                ("value_ratio", "Expected Value / Contract (%)", "Cap Hit vs Value Ratio"),
                ("points", "Points", "Cap Hit vs Points"),
                ("WS", "Win Shares", "Cap Hit vs Win Shares"),
            ]

            for y_col, y_label, title in metrics:
                if y_col not in sv.columns or sv[y_col].isna().all():
                    continue
                fig = px.scatter(sv, x="cap_hit_m", y=y_col, color="position",
                                 hover_name="player_name",
                                 labels={"cap_hit_m": "Cap Hit ($M)", y_col: y_label},
                                 title=f"{title} — {season_label(szn)}")
                fig.update_traces(marker=dict(size=5, opacity=0.7))
                fig.update_layout(height=550)
                if y_col == "value_ratio":
                    fig.add_hline(y=100, line_dash="dash", line_color="#999", line_width=1.5,
                                  annotation_text="Fair Value (1:1)", annotation_position="top left")
                style_fig(fig)
                st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# FREE AGENT PROJECTIONS
# ═══════════════════════════════════════════════════════════════════════════════

with _tabs[7]:
    st.header("2026 Free Agent Contract Projections")

    st.caption("XGBoost model predictions based on current-season stats, age, and market position")

    fa = data["fa_proj"]
    if fa is not None:
        tab1, tab2 = st.tabs(["UFA Targets", "RFA Targets"])

        def _fa_table(sub, n=40):
            """Render FA projection table with confidence intervals."""
            display = sub.head(n).copy()
            show_cols = ["Player", "POS", "Age", "GP", "WAR", "Points", "Current_Cap_Hit",
                         "AAV_Lo", "Pred_AAV", "AAV_Hi", "Term_Lo", "Pred_Term", "Term_Hi"]
            show_cols = [c for c in show_cols if c in display.columns]
            out = display[show_cols].copy()
            for c in ["AAV_Lo", "Pred_AAV", "AAV_Hi"]:
                if c in out.columns:
                    out[c] = out[c].apply(lambda v: f"${v:,.0f}")
            for c in ["Term_Lo", "Pred_Term", "Term_Hi"]:
                if c in out.columns:
                    out[c] = out[c].apply(lambda v: f"{v}yr")
            st.dataframe(out, use_container_width=True, hide_index=True)

        with tab1:
            ufas = fa[fa["Status"] == "UFA"].sort_values("Pred_AAV", ascending=False)
            st.subheader(f"Top UFAs ({len(ufas)} players)")
            _fa_table(ufas, 40)

        with tab2:
            rfas = fa[fa["Status"] == "RFA"].sort_values("Pred_AAV", ascending=False)
            st.subheader(f"Top RFAs ({len(rfas)} players)")
            _fa_table(rfas, 30)
    else:
        st.warning("FA projections not found. Run `python contracts/fa_projections.py` first.")


# ═══════════════════════════════════════════════════════════════════════════════
# PLAYER PROJECTIONS
# ═══════════════════════════════════════════════════════════════════════════════

with _tabs[8]:
    st.header("Player Projections — 2026-27")
    st.caption("Ridge regression projections based on carry-forward ratings, age curves, and current production")

    proj = data["player_proj"]
    if proj is not None:
        c1, c2, c3 = st.columns([1.2, 1.2, 1])
        pos = c1.selectbox("Position", ["ALL", "F", "D"], key="proj_pos")
        min_gp = c2.number_input("Min GP (current)", 1, 82, 30, key="proj_mingp")
        sort_by = c3.selectbox("Sort By", ["proj_WAR", "proj_Points", "proj_Goals",
                               "proj_WAR_82", "war_delta"], key="proj_sort")

        pf = proj.copy()
        if pos == "F":
            pf = pf[pf["position"].isin(["C", "L", "R"])]
        elif pos == "D":
            pf = pf[pf["position"] == "D"]
        pf = pf[pf["GP"] >= min_gp]

        if sort_by in pf.columns:
            pf = pf.sort_values(sort_by, ascending=False)

        # Table with CI columns
        show_cols = ["player_name", "position", "age_next", "GP",
                     "curr_WAR", "proj_WAR_lo", "proj_WAR", "proj_WAR_hi",
                     "proj_WAR_82",
                     "curr_points", "proj_Points",
                     "curr_goals", "proj_Goals",
                     "proj_GP",
                     "proj_EV_O_GAR", "proj_EV_D_GAR", "proj_PP_GAR"]
        show_cols = [c for c in show_cols if c in pf.columns]
        st.dataframe(pf[show_cols].head(50).round(2), use_container_width=True, hide_index=True)

        # WAR projection with confidence range (horizontal bar)
        st.subheader("Projected WAR with 80% Confidence Range")
        pf_plot = pf[pf["GP"] >= 40].head(30).copy()

        if "proj_WAR_lo" in pf_plot.columns and "proj_WAR_hi" in pf_plot.columns:
            fig = go.Figure()
            # CI range as thin horizontal bars
            fig.add_trace(go.Bar(
                y=pf_plot["player_name"], x=pf_plot["proj_WAR_hi"] - pf_plot["proj_WAR_lo"],
                base=pf_plot["proj_WAR_lo"], orientation="h",
                marker_color="rgba(74, 144, 226, 0.25)", name="80% CI",
                hovertemplate="%{y}<br>Range: %{base:.2f} – %{x:.2f}<extra></extra>",
            ))
            # Point estimate as markers
            fig.add_trace(go.Scatter(
                y=pf_plot["player_name"], x=pf_plot["proj_WAR"], mode="markers",
                marker=dict(color="#4a90d9", size=8), name="Projected WAR",
                hovertemplate="%{y}<br>Proj WAR: %{x:.2f}<extra></extra>",
            ))
            # Current WAR as reference markers
            fig.add_trace(go.Scatter(
                y=pf_plot["player_name"], x=pf_plot["curr_WAR"], mode="markers",
                marker=dict(color="#999", size=6, symbol="diamond"), name="Current WAR",
                hovertemplate="%{y}<br>Curr WAR: %{x:.2f}<extra></extra>",
            ))
            fig.update_layout(
                yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
                xaxis_title="WAR",
                title=f"Projected WAR — {int(LATEST_SEASON)+1}-{str(int(LATEST_SEASON)+2)[-2:]}",
                height=max(450, len(pf_plot) * 22),
            )
            style_fig(fig)
            st.plotly_chart(fig, use_container_width=True)

        # Current vs Projected Points scatter
        if "curr_points" in pf.columns and "proj_Points" in pf.columns:
            st.subheader("Current Points vs Projected Points")
            pf_pts = pf[pf["GP"] >= 40].copy()
            fig2 = px.scatter(pf_pts, x="curr_points", y="proj_Points", color="position",
                              hover_name="player_name",
                              labels={"curr_points": "Current Points", "proj_Points": "Projected Points"},
                              title="Current vs Projected Points")
            fig2.update_traces(marker=dict(size=6, opacity=0.7))
            pts_range = [0, max(pf_pts["curr_points"].max(), pf_pts["proj_Points"].max())]
            fig2.add_shape(type="line", x0=0, y0=0, x1=pts_range[1], y1=pts_range[1],
                           line=dict(color="#999", dash="dash", width=1))
            fig2.update_layout(height=550)
            style_fig(fig2)
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Player projections not found. Run `python contracts/player_projections.py` first.")


# ═══════════════════════════════════════════════════════════════════════════════
# DRAFT PICK VALUE
# ═══════════════════════════════════════════════════════════════════════════════

with _tabs[9]:
    st.header("Draft Pick Value Chart")
    st.caption("Expected WAR and surplus value by draft position, based on historical outcomes (2005-2024 drafts)")

    dpv = data["draft_pick_value"]
    dpv_detail = data["draft_pick_detail"]
    draft_raw = data["draft_picks_raw"]

    if dpv is not None:
        # ── Pick-level value chart ──
        st.subheader("Expected Value by Pick Range")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=dpv["pick_range"], y=dpv["total_surplus_7yr"] / 1e6,
            marker_color=[COLORS["EV_O"] if v > 0 else COLORS["EV_D"] for v in dpv["total_surplus_7yr"]],
            opacity=0.85,
            hovertemplate="%{x}<br>Surplus: $%{y:.1f}M<br>"
                          "NHL Rate: %{customdata[0]:.0%}<br>"
                          "Mean WAR: %{customdata[1]:.2f}<extra></extra>",
            customdata=dpv[["nhl_rate", "mean_war_7yr"]].values,
        ))
        fig.update_layout(
            xaxis_title="Draft Position", yaxis_title="Expected 7-Year Surplus ($M)",
            title="Draft Pick Value — Expected Surplus Over First 7 Pro Years",
            height=450, showlegend=False,
        )
        fig.add_hline(y=0, line_color="#999", line_width=1)
        style_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

        # Table
        display = dpv[["pick_range", "players", "nhl_rate", "mean_war_7yr",
                        "median_war_7yr", "total_surplus_7yr", "surplus_per_year"]].copy()
        display["nhl_rate"] = display["nhl_rate"].apply(lambda v: f"{v:.0%}")
        display["total_surplus_7yr"] = display["total_surplus_7yr"].apply(lambda v: f"${v:,.0f}")
        display["surplus_per_year"] = display["surplus_per_year"].apply(lambda v: f"${v:,.0f}")
        st.dataframe(display, use_container_width=True, hide_index=True)

        # ── WAR development curve by round ──
        if dpv_detail is not None:
            st.subheader("WAR Development by Pro Year")
            st.caption("How quickly different draft rounds develop into NHL contributors")

            fig2 = go.Figure()
            round_colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#d62728",
                            5: "#9467bd", 6: "#8c564b", 7: "#e377c2"}
            for rnd in range(1, 8):
                rnd_data = dpv_detail[dpv_detail["Draft Round"] == rnd].sort_values("pro_year")
                if len(rnd_data):
                    fig2.add_trace(go.Scatter(
                        x=rnd_data["pro_year"], y=rnd_data["mean_WAR"],
                        name=f"Round {rnd}", mode="lines+markers",
                        line=dict(color=round_colors[rnd], width=2.5),
                        marker=dict(size=6),
                    ))
            fig2.update_layout(
                xaxis_title="Pro Year", yaxis_title="Mean WAR",
                title="WAR by Pro Year and Draft Round", height=450,
                xaxis=dict(dtick=1),
            )
            style_fig(fig2)
            st.plotly_chart(fig2, use_container_width=True)

        # ── NHL appearance rate ──
        st.subheader("NHL Appearance Rate")
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=dpv["pick_range"], y=dpv["nhl_rate"] * 100,
            marker_color=COLORS["EV_O"], opacity=0.85,
        ))
        fig3.update_layout(
            xaxis_title="Draft Position", yaxis_title="% Reaching NHL (20+ GP season)",
            title="NHL Appearance Rate by Draft Position", height=400,
            showlegend=False,
        )
        style_fig(fig3)
        st.plotly_chart(fig3, use_container_width=True)

        # ── Notable picks by value ──
        if draft_raw is not None:
            st.subheader("Notable Picks — Best & Worst Value")
            dw_data = data["unified"]

            # Join draft picks to career WAR
            dr = draft_raw[draft_raw["playerId"].notna()].copy()
            dr["player_id"] = dr["playerId"].astype(int)
            career = dw_data.groupby("player_id").agg(
                total_WAR=("WAR", "sum"), seasons=("season", "count"),
                total_GP=("GP", "sum"),
            ).reset_index()
            dr = dr.merge(career, on="player_id", how="left")
            dr = dr[dr["total_GP"].notna() & (dr["total_GP"] > 0)]

            # WAR per pick (value relative to slot)
            pick_avg = dpv.set_index("pick_lo")["mean_war_7yr"]

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Best Value Picks** (highest WAR for their slot)")
                best = dr.nlargest(15, "total_WAR")
                show = best[["playerName", "draftYear", "overallPickNumber", "triCode",
                             "total_WAR", "seasons"]].copy()
                show.columns = ["Player", "Draft Year", "Pick", "Team", "Career WAR", "Seasons"]
                show["Career WAR"] = show["Career WAR"].round(2)
                st.dataframe(show, use_container_width=True, hide_index=True)

            with c2:
                st.markdown("**Top 10 Picks with Lowest WAR** (busts)")
                top10 = dr[dr["overallPickNumber"] <= 10].copy()
                worst = top10.nsmallest(15, "total_WAR")
                show2 = worst[["playerName", "draftYear", "overallPickNumber", "triCode",
                               "total_WAR", "seasons"]].copy()
                show2.columns = ["Player", "Draft Year", "Pick", "Team", "Career WAR", "Seasons"]
                show2["Career WAR"] = show2["Career WAR"].round(2)
                st.dataframe(show2, use_container_width=True, hide_index=True)
    else:
        st.warning("Draft pick value data not found. Run `python contracts/draft_pick_value.py` first.")


# ═══════════════════════════════════════════════════════════════════════════════
# TRADE EVALUATOR
# ═══════════════════════════════════════════════════════════════════════════════

with _tabs[10]:
    st.header("Trade Evaluator")
    st.caption("Value players and picks in dollars, then compare trade sides")

    # Load trade valuation components
    sys.path.insert(0, BASE)
    from contracts.trade_evaluator import value_player, value_player_production, value_draft_pick

    CURRENT_SEASON = int(LATEST_SEASON)

    st.subheader("Evaluate a Trade")

    val_mode = st.radio("Valuation Method", ["Surplus (cap-adjusted)", "Production (raw output)"],
                        horizontal=True, key="val_mode",
                        help="Surplus subtracts cap cost. Production values raw on-ice output like picks.")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Team A sends:**")
        team_a_input = st.text_area("Assets (one per line — player names or picks like '2025 1st')",
                                     value="Jason Robertson\n2026 2nd", height=120, key="trade_a")

    with c2:
        st.markdown("**Team B sends:**")
        team_b_input = st.text_area("Assets (one per line)",
                                     value="Connor Bedard", height=120, key="trade_b")

    if st.button("Evaluate Trade", key="eval_trade"):
        a_assets = [a.strip() for a in team_a_input.strip().split("\n") if a.strip()]
        b_assets = [a.strip() for a in team_b_input.strip().split("\n") if a.strip()]

        use_production = "Production" in val_mode

        def classify_and_value(asset):
            parts = asset.split()
            if len(parts) == 2 and parts[1] in ("1st", "2nd", "3rd", "4th", "5th", "6th", "7th"):
                val, desc = value_draft_pick(asset, CURRENT_SEASON)
                return val, desc, "pick"
            else:
                if use_production:
                    val, desc = value_player_production(asset, CURRENT_SEASON)
                else:
                    val, desc = value_player(asset, CURRENT_SEASON)
                return val, desc, "player"

        a_total, b_total = 0, 0
        a_rows, b_rows = [], []

        for asset in a_assets:
            val, desc, atype = classify_and_value(asset)
            a_total += val
            a_rows.append({"Asset": asset, "Type": atype, "Value ($)": f"${val:,.0f}", "Details": desc})

        for asset in b_assets:
            val, desc, atype = classify_and_value(asset)
            b_total += val
            b_rows.append({"Asset": asset, "Type": atype, "Value ($)": f"${val:,.0f}", "Details": desc})

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Team A sends:**")
            st.dataframe(pd.DataFrame(a_rows), use_container_width=True, hide_index=True)
            st.metric("Total Value", f"${a_total:,.0f}")

        with c2:
            st.markdown("**Team B sends:**")
            st.dataframe(pd.DataFrame(b_rows), use_container_width=True, hide_index=True)
            st.metric("Total Value", f"${b_total:,.0f}")

        net = b_total - a_total
        if net > 0:
            st.success(f"Team A wins by ${abs(net):,.0f}")
        elif net < 0:
            st.error(f"Team B wins by ${abs(net):,.0f}")
        else:
            st.info("Even trade")

    # Historical trades
    st.divider()
    st.subheader("Historical Trades")

    trades_path = os.path.join(BASE, "data", "trades.csv")
    if os.path.exists(trades_path):
        trades_df = pd.read_csv(trades_path)
        yr = st.selectbox("Draft Year", sorted(trades_df["draft_year"].unique(), reverse=True), key="trade_yr")
        yr_trades = trades_df[trades_df["draft_year"] == yr]
        st.dataframe(yr_trades[["trade_id", "team_1", "team_2", "team_1_sends", "team_2_sends"]].head(30),
                     use_container_width=True, hide_index=True)
    else:
        st.info("No historical trades. Add data/trades.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# RESEARCH
# ═══════════════════════════════════════════════════════════════════════════════

with _tabs[11]:
    st.header("Research")

    CONTENT_DIR = os.path.join(BASE, "content")

    # Load all markdown files from content/
    posts = []
    if os.path.isdir(CONTENT_DIR):
        for fname in sorted(os.listdir(CONTENT_DIR)):
            if fname.endswith(".md"):
                fpath = os.path.join(CONTENT_DIR, fname)
                with open(fpath) as f:
                    raw = f.read()
                # Parse frontmatter
                title, date = fname.replace(".md", ""), ""
                body = raw
                if raw.startswith("---"):
                    parts = raw.split("---", 2)
                    if len(parts) >= 3:
                        for line in parts[1].strip().split("\n"):
                            if line.startswith("title:"):
                                title = line.split(":", 1)[1].strip()
                            elif line.startswith("date:"):
                                date = line.split(":", 1)[1].strip()
                        body = parts[2].strip()
                posts.append({"title": title, "date": date, "body": body, "file": fname})

    if posts:
        # Article selector
        titles = [p["title"] for p in posts]
        selected = st.selectbox("Select article", titles, key="research_article")
        post = posts[titles.index(selected)]

        st.caption(post["date"])
        st.markdown(post["body"])
    else:
        st.info("No articles yet. Add markdown files to the `content/` folder.")
