"""
NHL WAR Dashboard — Full metric explorer.

Includes all metric layers:
  - Rolling RAPM (BPR_O, BPR_D, BPR) — 3-season windows
  - BPM components (GV_O/D, OOI_O/D) — box-score models
  - Composite (PV_O/D, IV_O/D, composite_O/D) — predictive blend
  - Daily ratings (EV_O/D, PP, PK, PEN) — Bayesian-smoothed per-game
  - Season WAR (GAR components, WAR_O/D, WAR/82)

Run: python dashboard/app.py
Then open http://127.0.0.1:8050
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, Input, Output, dash_table

# ── Load all data ─────────────────────────────────────────────────────────────
daily = pd.read_csv("output/v5_daily_ratings.csv")
daily["game_date"] = pd.to_datetime(daily["game_date"])
daily["total_gar"] = daily["EV_O_gar"] + daily["EV_D_gar"] + daily["PP_gar"] + daily["PK_gar"] + daily["PEN_gar"]

daily_war = pd.read_csv("output/v5_daily_war.csv")

# BPM components (GV, OOI, RAPM per season — through 2024)
bpm = pd.read_csv("output/v4_bpm_player_seasons.csv")

# Composite (PV, IV, composite — through 2024)
composite = pd.read_csv("output/v5_composite_player_seasons.csv")

# Composite WAR (through 2024)
composite_war = pd.read_csv("output/v5_season_war.csv")

# Rolling RAPM (latest window per player)
rapm_latest = pd.read_csv("output/v3_rolling_rapm_latest.csv")

# Team ratings
team_season = pd.read_csv("output/v6_team_season_ratings.csv")
team_games = pd.read_csv("output/v6_team_game_ratings.csv")
team_games["game_date"] = pd.to_datetime(team_games["game_date"])
TEAM_SEASONS = sorted(team_season["season"].unique())
TEAM_LATEST = max(TEAM_SEASONS)
ALL_TEAMS = sorted(team_season["team"].unique())

# All seasons available
ALL_SEASONS = sorted(set(daily_war["season"].unique()) | set(composite_war["season"].unique()))
LATEST_SEASON = max(ALL_SEASONS)

# Build unified season WAR: use daily_war for all seasons (it has 5-component breakdown)
# Merge in composite/BPM columns for seasons that have them
unified = daily_war.copy()
# Merge composite columns
comp_cols_to_add = ["composite_O", "composite_D", "PV_O", "PV_D", "IV_O", "IV_D"]
comp_merge = composite[["player_id", "season"] + [c for c in comp_cols_to_add if c in composite.columns]]
unified = unified.merge(comp_merge, on=["player_id", "season"], how="left")

# Merge BPM columns
bpm_cols_to_add = ["GV_O", "GV_D", "OOI_O", "OOI_D", "RAPM_O", "RAPM_D", "RAPM_O_se", "RAPM_D_se",
                   "PP_O", "PK_D", "ozPct", "dzPct"]
bpm_merge = bpm[["player_id", "season"] + [c for c in bpm_cols_to_add if c in bpm.columns]]
unified = unified.merge(bpm_merge, on=["player_id", "season"], how="left")

# Merge RAPM latest (career-level BPR)
rapm_cols = ["player_id", "BPR_O", "BPR_D", "BPR", "BPR_se"]
rapm_merge = rapm_latest[[c for c in rapm_cols if c in rapm_latest.columns]]
unified = unified.merge(rapm_merge, on="player_id", how="left")

def season_label(s):
    return f"{s}-{str(s+1)[-2:]}"

# Per-season sorted copies
season_dfs = {}
for szn in ALL_SEASONS:
    sdf = unified[unified["season"] == szn].copy()
    sdf = sdf.sort_values("WAR", ascending=False).reset_index(drop=True)
    sdf["rank"] = sdf.index + 1
    season_dfs[szn] = sdf

# Player list from latest season
latest_df = season_dfs[LATEST_SEASON]
player_options = [
    {"label": f"{r['player_name']} ({r['position']}, #{r['rank']})",
     "value": r["player_name"]}
    for _, r in latest_df.head(400).iterrows()
]

# All unique player names across all seasons for search
all_players = sorted(unified["player_name"].unique())
all_player_options = [{"label": p, "value": p} for p in all_players]

COLORS = {
    "EV_O": "#1f77b4", "EV_D": "#d62728", "PP": "#ff7f0e",
    "PK": "#9467bd", "PEN": "#2ca02c",
    "GV": "#17becf", "OOI": "#bcbd22", "RAPM": "#e377c2",
    "composite": "#1a1a2e", "PV": "#ff9896", "IV": "#aec7e8",
}
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


# ── Helpers (must be defined before layout) ───────────────────────────────────

def _dropdown(label, id_, options, value, width):
    return html.Div([
        html.Label(label, style={"fontWeight": "bold", "fontSize": "13px"}),
        dcc.Dropdown(id=id_, options=options, value=value, clearable=False,
                     style={"width": width}),
    ])


def _metric_options():
    return [
        {"label": "WAR", "value": "WAR"},
        {"label": "WAR/82", "value": "WAR_82"},
        {"label": "WAR_O", "value": "WAR_O"},
        {"label": "WAR_D", "value": "WAR_D"},
        {"label": "EV_O GAR", "value": "EV_O_GAR"},
        {"label": "EV_D GAR", "value": "EV_D_GAR"},
        {"label": "PP GAR", "value": "PP_GAR"},
        {"label": "PK GAR", "value": "PK_GAR"},
        {"label": "PEN GAR", "value": "PEN_GAR"},
        {"label": "Composite O", "value": "composite_O"},
        {"label": "Composite D", "value": "composite_D"},
        {"label": "PV_O", "value": "PV_O"},
        {"label": "PV_D", "value": "PV_D"},
        {"label": "IV_O", "value": "IV_O"},
        {"label": "IV_D", "value": "IV_D"},
        {"label": "GV_O", "value": "GV_O"},
        {"label": "GV_D", "value": "GV_D"},
        {"label": "OOI_O", "value": "OOI_O"},
        {"label": "OOI_D", "value": "OOI_D"},
        {"label": "RAPM_O", "value": "RAPM_O"},
        {"label": "RAPM_D", "value": "RAPM_D"},
        {"label": "BPR_O (career)", "value": "BPR_O"},
        {"label": "BPR_D (career)", "value": "BPR_D"},
        {"label": "BPR (career)", "value": "BPR"},
        {"label": "PP_O (RAPM)", "value": "PP_O"},
        {"label": "PK_D (RAPM)", "value": "PK_D"},
        {"label": "EV_O Rate", "value": "EV_O_rate"},
        {"label": "EV_D Rate", "value": "EV_D_rate"},
        {"label": "PP Rate", "value": "PP_rate"},
        {"label": "PK Rate", "value": "PK_rate"},
        {"label": "PEN Rate", "value": "PEN_rate"},
        {"label": "GP", "value": "GP"},
        {"label": "OZ%", "value": "ozPct"},
    ]


def _stat_box(label, value, color, small=False):
    if pd.isna(value):
        disp = "—"
    elif abs(value) < 100:
        disp = f"{value:+.2f}"
    else:
        disp = f"{value:.0f}"
    sz = "18px" if small else "22px"
    return html.Div(style={"textAlign": "center", "minWidth": "60px"}, children=[
        html.Div(disp, style={"fontSize": sz, "fontWeight": "bold", "color": color}),
        html.Div(label, style={"fontSize": "10px", "color": "#888", "marginTop": "2px"}),
    ])


def _hex_to_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _pos_filter(df, pos):
    if pos == "F":
        return df[df["position"].isin(["C", "L", "R"])]
    elif pos == "C":
        return df[df["position"] == "C"]
    elif pos == "W":
        return df[df["position"].isin(["L", "R"])]
    elif pos == "D":
        return df[df["position"] == "D"]
    return df


def _empty_fig(msg="No data"):
    fig = go.Figure()
    fig.update_layout(annotations=[dict(text=msg, showarrow=False, font=dict(size=18))],
                      plot_bgcolor="white")
    return fig


def _style_fig(fig):
    fig.update_layout(plot_bgcolor="white", hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
    fig.update_yaxes(gridcolor="#eee", zeroline=True, zerolinecolor="gray")
    fig.update_xaxes(gridcolor="#eee")
    return fig


# ── App ───────────────────────────────────────────────────────────────────────
app = Dash(__name__)
app.title = "NHL WAR Dashboard"

app.layout = html.Div(style={"fontFamily": "system-ui, -apple-system, sans-serif",
                              "maxWidth": "1400px", "margin": "0 auto", "padding": "20px",
                              "backgroundColor": "#fafafa"}, children=[

    html.H1("NHL Player Ratings Dashboard",
            style={"textAlign": "center", "marginBottom": "5px", "color": "#1a1a2e"}),
    html.P("RAPM + BPM + Composite + Daily Bayesian Ratings",
           style={"textAlign": "center", "color": "#666", "marginBottom": "25px"}),

    dcc.Tabs(id="tabs", value="leaderboard", children=[

        # ══════════════════════════════════════════════════════════════════════
        # TAB 1: LEADERBOARD
        # ══════════════════════════════════════════════════════════════════════
        dcc.Tab(label="Leaderboard", value="leaderboard", children=[
            html.Div(style={"padding": "20px"}, children=[

                html.Div(style={"display": "flex", "gap": "15px", "marginBottom": "15px",
                                "flexWrap": "wrap", "alignItems": "center"}, children=[
                    _dropdown("Season", "lb-season",
                              [{"label": season_label(s), "value": s} for s in ALL_SEASONS],
                              LATEST_SEASON, "130px"),
                    _dropdown("Position", "lb-pos", [
                        {"label": "All", "value": "ALL"}, {"label": "Forwards", "value": "F"},
                        {"label": "Centers", "value": "C"}, {"label": "Wingers", "value": "W"},
                        {"label": "Defensemen", "value": "D"},
                    ], "ALL", "150px"),
                    html.Div([
                        html.Label("Min GP", style={"fontWeight": "bold", "fontSize": "13px"}),
                        dcc.Input(id="lb-mingp", type="number", value=20, min=1, max=82,
                                  style={"width": "65px", "padding": "6px"}),
                    ]),
                    _dropdown("Sort By", "lb-sort", [
                        {"label": "WAR", "value": "WAR"},
                        {"label": "WAR/82", "value": "WAR_82"},
                        {"label": "WAR_O", "value": "WAR_O"},
                        {"label": "WAR_D", "value": "WAR_D"},
                        {"label": "EV_O GAR", "value": "EV_O_GAR"},
                        {"label": "EV_D GAR", "value": "EV_D_GAR"},
                        {"label": "PP GAR", "value": "PP_GAR"},
                        {"label": "PK GAR", "value": "PK_GAR"},
                        {"label": "PEN GAR", "value": "PEN_GAR"},
                        {"label": "Composite O", "value": "composite_O"},
                        {"label": "Composite D", "value": "composite_D"},
                        {"label": "GV_O", "value": "GV_O"},
                        {"label": "OOI_O", "value": "OOI_O"},
                        {"label": "RAPM_O", "value": "RAPM_O"},
                        {"label": "BPR", "value": "BPR"},
                    ], "WAR", "170px"),
                    _dropdown("View", "lb-view", [
                        {"label": "GAR Components", "value": "gar"},
                        {"label": "Underlying Metrics", "value": "metrics"},
                        {"label": "Full Table", "value": "table"},
                    ], "gar", "180px"),
                    html.Div([
                        html.Label("Top N", style={"fontWeight": "bold", "fontSize": "13px"}),
                        dcc.Input(id="lb-topn", type="number", value=30, min=5, max=200,
                                  style={"width": "65px", "padding": "6px"}),
                    ]),
                ]),

                dcc.Graph(id="lb-chart", style={"height": "700px"}),
                html.Div(id="lb-table-container", style={"marginTop": "10px"}),
            ]),
        ]),

        # ══════════════════════════════════════════════════════════════════════
        # TAB 2: PLAYER PROFILE
        # ══════════════════════════════════════════════════════════════════════
        dcc.Tab(label="Player Profile", value="player", children=[
            html.Div(style={"padding": "20px"}, children=[

                html.Div(style={"display": "flex", "gap": "20px", "marginBottom": "20px",
                                "alignItems": "center"}, children=[
                    html.Div([
                        html.Label("Player", style={"fontWeight": "bold", "fontSize": "13px"}),
                        dcc.Dropdown(id="pp-player", options=player_options,
                                     value="Connor McDavid", clearable=False,
                                     style={"width": "350px"}),
                    ]),
                    html.Div([
                        html.Label("Season", style={"fontWeight": "bold", "fontSize": "13px"}),
                        dcc.Dropdown(id="pp-season",
                                     options=[{"label": season_label(s), "value": s} for s in ALL_SEASONS],
                                     value=LATEST_SEASON, clearable=False, style={"width": "130px"}),
                    ]),
                ]),

                # Summary cards row
                html.Div(id="pp-card", style={"marginBottom": "20px"}),

                # Metric history across seasons
                html.Div(id="pp-history", style={"marginBottom": "10px"}),
                dcc.Graph(id="pp-history-chart", style={"height": "350px"}),

                # Daily charts (only for seasons with daily data)
                dcc.Graph(id="pp-cum-gar", style={"height": "380px"}),
                dcc.Graph(id="pp-rates", style={"height": "320px"}),
                dcc.Graph(id="pp-bars", style={"height": "280px"}),
            ]),
        ]),

        # ══════════════════════════════════════════════════════════════════════
        # TAB 3: COMPARE
        # ══════════════════════════════════════════════════════════════════════
        dcc.Tab(label="Compare", value="compare", children=[
            html.Div(style={"padding": "20px"}, children=[

                html.Div(style={"display": "flex", "gap": "15px", "marginBottom": "20px",
                                "flexWrap": "wrap", "alignItems": "flex-end"}, children=[
                    html.Div([
                        html.Label("Players (up to 6)", style={"fontWeight": "bold", "fontSize": "13px"}),
                        dcc.Dropdown(id="cmp-players", options=player_options,
                                     value=["Connor McDavid", "Leon Draisaitl", "Nikita Kucherov",
                                            "Nathan MacKinnon", "Kirill Kaprizov"],
                                     multi=True, style={"width": "650px"}),
                    ]),
                    _dropdown("Season", "cmp-season",
                              [{"label": season_label(s), "value": s} for s in ALL_SEASONS],
                              LATEST_SEASON, "130px"),
                    _dropdown("Chart", "cmp-metric", [
                        {"label": "Cumul. Total GAR", "value": "total_gar"},
                        {"label": "Cumul. PP GAR", "value": "PP_gar"},
                        {"label": "Cumul. EV_O GAR", "value": "EV_O_gar"},
                        {"label": "Cumul. PK GAR", "value": "PK_gar"},
                        {"label": "Rate: EV_O", "value": "EV_O"},
                        {"label": "Rate: PP", "value": "PP"},
                        {"label": "Rate: PK", "value": "PK"},
                    ], "total_gar", "200px"),
                ]),

                dcc.Graph(id="cmp-time", style={"height": "450px"}),
                dcc.Graph(id="cmp-radar", style={"height": "420px"}),

                # Metric table comparison
                html.H4("All Metrics Side-by-Side", style={"marginTop": "20px"}),
                html.Div(id="cmp-table", style={"marginTop": "5px"}),
            ]),
        ]),

        # ══════════════════════════════════════════════════════════════════════
        # TAB 4: METRIC EXPLORER
        # ══════════════════════════════════════════════════════════════════════
        dcc.Tab(label="Metric Explorer", value="metrics", children=[
            html.Div(style={"padding": "20px"}, children=[

                html.P("Scatter any two metrics against each other. Explore relationships between "
                       "RAPM, BPM components, composite ratings, and daily WAR.",
                       style={"color": "#666", "marginBottom": "15px"}),

                html.Div(style={"display": "flex", "gap": "15px", "marginBottom": "15px",
                                "flexWrap": "wrap", "alignItems": "center"}, children=[
                    _dropdown("Season", "mx-season",
                              [{"label": season_label(s), "value": s} for s in ALL_SEASONS],
                              2024, "130px"),
                    _dropdown("X Axis", "mx-x", _metric_options(), "composite_O", "200px"),
                    _dropdown("Y Axis", "mx-y", _metric_options(), "WAR", "200px"),
                    _dropdown("Color", "mx-color", [
                        {"label": "Position", "value": "position"},
                        {"label": "WAR", "value": "WAR"},
                        {"label": "GP", "value": "GP"},
                    ], "position", "130px"),
                    _dropdown("Position", "mx-pos", [
                        {"label": "All", "value": "ALL"}, {"label": "Forwards", "value": "F"},
                        {"label": "Defensemen", "value": "D"},
                    ], "ALL", "130px"),
                    html.Div([
                        html.Label("Min GP", style={"fontWeight": "bold", "fontSize": "13px"}),
                        dcc.Input(id="mx-mingp", type="number", value=30, min=1, max=82,
                                  style={"width": "65px", "padding": "6px"}),
                    ]),
                ]),

                dcc.Graph(id="mx-scatter", style={"height": "600px"}),
            ]),
        ]),

        # ══════════════════════════════════════════════════════════════════════
        # TAB 5: TEAM RATINGS
        # ══════════════════════════════════════════════════════════════════════
        dcc.Tab(label="Team Ratings", value="teams", children=[
            html.Div(style={"padding": "20px"}, children=[

                html.P("Team-level ratings aggregated from player metrics. "
                       "Compare roster strength, model predictions, and actual results.",
                       style={"color": "#666", "marginBottom": "15px"}),

                html.Div(style={"display": "flex", "gap": "15px", "marginBottom": "15px",
                                "flexWrap": "wrap", "alignItems": "center"}, children=[
                    _dropdown("Season", "tm-season",
                              [{"label": season_label(s), "value": s} for s in TEAM_SEASONS],
                              TEAM_LATEST, "130px"),
                    _dropdown("Sort By", "tm-sort", [
                        {"label": "Win %", "value": "win_pct"},
                        {"label": "Goal Diff", "value": "actual_GD"},
                        {"label": "Team Strength", "value": "team_strength"},
                        {"label": "Roster EV_O", "value": "roster_EV_O"},
                        {"label": "Roster EV_D", "value": "roster_EV_D"},
                        {"label": "Goalie", "value": "goalie_GA_G"},
                    ], "win_pct", "180px"),
                    _dropdown("View", "tm-view", [
                        {"label": "Strength Breakdown", "value": "breakdown"},
                        {"label": "Predicted vs Actual", "value": "pred_vs_actual"},
                        {"label": "Season Trajectory", "value": "trajectory"},
                    ], "breakdown", "200px"),
                ]),

                dcc.Graph(id="tm-chart", style={"height": "700px"}),
                html.Div(id="tm-table-container", style={"marginTop": "10px"}),

                # Team game-level detail
                html.Div(style={"display": "flex", "gap": "15px", "marginTop": "25px",
                                "marginBottom": "10px", "alignItems": "center"}, children=[
                    _dropdown("Team Detail", "tm-team",
                              [{"label": t, "value": t} for t in ALL_TEAMS],
                              ALL_TEAMS[0], "130px"),
                ]),
                dcc.Graph(id="tm-game-chart", style={"height": "400px"}),
            ]),
        ]),
    ]),
])


# ── Callbacks ─────────────────────────────────────────────────────────────────

# ── LEADERBOARD ──────────────────────────────────────────────────────────────

@app.callback(
    [Output("lb-chart", "figure"),
     Output("lb-table-container", "children")],
    [Input("lb-season", "value"), Input("lb-pos", "value"), Input("lb-mingp", "value"),
     Input("lb-sort", "value"), Input("lb-view", "value"), Input("lb-topn", "value")],
)
def update_leaderboard(season, pos, min_gp, sort_col, view, top_n):
    df = unified[unified["season"] == season].copy()
    df = _pos_filter(df, pos)
    df = df[df["GP"] >= (min_gp or 1)]

    # If sort column doesn't exist for this season, fall back to WAR
    if sort_col not in df.columns or df[sort_col].isna().all():
        sort_col = "WAR"

    df = df.sort_values(sort_col, ascending=False).head(top_n or 30)
    df = df.reset_index(drop=True)
    df["rank"] = df.index + 1
    slabel = season_label(season)

    if view == "gar":
        # Stacked GAR bar chart
        fig = go.Figure()
        comps = [
            ("EV_O_GAR", "EV Offense", COLORS["EV_O"]),
            ("PP_GAR",   "Power Play", COLORS["PP"]),
            ("PEN_GAR",  "Penalties",  COLORS["PEN"]),
            ("PK_GAR",   "Penalty Kill", COLORS["PK"]),
            ("EV_D_GAR", "EV Defense", COLORS["EV_D"]),
        ]
        for col, label, color in comps:
            if col in df.columns:
                fig.add_trace(go.Bar(
                    y=df["player_name"], x=df[col], name=label, orientation="h",
                    marker_color=color, opacity=0.8,
                    hovertemplate="%{y}: %{x:.2f}<extra>" + label + "</extra>",
                ))
        fig.update_layout(
            barmode="relative",
            yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
            xaxis_title="Goals Above Replacement (GAR)",
            title=f"GAR Components — Top {len(df)} by {sort_col} ({slabel})",
            margin=dict(l=160, r=20, t=60, b=40),
        )

    elif view == "metrics":
        # Grouped bar: underlying metric values
        fig = go.Figure()
        metric_comps = [
            ("GV_O",   "GV_O",   COLORS["GV"]),
            ("OOI_O",  "OOI_O",  COLORS["OOI"]),
            ("RAPM_O", "RAPM_O", COLORS["RAPM"]),
            ("composite_O", "Composite O", COLORS["composite"]),
        ]
        for col, label, color in metric_comps:
            if col in df.columns and not df[col].isna().all():
                fig.add_trace(go.Bar(
                    y=df["player_name"], x=df[col], name=label, orientation="h",
                    marker_color=color, opacity=0.8,
                ))
        fig.update_layout(
            barmode="group",
            yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
            xaxis_title="Metric Value (per-60 or rate)",
            title=f"Underlying Metrics — Top {len(df)} by {sort_col} ({slabel})",
            margin=dict(l=160, r=20, t=60, b=40),
        )

    else:  # table view
        fig = _empty_fig("See table below")

    _style_fig(fig)

    # Table
    table_cols = ["rank", "player_name", "position", "GP"]
    # Add all available metric columns
    optional = ["EV_O_GAR", "EV_D_GAR", "PP_GAR", "PK_GAR", "PEN_GAR",
                "WAR_O", "WAR_D", "WAR", "WAR_82",
                "composite_O", "composite_D", "GV_O", "GV_D", "OOI_O", "OOI_D",
                "RAPM_O", "RAPM_D", "BPR_O", "BPR_D", "BPR",
                "EV_O_rate", "EV_D_rate", "PP_rate", "PK_rate", "PEN_rate"]
    for c in optional:
        if c in df.columns and not df[c].isna().all():
            table_cols.append(c)

    table = dash_table.DataTable(
        data=df[table_cols].round(3).to_dict("records"),
        columns=[{"name": c, "id": c} for c in table_cols],
        sort_action="native", filter_action="native",
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "center", "padding": "4px", "fontSize": "11px"},
        style_header={"fontWeight": "bold", "backgroundColor": "#f0f0f0"},
        style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#f9f9f9"}],
        page_size=50,
    )

    return fig, table


# ── PLAYER PROFILE ───────────────────────────────────────────────────────────

@app.callback(
    [Output("pp-card", "children"),
     Output("pp-history-chart", "figure"),
     Output("pp-cum-gar", "figure"),
     Output("pp-rates", "figure"),
     Output("pp-bars", "figure")],
    [Input("pp-player", "value"), Input("pp-season", "value")],
)
def update_player(player_name, season):
    empty = _empty_fig()

    # All-seasons data for this player
    ph = unified[unified["player_name"] == player_name].sort_values("season")

    # Current season daily data
    pg = daily[(daily["player_name"] == player_name) & (daily["season"] == season)].copy()
    pw_row = unified[(unified["player_name"] == player_name) & (unified["season"] == season)]

    # ── Summary card ──
    if len(pw_row):
        r = pw_row.iloc[0]
        card_items = [
            html.Div([
                html.H2(player_name, style={"margin": "0 0 5px 0"}),
                html.Span(f"{r['position']} — {int(r['GP'])} GP — {season_label(season)}",
                          style={"color": "#666", "fontSize": "15px"}),
            ]),
            _stat_box("WAR", r.get("WAR", np.nan), "#1a1a2e"),
            _stat_box("WAR/82", r.get("WAR_82", np.nan), "#1a1a2e"),
        ]
        # Daily GAR components
        for lbl, col, clr in [("EV_O", "EV_O_GAR", COLORS["EV_O"]),
                                ("EV_D", "EV_D_GAR", COLORS["EV_D"]),
                                ("PP", "PP_GAR", COLORS["PP"]),
                                ("PK", "PK_GAR", COLORS["PK"]),
                                ("PEN", "PEN_GAR", COLORS["PEN"])]:
            card_items.append(_stat_box(lbl, r.get(col, np.nan), clr, small=True))

        # Separator + underlying metrics if available
        has_comp = not pd.isna(r.get("composite_O", np.nan))
        if has_comp:
            card_items.append(html.Div(style={"borderLeft": "2px solid #ddd",
                                               "height": "50px", "margin": "0 5px"}))
            for lbl, col, clr in [("Comp O", "composite_O", COLORS["composite"]),
                                    ("Comp D", "composite_D", COLORS["composite"]),
                                    ("GV_O", "GV_O", COLORS["GV"]),
                                    ("OOI_O", "OOI_O", COLORS["OOI"]),
                                    ("RAPM_O", "RAPM_O", COLORS["RAPM"])]:
                card_items.append(_stat_box(lbl, r.get(col, np.nan), clr, small=True))

        has_bpr = not pd.isna(r.get("BPR", np.nan))
        if has_bpr:
            card_items.append(html.Div(style={"borderLeft": "2px solid #ddd",
                                               "height": "50px", "margin": "0 5px"}))
            for lbl, col, clr in [("BPR_O", "BPR_O", COLORS["RAPM"]),
                                    ("BPR_D", "BPR_D", COLORS["RAPM"]),
                                    ("BPR", "BPR", COLORS["RAPM"])]:
                card_items.append(_stat_box(lbl, r.get(col, np.nan), clr, small=True))

        card = html.Div(style={"display": "flex", "gap": "20px", "flexWrap": "wrap",
                                "padding": "15px", "backgroundColor": "white",
                                "borderRadius": "8px", "boxShadow": "0 1px 3px rgba(0,0,0,0.1)",
                                "alignItems": "center"}, children=card_items)
    else:
        card = html.Div("No data for this player/season.", style={"color": "#999"})

    # ── Season history chart ──
    if len(ph) > 1:
        fig_hist = make_subplots(rows=1, cols=2, subplot_titles=("WAR by Season", "Metric Components"),
                                 horizontal_spacing=0.12)

        fig_hist.add_trace(go.Bar(
            x=[season_label(s) for s in ph["season"]], y=ph["WAR_O"],
            name="WAR_O", marker_color=COLORS["EV_O"], opacity=0.8,
        ), row=1, col=1)
        fig_hist.add_trace(go.Bar(
            x=[season_label(s) for s in ph["season"]], y=ph["WAR_D"],
            name="WAR_D", marker_color=COLORS["EV_D"], opacity=0.8,
        ), row=1, col=1)

        # Right panel: metric evolution
        for col, label, color in [("composite_O", "Comp O", COLORS["composite"]),
                                   ("GV_O", "GV_O", COLORS["GV"]),
                                   ("OOI_O", "OOI_O", COLORS["OOI"]),
                                   ("RAPM_O", "RAPM_O", COLORS["RAPM"])]:
            if col in ph.columns:
                valid = ph.dropna(subset=[col])
                if len(valid):
                    fig_hist.add_trace(go.Scatter(
                        x=[season_label(s) for s in valid["season"]], y=valid[col],
                        name=label, line=dict(color=color, width=2), mode="lines+markers",
                    ), row=1, col=2)

        fig_hist.update_layout(barmode="relative", height=350, plot_bgcolor="white",
                                legend=dict(orientation="h", yanchor="bottom", y=1.08,
                                            xanchor="center", x=0.5))
        fig_hist.update_yaxes(gridcolor="#eee", zeroline=True, zerolinecolor="gray")
    else:
        fig_hist = _empty_fig("Need multiple seasons for history")

    # ── Daily charts ──
    if len(pg) == 0:
        return card, fig_hist, empty, empty, empty

    for comp in ["EV_O_gar", "EV_D_gar", "PP_gar", "PK_gar", "PEN_gar", "total_gar"]:
        pg[f"cum_{comp}"] = pg[comp].cumsum()

    dates = pg["game_date"]

    # Cumulative GAR
    fig_cum = go.Figure()
    for col, label, color in [("cum_EV_O_gar", "EV Offense", COLORS["EV_O"]),
                               ("cum_PP_gar", "Power Play", COLORS["PP"]),
                               ("cum_PEN_gar", "Penalties", COLORS["PEN"]),
                               ("cum_PK_gar", "Penalty Kill", COLORS["PK"]),
                               ("cum_EV_D_gar", "EV Defense", COLORS["EV_D"])]:
        vals = pg[col].values
        fig_cum.add_trace(go.Scatter(
            x=dates, y=vals, name=f"{label} ({vals[-1]:+.1f})",
            stackgroup="pos" if vals[-1] >= 0 else "neg",
            line=dict(width=0.5, color=color), fillcolor=_hex_to_rgba(color, 0.4),
        ))
    fig_cum.add_trace(go.Scatter(
        x=dates, y=pg["cum_total_gar"].values,
        name=f"Total ({pg['cum_total_gar'].iloc[-1]:.1f})",
        line=dict(color="black", width=2.5),
    ))
    fig_cum.update_layout(title="Cumulative GAR", yaxis_title="Cumulative GAR",
                           xaxis=dict(tickformat="%b %d"))
    _style_fig(fig_cum)

    # Rates
    fig_rates = go.Figure()
    for col, label, color in [("EV_O", "EV Offense", COLORS["EV_O"]),
                               ("PP", "Power Play", COLORS["PP"]),
                               ("PEN", "Penalties", COLORS["PEN"]),
                               ("PK", "Penalty Kill", COLORS["PK"]),
                               ("EV_D", "EV Defense", COLORS["EV_D"])]:
        fig_rates.add_trace(go.Scatter(x=dates, y=pg[col], name=label,
                                        line=dict(color=color, width=2)))
    fig_rates.update_layout(title="Bayesian-Smoothed Rates (per-60)",
                             yaxis_title="Rate (per-60)", xaxis=dict(tickformat="%b %d"))
    _style_fig(fig_rates)

    # Per-game bars
    fig_bars = go.Figure()
    for col, label, color in [("EV_O_gar", "EV_O", COLORS["EV_O"]),
                               ("PP_gar", "PP", COLORS["PP"]),
                               ("PEN_gar", "PEN", COLORS["PEN"]),
                               ("PK_gar", "PK", COLORS["PK"]),
                               ("EV_D_gar", "EV_D", COLORS["EV_D"])]:
        fig_bars.add_trace(go.Bar(x=dates, y=pg[col], name=label,
                                   marker_color=color, opacity=0.7))
    fig_bars.update_layout(barmode="relative", title="Per-Game GAR",
                            yaxis_title="Game GAR", xaxis=dict(tickformat="%b %d"))
    _style_fig(fig_bars)

    return card, fig_hist, fig_cum, fig_rates, fig_bars


# ── COMPARE ──────────────────────────────────────────────────────────────────

@app.callback(
    [Output("cmp-time", "figure"),
     Output("cmp-radar", "figure"),
     Output("cmp-table", "children")],
    [Input("cmp-players", "value"), Input("cmp-season", "value"),
     Input("cmp-metric", "value")],
)
def update_compare(players, season, metric):
    empty = _empty_fig("Select players")
    if not players:
        return empty, empty, html.Div()

    players = players[:6]
    slabel = season_label(season)
    is_cum = metric in ["total_gar", "PP_gar", "EV_O_gar", "EV_D_gar", "PK_gar", "PEN_gar"]

    # Time series
    fig = go.Figure()
    for i, name in enumerate(players):
        pg = daily[(daily["player_name"] == name) & (daily["season"] == season)].copy()
        if len(pg) == 0:
            continue
        y = pg[metric].cumsum() if is_cum else pg[metric]
        fig.add_trace(go.Scatter(x=pg["game_date"], y=y, name=name,
                                  line=dict(color=PALETTE[i % len(PALETTE)], width=2.5)))

    mlabel = metric.replace("_gar", " GAR").replace("_", " ").upper()
    fig.update_layout(
        title=f"{'Cumulative ' if is_cum else ''}{mlabel} — {slabel}",
        yaxis_title=f"{'Cumulative ' if is_cum else ''}{mlabel}",
        xaxis=dict(tickformat="%b %d"),
    )
    _style_fig(fig)

    # Radar
    gar_cats = ["EV_O_GAR", "EV_D_GAR", "PP_GAR", "PK_GAR", "PEN_GAR"]
    gar_labels = ["EV Off", "EV Def", "PP", "PK", "PEN"]

    fig_radar = go.Figure()
    for i, name in enumerate(players):
        pw = unified[(unified["player_name"] == name) & (unified["season"] == season)]
        if len(pw) == 0:
            continue
        r = pw.iloc[0]
        vals = [r.get(c, 0) for c in gar_cats]
        vals.append(vals[0])
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=gar_labels + [gar_labels[0]],
            name=f"{name} ({r.get('WAR', 0):+.1f})",
            line=dict(color=PALETTE[i % len(PALETTE)], width=2),
            fill="toself", opacity=0.25,
        ))
    fig_radar.update_layout(
        title=f"GAR Component Radar — {slabel}",
        polar=dict(radialaxis=dict(visible=True, gridcolor="#ddd")),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )

    # Side-by-side table
    rows = []
    for name in players:
        pw = unified[(unified["player_name"] == name) & (unified["season"] == season)]
        if len(pw) == 0:
            continue
        r = pw.iloc[0]
        row = {"Player": name, "Pos": r["position"], "GP": int(r["GP"]),
               "WAR": r.get("WAR", np.nan), "WAR/82": r.get("WAR_82", np.nan)}
        for c in ["EV_O_GAR", "EV_D_GAR", "PP_GAR", "PK_GAR", "PEN_GAR",
                   "composite_O", "composite_D", "GV_O", "OOI_O", "RAPM_O",
                   "BPR_O", "BPR_D", "BPR"]:
            val = r.get(c, np.nan)
            row[c] = round(val, 3) if not pd.isna(val) else None
        rows.append(row)

    if rows:
        tdf = pd.DataFrame(rows)
        # Drop columns that are all None
        tdf = tdf.dropna(axis=1, how="all")
        table = dash_table.DataTable(
            data=tdf.round(3).to_dict("records"),
            columns=[{"name": c, "id": c} for c in tdf.columns],
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "center", "padding": "5px", "fontSize": "12px"},
            style_header={"fontWeight": "bold", "backgroundColor": "#f0f0f0"},
        )
    else:
        table = html.Div("No data", style={"color": "#999"})

    return fig, fig_radar, table


# ── METRIC EXPLORER ──────────────────────────────────────────────────────────

@app.callback(
    Output("mx-scatter", "figure"),
    [Input("mx-season", "value"), Input("mx-x", "value"), Input("mx-y", "value"),
     Input("mx-color", "value"), Input("mx-pos", "value"), Input("mx-mingp", "value")],
)
def update_scatter(season, x_col, y_col, color_col, pos, min_gp):
    df = unified[unified["season"] == season].copy()
    df = _pos_filter(df, pos)
    df = df[df["GP"] >= (min_gp or 1)]

    # Drop rows where either axis is NaN
    if x_col not in df.columns or y_col not in df.columns:
        return _empty_fig(f"Metric not available for {season_label(season)}")

    df = df.dropna(subset=[x_col, y_col])
    if len(df) == 0:
        return _empty_fig("No data after filtering")

    slabel = season_label(season)

    if color_col == "position":
        fig = go.Figure()
        for p, color in [("C", "#1f77b4"), ("L", "#ff7f0e"), ("R", "#2ca02c"), ("D", "#d62728")]:
            sub = df[df["position"] == p]
            if len(sub) == 0:
                continue
            fig.add_trace(go.Scatter(
                x=sub[x_col], y=sub[y_col], mode="markers+text", name=p,
                marker=dict(color=color, size=8, opacity=0.7),
                text=sub["player_name"], textposition="top center", textfont=dict(size=8),
                hovertemplate="%{text}<br>" + x_col + ": %{x:.3f}<br>" + y_col + ": %{y:.3f}<extra></extra>",
            ))
    else:
        fig = go.Figure(go.Scatter(
            x=df[x_col], y=df[y_col], mode="markers+text",
            marker=dict(color=df[color_col], colorscale="RdYlBu", size=8,
                        opacity=0.7, colorbar=dict(title=color_col)),
            text=df["player_name"], textposition="top center", textfont=dict(size=8),
            hovertemplate="%{text}<br>" + x_col + ": %{x:.3f}<br>" + y_col + ": %{y:.3f}<extra></extra>",
        ))

    # Add correlation
    corr = df[[x_col, y_col]].corr().iloc[0, 1]
    fig.update_layout(
        title=f"{x_col} vs {y_col} — {slabel} (r={corr:.3f}, n={len(df)})",
        xaxis_title=x_col, yaxis_title=y_col,
        margin=dict(t=60, b=40),
    )
    _style_fig(fig)

    return fig


# ── TEAM RATINGS ─────────────────────────────────────────────────────────────

@app.callback(
    [Output("tm-chart", "figure"),
     Output("tm-table-container", "children")],
    [Input("tm-season", "value"), Input("tm-sort", "value"), Input("tm-view", "value")],
)
def update_team_ratings(season, sort_col, view):
    df = team_season[team_season["season"] == season].copy()
    df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    slabel = season_label(season)

    if view == "breakdown":
        fig = go.Figure()
        comps = [
            ("roster_EV_O", "EV Offense", COLORS["EV_O"]),
            ("roster_EV_D", "EV Defense", COLORS["EV_D"]),
            ("roster_PP", "Power Play", COLORS["PP"]),
            ("roster_PK", "Penalty Kill", COLORS["PK"]),
            ("roster_PEN", "Penalties", COLORS["PEN"]),
        ]
        for col, label, color in comps:
            if col in df.columns:
                fig.add_trace(go.Bar(
                    y=df["team"], x=df[col], name=label, orientation="h",
                    marker_color=color, opacity=0.8,
                    hovertemplate="%{y}: %{x:.4f}<extra>" + label + "</extra>",
                ))
        fig.update_layout(
            barmode="relative",
            yaxis=dict(autorange="reversed", tickfont=dict(size=12)),
            xaxis_title="Roster Strength Component",
            title=f"Team Strength Breakdown — {slabel} (sorted by {sort_col})",
            margin=dict(l=80, r=20, t=60, b=40),
        )

    elif view == "pred_vs_actual":
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["team_strength"], y=df["win_pct"], mode="markers+text",
            marker=dict(size=12, color=df["actual_GD"], colorscale="RdYlBu",
                        colorbar=dict(title="Goal Diff"), line=dict(width=1, color="gray")),
            text=df["team"], textposition="top center", textfont=dict(size=10),
            hovertemplate="%{text}<br>Strength: %{x:.3f}<br>Win%%: %{y:.3f}<br>"
                          "GD: %{marker.color:+d}<extra></extra>",
        ))
        # Add trend line
        from numpy.polynomial.polynomial import polyfit
        if len(df) > 2:
            b, m = polyfit(df["team_strength"], df["win_pct"], 1)
            x_range = np.linspace(df["team_strength"].min(), df["team_strength"].max(), 50)
            fig.add_trace(go.Scatter(x=x_range, y=b + m * x_range, mode="lines",
                                     line=dict(color="gray", dash="dash", width=1),
                                     showlegend=False))
        corr = df[["team_strength", "win_pct"]].corr().iloc[0, 1]
        fig.update_layout(
            xaxis_title="Roster Aggregate Strength",
            yaxis_title="Actual Win %",
            title=f"Team Strength vs Actual Win% — {slabel} (r={corr:.3f})",
            margin=dict(t=60, b=40),
        )

    else:  # trajectory
        # Rolling win% from game-level data
        gdf = team_games[team_games["season"] == season].copy()
        fig = go.Figure()
        teams_sorted = df["team"].tolist()
        top_teams = teams_sorted[:5]
        bottom_teams = teams_sorted[-3:]
        highlight = set(top_teams + bottom_teams)

        for team in teams_sorted:
            tg = gdf[(gdf["home_team"] == team) | (gdf["away_team"] == team)].sort_values("game_date")
            wins = []
            for _, row in tg.iterrows():
                if row["home_team"] == team:
                    wins.append(1 if row["home_win"] == 1 else 0)
                else:
                    wins.append(1 if row["home_win"] == 0 else 0)
            if not wins:
                continue
            cum_wins = np.cumsum(wins)
            gp_range = np.arange(1, len(wins) + 1)
            rolling_pct = cum_wins / gp_range

            show = team in highlight
            fig.add_trace(go.Scatter(
                x=gp_range, y=rolling_pct, name=team, mode="lines",
                line=dict(width=2.5 if show else 1, color=None if show else "lightgray"),
                opacity=1.0 if show else 0.3,
                showlegend=show,
            ))

        fig.update_layout(
            xaxis_title="Games Played", yaxis_title="Cumulative Win %",
            title=f"Season Win% Trajectory — {slabel} (top 5 / bottom 3 highlighted)",
            margin=dict(t=60, b=40),
        )
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)

    _style_fig(fig)

    # Table
    table_cols = ["rank", "team", "GP", "actual_wins", "win_pct", "actual_GD",
                  "team_strength", "roster_EV_O", "roster_EV_D", "roster_PP",
                  "roster_PK", "roster_PEN", "goalie_GA_G"]
    table_cols = [c for c in table_cols if c in df.columns]
    table = dash_table.DataTable(
        data=df[table_cols].round(4).to_dict("records"),
        columns=[{"name": c, "id": c} for c in table_cols],
        sort_action="native", filter_action="native",
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "center", "padding": "4px", "fontSize": "11px"},
        style_header={"fontWeight": "bold", "backgroundColor": "#f0f0f0"},
        style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#f9f9f9"}],
        page_size=32,
    )

    return fig, table


@app.callback(
    Output("tm-game-chart", "figure"),
    [Input("tm-team", "value"), Input("tm-season", "value")],
)
def update_team_game_detail(team, season):
    gdf = team_games[team_games["season"] == season].copy()
    tg = gdf[(gdf["home_team"] == team) | (gdf["away_team"] == team)].sort_values("game_date")

    if len(tg) == 0:
        return _empty_fig(f"No game data for {team}")

    # Build per-game series
    dates, gds, pred_gds, wins = [], [], [], []
    for _, row in tg.iterrows():
        dates.append(row["game_date"])
        if row["home_team"] == team:
            gds.append(row["goal_diff"])
            pred_gds.append(row["pred_gd_xgb"])
            wins.append(1 if row["home_win"] == 1 else 0)
        else:
            gds.append(-row["goal_diff"])
            pred_gds.append(-row["pred_gd_xgb"])
            wins.append(1 if row["home_win"] == 0 else 0)

    cum_gd = np.cumsum(gds)
    cum_wins = np.cumsum(wins)
    gp = np.arange(1, len(wins) + 1)

    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        f"{team} Cumulative Goal Diff", f"{team} Win% vs XGBoost Predicted"),
        horizontal_spacing=0.12)

    fig.add_trace(go.Scatter(
        x=dates, y=cum_gd, mode="lines", name="Actual GD",
        line=dict(color=COLORS["EV_O"], width=2.5),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=dates, y=np.cumsum(pred_gds), mode="lines", name="Predicted GD (XGB)",
        line=dict(color=COLORS["EV_D"], width=2, dash="dash"),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=gp, y=cum_wins / gp, mode="lines", name="Actual Win%",
        line=dict(color=COLORS["EV_O"], width=2.5),
    ), row=1, col=2)

    # Predicted win% from XGB
    pred_wins = []
    for _, row in tg.iterrows():
        if row["home_team"] == team:
            pred_wins.append(row["pred_win_xgb"])
        else:
            pred_wins.append(1 - row["pred_win_xgb"])
    cum_pred_wins = np.cumsum(pred_wins)
    fig.add_trace(go.Scatter(
        x=gp, y=cum_pred_wins / gp, mode="lines", name="XGB Expected Win%",
        line=dict(color=COLORS["EV_D"], width=2, dash="dash"),
    ), row=1, col=2)

    fig.update_layout(height=400, plot_bgcolor="white",
                       legend=dict(orientation="h", yanchor="bottom", y=1.08,
                                   xanchor="center", x=0.5))
    fig.update_yaxes(gridcolor="#eee", zeroline=True, zerolinecolor="gray")
    fig.update_xaxes(gridcolor="#eee")

    return fig


if __name__ == "__main__":
    print(f"\n  NHL Player Ratings Dashboard")
    print(f"  Seasons: {season_label(min(ALL_SEASONS))} through {season_label(LATEST_SEASON)}")
    print(f"  {len(unified)} total player-seasons")
    print(f"  Open http://127.0.0.1:8050\n")
    app.run(debug=True)
