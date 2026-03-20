"""
viz_v2.py — NHL RAPM v2 visualizations.

Same as v1 viz.py plus:
  - Prior vs posterior scatter (x=box prior, y=final BPR, color by rapm_weight)
  - Uncertainty funnel (x=TOI, y=BPR with error bars from posterior SE)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import plotly.express as px
from pathlib import Path

# ── config ──────────────────────────────────────────────────────────────────
BY_SEASON  = Path("output/v2_final_ratings_by_season.csv")
POOLED     = Path("output/v2_final_ratings.csv")
OUT_DIR    = Path("viz/output_v2")
OUT_DIR.mkdir(exist_ok=True)

TRAJECTORY_PLAYERS = [
    "Sidney.Crosby", "Alex.Ovechkin", "Connor.McDavid",
    "Leon.Draisaitl", "Nathan.MacKinnon", "Erik.Karlsson",
]

# ── load ─────────────────────────────────────────────────────────────────────
df = pd.read_csv(BY_SEASON)
pooled = pd.read_csv(POOLED)
df["display_name"] = df["player_name"].str.replace(".", " ", regex=False)
df["label"] = df["display_name"] + " (" + df["season"].astype(str) + ")"
pooled["display_name"] = pooled["player_name"].str.replace(".", " ", regex=False)

# ── 1. Top 10 per season tables ─────────────────────────────────────────────
seasons = sorted(df["season"].unique())

for season in seasons:
    sub = (df[df["season"] == season]
           .sort_values("BPR", ascending=False)
           .head(10)
           .reset_index(drop=True))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axis("off")

    has_se = "BPR_se" in sub.columns
    header = ["Rank", "Player", "Pos", "BPR_O", "BPR_D", "BPR"]
    if has_se:
        header.append("SE")

    table_data = []
    for i, row in sub.iterrows():
        r = [
            i + 1,
            row["display_name"],
            row["position"],
            f"{row['BPR_O']:+.3f}",
            f"{row['BPR_D']:+.3f}",
            f"{row['BPR']:+.3f}",
        ]
        if has_se:
            r.append(f"{row.get('BPR_se', 0):.3f}")
        table_data.append(r)

    tbl = ax.table(cellText=table_data, colLabels=header, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.5)

    for j in range(len(header)):
        tbl[0, j].set_facecolor("#1a1a2e")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, 11):
        color = "#f0f4ff" if i % 2 == 0 else "white"
        for j in range(len(header)):
            tbl[i, j].set_facecolor(color)

    nhl_season = f"{season}-{str(season+1)[-2:]}"
    ax.set_title(f"Top 10 Skaters — {nhl_season} (v2 Bayesian BPR)", fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"top10_{season}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved top10_{season}.png")

print(f"Top-10 tables: {len(seasons)} seasons written to {OUT_DIR}/")


# ── 2. Career trajectories ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
colors = plt.cm.tab10.colors
present = [p for p in TRAJECTORY_PLAYERS if p in df["player_name"].values]

for idx, pname in enumerate(present):
    sub = df[df["player_name"] == pname].sort_values("season")
    label = pname.replace(".", " ")
    ax.plot(sub["season"], sub["BPR"], marker="o", linewidth=2,
            color=colors[idx], label=label)
    if "BPR_se" in sub.columns:
        ax.fill_between(
            sub["season"],
            sub["BPR"] - sub["BPR_se"],
            sub["BPR"] + sub["BPR_se"],
            alpha=0.12, color=colors[idx],
        )

ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax.set_xlabel("Season (start year)", fontsize=12)
ax.set_ylabel("BPR (Offensive + Defensive)", fontsize=12)
ax.set_title("Career Trajectory — Elite Skaters (v2 Bayesian BPR)", fontsize=14, fontweight="bold")
ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
ax.xaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f"{int(x)}-{str(int(x)+1)[-2:]}"
))
plt.xticks(rotation=45)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "career_trajectories.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Career trajectory chart saved")


# ── 3. Interactive BPR_O vs BPR_D scatter ────────────────────────────────────
scatter_df = df.copy()
if "BPR_se" in scatter_df.columns:
    scatter_df["confidence"] = (1 / scatter_df["BPR_se"].clip(lower=0.05)).clip(upper=40)
else:
    scatter_df["confidence"] = 10.0
scatter_df["nhl_season"] = scatter_df["season"].apply(lambda s: f"{s}-{str(s+1)[-2:]}")

fig_scatter = px.scatter(
    scatter_df,
    x="BPR_O", y="BPR_D",
    color="position",
    size="confidence", size_max=14,
    hover_name="display_name",
    hover_data={
        "nhl_season": True, "BPR_O": ":.3f", "BPR_D": ":.3f",
        "BPR": ":.3f", "position": True, "confidence": False,
    },
    color_discrete_map={"F": "#2196F3", "D": "#FF5722"},
    opacity=0.65,
    title="v2 Bayesian BPR: Offensive vs Defensive — All Player-Seasons",
    labels={"BPR_O": "BPR Offensive", "BPR_D": "BPR Defensive", "nhl_season": "Season"},
)
fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
fig_scatter.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)
fig_scatter.update_layout(width=1000, height=750, plot_bgcolor="white", paper_bgcolor="white")
fig_scatter.update_xaxes(showgrid=True, gridcolor="#eeeeee", zeroline=False)
fig_scatter.update_yaxes(showgrid=True, gridcolor="#eeeeee", zeroline=False)
fig_scatter.write_html(str(OUT_DIR / "bpr_scatter.html"))
print(f"Interactive scatter saved")


# ── 4. Prior vs Posterior scatter (NEW in v2) ────────────────────────────────
if "prior_O" in pooled.columns and "prior_D" in pooled.columns:
    pooled["prior_BPR"] = pooled["prior_O"] + pooled["prior_D"]

    fig_prior = px.scatter(
        pooled,
        x="prior_BPR", y="BPR",
        color="rapm_weight" if "rapm_weight" in pooled.columns else "position",
        hover_name="display_name",
        hover_data={
            "prior_BPR": ":.3f", "BPR": ":.3f", "position": True,
        },
        opacity=0.65,
        title="Box Score Prior vs Bayesian RAPM (Pooled)",
        labels={"prior_BPR": "Box Score Prior (BPR)", "BPR": "Bayesian RAPM (BPR)"},
        color_continuous_scale="Viridis",
    )
    # Perfect agreement line
    bpr_range = [min(pooled["prior_BPR"].min(), pooled["BPR"].min()),
                 max(pooled["prior_BPR"].max(), pooled["BPR"].max())]
    fig_prior.add_shape(type="line", x0=bpr_range[0], y0=bpr_range[0],
                        x1=bpr_range[1], y1=bpr_range[1],
                        line=dict(dash="dash", color="gray", width=1))
    fig_prior.update_layout(width=900, height=700, plot_bgcolor="white", paper_bgcolor="white")
    fig_prior.update_xaxes(showgrid=True, gridcolor="#eeeeee")
    fig_prior.update_yaxes(showgrid=True, gridcolor="#eeeeee")
    fig_prior.write_html(str(OUT_DIR / "prior_vs_posterior.html"))
    print(f"Prior vs posterior scatter saved")


# ── 5. Uncertainty funnel (NEW in v2) ────────────────────────────────────────
if "BPR_se" in pooled.columns and "toi_5v5" in pooled.columns:
    funnel_df = pooled[pooled["toi_5v5"] > 0].copy()

    fig_funnel, ax = plt.subplots(figsize=(12, 7))
    pos_colors = {"F": "#2196F3", "D": "#FF5722"}

    for pos in ["F", "D"]:
        sub = funnel_df[funnel_df["position"] == pos]
        ax.errorbar(
            sub["toi_5v5"], sub["BPR"],
            yerr=sub["BPR_se"],
            fmt="o", markersize=3, alpha=0.4,
            color=pos_colors.get(pos, "gray"),
            ecolor=pos_colors.get(pos, "gray"),
            elinewidth=0.5, capsize=0, label=pos,
        )

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("5v5 TOI (minutes, pooled)", fontsize=12)
    ax.set_ylabel("BPR", fontsize=12)
    ax.set_title("Uncertainty Funnel — BPR vs TOI with Posterior SE", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig_funnel.tight_layout()
    fig_funnel.savefig(OUT_DIR / "uncertainty_funnel.png", dpi=150, bbox_inches="tight")
    plt.close(fig_funnel)
    print(f"Uncertainty funnel saved")


# ── 6. PP/PK tables + scatter (reuse v1 data) ────────────────────────────────
PP_FILE = Path("output/pp_rapm.csv")
if PP_FILE.exists():
    pp = pd.read_csv(PP_FILE)
    pp["display_name"] = pp["player_name"].str.replace(".", " ", regex=False)

    def make_st_table(data, sort_col, title, filename, header):
        sub = data.sort_values(sort_col, ascending=False).head(10).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.axis("off")
        rows = []
        for i, row in sub.iterrows():
            rows.append([i + 1, row["display_name"], row["position"]] +
                        [f"{row[c]:+.3f}" for c in header[3:]])
        tbl = ax.table(cellText=rows, colLabels=header, cellLoc="center", loc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 1.5)
        for j in range(len(header)):
            tbl[0, j].set_facecolor("#1a1a2e")
            tbl[0, j].set_text_props(color="white", fontweight="bold")
        for i in range(1, 11):
            color = "#f0f4ff" if i % 2 == 0 else "white"
            for j in range(len(header)):
                tbl[i, j].set_facecolor(color)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        fig.tight_layout()
        fig.savefig(OUT_DIR / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {filename}")

    make_st_table(pp, "PP_O", "Top 10 PP Specialists (PP_O)", "top10_pp.png",
                  ["Rank", "Player", "Pos", "PP_O", "PK_D", "PP_BPR"])
    make_st_table(pp, "PK_D", "Top 10 PK Specialists (PK_D)", "top10_pk.png",
                  ["Rank", "Player", "Pos", "PK_D", "PP_O", "PP_BPR"])

    fig_pp = px.scatter(
        pp, x="PP_O", y="PK_D", color="position",
        hover_name="display_name",
        hover_data={"PP_O": ":.3f", "PK_D": ":.3f", "PP_BPR": ":.3f", "position": True},
        color_discrete_map={"F": "#2196F3", "D": "#FF5722"},
        opacity=0.7,
        title="Special Teams: PP_O vs PK_D (Pooled Career)",
        labels={"PP_O": "PP Offensive", "PK_D": "PK Defensive"},
    )
    fig_pp.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    fig_pp.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)
    fig_pp.update_layout(width=1000, height=750, plot_bgcolor="white", paper_bgcolor="white")
    fig_pp.write_html(str(OUT_DIR / "pp_pk_scatter.html"))
    print(f"PP/PK scatter saved")

# ── 7. GAR component stacked bar chart (top 20 WAR) ──────────────────────────
GAR_POOLED = Path("output/v2_gar_pooled.csv")
if GAR_POOLED.exists():
    gar = pd.read_csv(GAR_POOLED)
    gar["display_name"] = gar["player_name"].str.replace(".", " ", regex=False)

    top20 = gar.sort_values("WAR", ascending=False).head(20).copy()
    top20 = top20.sort_values("WAR", ascending=True)  # bottom-to-top for horizontal bars

    comp_cols = ["xEV_O_GAR", "xEV_D_GAR", "FINISH_O_GAR", "FINISH_D_GAR",
                 "PP_GAR", "PK_GAR", "PEN_GAR", "FO_GAR"]
    comp_labels = ["xEV Offense", "xEV Defense", "Finishing (O)", "Finishing (D)",
                   "Power Play", "Penalty Kill", "Penalties", "Faceoffs"]
    comp_colors = ["#2196F3", "#1565C0", "#4CAF50", "#2E7D32",
                   "#FF9800", "#F57C00", "#9C27B0", "#607D8B"]

    fig, ax = plt.subplots(figsize=(13, 9))
    y_pos = range(len(top20))
    left_pos = np.zeros(len(top20))
    left_neg = np.zeros(len(top20))

    for col, label, color in zip(comp_cols, comp_labels, comp_colors):
        if col not in top20.columns:
            continue
        vals = top20[col].fillna(0).values
        pos_vals = np.where(vals > 0, vals, 0)
        neg_vals = np.where(vals < 0, vals, 0)

        if pos_vals.sum() > 0:
            ax.barh(y_pos, pos_vals, left=left_pos, height=0.7,
                    color=color, label=label, edgecolor="white", linewidth=0.3)
            left_pos += pos_vals
        if neg_vals.sum() < 0:
            ax.barh(y_pos, neg_vals, left=left_neg, height=0.7,
                    color=color, edgecolor="white", linewidth=0.3, alpha=0.6)
            left_neg += neg_vals

    # WAR labels on right
    for i, (_, row) in enumerate(top20.iterrows()):
        ax.text(left_pos[i] + 0.3, i, f"{row['WAR']:.1f} WAR",
                va="center", fontsize=9, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top20["display_name"], fontsize=10)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Goals Above Replacement (GAR)", fontsize=12)
    ax.set_title("Top 20 Skaters — Component-Level GAR Breakdown (Pooled Career)",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gar_components_top20.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("GAR component chart saved")


# ── 8. xGAR vs GAR scatter ──────────────────────────────────────────────────
    if "xGAR" in gar.columns:
        qualified = gar[gar["toi_5v5"] >= 200].copy()
        qualified["display_name"] = qualified["player_name"].str.replace(".", " ", regex=False)

        fig_xgar = px.scatter(
            qualified,
            x="xGAR", y="GAR",
            color="position",
            hover_name="display_name",
            hover_data={"xGAR": ":.1f", "GAR": ":.1f", "WAR": ":.1f",
                        "FINISH_O_GAR": ":.1f", "position": True},
            color_discrete_map={"F": "#2196F3", "D": "#FF5722"},
            opacity=0.6,
            title="xGAR vs GAR — Finishing Talent (deviation from diagonal)",
            labels={"xGAR": "Expected GAR (no finishing)", "GAR": "GAR (with finishing)"},
        )
        gar_range = [min(qualified["xGAR"].min(), qualified["GAR"].min()),
                     max(qualified["xGAR"].max(), qualified["GAR"].max())]
        fig_xgar.add_shape(type="line", x0=gar_range[0], y0=gar_range[0],
                           x1=gar_range[1], y1=gar_range[1],
                           line=dict(dash="dash", color="gray", width=1))
        fig_xgar.update_layout(width=900, height=700, plot_bgcolor="white", paper_bgcolor="white")
        fig_xgar.update_xaxes(showgrid=True, gridcolor="#eeeeee")
        fig_xgar.update_yaxes(showgrid=True, gridcolor="#eeeeee")
        fig_xgar.write_html(str(OUT_DIR / "xgar_vs_gar.html"))
        print("xGAR vs GAR scatter saved")


# ── 9. WAR leaderboard table (top 20) ────────────────────────────────────────
    top20_tbl = gar.sort_values("WAR", ascending=False).head(20).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")

    header = ["Rank", "Player", "Pos", "EV_O", "EV_D", "PP", "PK", "PEN", "FO", "GAR", "WAR"]
    rows = []
    for i, row in top20_tbl.iterrows():
        rows.append([
            i + 1,
            row["display_name"],
            row["position"],
            f"{row.get('EV_O_GAR', 0):+.1f}",
            f"{row.get('EV_D_GAR', 0):+.1f}",
            f"{row.get('PP_GAR', 0):+.1f}",
            f"{row.get('PK_GAR', 0):+.1f}",
            f"{row.get('PEN_GAR', 0):+.1f}",
            f"{row.get('FO_GAR', 0):+.1f}",
            f"{row['GAR']:+.1f}",
            f"{row['WAR']:.1f}",
        ])

    tbl = ax.table(cellText=rows, colLabels=header, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
    for j in range(len(header)):
        tbl[0, j].set_facecolor("#1a1a2e")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, 21):
        color = "#f0f4ff" if i % 2 == 0 else "white"
        for j in range(len(header)):
            tbl[i, j].set_facecolor(color)

    ax.set_title("Top 20 Skaters — Component GAR / WAR (Pooled Career)",
                 fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "war_leaderboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("WAR leaderboard table saved")


# ── 9b. Single-season WAR leaderboard (most recent complete season) ──────────
GAR_SEASON = Path("output/v2_gar_by_season.csv")
if GAR_SEASON.exists():
    gar_s = pd.read_csv(GAR_SEASON)
    gar_s["display_name"] = gar_s["player_name"].str.replace(".", " ", regex=False)

    # Most recent complete season: largest season with a reasonable player count
    season_counts = gar_s.groupby("season").size()
    # The current (partial) season has fewer players; pick the second-latest if latest is small
    all_seasons = sorted(season_counts.index)
    if len(all_seasons) >= 2 and season_counts[all_seasons[-1]] < 0.8 * season_counts[all_seasons[-2]]:
        latest_full = all_seasons[-2]
    else:
        latest_full = all_seasons[-1]

    nhl_season_str = f"{latest_full}-{str(latest_full + 1)[-2:]}"
    ssn = gar_s[gar_s["season"] == latest_full].sort_values("WAR", ascending=False)
    top20_ssn = ssn.head(20).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")

    header = ["Rank", "Player", "Pos", "EV_O", "EV_D", "PP", "PK", "PEN", "FO", "GAR", "WAR"]
    rows = []
    for i, row in top20_ssn.iterrows():
        rows.append([
            i + 1,
            row["display_name"],
            row["position"],
            f"{row.get('EV_O_GAR', 0):+.1f}",
            f"{row.get('EV_D_GAR', 0):+.1f}",
            f"{row.get('PP_GAR', 0):+.1f}",
            f"{row.get('PK_GAR', 0):+.1f}",
            f"{row.get('PEN_GAR', 0):+.1f}",
            f"{row.get('FO_GAR', 0):+.1f}",
            f"{row['GAR']:+.1f}",
            f"{row['WAR']:.1f}",
        ])

    tbl = ax.table(cellText=rows, colLabels=header, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
    for j in range(len(header)):
        tbl[0, j].set_facecolor("#1a1a2e")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, 21):
        color = "#f0f4ff" if i % 2 == 0 else "white"
        for j in range(len(header)):
            tbl[i, j].set_facecolor(color)

    ax.set_title(f"Top 20 Skaters — Component GAR / WAR ({nhl_season_str})",
                 fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"war_leaderboard_{latest_full}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"WAR leaderboard ({nhl_season_str}) saved")


# ── 10. Goalie WAR leaderboard ───────────────────────────────────────────────
GOALIE_WAR_FILE = Path("output/v2_goalie_war.csv")
if GOALIE_WAR_FILE.exists():
    gwar = pd.read_csv(GOALIE_WAR_FILE)
    top15g = gwar.sort_values("GOALIE_GAR_per60", ascending=False).head(15).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axis("off")
    header = ["Rank", "Goalie", "Rate", "GAR/60", "WAR/60"]
    rows = []
    for i, row in top15g.iterrows():
        rows.append([
            i + 1,
            row["goalie_name"],
            f"{row['goalie_rate']:+.4f}",
            f"{row['GOALIE_GAR_per60']:+.4f}",
            f"{row['GOALIE_WAR_per60']:+.4f}",
        ])
    tbl = ax.table(cellText=rows, colLabels=header, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.5)
    for j in range(len(header)):
        tbl[0, j].set_facecolor("#1a1a2e")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, 16):
        color = "#f0f4ff" if i % 2 == 0 else "white"
        for j in range(len(header)):
            tbl[i, j].set_facecolor(color)
    ax.set_title("Top 15 Goalies — RAPM-Based WAR Rate (Pooled)",
                 fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "goalie_war_leaderboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Goalie WAR leaderboard saved")

print("\nAll visualizations written to viz/output_v2/")
