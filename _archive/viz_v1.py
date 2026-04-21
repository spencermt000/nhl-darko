"""
viz.py — NHL RAPM visualizations
  1. Top 10 per season tables (saved as PNG)
  2. Career trajectories for HoF / elite players (line chart)
  3. Interactive BPR_O vs BPR_D scatter (HTML, hover = name + season)
  4. Top 10 PP specialists table (PNG)
  5. Top 10 PK specialists table (PNG)
  6. Interactive PP_O vs PK_D scatter (HTML)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ── config ──────────────────────────────────────────────────────────────────
BY_SEASON  = Path("output/final_ratings_by_season.csv")
POOLED     = Path("output/final_ratings.csv")
OUT_DIR    = Path("viz/output_v1")
OUT_DIR.mkdir(exist_ok=True)

TRAJECTORY_PLAYERS = [
    "Sidney.Crosby", "Alex.Ovechkin", "Connor.McDavid",
    "Leon.Draisaitl", "Nathan.MacKinnon", "Erik.Karlsson",
]

METRIC_LABEL = "final_BPR"   # column used for career trajectory y-axis

# ── load ─────────────────────────────────────────────────────────────────────
df = pd.read_csv(BY_SEASON)
df["display_name"] = df["player_name"].str.replace(".", " ", regex=False)
df["label"] = df["display_name"] + " (" + df["season"].astype(str) + ")"

# ── 1. Top 10 per season ─────────────────────────────────────────────────────
seasons = sorted(df["season"].unique())

for season in seasons:
    sub = (df[df["season"] == season]
           .sort_values("final_BPR", ascending=False)
           .head(10)
           .reset_index(drop=True))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axis("off")

    table_data = [["Rank", "Player", "Pos", "BPR_O", "BPR_D", "BPR", "±SE"]]
    for i, row in sub.iterrows():
        table_data.append([
            i + 1,
            row["display_name"],
            row["position"],
            f"{row['final_BPR_O']:+.3f}",
            f"{row['final_BPR_D']:+.3f}",
            f"{row['final_BPR']:+.3f}",
            f"±{row['BPR_se']:.3f}",
        ])

    tbl = ax.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.5)

    # header row styling
    for j in range(len(table_data[0])):
        tbl[0, j].set_facecolor("#1a1a2e")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # alternating row shading
    for i in range(1, 11):
        color = "#f0f4ff" if i % 2 == 0 else "white"
        for j in range(len(table_data[0])):
            tbl[i, j].set_facecolor(color)

    nhl_season = f"{season}-{str(season+1)[-2:]}"
    ax.set_title(f"Top 10 Skaters — {nhl_season} (5v5 BPR)", fontsize=13, fontweight="bold", pad=12)
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
    ax.plot(sub["season"], sub[METRIC_LABEL], marker="o", linewidth=2,
            color=colors[idx], label=label)
    # confidence band
    ax.fill_between(
        sub["season"],
        sub[METRIC_LABEL] - sub["BPR_se"],
        sub[METRIC_LABEL] + sub["BPR_se"],
        alpha=0.12, color=colors[idx],
    )

ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax.set_xlabel("Season (start year)", fontsize=12)
ax.set_ylabel("BPR (Offensive + Defensive)", fontsize=12)
ax.set_title("Career Trajectory — Elite Skaters (5v5 BPR)", fontsize=14, fontweight="bold")
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
print(f"Career trajectory chart saved → {OUT_DIR}/career_trajectories.png")


# ── 3. Interactive BPR_O vs BPR_D scatter ────────────────────────────────────
# Color by position, size by 1/BPR_se (larger = more confident)
scatter_df = df.copy()
scatter_df["confidence"] = (1 / scatter_df["BPR_se"].clip(lower=0.05)).clip(upper=40)
scatter_df["nhl_season"] = scatter_df["season"].apply(
    lambda s: f"{s}-{str(s+1)[-2:]}"
)

fig_scatter = px.scatter(
    scatter_df,
    x="final_BPR_O",
    y="final_BPR_D",
    color="position",
    size="confidence",
    size_max=14,
    hover_name="display_name",
    hover_data={
        "nhl_season": True,
        "final_BPR_O": ":.3f",
        "final_BPR_D": ":.3f",
        "final_BPR": ":.3f",
        "BPR_se": ":.3f",
        "position": True,
        "confidence": False,
    },
    color_discrete_map={"F": "#2196F3", "D": "#FF5722", "G": "#4CAF50"},
    opacity=0.65,
    title="5v5 BPR: Offensive vs Defensive Rating — All Player-Seasons",
    labels={
        "final_BPR_O": "BPR Offensive",
        "final_BPR_D": "BPR Defensive",
        "nhl_season": "Season",
    },
)

# quadrant lines
fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
fig_scatter.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)

# quadrant labels
for text, x, y in [
    ("Two-Way", 0.45, 0.45),
    ("Offensive", 0.45, -0.45),
    ("Defensive", -0.35, 0.45),
    ("Below Avg", -0.35, -0.45),
]:
    fig_scatter.add_annotation(
        x=x, y=y, text=text, showarrow=False,
        font=dict(size=11, color="rgba(100,100,100,0.5)"),
        xref="x", yref="y",
    )

fig_scatter.update_layout(
    width=1000, height=750,
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="Arial", size=12),
    legend_title_text="Position",
)
fig_scatter.update_xaxes(showgrid=True, gridcolor="#eeeeee", zeroline=False)
fig_scatter.update_yaxes(showgrid=True, gridcolor="#eeeeee", zeroline=False)

out_path = OUT_DIR / "bpr_scatter.html"
fig_scatter.write_html(str(out_path))
print(f"Interactive scatter saved → {out_path}")


# ── 4 & 5. PP / PK top-10 tables ─────────────────────────────────────────────
PP_RAPM = Path("output/pp_rapm.csv")
pp = pd.read_csv(PP_RAPM)
pp["display_name"] = pp["player_name"].str.replace(".", " ", regex=False)

def make_special_teams_table(data, sort_col, title, filename, header):
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

make_special_teams_table(
    pp, "PP_O",
    "Top 10 PP Specialists — Career (PP_O)",
    "top10_pp_specialists.png",
    ["Rank", "Player", "Pos", "PP_O", "PK_D", "PP_BPR"],
)
make_special_teams_table(
    pp, "PK_D",
    "Top 10 PK Specialists — Career (PK_D)",
    "top10_pk_specialists.png",
    ["Rank", "Player", "Pos", "PK_D", "PP_O", "PP_BPR"],
)

# ── 6. Interactive PP_O vs PK_D scatter ──────────────────────────────────────
fig_pp = px.scatter(
    pp,
    x="PP_O",
    y="PK_D",
    color="position",
    hover_name="display_name",
    hover_data={
        "PP_O": ":.3f",
        "PK_D": ":.3f",
        "PP_BPR": ":.3f",
        "position": True,
    },
    color_discrete_map={"F": "#2196F3", "D": "#FF5722"},
    opacity=0.7,
    title="Special Teams Rating: PP Offense vs PK Defense (Pooled Career)",
    labels={"PP_O": "PP Offensive (PP_O)", "PK_D": "PK Defensive (PK_D)"},
)

fig_pp.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
fig_pp.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)

for text, x, y in [
    ("Elite PP + PK", 0.4, 0.12),
    ("PP Specialist", 0.4, -0.05),
    ("PK Specialist", -0.15, 0.12),
    ("Below Avg ST", -0.15, -0.05),
]:
    fig_pp.add_annotation(
        x=x, y=y, text=text, showarrow=False,
        font=dict(size=11, color="rgba(100,100,100,0.5)"),
        xref="x", yref="y",
    )

fig_pp.update_layout(
    width=1000, height=750,
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="Arial", size=12),
    legend_title_text="Position",
)
fig_pp.update_xaxes(showgrid=True, gridcolor="#eeeeee", zeroline=False)
fig_pp.update_yaxes(showgrid=True, gridcolor="#eeeeee", zeroline=False)

out_path_pp = OUT_DIR / "pp_pk_scatter.html"
fig_pp.write_html(str(out_path_pp))
print(f"PP/PK scatter saved → {out_path_pp}")

print("\nAll done. Open viz/output_v1/ to view.")
