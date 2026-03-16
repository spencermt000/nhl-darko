"""Quick viz: McDavid 2025-26 game-by-game GAR breakdown."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

d = pd.read_csv("data/v5_daily_ratings.csv")
mc = d[(d["player_name"] == "Connor McDavid") & (d["season"] == 2025)].copy()
mc["game_date"] = pd.to_datetime(mc["game_date"])

# Cumulative GARs
for comp in ["EV_O_gar", "EV_D_gar", "PP_gar", "PK_gar", "PEN_gar"]:
    mc[f"cum_{comp}"] = mc[comp].cumsum()

fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={"height_ratios": [2, 1.2, 1.2]})
fig.suptitle("Connor McDavid — 2025-26 Season", fontsize=16, fontweight="bold", y=0.95)

# ── Panel 1: Stacked cumulative GAR ──────────────────────────────────────────
ax = axes[0]
dates = mc["game_date"].values

components = [
    ("cum_EV_O_gar", "EV Offense", "#1f77b4"),
    ("cum_PP_gar",   "Power Play", "#ff7f0e"),
    ("cum_PEN_gar",  "Penalties",  "#2ca02c"),
    ("cum_PK_gar",   "Penalty Kill", "#9467bd"),
    ("cum_EV_D_gar", "EV Defense", "#d62728"),
]

# Stack positives and negatives separately
pos_cols = []
neg_cols = []
for col, label, color in components:
    final_val = mc[col].iloc[-1]
    if final_val >= 0:
        pos_cols.append((col, label, color))
    else:
        neg_cols.append((col, label, color))

# Plot positive stacked area
bottom_pos = np.zeros(len(mc))
for col, label, color in pos_cols:
    vals = mc[col].values
    ax.fill_between(dates, bottom_pos, bottom_pos + vals, alpha=0.4, color=color, label=f"{label} ({vals[-1]:+.1f})")
    ax.plot(dates, bottom_pos + vals, color=color, linewidth=0.8, alpha=0.7)
    bottom_pos += vals

# Plot negative stacked below zero
bottom_neg = np.zeros(len(mc))
for col, label, color in neg_cols:
    vals = mc[col].values
    ax.fill_between(dates, bottom_neg + vals, bottom_neg, alpha=0.4, color=color, label=f"{label} ({vals[-1]:+.1f})")
    ax.plot(dates, bottom_neg + vals, color=color, linewidth=0.8, alpha=0.7)
    bottom_neg += vals

# Total GAR line
total_cum = mc["cum_EV_O_gar"] + mc["cum_EV_D_gar"] + mc["cum_PP_gar"] + mc["cum_PK_gar"] + mc["cum_PEN_gar"]
ax.plot(dates, total_cum.values, color="black", linewidth=2.5, label=f"Total GAR ({total_cum.iloc[-1]:.1f})")

ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
ax.set_ylabel("Cumulative GAR", fontsize=11)
ax.set_title("Cumulative Goals Above Replacement — 5-Component Breakdown", fontsize=12)
ax.legend(loc="upper left", fontsize=9, ncol=2)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.grid(axis="y", alpha=0.3)

# ── Panel 2: Smoothed rates over time ────────────────────────────────────────
ax2 = axes[1]

rate_components = [
    ("EV_O", "EV Offense", "#1f77b4"),
    ("PP",   "Power Play", "#ff7f0e"),
    ("PEN",  "Penalties",  "#2ca02c"),
    ("PK",   "Penalty Kill", "#9467bd"),
    ("EV_D", "EV Defense", "#d62728"),
]

for col, label, color in rate_components:
    ax2.plot(dates, mc[col].values, color=color, linewidth=1.5, alpha=0.85, label=label)

ax2.axhline(0, color="gray", linewidth=0.5, linestyle="--")
ax2.set_ylabel("Rate (per-60, goals)", fontsize=11)
ax2.set_title("Bayesian-Smoothed Component Rates", fontsize=12)
ax2.legend(loc="upper right", fontsize=9, ncol=3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax2.xaxis.set_major_locator(mdates.MonthLocator())
ax2.grid(axis="y", alpha=0.3)

# ── Panel 3: Per-game GAR (bar chart) ────────────────────────────────────────
ax3 = axes[2]

mc["total_gar"] = mc["EV_O_gar"] + mc["EV_D_gar"] + mc["PP_gar"] + mc["PK_gar"] + mc["PEN_gar"]

colors_bar = ["#1f77b4", "#d62728", "#ff7f0e", "#9467bd", "#2ca02c"]
labels_bar = ["EV_O", "EV_D", "PP", "PK", "PEN"]
gar_cols = ["EV_O_gar", "EV_D_gar", "PP_gar", "PK_gar", "PEN_gar"]

bottom_pos = np.zeros(len(mc))
bottom_neg = np.zeros(len(mc))

for col, label, color in zip(gar_cols, labels_bar, colors_bar):
    vals = mc[col].values
    pos_vals = np.where(vals >= 0, vals, 0)
    neg_vals = np.where(vals < 0, vals, 0)
    ax3.bar(dates, pos_vals, bottom=bottom_pos, color=color, alpha=0.6, width=1.5, label=label)
    ax3.bar(dates, neg_vals, bottom=bottom_neg, color=color, alpha=0.6, width=1.5)
    bottom_pos += pos_vals
    bottom_neg += neg_vals

ax3.axhline(0, color="gray", linewidth=0.5, linestyle="--")
ax3.set_ylabel("Game GAR", fontsize=11)
ax3.set_title("Per-Game GAR Contributions", fontsize=12)
ax3.legend(loc="upper right", fontsize=8, ncol=5)
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax3.xaxis.set_major_locator(mdates.MonthLocator())
ax3.grid(axis="y", alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("data/viz_mcdavid_2025.png", dpi=150, bbox_inches="tight")
print("Saved → data/viz_mcdavid_2025.png")
