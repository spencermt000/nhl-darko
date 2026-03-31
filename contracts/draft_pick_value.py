"""
Draft Pick Value Chart — Expected surplus value by draft round.

Quantifies how much surplus value a draft pick at each round is expected
to produce over their first 7 years (ELC + bridge/extension). This enables
trade evaluation: is pick X worth player Y?

Uses career WAR data joined with draft round info from contracts.csv.

Usage: python contracts/draft_pick_value.py
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(BASE, "output")

# ── 1. Load data ─────────────────────────────────────────────────────────────

dw = pd.read_csv(os.path.join(OUTPUT, "v5_daily_war.csv"))
cf = pd.read_csv(os.path.join(OUTPUT, "v6_carry_forward.csv"),
                 usecols=["player_id", "season", "age"])

# Get draft info from NHL API data (exact pick numbers + NHL player IDs)
draft_path = os.path.join(BASE, "data", "nhl_draft_picks.csv")
draft_raw = pd.read_csv(draft_path)
draft_raw = draft_raw[draft_raw["playerId"].notna()].copy()
draft_raw["player_id"] = draft_raw["playerId"].astype(int)

draft_info = draft_raw[["player_id", "playerName", "draftYear", "roundNumber",
                         "overallPickNumber", "pickInRound", "position", "triCode"]].copy()
draft_info = draft_info.rename(columns={
    "draftYear": "Draft Year", "roundNumber": "Draft Round",
    "overallPickNumber": "Overall Pick", "pickInRound": "Pick In Round",
    "playerName": "Player", "triCode": "Draft Team",
})
draft_info = draft_info[draft_info["Draft Round"] <= 7]
draft_info = draft_info.drop_duplicates("player_id", keep="first")

print(f"Players with draft info (exact pick numbers): {len(draft_info)}")

# ── 2. Build career trajectories ────────────────────────────────────────────

# For each drafted player, track their WAR by "pro year" (seasons since draft)
player_war = dw[["player_id", "season", "GP", "WAR", "WAR_82"]].copy()
player_war = player_war.merge(
    draft_info[["player_id", "Draft Year", "Draft Round", "Overall Pick"]],
    on="player_id", how="inner"
)

# Pro year = seasons since draft year
player_war["pro_year"] = player_war["season"] - player_war["Draft Year"]
# Only look at years 1-8 (ELC through first extension)
player_war = player_war[(player_war["pro_year"] >= 1) & (player_war["pro_year"] <= 8)].copy()

# Merge age
player_war = player_war.merge(cf[["player_id", "season", "age"]], on=["player_id", "season"], how="left")

print(f"Player-season observations: {len(player_war)}")
print(f"Unique players: {player_war['player_id'].nunique()}")

# ── 3. Value by draft round ────────────────────────────────────────────────

print(f"\n{'='*70}")
print("WAR BY DRAFT ROUND AND PRO YEAR")
print(f"{'='*70}\n")

# Average WAR by round and pro year
pivot = player_war.groupby(["Draft Round", "pro_year"]).agg(
    mean_WAR=("WAR", "mean"),
    mean_WAR_82=("WAR_82", "mean"),
    median_WAR=("WAR", "median"),
    players=("player_id", "nunique"),
    mean_GP=("GP", "mean"),
).reset_index()

print(f"{'Round':>5s} {'Year':>4s} {'Players':>7s} {'Mean GP':>7s} {'Mean WAR':>8s} {'WAR/82':>7s} {'Med WAR':>8s}")
print("-" * 55)
for _, r in pivot.sort_values(["Draft Round", "pro_year"]).iterrows():
    print(f"{int(r['Draft Round']):5d} {int(r['pro_year']):4d} {int(r['players']):7d} "
          f"{r['mean_GP']:7.0f} {r['mean_WAR']:8.3f} {r['mean_WAR_82']:7.3f} {r['median_WAR']:8.3f}")

# ── 4. Total expected value per draft pick ──────────────────────────────────

# For each round, calculate:
# 1. NHL appearance rate (what % of picks make the NHL?)
# 2. Expected total WAR over first 7 pro years
# 3. Expected surplus (using ELC + RFA cap hits)

DOLLARS_PER_WAR = 17_500_000

# Compute actual average cap hits by pro year from our data
active = pd.read_csv(os.path.join(BASE, "contracts", "active_contracts_by_season.csv"))
active = active.merge(draft_info[["player_id", "Draft Year"]], on="player_id", how="inner")
active["pro_year"] = active["season"] - active["Draft Year"]
active = active[(active["pro_year"] >= 1) & (active["pro_year"] <= 8)]
AVG_CAP_BY_YEAR = active.groupby("pro_year")["cap_hit"].mean().to_dict()
print("Empirical avg cap hit by pro year:")
for yr in range(1, 9):
    print(f"  Year {yr}: ${AVG_CAP_BY_YEAR.get(yr, 0):,.0f}")
print()

print(f"\n{'='*70}")
print("DRAFT PICK VALUE CHART — BY OVERALL PICK")
print(f"{'='*70}\n")

# Group by pick ranges for sufficient sample size
pick_bins = [(1, 1), (2, 3), (4, 5), (6, 10), (11, 15), (16, 20), (21, 31),
             (32, 64), (65, 96), (97, 128), (129, 160), (161, 224)]

pick_values = []

for lo, hi in pick_bins:
    bin_data = player_war[(player_war["Overall Pick"] >= lo) & (player_war["Overall Pick"] <= hi)]
    unique_players = bin_data["player_id"].nunique()
    if unique_players == 0:
        continue

    total_war_by_player = bin_data.groupby("player_id")["WAR"].sum()
    mean_career_war = total_war_by_player.mean()
    median_career_war = total_war_by_player.median()

    nhl_regulars = bin_data[bin_data["GP"] >= 20]["player_id"].nunique()
    nhl_rate = nhl_regulars / unique_players

    total_surplus = 0
    for yr in range(1, 9):
        yr_data = bin_data[bin_data["pro_year"] == yr]
        if len(yr_data):
            avg_war = yr_data["WAR"].mean()
            market_value = max(0, avg_war) * DOLLARS_PER_WAR
            cap_cost = AVG_CAP_BY_YEAR.get(yr, 2_000_000)
            total_surplus += market_value - cap_cost

    label = f"#{lo}" if lo == hi else f"#{lo}-{hi}"
    pick_values.append({
        "pick_range": label,
        "pick_lo": lo,
        "pick_hi": hi,
        "players": unique_players,
        "nhl_rate": nhl_rate,
        "mean_war_7yr": mean_career_war,
        "median_war_7yr": median_career_war,
        "total_surplus_7yr": total_surplus,
        "surplus_per_year": total_surplus / 7,
    })

pv = pd.DataFrame(pick_values)

print(f"{'Picks':>10s} {'Players':>8s} {'NHL%':>5s} {'Mean WAR':>9s} {'Med WAR':>9s} "
      f"{'Total Surplus':>14s} {'$/yr':>12s}")
print("-" * 75)
for _, r in pv.iterrows():
    print(f"{r['pick_range']:>10s} {int(r['players']):8d} {r['nhl_rate']:4.0%} "
          f"{r['mean_war_7yr']:9.2f} {r['median_war_7yr']:9.2f} "
          f"${r['total_surplus_7yr']:>13,.0f} ${r['surplus_per_year']:>11,.0f}")

# Also do by-round summary
round_values = []
for rnd in range(1, 8):
    rnd_data = player_war[player_war["Draft Round"] == rnd]
    unique_players = rnd_data["player_id"].nunique()
    if unique_players == 0:
        continue
    total_war_by_player = rnd_data.groupby("player_id")["WAR"].sum()
    nhl_regulars = rnd_data[rnd_data["GP"] >= 20]["player_id"].nunique()
    nhl_rate = nhl_regulars / unique_players

    total_surplus = 0
    for yr in range(1, 9):
        yr_data = rnd_data[rnd_data["pro_year"] == yr]
        if len(yr_data):
            avg_war = yr_data["WAR"].mean()
            market_value = max(0, avg_war) * DOLLARS_PER_WAR
            cap_cost = AVG_CAP_BY_YEAR.get(yr, 2_000_000)
            total_surplus += market_value - cap_cost

    round_values.append({
        "round": rnd, "players": unique_players, "nhl_rate": nhl_rate,
        "mean_war_7yr": total_war_by_player.mean(),
        "median_war_7yr": total_war_by_player.median(),
        "total_surplus_7yr": total_surplus,
        "surplus_per_year": total_surplus / 7,
    })

rv = pd.DataFrame(round_values)

# ── 5. Relative value (Round 1 = 100) ──────────────────────────────────────

r1_surplus = rv.iloc[0]["total_surplus_7yr"] if rv.iloc[0]["total_surplus_7yr"] > 0 else 1
rv["relative_value"] = rv["total_surplus_7yr"] / r1_surplus * 100

print(f"\n{'='*70}")
print("RELATIVE DRAFT PICK VALUE (Round 1 = 100)")
print(f"{'='*70}\n")

for _, r in rv.iterrows():
    bar = "#" * int(r["relative_value"] / 2)
    print(f"  Round {int(r['round'])}: {r['relative_value']:6.1f}  {bar}")

# ── 6. Trade value implications ─────────────────────────────────────────────

print(f"\n{'='*70}")
print("TRADE VALUE EQUIVALENCIES")
print(f"{'='*70}\n")

print("Based on expected surplus, approximate trade equivalencies:")
for i, r in rv.iterrows():
    if r["round"] == 1:
        continue
    ratio = r1_surplus / r["total_surplus_7yr"] if r["total_surplus_7yr"] > 0 else float("inf")
    if ratio < 20:
        print(f"  1 first-round pick ≈ {ratio:.1f} round-{int(r['round'])} picks")

# ── 7. Year-by-year surplus curve ───────────────────────────────────────────

print(f"\n{'='*70}")
print("SURPLUS CURVE BY PRO YEAR (Round 1 picks)")
print(f"{'='*70}\n")

r1_data = player_war[player_war["Draft Round"] == 1]
print(f"{'Year':>4s} {'Players':>8s} {'Mean GP':>7s} {'Mean WAR':>8s} {'Cap Cost':>10s} "
      f"{'Mkt Value':>10s} {'Surplus':>10s}")
print("-" * 65)
for yr in range(1, 9):
    yr_data = r1_data[r1_data["pro_year"] == yr]
    if len(yr_data):
        avg_war = yr_data["WAR"].mean()
        mkt = max(0, avg_war) * DOLLARS_PER_WAR
        cap = AVG_CAP_BY_YEAR.get(yr, 5_000_000)
        print(f"{yr:4d} {yr_data['player_id'].nunique():8d} {yr_data['GP'].mean():7.0f} "
              f"{avg_war:8.3f} ${cap:>9,.0f} ${mkt:>9,.0f} ${mkt - cap:>+9,.0f}")

# ── 8. Save ─────────────────────────────────────────────────────────────────

# Save pick-level chart
pv.to_csv(os.path.join(BASE, "contracts", "draft_pick_value_chart.csv"), index=False)

# Save round-level chart
rv.to_csv(os.path.join(BASE, "contracts", "draft_round_value_chart.csv"), index=False)

# Save detailed year-by-round data
pivot.to_csv(os.path.join(BASE, "contracts", "draft_pick_value_detail.csv"), index=False)

print(f"\nSaved draft_pick_value_chart.csv, draft_round_value_chart.csv, draft_pick_value_detail.csv")
