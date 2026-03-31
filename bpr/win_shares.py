"""
Win Shares — Allocate actual team wins to individual players.

Unlike WAR (which measures value above replacement and can be negative),
Win Shares distributes a team's actual wins proportionally based on each
player's GAR contributions. Properties:
  - Always >= 0
  - Sums to team wins per team-season
  - Split into OWS (offensive) and DWS (defensive)

Usage: python bpr/win_shares.py
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(BASE, "output")
DATA = os.path.join(BASE, "data")

# ── 1. Load data ─────────────────────────────────────────────────────────────

print("Loading data...")

war = pd.read_csv(os.path.join(OUTPUT, "v5_daily_war.csv"))
team_season = pd.read_csv(os.path.join(OUTPUT, "v6_team_season_ratings.csv"))

# Skaters by game (for player-team mapping)
sbg = pd.read_csv(os.path.join(DATA, "skaters_by_game.csv"),
                  usecols=["playerId", "season", "playerTeam", "situation", "icetime"],
                  low_memory=False)
sbg_2025 = os.path.join(DATA, "skaters_by_game2025.csv")
if os.path.exists(sbg_2025):
    sbg2 = pd.read_csv(sbg_2025,
                       usecols=["playerId", "season", "playerTeam", "situation", "icetime"],
                       low_memory=False)
    sbg = pd.concat([sbg, sbg2], ignore_index=True)

# ── 2. Build player-team mapping ────────────────────────────────────────────
# Pattern from validate_war.py:112-120

all_sit = sbg[sbg["situation"] == "all"].copy()
all_sit["icetime"] = pd.to_numeric(all_sit["icetime"], errors="coerce")

player_team = (
    all_sit.groupby(["playerId", "season", "playerTeam"])["icetime"]
    .sum()
    .reset_index()
)
# Primary team = most icetime per player-season
idx = player_team.groupby(["playerId", "season"])["icetime"].idxmax()
player_primary = player_team.loc[idx, ["playerId", "season", "playerTeam"]].copy()
player_primary = player_primary.rename(columns={"playerId": "player_id", "playerTeam": "team"})

print(f"Player-team mappings: {len(player_primary)}")

# ── 3. Merge GAR with team info ─────────────────────────────────────────────

df = war.merge(player_primary, on=["player_id", "season"], how="left")
df = df.dropna(subset=["team"])
print(f"Players with team assignment: {len(df)}")

# Merge team actual wins
df = df.merge(
    team_season[["team", "season", "actual_wins"]],
    on=["team", "season"],
    how="left",
)
df = df.dropna(subset=["actual_wins"])
print(f"Players with team wins data: {len(df)}")

# ── 4. Compute Win Shares ───────────────────────────────────────────────────

# Floor individual GAR_O and GAR_D at 0
df["GAR_O_pos"] = df["GAR_O"].clip(lower=0)
df["GAR_D_pos"] = df["GAR_D"].clip(lower=0)

results = []

for (team, season), grp in df.groupby(["team", "season"]):
    team_wins = grp["actual_wins"].iloc[0]

    team_gar_o = grp["GAR_O_pos"].sum()
    team_gar_d = grp["GAR_D_pos"].sum()
    team_gar_total = team_gar_o + team_gar_d

    if team_gar_total > 0:
        # Allocate wins proportionally to floored GAR
        ows = (grp["GAR_O_pos"] / team_gar_total) * team_wins
        dws = (grp["GAR_D_pos"] / team_gar_total) * team_wins
    else:
        # Fallback: allocate by TOI share
        total_toi = grp["toi_5v5"].sum() + grp["toi_pp"].sum() + grp["toi_pk"].sum()
        if total_toi > 0:
            player_toi = grp["toi_5v5"] + grp["toi_pp"] + grp["toi_pk"]
            share = player_toi / total_toi
        else:
            share = 1.0 / len(grp)
        ows = share * team_wins * 0.5
        dws = share * team_wins * 0.5

    for i, (idx, row) in enumerate(grp.iterrows()):
        ws = ows.iloc[i] + dws.iloc[i]
        ws_82 = ws * 82 / row["GP"] if row["GP"] > 0 else 0
        results.append({
            "player_id": int(row["player_id"]),
            "player_name": row["player_name"],
            "position": row["position"],
            "team": team,
            "season": season,
            "GP": row["GP"],
            "GAR_O": row["GAR_O"],
            "GAR_D": row["GAR_D"],
            "OWS": round(ows.iloc[i], 3),
            "DWS": round(dws.iloc[i], 3),
            "WS": round(ws, 3),
            "WS_82": round(ws_82, 3),
        })

ws_df = pd.DataFrame(results)
ws_df = ws_df.sort_values(["season", "WS"], ascending=[True, False]).reset_index(drop=True)

print(f"\nWin Shares computed: {len(ws_df)} player-seasons")

# ── 5. Verification ─────────────────────────────────────────────────────────

print(f"\n{'='*70}")
print("VERIFICATION: Team WS Sum vs Actual Wins")
print(f"{'='*70}\n")

team_check = ws_df.groupby(["team", "season"]).agg(
    ws_sum=("WS", "sum"),
).reset_index()
team_check = team_check.merge(team_season[["team", "season", "actual_wins"]], on=["team", "season"])
team_check["diff"] = (team_check["ws_sum"] - team_check["actual_wins"]).abs()

print(f"Max absolute difference: {team_check['diff'].max():.6f}")
print(f"Mean absolute difference: {team_check['diff'].mean():.6f}")
print(f"All within 0.01: {(team_check['diff'] < 0.01).all()}")

# Non-negativity check
print(f"\nMin WS value: {ws_df['WS'].min():.6f} (should be >= 0)")
print(f"Negative WS count: {(ws_df['WS'] < 0).sum()}")

# ── 6. Leaderboards ─────────────────────────────────────────────────────────

print(f"\n{'='*70}")
print("TOP 30 WIN SHARE SEASONS (WS)")
print(f"{'='*70}\n")

top = ws_df.nlargest(30, "WS")
print(f"{'Player':25s} {'Team':>4s} {'Szn':>7s} {'GP':>3s} {'OWS':>6s} {'DWS':>6s} {'WS':>6s} {'WS/82':>6s}  {'WAR':>5s}")
print("-" * 80)
for _, r in top.iterrows():
    # Get WAR for comparison
    w = war[(war["player_id"] == r["player_id"]) & (war["season"] == r["season"])]
    war_val = w["WAR"].iloc[0] if len(w) else 0
    print(f"{r['player_name']:25s} {r['team']:>4s} {int(r['season'])}-{str(int(r['season'])+1)[-2:]:>2s} "
          f"{int(r['GP']):>3d} {r['OWS']:>6.2f} {r['DWS']:>6.2f} {r['WS']:>6.2f} {r['WS_82']:>6.2f}  {war_val:>5.2f}")

print(f"\n{'='*70}")
print("TOP 30 WIN SHARE RATE SEASONS (WS/82, min 40 GP)")
print(f"{'='*70}\n")

qualified = ws_df[ws_df["GP"] >= 40]
top_rate = qualified.nlargest(30, "WS_82")
print(f"{'Player':25s} {'Team':>4s} {'Szn':>7s} {'GP':>3s} {'OWS':>6s} {'DWS':>6s} {'WS':>6s} {'WS/82':>6s}")
print("-" * 75)
for _, r in top_rate.iterrows():
    print(f"{r['player_name']:25s} {r['team']:>4s} {int(r['season'])}-{str(int(r['season'])+1)[-2:]:>2s} "
          f"{int(r['GP']):>3d} {r['OWS']:>6.2f} {r['DWS']:>6.2f} {r['WS']:>6.2f} {r['WS_82']:>6.2f}")

# Latest season leaderboard
latest = ws_df["season"].max()
print(f"\n{'='*70}")
print(f"TOP 30 WIN SHARES — {int(latest)}-{str(int(latest)+1)[-2:]} SEASON")
print(f"{'='*70}\n")

latest_top = ws_df[(ws_df["season"] == latest) & (ws_df["GP"] >= 20)].nlargest(30, "WS")
print(f"{'#':>2s} {'Player':25s} {'Team':>4s} {'GP':>3s} {'OWS':>6s} {'DWS':>6s} {'WS':>6s} {'WS/82':>6s}")
print("-" * 60)
for i, (_, r) in enumerate(latest_top.iterrows(), 1):
    print(f"{i:>2d} {r['player_name']:25s} {r['team']:>4s} {int(r['GP']):>3d} "
          f"{r['OWS']:>6.2f} {r['DWS']:>6.2f} {r['WS']:>6.2f} {r['WS_82']:>6.2f}")

# WAR vs WS correlation
print(f"\n{'='*70}")
print("WAR vs WS CORRELATION")
print(f"{'='*70}\n")

merged = ws_df.merge(war[["player_id", "season", "WAR"]], on=["player_id", "season"])
corr = merged[["WS", "WAR"]].corr().iloc[0, 1]
print(f"Pearson r (WS vs WAR): {corr:.4f}")

qualified_merged = merged[merged["GP"] >= 40]
corr_q = qualified_merged[["WS", "WAR"]].corr().iloc[0, 1]
print(f"Pearson r (WS vs WAR, min 40 GP): {corr_q:.4f}")

# ── 7. Save output ──────────────────────────────────────────────────────────

out_path = os.path.join(OUTPUT, "win_shares_by_season.csv")
ws_df.to_csv(out_path, index=False)
print(f"\nSaved {len(ws_df)} player-seasons to {out_path}")
