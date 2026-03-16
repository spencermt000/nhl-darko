"""
bpm.py — Box Plus-Minus with two calibration targets.

Two independent box-score models, each producing per-player-season ratings:

  MODEL 1 — "Goal Value" (GV):
    Empirical regression: team-level individual stats → team goals.
    Derives the marginal goal value of each box-score stat, then applies
    those weights to individual players. Pure production-based.
    → GV_O, GV_D per player-season (goals above average per 60 min)

  MODEL 2 — "On/Off Impact" (OOI):
    Regression: player box-score stats → on/off xGF differential.
    Captures how much a player's individual profile predicts their
    on-ice impact relative to when they're off the ice.
    → OOI_O, OOI_D per player-season

Both are combined with rolling RAPM (already built) and special teams
in a downstream script.

Inputs:
  data/skaters_by_game.csv       Per-game box scores
  data/v3_rolling_rapm_latest.csv  Latest-window RAPM per player
  data/pp_rapm.csv               PP/PK RAPM coefficients

Outputs:
  data/v4_bpm_player_seasons.csv   Per-player-season BPM components
"""

import sys
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

# ── Config ───────────────────────────────────────────────────────────────────
MIN_GP = 20          # minimum games for season inclusion
MIN_TOI_GAME = 2.0   # minimum 5v5 TOI per game (minutes)
MIN_TOI_SEASON = 100  # minimum 5v5 TOI per season (minutes)

# ── 1. Load data ─────────────────────────────────────────────────────────────
print("Loading data...", file=sys.stderr)

sbg = pd.read_csv("data/skaters_by_game.csv")
sbg = sbg.rename(columns={"playerId": "player_id", "name": "player_name", "gameId": "game_id"})

# 5v5 only, 2015+
ev = sbg[(sbg["situation"] == "5on5") & (sbg["season"] >= 2015)].copy()
ev["toi_min"] = ev["icetime"] / 60.0
ev = ev[ev["toi_min"] >= MIN_TOI_GAME].copy()
print(f"  5v5 player-games: {len(ev):,}", file=sys.stderr)

# All-situations for TOI breakdown
all_sit = sbg[sbg["situation"].isin(["5on5", "5on4", "4on5"])]

# ── 2. Player-season aggregation ─────────────────────────────────────────────
print("\nAggregating to player-seasons...", file=sys.stderr)


def safe_div(a, b, fill=0.0):
    return np.where(b > 0, a / b, fill)


# Aggregate 5v5 stats to player-season
agg_cols = {
    "toi_min": "sum",
    "game_id": "count",  # GP
    # Individual offense
    "I_F_xGoals": "sum", "I_F_goals": "sum",
    "I_F_primaryAssists": "sum", "I_F_secondaryAssists": "sum",
    "I_F_shotsOnGoal": "sum", "I_F_shotAttempts": "sum",
    "I_F_highDangerShots": "sum", "I_F_highDangerxGoals": "sum",
    "I_F_highDangerGoals": "sum",
    "I_F_mediumDangerShots": "sum", "I_F_mediumDangerxGoals": "sum",
    "I_F_lowDangerShots": "sum",
    "I_F_rebounds": "sum", "I_F_reboundGoals": "sum",
    "I_F_xGoals_with_earned_rebounds": "sum",
    "I_F_playContinuedInZone": "sum", "I_F_playContinuedOutsideZone": "sum",
    "I_F_points": "sum",
    "I_F_unblockedShotAttempts": "sum",
    # Individual defense
    "shotsBlockedByPlayer": "sum",
    "I_F_takeaways": "sum", "I_F_giveaways": "sum", "I_F_dZoneGiveaways": "sum",
    "I_F_hits": "sum",
    # Zone deployment
    "I_F_oZoneShiftStarts": "sum", "I_F_dZoneShiftStarts": "sum",
    "I_F_neutralZoneShiftStarts": "sum",
    "I_F_oZoneShiftEnds": "sum", "I_F_dZoneShiftEnds": "sum",
    # Faceoffs & penalties
    "faceoffsWon": "sum", "faceoffsLost": "sum",
    "penalties": "sum", "penaltiesDrawn": "sum",
    # On-ice FOR
    "OnIce_F_xGoals": "sum", "OnIce_F_goals": "sum",
    "OnIce_F_shotsOnGoal": "sum", "OnIce_F_shotAttempts": "sum",
    "OnIce_F_highDangerShots": "sum", "OnIce_F_highDangerxGoals": "sum",
    "OnIce_F_rebounds": "sum", "OnIce_F_unblockedShotAttempts": "sum",
    # On-ice AGAINST
    "OnIce_A_xGoals": "sum", "OnIce_A_goals": "sum",
    "OnIce_A_shotsOnGoal": "sum", "OnIce_A_shotAttempts": "sum",
    "OnIce_A_highDangerShots": "sum", "OnIce_A_highDangerxGoals": "sum",
    "OnIce_A_blockedShotAttempts": "sum", "OnIce_A_unblockedShotAttempts": "sum",
    # Off-ice
    "OffIce_F_xGoals": "sum", "OffIce_A_xGoals": "sum",
    # After-shift
    "xGoalsForAfterShifts": "sum", "xGoalsAgainstAfterShifts": "sum",
}

# Ensure columns exist before aggregating
agg_cols = {k: v for k, v in agg_cols.items() if k in ev.columns}

ps = ev.groupby(["player_id", "season"]).agg(agg_cols).reset_index()
ps = ps.rename(columns={"game_id": "GP"})

# Get player name and position from first appearance
name_pos = ev.groupby("player_id")[["player_name", "position"]].first().reset_index()
ps = ps.merge(name_pos, on="player_id", how="left")

# Filter: minimum GP and TOI
ps = ps[(ps["GP"] >= MIN_GP) & (ps["toi_min"] >= MIN_TOI_SEASON)].copy()
print(f"  Qualified player-seasons: {len(ps):,}", file=sys.stderr)

# Per-60 rate helper
toi60 = ps["toi_min"] / 60.0


# ═══════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING (shared features for both models)
# ═══════════════════════════════════════════════════════════════════════════
print("\nEngineering features...", file=sys.stderr)

# --- Offensive features ---
# Individual production rates (per 60 min)
ps["ixG_60"] = ps["I_F_xGoals"] / toi60
ps["iGoals_60"] = ps["I_F_goals"] / toi60
ps["iA1_60"] = ps["I_F_primaryAssists"] / toi60
ps["iA2_60"] = ps["I_F_secondaryAssists"] / toi60
ps["iSOG_60"] = ps["I_F_shotsOnGoal"] / toi60
ps["iSA_60"] = ps["I_F_shotAttempts"] / toi60
ps["iHDShots_60"] = ps["I_F_highDangerShots"] / toi60
ps["iHDxG_60"] = ps["I_F_highDangerxGoals"] / toi60
ps["iMDShots_60"] = ps["I_F_mediumDangerShots"] / toi60
ps["iPoints_60"] = ps["I_F_points"] / toi60
ps["iRebounds_60"] = ps["I_F_rebounds"] / toi60
ps["ixG_ER_60"] = ps["I_F_xGoals_with_earned_rebounds"] / toi60

# Primary points (goals + A1)
ps["iPrimPts_60"] = (ps["I_F_goals"] + ps["I_F_primaryAssists"]) / toi60

# Finishing talent (goals - xGoals per 60)
ps["finish_60"] = (ps["I_F_goals"] - ps["I_F_xGoals"]) / toi60

# Shot quality
ps["xG_per_shot"] = safe_div(ps["I_F_xGoals"], ps["I_F_shotAttempts"])
ps["HD_shot_pct"] = safe_div(ps["I_F_highDangerShots"], ps["I_F_shotAttempts"])
ps["shooting_pct"] = safe_div(ps["I_F_goals"], ps["I_F_shotsOnGoal"])

# Market share (individual / on-ice team total)
ps["ixG_share"] = safe_div(ps["I_F_xGoals"], ps["OnIce_F_xGoals"].clip(lower=0.01))
ps["iGoals_share"] = safe_div(ps["I_F_goals"], ps["OnIce_F_goals"].clip(lower=0.01))
ps["iSOG_share"] = safe_div(ps["I_F_shotsOnGoal"], ps["OnIce_F_shotsOnGoal"].clip(lower=1))
ps["iSA_share"] = safe_div(ps["I_F_shotAttempts"], ps["OnIce_F_shotAttempts"].clip(lower=1))
ps["iHD_share"] = safe_div(ps["I_F_highDangerShots"], ps["OnIce_F_highDangerShots"].clip(lower=0.01))
ps["iPts_share"] = safe_div(ps["I_F_points"], ps["OnIce_F_goals"].clip(lower=0.01))

# Rebound creation
ps["rebound_rate"] = safe_div(ps["I_F_rebounds"], ps["I_F_shotsOnGoal"].clip(lower=1))
ps["xG_reb_bonus_60"] = (ps["I_F_xGoals_with_earned_rebounds"] - ps["I_F_xGoals"]) / toi60

# Play continuation
play_cont = ps["I_F_playContinuedInZone"] + ps["I_F_playContinuedOutsideZone"]
ps["play_cont_zone_pct"] = safe_div(ps["I_F_playContinuedInZone"], play_cont.clip(lower=1))

# Zone deployment
total_shifts = (ps["I_F_oZoneShiftStarts"] + ps["I_F_dZoneShiftStarts"] +
                ps["I_F_neutralZoneShiftStarts"]).clip(lower=1)
ps["ozPct"] = ps["I_F_oZoneShiftStarts"] / total_shifts
ps["dzPct"] = ps["I_F_dZoneShiftStarts"] / total_shifts

# Penalties drawn
ps["penDrawn_60"] = ps["penaltiesDrawn"] / toi60

# Faceoffs
ps["foWon_60"] = ps["faceoffsWon"] / toi60
ps["foWon_pct"] = safe_div(ps["faceoffsWon"], (ps["faceoffsWon"] + ps["faceoffsLost"]).clip(lower=1))

# --- Defensive features ---
ps["blocks_60"] = ps["shotsBlockedByPlayer"] / toi60
ps["takes_60"] = ps["I_F_takeaways"] / toi60
ps["gives_60"] = ps["I_F_giveaways"] / toi60
ps["dzGives_60"] = ps["I_F_dZoneGiveaways"] / toi60
ps["hits_60"] = ps["I_F_hits"] / toi60
ps["penTaken_60"] = ps["penalties"] / toi60

# Puck management
ps["give_take_ratio"] = safe_div(ps["I_F_giveaways"], (ps["I_F_giveaways"] + ps["I_F_takeaways"]).clip(lower=1))
ps["dzGive_pct"] = safe_div(ps["I_F_dZoneGiveaways"], ps["I_F_giveaways"].clip(lower=1))
ps["blocks_share"] = safe_div(ps["shotsBlockedByPlayer"], ps["OnIce_A_blockedShotAttempts"].clip(lower=1))

# On-ice rates (per 60)
ps["oiXGF_60"] = ps["OnIce_F_xGoals"] / toi60
ps["oiXGA_60"] = ps["OnIce_A_xGoals"] / toi60
ps["oiGF_60"] = ps["OnIce_F_goals"] / toi60
ps["oiGA_60"] = ps["OnIce_A_goals"] / toi60
ps["oiSOGF_60"] = ps["OnIce_F_shotsOnGoal"] / toi60
ps["oiSOGA_60"] = ps["OnIce_A_shotsOnGoal"] / toi60
ps["oiHDxGF_60"] = ps["OnIce_F_highDangerxGoals"] / toi60
ps["oiHDxGA_60"] = ps["OnIce_A_highDangerxGoals"] / toi60

# Off-ice rates (per 60 of off-ice time, approximated)
# off-ice TOI ≈ total game time - on-ice time; use a rough 60-min game estimate
# Better: use off-ice totals directly as rates per player-toi (proxy)
ps["offXGF_60"] = ps["OffIce_F_xGoals"] / toi60
ps["offXGA_60"] = ps["OffIce_A_xGoals"] / toi60

# On/Off differentials
ps["xGF_on_off"] = ps["oiXGF_60"] - ps["offXGF_60"]
ps["xGA_on_off"] = ps["oiXGA_60"] - ps["offXGA_60"]

# After-shift impact
ps["xGF_after_60"] = ps["xGoalsForAfterShifts"] / toi60
ps["xGA_after_60"] = ps["xGoalsAgainstAfterShifts"] / toi60

# Position indicator
ps["isD"] = (ps["position"] == "D").astype(float)

# DZ shift end %
shift_ends = (ps["I_F_oZoneShiftEnds"] + ps["I_F_dZoneShiftEnds"]).clip(lower=1)
ps["dzEndPct"] = ps["I_F_dZoneShiftEnds"] / shift_ends

# Faceoff lost rate (defensive cost)
ps["foLost_60"] = ps["faceoffsLost"] / toi60


# ═══════════════════════════════════════════════════════════════════════════
# 4. MODEL 1 — GOAL VALUE (GV): box scores → marginal goal value
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Model 1: Goal Value ──", file=sys.stderr)
print("Regressing on-ice goals on individual box-score stats...", file=sys.stderr)

# Target: on-ice goal differential per 60 (= what actually happened when you played)
# Split into O and D targets
ps["target_GV_O"] = ps["oiGF_60"]
ps["target_GV_D"] = -ps["oiGA_60"]  # negative = fewer goals against = good defense

GV_O_FEATURES = [
    "ixG_60", "iGoals_60", "iA1_60", "iA2_60", "iSOG_60", "iSA_60",
    "iHDShots_60", "iHDxG_60", "iMDShots_60", "iPrimPts_60",
    "finish_60", "xG_per_shot", "HD_shot_pct", "shooting_pct",
    "ixG_share", "iGoals_share", "iSOG_share", "iSA_share", "iHD_share", "iPts_share",
    "iRebounds_60", "xG_reb_bonus_60", "rebound_rate",
    "play_cont_zone_pct",
    "ozPct", "dzPct",
    "penDrawn_60", "foWon_60", "foWon_pct",
    "isD",
]

GV_D_FEATURES = [
    "blocks_60", "takes_60", "gives_60", "dzGives_60", "hits_60",
    "penTaken_60",
    "give_take_ratio", "dzGive_pct", "blocks_share",
    "ozPct", "dzPct", "dzEndPct",
    "foLost_60", "foWon_pct",
    "xGF_after_60", "xGA_after_60",
    "isD",
]

X_GV_O = ps[GV_O_FEATURES].fillna(0).values
X_GV_D = ps[GV_D_FEATURES].fillna(0).values
y_GV_O = ps["target_GV_O"].values
y_GV_D = ps["target_GV_D"].values
w_gv = np.sqrt(ps["toi_min"].values / ps["toi_min"].median())

# Ridge regression with CV
from sklearn.preprocessing import StandardScaler

scaler_GV_O = StandardScaler()
X_GV_O_s = scaler_GV_O.fit_transform(X_GV_O)
model_GV_O = RidgeCV(alphas=np.logspace(-1, 3, 20), fit_intercept=True)
model_GV_O.fit(X_GV_O_s, y_GV_O, sample_weight=w_gv)
pred_GV_O = model_GV_O.predict(X_GV_O_s)
r2_GV_O = 1 - np.sum(w_gv * (y_GV_O - pred_GV_O)**2) / np.sum(w_gv * (y_GV_O - np.average(y_GV_O, weights=w_gv))**2)
print(f"  GV Offense R²: {r2_GV_O:.3f} (alpha={model_GV_O.alpha_:.1f})", file=sys.stderr)

scaler_GV_D = StandardScaler()
X_GV_D_s = scaler_GV_D.fit_transform(X_GV_D)
model_GV_D = RidgeCV(alphas=np.logspace(-1, 3, 20), fit_intercept=True)
model_GV_D.fit(X_GV_D_s, y_GV_D, sample_weight=w_gv)
pred_GV_D = model_GV_D.predict(X_GV_D_s)
r2_GV_D = 1 - np.sum(w_gv * (y_GV_D - pred_GV_D)**2) / np.sum(w_gv * (y_GV_D - np.average(y_GV_D, weights=w_gv))**2)
print(f"  GV Defense R²: {r2_GV_D:.3f} (alpha={model_GV_D.alpha_:.1f})", file=sys.stderr)

# Extract original-scale coefficients for per-player scoring
coef_GV_O = model_GV_O.coef_ / scaler_GV_O.scale_
int_GV_O = model_GV_O.intercept_ - np.sum(coef_GV_O * scaler_GV_O.mean_)
coef_GV_D = model_GV_D.coef_ / scaler_GV_D.scale_
int_GV_D = model_GV_D.intercept_ - np.sum(coef_GV_D * scaler_GV_D.mean_)

# Apply — GV score = prediction - league average → goals above average per 60
ps["GV_O_raw"] = (ps[GV_O_FEATURES].fillna(0).values @ coef_GV_O + int_GV_O)
ps["GV_D_raw"] = (ps[GV_D_FEATURES].fillna(0).values @ coef_GV_D + int_GV_D)

# Center at zero (above/below average)
ps["GV_O"] = np.round(ps["GV_O_raw"] - ps["GV_O_raw"].mean(), 4)
ps["GV_D"] = np.round(ps["GV_D_raw"] - ps["GV_D_raw"].mean(), 4)

# Top features
print(f"\n  Top GV_O features (original-scale coefs):", file=sys.stderr)
for feat, c in sorted(zip(GV_O_FEATURES, coef_GV_O), key=lambda x: abs(x[1]), reverse=True)[:10]:
    print(f"    {feat:25s} {c:+.4f}", file=sys.stderr)

print(f"\n  Top GV_D features:", file=sys.stderr)
for feat, c in sorted(zip(GV_D_FEATURES, coef_GV_D), key=lambda x: abs(x[1]), reverse=True)[:10]:
    print(f"    {feat:25s} {c:+.4f}", file=sys.stderr)


# ═══════════════════════════════════════════════════════════════════════════
# 5. MODEL 2 — ON/OFF IMPACT (OOI): box scores → on/off differential
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Model 2: On/Off Impact ──", file=sys.stderr)
print("Regressing on/off xGF differential on individual box-score stats...", file=sys.stderr)

# Target: on-ice xGF/60 minus off-ice xGF/60 (for offense)
# and off-ice xGA/60 minus on-ice xGA/60 (for defense — positive = good)
ps["target_OOI_O"] = ps["xGF_on_off"]
ps["target_OOI_D"] = -ps["xGA_on_off"]  # flip: positive = suppresses goals

# Use same feature sets
X_OOI_O = ps[GV_O_FEATURES].fillna(0).values
X_OOI_D = ps[GV_D_FEATURES].fillna(0).values
y_OOI_O = ps["target_OOI_O"].values
y_OOI_D = ps["target_OOI_D"].values

scaler_OOI_O = StandardScaler()
X_OOI_O_s = scaler_OOI_O.fit_transform(X_OOI_O)
model_OOI_O = RidgeCV(alphas=np.logspace(-1, 3, 20), fit_intercept=True)
model_OOI_O.fit(X_OOI_O_s, y_OOI_O, sample_weight=w_gv)
pred_OOI_O = model_OOI_O.predict(X_OOI_O_s)
r2_OOI_O = 1 - np.sum(w_gv * (y_OOI_O - pred_OOI_O)**2) / np.sum(w_gv * (y_OOI_O - np.average(y_OOI_O, weights=w_gv))**2)
print(f"  OOI Offense R²: {r2_OOI_O:.3f} (alpha={model_OOI_O.alpha_:.1f})", file=sys.stderr)

scaler_OOI_D = StandardScaler()
X_OOI_D_s = scaler_OOI_D.fit_transform(X_OOI_D)
model_OOI_D = RidgeCV(alphas=np.logspace(-1, 3, 20), fit_intercept=True)
model_OOI_D.fit(X_OOI_D_s, y_OOI_D, sample_weight=w_gv)
pred_OOI_D = model_OOI_D.predict(X_OOI_D_s)
r2_OOI_D = 1 - np.sum(w_gv * (y_OOI_D - pred_OOI_D)**2) / np.sum(w_gv * (y_OOI_D - np.average(y_OOI_D, weights=w_gv))**2)
print(f"  OOI Defense R²: {r2_OOI_D:.3f} (alpha={model_OOI_D.alpha_:.1f})", file=sys.stderr)

coef_OOI_O = model_OOI_O.coef_ / scaler_OOI_O.scale_
int_OOI_O = model_OOI_O.intercept_ - np.sum(coef_OOI_O * scaler_OOI_O.mean_)
coef_OOI_D = model_OOI_D.coef_ / scaler_OOI_D.scale_
int_OOI_D = model_OOI_D.intercept_ - np.sum(coef_OOI_D * scaler_OOI_D.mean_)

ps["OOI_O_raw"] = (ps[GV_O_FEATURES].fillna(0).values @ coef_OOI_O + int_OOI_O)
ps["OOI_D_raw"] = (ps[GV_D_FEATURES].fillna(0).values @ coef_OOI_D + int_OOI_D)
ps["OOI_O"] = np.round(ps["OOI_O_raw"] - ps["OOI_O_raw"].mean(), 4)
ps["OOI_D"] = np.round(ps["OOI_D_raw"] - ps["OOI_D_raw"].mean(), 4)

print(f"\n  Top OOI_O features:", file=sys.stderr)
for feat, c in sorted(zip(GV_O_FEATURES, coef_OOI_O), key=lambda x: abs(x[1]), reverse=True)[:10]:
    print(f"    {feat:25s} {c:+.4f}", file=sys.stderr)

print(f"\n  Top OOI_D features:", file=sys.stderr)
for feat, c in sorted(zip(GV_D_FEATURES, coef_OOI_D), key=lambda x: abs(x[1]), reverse=True)[:10]:
    print(f"    {feat:25s} {c:+.4f}", file=sys.stderr)


# ═══════════════════════════════════════════════════════════════════════════
# 6. MERGE IN RAPM + SPECIAL TEAMS
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Merging RAPM and special teams ──", file=sys.stderr)

# Rolling RAPM — match player to their best window for each season
rapm = pd.read_csv("data/v3_rolling_rapm.csv")
rapm["player_id"] = rapm["player_id"].astype(int)

# For each player-season, find the window that contains that season
rapm_matched = []
for _, row in ps[["player_id", "season"]].iterrows():
    pid, szn = int(row["player_id"]), int(row["season"])
    windows = rapm[(rapm["player_id"] == pid) &
                   (rapm["window_start"] <= szn) &
                   (rapm["window_end"] >= szn)]
    if len(windows) > 0:
        # Use the most recent window
        best = windows.sort_values("window_end", ascending=False).iloc[0]
        rapm_matched.append({
            "player_id": pid, "season": szn,
            "RAPM_O": best["BPR_O"], "RAPM_D": best["BPR_D"],
            "RAPM_O_se": best["BPR_O_se"], "RAPM_D_se": best["BPR_D_se"],
        })
    else:
        rapm_matched.append({
            "player_id": pid, "season": szn,
            "RAPM_O": np.nan, "RAPM_D": np.nan,
            "RAPM_O_se": np.nan, "RAPM_D_se": np.nan,
        })

rapm_df = pd.DataFrame(rapm_matched)
ps = ps.merge(rapm_df, on=["player_id", "season"], how="left")

# Fill missing RAPM with 0 (replacement level assumption)
ps["RAPM_O"] = ps["RAPM_O"].fillna(0)
ps["RAPM_D"] = ps["RAPM_D"].fillna(0)
ps["RAPM_O_se"] = ps["RAPM_O_se"].fillna(0.5)  # high uncertainty
ps["RAPM_D_se"] = ps["RAPM_D_se"].fillna(0.5)

print(f"  RAPM matched: {(~rapm_df['RAPM_O'].isna()).sum():,} / {len(rapm_df):,}", file=sys.stderr)

# Special teams
pp_rapm = pd.read_csv("data/pp_rapm.csv")[["player_id", "PP_O", "PK_D"]]
pp_rapm["player_id"] = pp_rapm["player_id"].astype(int)
ps = ps.merge(pp_rapm, on="player_id", how="left")
ps["PP_O"] = ps["PP_O"].fillna(0)
ps["PK_D"] = ps["PK_D"].fillna(0)

# TOI by situation
sit_toi = (
    all_sit.groupby(["player_id", "season", "situation"])["icetime"]
    .sum().unstack("situation", fill_value=0).reset_index()
)
sit_toi.columns.name = None
rename_map = {}
for c in sit_toi.columns:
    if c == "5on5": rename_map[c] = "toi_5v5_sec"
    elif c == "5on4": rename_map[c] = "toi_pp_sec"
    elif c == "4on5": rename_map[c] = "toi_pk_sec"
sit_toi = sit_toi.rename(columns=rename_map)
for col in ["toi_5v5_sec", "toi_pp_sec", "toi_pk_sec"]:
    if col not in sit_toi.columns:
        sit_toi[col] = 0
sit_toi["player_id"] = sit_toi["player_id"].astype(int)
sit_toi["toi_pp_min"] = (sit_toi["toi_pp_sec"] / 60).round(1)
sit_toi["toi_pk_min"] = (sit_toi["toi_pk_sec"] / 60).round(1)

ps = ps.merge(sit_toi[["player_id", "season", "toi_pp_min", "toi_pk_min"]],
              on=["player_id", "season"], how="left")
ps["toi_pp_min"] = ps["toi_pp_min"].fillna(0)
ps["toi_pk_min"] = ps["toi_pk_min"].fillna(0)


# ═══════════════════════════════════════════════════════════════════════════
# 7. OUTPUT: all components per player-season
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Building output ──", file=sys.stderr)

out = ps[[
    "player_id", "player_name", "position", "season", "GP", "toi_min",
    "toi_pp_min", "toi_pk_min",
    # Model 1: Goal Value
    "GV_O", "GV_D",
    # Model 2: On/Off Impact
    "OOI_O", "OOI_D",
    # RAPM
    "RAPM_O", "RAPM_D", "RAPM_O_se", "RAPM_D_se",
    # Special teams
    "PP_O", "PK_D",
    # Zone deployment / role context
    "ozPct", "dzPct",
]].copy()

# Round
for c in ["GV_O", "GV_D", "OOI_O", "OOI_D", "RAPM_O", "RAPM_D",
           "RAPM_O_se", "RAPM_D_se", "PP_O", "PK_D", "ozPct", "dzPct"]:
    out[c] = out[c].round(4)
out["toi_min"] = out["toi_min"].round(1)

out = out.sort_values(["season", "player_name"]).reset_index(drop=True)
out.to_csv("data/v4_bpm_player_seasons.csv", index=False)
print(f"  {len(out):,} player-seasons → data/v4_bpm_player_seasons.csv", file=sys.stderr)


# ═══════════════════════════════════════════════════════════════════════════
# 8. LEADERBOARDS
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Leaderboards ──", file=sys.stderr)

key_players = [
    "Connor McDavid", "Nikita Kucherov", "Auston Matthews", "Nathan MacKinnon",
    "Kirill Kaprizov", "Cale Makar", "Leon Draisaitl", "David Pastrnak",
    "Matthew Tkachuk", "Sidney Crosby", "Adam Fox", "Charlie McAvoy",
    "Mikko Rantanen", "Jack Hughes", "Miro Heiskanen", "Patrice Bergeron",
    "Sam Reinhart", "Aleksander Barkov", "Mark Stone",
]

for szn in sorted(out["season"].unique())[-2:]:
    s = out[out["season"] == szn].copy()
    nhl = f"{szn}-{str(szn+1)[-2:]}"

    # Rank by each metric
    s = s.sort_values("GV_O", ascending=False).reset_index(drop=True)
    s["GV_O_rank"] = s.index + 1
    s = s.sort_values("OOI_O", ascending=False).reset_index(drop=True)
    s["OOI_O_rank"] = s.index + 1
    s = s.sort_values("RAPM_O", ascending=False).reset_index(drop=True)
    s["RAPM_O_rank"] = s.index + 1

    print(f"\n{'='*100}", file=sys.stderr)
    print(f"  {nhl} — Key Players", file=sys.stderr)
    print(f"{'='*100}", file=sys.stderr)
    print(f"  {'Name':25s} {'Pos':3s} {'GP':>3s} {'GV_O':>7s} {'rk':>4s} {'OOI_O':>7s} {'rk':>4s} "
          f"{'RAPM_O':>7s} {'rk':>4s}  {'GV_D':>7s} {'OOI_D':>7s} {'RAPM_D':>7s}", file=sys.stderr)
    print(f"  {'-'*95}", file=sys.stderr)

    for name in key_players:
        row = s[s["player_name"] == name]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        print(f"  {name:25s} {r['position']:3s} {r['GP']:3.0f} "
              f"{r['GV_O']:+7.4f} {r['GV_O_rank']:4.0f} "
              f"{r['OOI_O']:+7.4f} {r['OOI_O_rank']:4.0f} "
              f"{r['RAPM_O']:+7.4f} {r['RAPM_O_rank']:4.0f}  "
              f"{r['GV_D']:+7.4f} {r['OOI_D']:+7.4f} {r['RAPM_D']:+7.4f}",
              file=sys.stderr)

    # Top 15 by GV_O
    print(f"\n  Top 15 GV_O ({nhl}):", file=sys.stderr)
    top = s.sort_values("GV_O", ascending=False).head(15)
    for _, r in top.iterrows():
        print(f"    {r['player_name']:25s} {r['position']:3s} GP={r['GP']:2.0f} "
              f"GV_O={r['GV_O']:+.4f} OOI_O={r['OOI_O']:+.4f} RAPM_O={r['RAPM_O']:+.4f}",
              file=sys.stderr)

    # Top 15 by OOI_O
    print(f"\n  Top 15 OOI_O ({nhl}):", file=sys.stderr)
    top = s.sort_values("OOI_O", ascending=False).head(15)
    for _, r in top.iterrows():
        print(f"    {r['player_name']:25s} {r['position']:3s} GP={r['GP']:2.0f} "
              f"GV_O={r['GV_O']:+.4f} OOI_O={r['OOI_O']:+.4f} RAPM_O={r['RAPM_O']:+.4f}",
              file=sys.stderr)

print("\n\nDone.", file=sys.stderr)
