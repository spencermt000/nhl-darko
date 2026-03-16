"""
epm.py — XGBoost-based EPM hockey metric with separate WAR_O and WAR_D.

Architecture (modeled on NBA EPM, adapted for hockey):
  Two separate XGBoost models trained on multi-representation box score features:

  OFFENSIVE MODEL (box scores → RAPM_O):
    XGBoost captures non-linear interactions (e.g. high ixG + high HD% → extra
    impact beyond their sum). CV R² ~0.28 vs Ridge's 0.20 — 38% improvement.
    Uses the MODEL PREDICTION as xGI_O.

  DEFENSIVE MODEL (box scores → RAPM_D):
    Same approach. CV R² ~0.34 vs Ridge's 0.30.

  Each stat in 3 representations: raw count, per-10-min rate, market share.
  The model prediction naturally regresses extreme RAPM toward box-score
  expectation.

The metric: xGI/10 (Expected Goals Impact per 10 minutes)
  Separate WAR_O and WAR_D components.

Inputs:
  data/skaters_by_game.csv           Per-game box scores (157 columns)
  data/v3_rolling_rapm.csv           Rolling 3-season RAPM (training target)

Outputs:
  data/v4_epm_model.json             Model config and feature importance
  data/v4_daily_epm.csv              Per-player-game smoothed xGI
  data/v4_season_war.csv             Season-aggregated WAR_O + WAR_D
"""

import sys
import json
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# ── Config ───────────────────────────────────────────────────────────────────
GOALS_TO_WINS = 6.0
RL_PERCENTILE = 17
BPR_SCALE = 1.0 / 6.0  # BPR (per 60) → xGI (per 10)

# Smoothing
DECAY_HALFLIFE = 30
GAME_EVIDENCE_SCALE = 1.5
PRIOR_SE_O = 0.08
PRIOR_SE_D = 0.05

MODEL_PATH_O = "data/v4_epm_model_O.pkl"
MODEL_PATH_D = "data/v4_epm_model_D.pkl"
MODEL_META_PATH = "data/v4_epm_features.json"

APPLY_ONLY = "--apply-only" in sys.argv

# ── 1. Load data ─────────────────────────────────────────────────────────────
print("Loading data...", file=sys.stderr)

sbg_cols = [
    "playerId", "name", "gameId", "season", "gameDate", "position", "situation",
    "icetime",
    # Individual offense
    "I_F_xGoals", "I_F_goals", "I_F_primaryAssists", "I_F_secondaryAssists",
    "I_F_shotsOnGoal", "I_F_missedShots", "I_F_shotAttempts",
    "I_F_highDangerShots", "I_F_highDangerxGoals", "I_F_highDangerGoals",
    "I_F_mediumDangerShots", "I_F_mediumDangerxGoals",
    "I_F_lowDangerShots",
    "I_F_rebounds", "I_F_reboundGoals",
    "I_F_xGoals_with_earned_rebounds",
    "I_F_playContinuedInZone", "I_F_playContinuedOutsideZone",
    "I_F_points",
    # Individual defense / other
    "shotsBlockedByPlayer", "I_F_takeaways", "I_F_giveaways", "I_F_dZoneGiveaways",
    "I_F_hits",
    # Zone deployment
    "I_F_oZoneShiftStarts", "I_F_dZoneShiftStarts", "I_F_neutralZoneShiftStarts",
    "I_F_oZoneShiftEnds", "I_F_dZoneShiftEnds",
    # Faceoffs & penalties
    "faceoffsWon", "faceoffsLost",
    "penalties", "penaltiesDrawn",
    # On-ice FOR
    "OnIce_F_xGoals", "OnIce_F_goals", "OnIce_F_shotsOnGoal", "OnIce_F_shotAttempts",
    "OnIce_F_highDangerShots", "OnIce_F_highDangerxGoals",
    "OnIce_F_rebounds", "OnIce_F_unblockedShotAttempts",
    # On-ice AGAINST
    "OnIce_A_xGoals", "OnIce_A_goals", "OnIce_A_shotsOnGoal", "OnIce_A_shotAttempts",
    "OnIce_A_highDangerShots", "OnIce_A_highDangerxGoals",
    "OnIce_A_blockedShotAttempts", "OnIce_A_unblockedShotAttempts",
    # Relative metrics
    "onIce_xGoalsPercentage", "offIce_xGoalsPercentage",
    "onIce_corsiPercentage", "offIce_corsiPercentage",
    "onIce_fenwickPercentage", "offIce_fenwickPercentage",
    # Off-ice
    "OffIce_F_xGoals", "OffIce_A_xGoals",
    # After shifts
    "xGoalsForAfterShifts", "xGoalsAgainstAfterShifts",
]

sbg = pd.read_csv("data/skaters_by_game.csv", usecols=sbg_cols)
sbg = sbg.rename(columns={"playerId": "player_id", "gameId": "game_id", "name": "player_name"})

sbg_5v5 = sbg[(sbg["situation"] == "5on5") & (sbg["season"] >= 2015)].copy()
sbg_5v5["toi_min"] = sbg_5v5["icetime"] / 60.0
sbg_5v5["game_date"] = pd.to_datetime(sbg_5v5["gameDate"].astype(str), format="%Y%m%d")
sbg_5v5 = sbg_5v5[sbg_5v5["toi_min"] >= 2.0].copy()

print(f"  5v5 player-games: {len(sbg_5v5):,}", file=sys.stderr)

# Rolling RAPM (training target — not needed in apply-only mode)
if not APPLY_ONLY:
    rolling = pd.read_csv("data/v3_rolling_rapm.csv")
    rolling["player_id"] = rolling["player_id"].astype(int)
    print(f"  Rolling RAPM: {len(rolling):,} player-windows", file=sys.stderr)


# ── 2. Feature engineering ──────────────────────────────────────────────────
print("\nEngineering features...", file=sys.stderr)

toi10 = sbg_5v5["toi_min"].clip(lower=2.0) / 10.0
df = sbg_5v5  # alias for brevity

# Helper: safe division
def sdiv(a, b, fill=0.0):
    return np.where(b > 0, a / b, fill)

# ═══════════════════════════════════════════════════════════════════════════
# OFFENSIVE FEATURES
# ═══════════════════════════════════════════════════════════════════════════

O_FEATURES = []

def add_feature(name, values):
    df[name] = values.astype(float)
    O_FEATURES.append(name)

# --- Shot generation (3 reps each) ---
for stat, col in [("ixG", "I_F_xGoals"), ("iGoals", "I_F_goals"),
                   ("iSOG", "I_F_shotsOnGoal"), ("iSA", "I_F_shotAttempts"),
                   ("iHDShots", "I_F_highDangerShots"), ("iHDxG", "I_F_highDangerxGoals"),
                   ("iMDShots", "I_F_mediumDangerShots")]:
    raw = df[col].fillna(0)
    add_feature(f"O_{stat}_raw", raw)
    add_feature(f"O_{stat}_rate", raw / toi10)
    # Market share: individual / on-ice team total (where applicable)
    if "xG" in stat or "xGoals" in col:
        add_feature(f"O_{stat}_share", sdiv(raw, df["OnIce_F_xGoals"].clip(lower=0.01)))
    elif "SOG" in stat:
        add_feature(f"O_{stat}_share", sdiv(raw, df["OnIce_F_shotsOnGoal"].clip(lower=1)))
    elif "SA" in stat and "HD" not in stat:
        add_feature(f"O_{stat}_share", sdiv(raw, df["OnIce_F_shotAttempts"].clip(lower=1)))
    elif "HDShots" in stat:
        add_feature(f"O_{stat}_share", sdiv(raw, df["OnIce_F_highDangerShots"].clip(lower=0.01)))

# --- Finishing talent ---
add_feature("O_finish_raw", df["I_F_goals"].fillna(0) - df["I_F_xGoals"].fillna(0))
add_feature("O_finish_rate", (df["I_F_goals"].fillna(0) - df["I_F_xGoals"].fillna(0)) / toi10)
add_feature("O_shooting_pct", sdiv(df["I_F_goals"].fillna(0), df["I_F_shotsOnGoal"].clip(lower=0.01)))
add_feature("O_HD_goals_rate", df["I_F_highDangerGoals"].fillna(0) / toi10)

# --- Shot quality ---
add_feature("O_xG_per_shot", sdiv(df["I_F_xGoals"].fillna(0), df["I_F_shotAttempts"].clip(lower=0.01)))
add_feature("O_HD_shot_pct", sdiv(df["I_F_highDangerShots"].fillna(0), df["I_F_shotAttempts"].clip(lower=0.01)))

# --- Playmaking (3 reps) ---
for stat, col in [("iA1", "I_F_primaryAssists"), ("iA2", "I_F_secondaryAssists")]:
    raw = df[col].fillna(0)
    add_feature(f"O_{stat}_raw", raw)
    add_feature(f"O_{stat}_rate", raw / toi10)
    add_feature(f"O_{stat}_share", sdiv(raw, df["OnIce_F_goals"].clip(lower=0.01)))

# Primary points (goals + A1) — strongest involvement signal
pp_raw = df["I_F_goals"].fillna(0) + df["I_F_primaryAssists"].fillna(0)
add_feature("O_primary_pts_raw", pp_raw)
add_feature("O_primary_pts_rate", pp_raw / toi10)
add_feature("O_primary_pts_share", sdiv(pp_raw, df["OnIce_F_goals"].clip(lower=0.01)))

# Total points
tp_raw = df["I_F_points"].fillna(0)
add_feature("O_points_raw", tp_raw)
add_feature("O_points_rate", tp_raw / toi10)
add_feature("O_points_share", sdiv(tp_raw, df["OnIce_F_goals"].clip(lower=0.01)))

# --- Rebound creation ---
add_feature("O_rebounds_raw", df["I_F_rebounds"].fillna(0))
add_feature("O_rebounds_rate", df["I_F_rebounds"].fillna(0) / toi10)
add_feature("O_rebound_goals_rate", df["I_F_reboundGoals"].fillna(0) / toi10)

# xG with earned rebounds (more complete offensive picture)
xg_er = df["I_F_xGoals_with_earned_rebounds"].fillna(0)
add_feature("O_xG_earned_reb_rate", xg_er / toi10)
add_feature("O_xG_earned_reb_bonus", (xg_er - df["I_F_xGoals"].fillna(0)) / toi10)

# --- Offensive zone play continuation ---
pic = df["I_F_playContinuedInZone"].fillna(0)
poc = df["I_F_playContinuedOutsideZone"].fillna(0)
add_feature("O_play_cont_zone_rate", pic / toi10)
add_feature("O_play_cont_zone_pct", sdiv(pic, pic + poc + df["I_F_goals"].fillna(0) + 0.01))

# --- Zone deployment ---
total_shifts = (df["I_F_oZoneShiftStarts"].fillna(0) +
                df["I_F_dZoneShiftStarts"].fillna(0) +
                df["I_F_neutralZoneShiftStarts"].fillna(0)).clip(lower=1)
add_feature("O_ozPct", df["I_F_oZoneShiftStarts"].fillna(0) / total_shifts)
add_feature("O_dzPct", df["I_F_dZoneShiftStarts"].fillna(0) / total_shifts)

# --- Penalties drawn (creates PP opportunities = offensive value) ---
add_feature("O_penDrawn_raw", df["penaltiesDrawn"].fillna(0))
add_feature("O_penDrawn_rate", df["penaltiesDrawn"].fillna(0) / toi10)

# --- On-ice relative metrics (marginal offensive impact) ---
add_feature("O_xGPct_rel", df["onIce_xGoalsPercentage"].fillna(50) - df["offIce_xGoalsPercentage"].fillna(50))
add_feature("O_corsi_rel", df["onIce_corsiPercentage"].fillna(50) - df["offIce_corsiPercentage"].fillna(50))
add_feature("O_fenwick_rel", df["onIce_fenwickPercentage"].fillna(50) - df["offIce_fenwickPercentage"].fillna(50))

# On-ice xGF rate and relative
add_feature("O_oiXGF_rate", df["OnIce_F_xGoals"].fillna(0) / toi10)
oiXGF_off = df["OffIce_F_xGoals"].fillna(0)
oi_toi = df["toi_min"].clip(lower=2)
# Off-ice rate approximation: total game xGF - on-ice xGF, over (60 - toi) min
# Simplified: use the percentage metrics instead
add_feature("O_oiHDxGF_rate", df["OnIce_F_highDangerxGoals"].fillna(0) / toi10)

# After-shift impact (do good things happen right after your shift?)
add_feature("O_xGF_after_shifts", df["xGoalsForAfterShifts"].fillna(0) / toi10)

# --- Position ---
add_feature("O_isD", (df["position"] == "D").astype(float))

# --- Faceoffs (offensive zone possession) ---
add_feature("O_foWon_rate", df["faceoffsWon"].fillna(0) / toi10)

print(f"  {len(O_FEATURES)} offensive features", file=sys.stderr)


# ═══════════════════════════════════════════════════════════════════════════
# DEFENSIVE FEATURES
# ═══════════════════════════════════════════════════════════════════════════

D_FEATURES = []

def add_d_feature(name, values):
    df[name] = values.astype(float)
    D_FEATURES.append(name)

# --- Blocked shots (3 reps) ---
blk = df["shotsBlockedByPlayer"].fillna(0)
add_d_feature("D_blocks_raw", blk)
add_d_feature("D_blocks_rate", blk / toi10)
add_d_feature("D_blocks_share", sdiv(blk, df["OnIce_A_blockedShotAttempts"].clip(lower=0.01)))

# --- Takeaways (3 reps) ---
takes = df["I_F_takeaways"].fillna(0)
add_d_feature("D_takes_raw", takes)
add_d_feature("D_takes_rate", takes / toi10)
add_d_feature("D_takes_per_oiSA", sdiv(takes, df["OnIce_A_shotAttempts"].clip(lower=1)))

# --- Giveaways / puck management ---
gives = df["I_F_giveaways"].fillna(0)
dz_gives = df["I_F_dZoneGiveaways"].fillna(0)
add_d_feature("D_gives_raw", gives)
add_d_feature("D_gives_rate", gives / toi10)
add_d_feature("D_dzGive_raw", dz_gives)
add_d_feature("D_dzGive_rate", dz_gives / toi10)
# Giveaway ratio: giveaways / (giveaways + takeaways) — puck management quality
add_d_feature("D_give_ratio", sdiv(gives, gives + takes + 0.01))
# DZ giveaway share of all giveaways
add_d_feature("D_dzGive_pct", sdiv(dz_gives, gives.clip(lower=0.01)))

# --- Hits (3 reps) ---
hits = df["I_F_hits"].fillna(0)
add_d_feature("D_hits_raw", hits)
add_d_feature("D_hits_rate", hits / toi10)

# --- On-ice shot/xG suppression ---
add_d_feature("D_oiXGA_rate", df["OnIce_A_xGoals"].fillna(0) / toi10)
add_d_feature("D_oiSOGA_rate", df["OnIce_A_shotsOnGoal"].fillna(0) / toi10)
add_d_feature("D_oiSAA_rate", df["OnIce_A_shotAttempts"].fillna(0) / toi10)
add_d_feature("D_oiHDxGA_rate", df["OnIce_A_highDangerxGoals"].fillna(0) / toi10)
add_d_feature("D_oiHDSA_rate", df["OnIce_A_highDangerShots"].fillna(0) / toi10)
add_d_feature("D_oiGA_rate", df["OnIce_A_goals"].fillna(0) / toi10)

# Relative suppression (on-ice vs off-ice)
add_d_feature("D_xGPct_rel", df["onIce_xGoalsPercentage"].fillna(50) - df["offIce_xGoalsPercentage"].fillna(50))
add_d_feature("D_corsi_rel", df["onIce_corsiPercentage"].fillna(50) - df["offIce_corsiPercentage"].fillna(50))

# HD share of opponent shots (lower = better — forcing low-danger attempts)
add_d_feature("D_opp_HD_pct", sdiv(
    df["OnIce_A_highDangerShots"].fillna(0),
    df["OnIce_A_shotAttempts"].clip(lower=1)
))

# After-shift impact (defensive)
add_d_feature("D_xGA_after_shifts", df["xGoalsAgainstAfterShifts"].fillna(0) / toi10)

# --- Zone deployment ---
add_d_feature("D_ozPct", df["I_F_oZoneShiftStarts"].fillna(0) / total_shifts)
add_d_feature("D_dzPct", df["I_F_dZoneShiftStarts"].fillna(0) / total_shifts)

# D-zone shift end % (how often shifts end in your own zone — bad)
total_shift_ends = (df["I_F_oZoneShiftEnds"].fillna(0) +
                    df["I_F_dZoneShiftEnds"].fillna(0) + 0.01)
add_d_feature("D_dzEndPct", df["I_F_dZoneShiftEnds"].fillna(0) / total_shift_ends)

# --- Penalties taken (defensive liability) ---
add_d_feature("D_penTaken_raw", df["penalties"].fillna(0))
add_d_feature("D_penTaken_rate", df["penalties"].fillna(0) / toi10)

# --- Faceoffs (defensive zone possession) ---
add_d_feature("D_foLost_rate", df["faceoffsLost"].fillna(0) / toi10)

# --- Position ---
add_d_feature("D_isD", (df["position"] == "D").astype(float))

print(f"  {len(D_FEATURES)} defensive features", file=sys.stderr)
print(f"  Total: {len(O_FEATURES) + len(D_FEATURES)} features", file=sys.stderr)


if APPLY_ONLY:
    # ── Load saved models ────────────────────────────────────────────────────
    print("\n[--apply-only] Loading saved XGBoost models...", file=sys.stderr)
    with open(MODEL_PATH_O, "rb") as f:
        model_O = pickle.load(f)
    with open(MODEL_PATH_D, "rb") as f:
        model_D = pickle.load(f)
    print("  Models loaded.", file=sys.stderr)

else:
    # ── 3. Build training set ───────────────────────────────────────────────────
    print("\nBuilding training set (player-window averages → RAPM)...", file=sys.stderr)

    ALL_FEATURES = O_FEATURES + D_FEATURES
    train_rows = []

    for _, rrow in rolling.iterrows():
        pid = rrow["player_id"]
        w_start = rrow["window_start"]
        w_end = rrow["window_end"]

        mask = (
            (df["player_id"] == pid) &
            (df["season"] >= w_start) &
            (df["season"] <= w_end)
        )
        pg = df.loc[mask, ALL_FEATURES + ["toi_min"]]

        if len(pg) < 20:
            continue

        weights = pg["toi_min"].values
        total_w = weights.sum()
        if total_w <= 0:
            continue

        feat_avg = {}
        for col in ALL_FEATURES:
            feat_avg[col] = float(np.average(pg[col].fillna(0).values, weights=weights))

        feat_avg["player_id"] = pid
        feat_avg["n_games"] = len(pg)
        feat_avg["BPR_O_target"] = rrow["BPR_O"]
        feat_avg["BPR_D_target"] = rrow["BPR_D"]
        train_rows.append(feat_avg)

    train_df = pd.DataFrame(train_rows)
    print(f"  {len(train_df):,} training samples", file=sys.stderr)


    # ── 4. Train offensive model (XGBoost) ──────────────────────────────────────
    print("\nTraining OFFENSIVE model (XGBoost: box scores → RAPM_O)...", file=sys.stderr)

    X_O = train_df[O_FEATURES].values
    y_O = train_df["BPR_O_target"].values
    sw = np.sqrt(train_df["n_games"].values)

    model_O = XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.6,
        reg_alpha=3, reg_lambda=15,
        random_state=42, n_jobs=-1,
    )
    model_O.fit(X_O, y_O, sample_weight=sw)
    pred_O = model_O.predict(X_O)
    r2_O = 1 - np.sum((y_O - pred_O) ** 2) / np.sum((y_O - np.mean(y_O)) ** 2)
    print(f"  R² offensive (train): {r2_O:.3f}", file=sys.stderr)

    # Feature importance
    feat_imp_O = model_O.feature_importances_
    imp_order_O = sorted(zip(O_FEATURES, feat_imp_O), key=lambda x: x[1], reverse=True)
    print(f"\n  Top 15 offensive features (importance):", file=sys.stderr)
    for feat, imp in imp_order_O[:15]:
        print(f"    {feat:30s} {imp:.4f}", file=sys.stderr)


    # ── 5. Train defensive model (XGBoost) ──────────────────────────────────────
    print("\nTraining DEFENSIVE model (XGBoost: box scores → RAPM_D)...", file=sys.stderr)

    X_D = train_df[D_FEATURES].values
    y_D = train_df["BPR_D_target"].values

    model_D = XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.6,
        reg_alpha=3, reg_lambda=15,
        random_state=42, n_jobs=-1,
    )
    model_D.fit(X_D, y_D, sample_weight=sw)
    pred_D = model_D.predict(X_D)
    r2_D = 1 - np.sum((y_D - pred_D) ** 2) / np.sum((y_D - np.mean(y_D)) ** 2)
    print(f"  R² defensive (train): {r2_D:.3f}", file=sys.stderr)

    feat_imp_D = model_D.feature_importances_
    imp_order_D = sorted(zip(D_FEATURES, feat_imp_D), key=lambda x: x[1], reverse=True)
    print(f"\n  Top 15 defensive features (importance):", file=sys.stderr)
    for feat, imp in imp_order_D[:15]:
        print(f"    {feat:30s} {imp:.4f}", file=sys.stderr)

    # Save models for --apply-only reuse
    with open(MODEL_PATH_O, "wb") as f:
        pickle.dump(model_O, f)
    with open(MODEL_PATH_D, "wb") as f:
        pickle.dump(model_D, f)
    with open(MODEL_META_PATH, "w") as f:
        json.dump({"O_FEATURES": O_FEATURES, "D_FEATURES": D_FEATURES}, f)
    print("\n  Models saved to disk.", file=sys.stderr)


# ── 6. Apply models to each game ────────────────────────────────────────────
print("\nApplying models to each game...", file=sys.stderr)

X_all_O = df[O_FEATURES].fillna(0).values
X_all_D = df[D_FEATURES].fillna(0).values

df["xGI_O_raw"] = np.round(model_O.predict(X_all_O) * BPR_SCALE, 4)
df["xGI_D_raw"] = np.round(model_D.predict(X_all_D) * BPR_SCALE, 4)
df["xGI_raw"] = (df["xGI_O_raw"] + df["xGI_D_raw"]).round(4)

print(f"  xGI_O: mean={df['xGI_O_raw'].mean():.4f}, std={df['xGI_O_raw'].std():.4f}", file=sys.stderr)
print(f"  xGI_D: mean={df['xGI_D_raw'].mean():.4f}, std={df['xGI_D_raw'].std():.4f}", file=sys.stderr)

# Save raw per-game predictions (used by daily.py for Bayesian smoothing)
raw_out = df[["player_id", "player_name", "position", "game_id", "season",
              "game_date", "toi_min", "xGI_O_raw", "xGI_D_raw"]].copy()
raw_out.to_csv("data/v4_epm_raw_per_game.csv", index=False)
print(f"  Saved {len(raw_out):,} rows → data/v4_epm_raw_per_game.csv", file=sys.stderr)

if not APPLY_ONLY:
    # Save model info
    model_info = {
        "architecture": "Separate O/D XGBoost models: box scores → RAPM (NBA EPM style)",
        "r2_O": round(r2_O, 4),
        "r2_D": round(r2_D, 4),
        "n_features_O": len(O_FEATURES),
        "n_features_D": len(D_FEATURES),
        "top_O_features": {f: round(float(c), 4) for f, c in imp_order_O[:10]},
        "top_D_features": {f: round(float(c), 4) for f, c in imp_order_D[:10]},
    }
    with open("data/v4_epm_model.json", "w") as f:
        json.dump(model_info, f, indent=2)


# ── 7. Bayesian smoothing (separate O and D) ────────────────────────────────
print("\nApplying Bayesian smoothing...", file=sys.stderr)

decay_rate = np.log(2) / DECAY_HALFLIFE
prior_prec_O = 1.0 / (PRIOR_SE_O ** 2)
prior_prec_D = 1.0 / (PRIOR_SE_D ** 2)

player_prior_O = (
    df.groupby("player_id")
    .apply(lambda g: np.average(g["xGI_O_raw"].values, weights=g["toi_min"].values), include_groups=False)
    .to_dict()
)
player_prior_D = (
    df.groupby("player_id")
    .apply(lambda g: np.average(g["xGI_D_raw"].values, weights=g["toi_min"].values), include_groups=False)
    .to_dict()
)

df = df.sort_values(["player_id", "game_date", "game_id"]).reset_index(drop=True)

smoothed = []
player_ids = df["player_id"].unique()

for pi, pid in enumerate(player_ids):
    if pi % 500 == 0 and pi > 0:
        print(f"  smoothed {pi:,} / {len(player_ids):,} players...", file=sys.stderr)

    pdf = df[df["player_id"] == pid].reset_index(drop=True)

    pr_o = player_prior_O.get(pid, 0.0)
    pr_d = player_prior_D.get(pid, 0.0)
    pname = pdf.iloc[0]["player_name"]
    pos = pdf.iloc[0]["position"]

    sw_o = 0.0
    sw_d = 0.0
    sum_wo = 0.0
    sum_wd = 0.0

    for gi in range(len(pdf)):
        row = pdf.iloc[gi]

        if gi > 0:
            date_gap = (row["game_date"] - pdf.iloc[gi - 1]["game_date"]).days
            game_gap = max(date_gap / 2.0, 1.0)
            decay = np.exp(-decay_rate * game_gap)
        else:
            decay = 1.0

        sw_o *= decay
        sw_d *= decay
        sum_wo *= decay
        sum_wd *= decay

        ev_w = GAME_EVIDENCE_SCALE * (row["toi_min"] / 15.0)
        sw_o += ev_w
        sw_d += ev_w
        sum_wo += ev_w * row["xGI_O_raw"]
        sum_wd += ev_w * row["xGI_D_raw"]

        tp_o = prior_prec_O + sw_o
        tp_d = prior_prec_D + sw_d
        post_o = (prior_prec_O * pr_o + sum_wo) / tp_o
        post_d = (prior_prec_D * pr_d + sum_wd) / tp_d

        smoothed.append({
            "player_id": pid,
            "player_name": pname,
            "position": pos,
            "game_id": row["game_id"],
            "season": row["season"],
            "game_date": row["game_date"],
            "game_number": gi + 1,
            "toi_min": round(row["toi_min"], 1),
            "xGI_O": round(post_o, 4),
            "xGI_D": round(post_d, 4),
            "xGI": round(post_o + post_d, 4),
            "xGI_O_se": round(np.sqrt(1.0 / tp_o), 4),
            "xGI_D_se": round(np.sqrt(1.0 / tp_d), 4),
        })

daily = pd.DataFrame(smoothed)
daily.to_csv("data/v4_daily_epm.csv", index=False)
print(f"\n  {len(daily):,} rows → data/v4_daily_epm.csv", file=sys.stderr)


# ── 8. Season-aggregated WAR_O + WAR_D ──────────────────────────────────────
print("\nComputing season WAR_O + WAR_D...", file=sys.stderr)

season_last = (
    daily.sort_values(["player_id", "season", "game_date"])
    .groupby(["player_id", "season"])
    .last()
    .reset_index()
)

# TOI
sit = pd.read_csv("data/skaters_by_game.csv", usecols=["playerId", "season", "situation", "icetime"])
sit = sit.rename(columns={"playerId": "player_id"})
sit_toi = (
    sit[sit["situation"].isin(["5on5", "5on4", "4on5"])]
    .groupby(["player_id", "season", "situation"])["icetime"]
    .sum().unstack("situation", fill_value=0).reset_index()
)
sit_toi.columns.name = None
sit_toi = sit_toi.rename(columns={"5on5": "toi_5v5", "5on4": "toi_pp", "4on5": "toi_pk"})
for col in ["toi_5v5", "toi_pp", "toi_pk"]:
    if col in sit_toi.columns:
        sit_toi[col] = (sit_toi[col] / 60).round(1)
sit_toi["player_id"] = sit_toi["player_id"].astype(int)

pp_rapm = pd.read_csv("data/pp_rapm.csv")[["player_id", "PP_O", "PK_D"]]
pp_rapm["player_id"] = pp_rapm["player_id"].astype(int)

sw = season_last.merge(sit_toi, on=["player_id", "season"], how="left")
sw = sw.merge(pp_rapm, on="player_id", how="left")

toi_5v5 = sw["toi_5v5"].fillna(0).values
toi_pp = sw["toi_pp"].fillna(0).values
toi_pk = sw["toi_pk"].fillna(0).values

sw["EV_O_GAR"] = (sw["xGI_O"] * toi_5v5 / 10).round(2)
sw["EV_D_GAR"] = (sw["xGI_D"] * toi_5v5 / 10).round(2)
sw["PP_GAR"] = (sw["PP_O"].fillna(0) * toi_pp / 60).round(2)
sw["PK_GAR"] = (sw["PK_D"].fillna(0) * toi_pk / 60).round(2)

sw["GAR_O"] = (sw["EV_O_GAR"] + sw["PP_GAR"]).round(2)
sw["GAR_D"] = (sw["EV_D_GAR"] + sw["PK_GAR"]).round(2)
sw["GAR_above_avg"] = (sw["GAR_O"] + sw["GAR_D"]).round(2)

total_toi = toi_5v5 + toi_pp + toi_pk
qualified = toi_5v5 >= 100

if qualified.sum() > 0:
    per60_O = sw.loc[qualified, "GAR_O"].values / (total_toi[qualified] / 60)
    per60_D = sw.loc[qualified, "GAR_D"].values / (total_toi[qualified] / 60)
    rl_O = float(np.percentile(per60_O, RL_PERCENTILE))
    rl_D = float(np.percentile(per60_D, RL_PERCENTILE))
    print(f"  Replacement level O: {rl_O:.4f} per 60", file=sys.stderr)
    print(f"  Replacement level D: {rl_D:.4f} per 60", file=sys.stderr)
else:
    rl_O = -0.10
    rl_D = -0.05

sw["WAR_O"] = ((sw["GAR_O"] - rl_O * total_toi / 60) / GOALS_TO_WINS).round(2)
sw["WAR_D"] = ((sw["GAR_D"] - rl_D * total_toi / 60) / GOALS_TO_WINS).round(2)
sw["WAR"] = (sw["WAR_O"] + sw["WAR_D"]).round(2)

gp = daily.groupby(["player_id", "season"]).size().reset_index(name="GP")
sw = sw.merge(gp, on=["player_id", "season"], how="left")

out_cols = [
    "player_id", "player_name", "position", "season", "GP",
    "toi_5v5", "toi_pp", "toi_pk",
    "xGI_O", "xGI_D", "xGI", "xGI_O_se", "xGI_D_se",
    "EV_O_GAR", "EV_D_GAR", "PP_GAR", "PK_GAR",
    "GAR_O", "GAR_D", "WAR_O", "WAR_D", "WAR",
]
out_cols = [c for c in out_cols if c in sw.columns]
sw_out = sw[out_cols].sort_values(["season", "WAR"], ascending=[True, False])
sw_out.to_csv("data/v4_season_war.csv", index=False)
print(f"  {len(sw_out):,} player-seasons → data/v4_season_war.csv", file=sys.stderr)

# Leaderboards
for szn in [2023, 2024]:
    s = sw_out[sw_out["season"] == szn].sort_values("WAR", ascending=False)
    if len(s) == 0:
        continue
    nhl = f"{szn}-{str(szn+1)[-2:]}"
    print(f"\nTop 25 WAR ({nhl}):", file=sys.stderr)
    show = ["player_name", "position", "GP", "xGI_O", "xGI_D", "xGI",
            "WAR_O", "WAR_D", "WAR"]
    show = [c for c in show if c in s.columns]
    print(s.head(25)[show].to_string(index=False), file=sys.stderr)

# Key players
latest_szn = sw_out["season"].max()
print(f"\n\nKey players ({latest_szn}-{str(latest_szn+1)[-2:]}):", file=sys.stderr)
sl = sw_out[sw_out["season"] == latest_szn].sort_values("WAR", ascending=False).reset_index(drop=True)
sl["rank"] = sl.index + 1
for name in ["Connor McDavid", "Nikita Kucherov", "Auston Matthews", "Nathan MacKinnon",
             "Kirill Kaprizov", "Cale Makar", "Leon Draisaitl", "David Pastrnak",
             "Matthew Tkachuk", "Sidney Crosby", "Adam Fox", "Charlie McAvoy",
             "Mikko Rantanen", "Jack Hughes", "Miro Heiskanen", "Patrice Bergeron"]:
    row = sl[sl["player_name"] == name]
    if len(row):
        r = row.iloc[0]
        print(f"  {r['rank']:3d}. {name:25s} xGI_O={r['xGI_O']:+.4f} xGI_D={r['xGI_D']:+.4f} "
              f"WAR_O={r['WAR_O']:+.2f} WAR_D={r['WAR_D']:+.2f} WAR={r['WAR']:.2f}", file=sys.stderr)

print("\nDone.", file=sys.stderr)
