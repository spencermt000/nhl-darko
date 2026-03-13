"""
blend.py — Uncertainty-weighted blend of RAPM + box score ratings.

DARKO-style: two independent signal sources combined by sample confidence.
  - RAPM weight increases with events on ice (more data → trust RAPM more)
  - Box score fills the gap for small-sample / new players
  - PP/PK contributions added via TOI-weighted situational blend

Inputs:
  data/rapm_results.csv          Pooled RAPM (1,999 skaters)
  data/rapm_by_season.csv        Per-season RAPM
  data/box_score_ratings.csv     Box score model output
  data/pp_rapm.csv               PP_O / PK_D ratings (pooled)
  data/skaters_by_game.csv       Situational TOI (5on4, 4on5 per player/season)

Outputs:
  data/final_ratings.csv         Pooled blended ratings (with PP/PK component)
  data/final_ratings_by_season.csv  Per-season blended ratings (with PP/PK component)
"""

import numpy as np
import pandas as pd

RAPM_POOLED    = "data/rapm_results.csv"
RAPM_SEASON    = "data/rapm_by_season.csv"
BOX_SCORE      = "data/box_score_ratings.csv"
PP_RAPM        = "data/pp_rapm.csv"
SKATERS_GAME   = "data/skaters_by_game.csv"
OUT_POOLED     = "data/final_ratings.csv"
OUT_SEASON     = "data/final_ratings_by_season.csv"

# Sigmoid blend on TOI (minutes on ice) — directly measures sample size.
# More ice time → trust RAPM more; less ice time → lean on box score.
#
# Per-season TOI (single season):
#   200 min → weight ≈ 0.14  (spot-starter, lean box score)
#   700 min → weight = 0.50  (midpoint: half a starter season)
#  1400 min → weight ≈ 0.97  (full-time star, trust RAPM)
SEASON_TOI_MIDPOINT = 700.0   # minutes at which RAPM weight = 0.50
SEASON_TOI_SCALE    = 0.005   # steepness of transition
#
# Pooled career TOI (sum across all seasons):
#   500 min  → weight ≈ 0.22  (cup-of-coffee career, lean box score)
#  2000 min  → weight = 0.50  (midpoint: ~2 solid seasons)
#  7000 min  → weight ≈ 0.99  (established vet, trust RAPM)
POOLED_TOI_MIDPOINT = 2000.0
POOLED_TOI_SCALE    = 0.001

def rapm_weight(toi, midpoint, scale):
    """Sigmoid over TOI (min): more ice time → higher RAPM weight."""
    return 1.0 / (1.0 + np.exp(-scale * (toi - midpoint)))

def blend(rapm_val, box_val, w):
    """Weighted blend; handles NaN gracefully."""
    rapm_ok = ~np.isnan(rapm_val)
    box_ok  = ~np.isnan(box_val)
    result  = np.full(len(rapm_val), np.nan)
    # Both available: blend
    both = rapm_ok & box_ok
    result[both] = w[both] * rapm_val[both] + (1 - w[both]) * box_val[both]
    # Only RAPM
    only_rapm = rapm_ok & ~box_ok
    result[only_rapm] = rapm_val[only_rapm]
    # Only box score
    only_box = ~rapm_ok & box_ok
    result[only_box] = box_val[only_box]
    return result


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading ratings...")
rapm_p = pd.read_csv(RAPM_POOLED)
rapm_s = pd.read_csv(RAPM_SEASON)
box    = pd.read_csv(BOX_SCORE)
pp     = pd.read_csv(PP_RAPM)[["player_id", "PP_O", "PK_D", "PP_BPR"]]
box["player_id"] = box["player_id"].astype(int)
rapm_p["player_id"] = rapm_p["player_id"].astype(int)
rapm_s["player_id"] = rapm_s["player_id"].astype(int)
pp["player_id"] = pp["player_id"].astype(int)

# Situational TOI: PP (5on4) and PK (4on5) minutes per player per season
sit = pd.read_csv(SKATERS_GAME, usecols=["playerId", "season", "situation", "icetime"])
sit = sit.rename(columns={"playerId": "player_id"})
sit_toi = (
    sit[sit["situation"].isin(["5on5", "5on4", "4on5"])]
    .groupby(["player_id", "season", "situation"])["icetime"]
    .sum()
    .unstack("situation", fill_value=0)
    .reset_index()
)
sit_toi.columns.name = None
sit_toi = sit_toi.rename(columns={"5on5": "toi_5v5", "5on4": "toi_pp", "4on5": "toi_pk"})
# Convert seconds → minutes
for col in ["toi_5v5", "toi_pp", "toi_pk"]:
    sit_toi[col] = (sit_toi[col] / 60).round(1)
sit_toi["player_id"] = sit_toi["player_id"].astype(int)

# Pooled situational TOI (sum across seasons)
sit_pooled = sit_toi.groupby("player_id")[["toi_5v5", "toi_pp", "toi_pk"]].sum().reset_index()

print(f"  RAPM pooled: {len(rapm_p):,} players")
print(f"  RAPM by-season: {len(rapm_s):,} player-seasons")
print(f"  Box score: {len(box):,} player-seasons")
print(f"  PP/PK RAPM: {len(pp):,} players")

# ── League-average situational weights (deployment-neutral) ──────────────────
# Use total skater-minutes in each situation across all players and seasons.
# These fixed weights let every player be compared on equal footing regardless
# of how their coach deploys them.
_lg = sit_pooled[["toi_5v5", "toi_pp", "toi_pk"]].sum()
_lg_total = _lg.sum()
W_ES = float(_lg["toi_5v5"] / _lg_total)
W_PP = float(_lg["toi_pp"]  / _lg_total)
W_PK = float(_lg["toi_pk"]  / _lg_total)
print(f"\nLeague-average TOI weights: ES={W_ES:.3f}  PP={W_PP:.3f}  PK={W_PK:.3f}")

# ── Pooled blend ──────────────────────────────────────────────────────────────
# For pooled, box score is aggregated across all seasons per player
box_agg = (
    box.sort_values("season")
    .groupby("player_id")
    .agg(
        box_O=("box_O", "mean"),
        box_D=("box_D", "mean"),
        box_BPR=("box_BPR", "mean"),
        toi=("toi", "sum"),
        position=("position", "last"),
    )
    .reset_index()
)

pooled = rapm_p.merge(box_agg[["player_id", "box_O", "box_D", "box_BPR", "toi"]], on="player_id", how="left")
pooled = pooled.merge(sit_pooled, on="player_id", how="left")
pooled = pooled.merge(pp, on="player_id", how="left")

pooled["rapm_w"] = rapm_weight(pooled["toi"].fillna(0).values, POOLED_TOI_MIDPOINT, POOLED_TOI_SCALE)

pooled["final_BPR_O"] = blend(pooled["BPR_O"].values, pooled["box_O"].values, pooled["rapm_w"].values).round(4)
pooled["final_BPR_D"] = blend(pooled["BPR_D"].values, pooled["box_D"].values, pooled["rapm_w"].values).round(4)
pooled["final_BPR"]   = (pooled["final_BPR_O"] + pooled["final_BPR_D"]).round(4)
pooled["rapm_weight"] = pooled["rapm_w"].round(3)

# ── PP/PK situational blend ───────────────────────────────────────────────────
# total_BPR = (toi_5v5 * 5v5_BPR + toi_pp * PP_O + toi_pk * PK_D) / total_toi
# Players without PP/PK data fall back to final_BPR (5v5-only)
t5  = pooled["toi_5v5"].fillna(0).values
tpp = pooled["toi_pp"].fillna(0).values
tpk = pooled["toi_pk"].fillna(0).values
ttotal = t5 + tpp + tpk

has_pp = ~pooled["PP_O"].isna().values
bpr5   = pooled["final_BPR"].values
pp_o   = pooled["PP_O"].fillna(0).values
pk_d   = pooled["PK_D"].fillna(0).values

safe_total = np.where(ttotal > 0, ttotal, 1.0)
total_bpr = np.where(
    has_pp & (ttotal > 0),
    (t5 * bpr5 + tpp * pp_o + tpk * pk_d) / safe_total,
    bpr5,
)
pooled["total_BPR"] = total_bpr.round(4)
pooled["pp_toi_frac"] = (tpp / safe_total * (ttotal > 0)).round(3)
pooled["pk_toi_frac"] = (tpk / safe_total * (ttotal > 0)).round(3)

# Deployment-neutral: league-average fixed weights
pooled["total_BPR_adj"] = np.where(
    has_pp,
    (W_ES * bpr5 + W_PP * pp_o + W_PK * pk_d).round(4),
    bpr5,
)

pooled.to_csv(OUT_POOLED, index=False)
print(f"\nPooled: {len(pooled):,} players → {OUT_POOLED}")
print("\nrapm_weight distribution (pooled):")
print(pooled["rapm_weight"].describe().round(3).to_string())
print("\nTop 20 (total_BPR_adj — deployment-neutral):")
cols_adj = ["player_name", "position", "toi", "final_BPR", "PP_O", "PK_D", "total_BPR", "total_BPR_adj"]
print(pooled.sort_values("total_BPR_adj", ascending=False).head(20)[cols_adj].to_string(index=False))
print("\nTop 20 (total_BPR — individual TOI weights):")
print(pooled.sort_values("total_BPR", ascending=False).head(20)[cols_adj].to_string(index=False))

# ── Per-season blend ──────────────────────────────────────────────────────────
season = rapm_s.merge(
    box[["player_id", "season", "box_O", "box_D", "box_BPR", "toi"]],
    on=["player_id", "season"], how="left",
)
# Per-season situational TOI; PP/PK RAPM is pooled (no per-season PP model)
season = season.merge(sit_toi, on=["player_id", "season"], how="left")
season = season.merge(pp, on="player_id", how="left")

season["rapm_w"] = rapm_weight(season["toi"].fillna(0).values, SEASON_TOI_MIDPOINT, SEASON_TOI_SCALE)

season["final_BPR_O"] = blend(season["BPR_O"].values, season["box_O"].values, season["rapm_w"].values).round(4)
season["final_BPR_D"] = blend(season["BPR_D"].values, season["box_D"].values, season["rapm_w"].values).round(4)
season["final_BPR"]   = (season["final_BPR_O"] + season["final_BPR_D"]).round(4)
season["rapm_weight"] = season["rapm_w"].round(3)

# TOI-weighted total BPR (5v5 + PP + PK)
t5s  = season["toi_5v5"].fillna(0).values
tpps = season["toi_pp"].fillna(0).values
tpks = season["toi_pk"].fillna(0).values
ttots = t5s + tpps + tpks

has_pps = ~season["PP_O"].isna().values
bpr5s   = season["final_BPR"].values
pp_os   = season["PP_O"].fillna(0).values
pk_ds   = season["PK_D"].fillna(0).values

safe_tots = np.where(ttots > 0, ttots, 1.0)
season["total_BPR"] = np.where(
    has_pps & (ttots > 0),
    (t5s * bpr5s + tpps * pp_os + tpks * pk_ds) / safe_tots,
    bpr5s,
).round(4)
season["pp_toi_frac"] = (tpps / safe_tots * (ttots > 0)).round(3)
season["pk_toi_frac"] = (tpks / safe_tots * (ttots > 0)).round(3)

# Deployment-neutral: league-average fixed weights
season["total_BPR_adj"] = np.where(
    has_pps,
    (W_ES * bpr5s + W_PP * pp_os + W_PK * pk_ds).round(4),
    bpr5s,
)

season.to_csv(OUT_SEASON, index=False)
print(f"\nPer-season: {len(season):,} player-seasons → {OUT_SEASON}")
print("\nrapm_weight distribution (per-season):")
print(season["rapm_weight"].describe().round(3).to_string())
print("\nTop 20 (total_BPR_adj, 2022+ seasons):")
recent = season[season["season"] >= 2022].sort_values("total_BPR_adj", ascending=False)
print(recent.head(20)[["player_name", "position", "season", "toi", "final_BPR",
                         "PP_O", "PK_D", "total_BPR", "total_BPR_adj"]].to_string(index=False))
