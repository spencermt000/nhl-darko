"""
composite.py — Final v2 ratings assembly.

No sigmoid blend needed — the Bayesian RAPM output IS the blend (prior baked in).
This script adds:
  - PP/PK integration (league-avg TOI weights, reuses v1 PP/PK RAPM)
  - GAR counting stat: (BPR - replacement_level) * toi / 60
  - Diagnostic columns: prior_O, prior_D, rapm_shift, rapm_weight

Inputs:
  data/v2_rapm_results.csv        Pooled prior-informed RAPM
  data/v2_rapm_by_season.csv      Per-season prior-informed RAPM
  data/v2_box_prior.csv           Box score priors (for diagnostics)
  data/pp_rapm.csv                PP_O / PK_D ratings (from v1)
  data/skaters_by_game.csv        Situational TOI

Outputs:
  data/v2_final_ratings.csv           Pooled composite ratings
  data/v2_final_ratings_by_season.csv Per-season composite ratings
"""

import sys
import numpy as np
import pandas as pd

# ── Config ──────────────────────────────────────────────────────────────────
RAPM_POOLED  = "output/v2_rapm_results.csv"
RAPM_SEASON  = "output/v2_rapm_by_season.csv"
PRIOR_FILE   = "output/v2_box_prior.csv"
PP_RAPM      = "output/pp_rapm.csv"
SKATERS_GAME = "data/skaters_by_game.csv"
OUT_POOLED   = "output/v2_final_ratings.csv"
OUT_SEASON   = "output/v2_final_ratings_by_season.csv"

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading v2 RAPM results...", file=sys.stderr)
rapm_p = pd.read_csv(RAPM_POOLED)
rapm_s = pd.read_csv(RAPM_SEASON)
rapm_p["player_id"] = rapm_p["player_id"].astype(int)
rapm_s["player_id"] = rapm_s["player_id"].astype(int)

print("Loading box score priors...", file=sys.stderr)
prior = pd.read_csv(PRIOR_FILE)
prior["player_id"] = prior["player_id"].astype(int)

# PP/PK RAPM (from v1 — reuse since we don't have a separate v2 PP model)
print("Loading PP/PK RAPM...", file=sys.stderr)
pp = pd.read_csv(PP_RAPM)[["player_id", "PP_O", "PK_D", "PP_BPR"]]
pp["player_id"] = pp["player_id"].astype(int)

# Situational TOI
print("Loading situational TOI...", file=sys.stderr)
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
for col in ["toi_5v5", "toi_pp", "toi_pk"]:
    sit_toi[col] = (sit_toi[col] / 60).round(1)
sit_toi["player_id"] = sit_toi["player_id"].astype(int)

# Pooled situational TOI
sit_pooled = sit_toi.groupby("player_id")[["toi_5v5", "toi_pp", "toi_pk"]].sum().reset_index()

print(f"  RAPM pooled: {len(rapm_p):,} players", file=sys.stderr)
print(f"  RAPM by-season: {len(rapm_s):,} player-seasons", file=sys.stderr)
print(f"  PP/PK RAPM: {len(pp):,} players", file=sys.stderr)

# ── League-average situational weights ─────────────────────────────────────
_lg = sit_pooled[["toi_5v5", "toi_pp", "toi_pk"]].sum()
_lg_total = _lg.sum()
W_ES = float(_lg["toi_5v5"] / _lg_total)
W_PP = float(_lg["toi_pp"]  / _lg_total)
W_PK = float(_lg["toi_pk"]  / _lg_total)
print(f"\nLeague-average TOI weights: ES={W_ES:.3f}  PP={W_PP:.3f}  PK={W_PK:.3f}", file=sys.stderr)

# ── Pooled composite ──────────────────────────────────────────────────────
pooled = rapm_p.copy()
pooled = pooled.merge(sit_pooled, on="player_id", how="left")
pooled = pooled.merge(pp, on="player_id", how="left")

# Diagnostic: how much did RAPM shift from the prior?
if "prior_O" in pooled.columns:
    pooled["rapm_shift_O"] = (pooled["BPR_O"] - pooled["prior_O"]).round(4)
    pooled["rapm_shift_D"] = (pooled["BPR_D"] - pooled["prior_D"]).round(4)

# Effective data weight: 1 - posterior_var/prior_var (how much the data moved us from prior)
# Higher = more data influence, lower = still near prior
if "BPR_se" in pooled.columns and "prior_O" in pooled.columns:
    # Rough prior var from calibration file (loaded from prior results)
    prior_pooled_agg = prior.groupby("player_id")[["prior_O", "prior_D"]].mean().reset_index()
    prior_var_O = float(prior_pooled_agg["prior_O"].var())
    prior_var_D = float(prior_pooled_agg["prior_D"].var())
    prior_var_total = prior_var_O + prior_var_D
    posterior_var = pooled["BPR_se"].fillna(1.0) ** 2
    pooled["rapm_weight"] = (1.0 - posterior_var / max(prior_var_total, 1e-6)).clip(0, 1).round(3)
else:
    pooled["rapm_weight"] = 0.5

# PP/PK situational blend (same approach as v1)
t5  = pooled["toi_5v5"].fillna(0).values
tpp = pooled["toi_pp"].fillna(0).values
tpk = pooled["toi_pk"].fillna(0).values
ttotal = t5 + tpp + tpk

has_pp = ~pooled["PP_O"].isna().values
bpr5   = pooled["BPR"].values
pp_o   = pooled["PP_O"].fillna(0).values
pk_d   = pooled["PK_D"].fillna(0).values

safe_total = np.where(ttotal > 0, ttotal, 1.0)
pooled["total_BPR"] = np.where(
    has_pp & (ttotal > 0),
    (t5 * bpr5 + tpp * pp_o + tpk * pk_d) / safe_total,
    bpr5,
).round(4)
pooled["pp_toi_frac"] = (tpp / safe_total * (ttotal > 0)).round(3)
pooled["pk_toi_frac"] = (tpk / safe_total * (ttotal > 0)).round(3)

# Deployment-neutral: league-average fixed weights
pooled["total_BPR_adj"] = np.where(
    has_pp,
    (W_ES * bpr5 + W_PP * pp_o + W_PK * pk_d).round(4),
    bpr5,
)

# ── Save pooled ───────────────────────────────────────────────────────────
pooled.to_csv(OUT_POOLED, index=False)
print(f"\nPooled: {len(pooled):,} players → {OUT_POOLED}", file=sys.stderr)

print("\nTop 20 (total_BPR_adj — deployment-neutral):", file=sys.stderr)
cols_show = ["player_name", "position", "BPR_O", "BPR_D", "BPR", "BPR_se",
             "PP_O", "PK_D", "total_BPR_adj", "rapm_weight"]
cols_show = [c for c in cols_show if c in pooled.columns]
print(pooled.sort_values("total_BPR_adj", ascending=False).head(20)[cols_show].to_string(index=False),
      file=sys.stderr)

# ── Per-season composite ──────────────────────────────────────────────────
season = rapm_s.copy()
season = season.merge(sit_toi, on=["player_id", "season"], how="left")
season = season.merge(pp, on="player_id", how="left")

# ── Blend per-season RAPM with pooled RAPM ────────────────────────────────
# Per-season RAPM uses the same ridge alpha as pooled (~10x more data), making
# single-season estimates over-regularized and noisy. Blend toward the more
# stable pooled career estimate, weighted by per-season 5v5 TOI.
#   blended = season_weight * per_season + (1 - season_weight) * pooled
#   season_weight = toi_5v5 / (toi_5v5 + BLEND_K)
# At k=800, a typical starter (800 min) gets 50% season / 50% pooled.
BLEND_K = 800  # minutes 5v5
blend_cols = ["xGF_O", "xGF_D", "GF_O", "GF_D", "SOG_O", "SOG_D",
              "TO_O", "TO_D", "GA_O", "GA_D", "BPR_O", "BPR_D", "BPR"]

# Merge pooled values with _pool suffix
pooled_for_blend = rapm_p[["player_id"] + blend_cols].copy()
pooled_for_blend = pooled_for_blend.rename(columns={c: f"{c}_pool" for c in blend_cols})
season = season.merge(pooled_for_blend, on="player_id", how="left")

toi_blend = season["toi_5v5"].fillna(0).values
season_weight = toi_blend / (toi_blend + BLEND_K)
season["blend_weight"] = season_weight.round(3)

for col in blend_cols:
    pool_col = f"{col}_pool"
    s_vals = season[col].fillna(0).values
    p_vals = season[pool_col].fillna(0).values
    season[col] = (season_weight * s_vals + (1 - season_weight) * p_vals).round(4)
    season = season.drop(columns=[pool_col])

print(f"\n  Per-season blending: k={BLEND_K}, median blend_weight={season['blend_weight'].median():.2f}", file=sys.stderr)

# ── Hybrid: blend RAPM with on-ice/off-ice relative rates ─────────────────
# RAPM controls for teammates/competition but compresses star players via ridge.
# Relative xGF/60 (on-ice minus off-ice) preserves magnitude but conflates
# player skill with deployment. Blending gets the best of both.
print("\n  Computing on-ice/off-ice relative rates...", file=sys.stderr)

rel_cols = ["playerId", "season", "situation", "icetime",
            "OnIce_F_xGoals", "OnIce_A_xGoals",
            "OffIce_F_xGoals", "OffIce_A_xGoals"]
rel_sbg = pd.read_csv(SKATERS_GAME, usecols=rel_cols)
rel_sbg = rel_sbg.rename(columns={"playerId": "player_id"})
rel_5v5 = rel_sbg[rel_sbg["situation"] == "5on5"].copy()

rel_agg = rel_5v5.groupby(["player_id", "season"]).agg(
    toi_sec=("icetime", "sum"),
    on_xgf=("OnIce_F_xGoals", "sum"),
    on_xga=("OnIce_A_xGoals", "sum"),
    off_xgf=("OffIce_F_xGoals", "sum"),
    off_xga=("OffIce_A_xGoals", "sum"),
).reset_index()

toi_min = rel_agg["toi_sec"] / 60
toi_hrs = toi_min / 60

# On-ice and off-ice per-60 rates
rel_agg["on_xgf_60"] = np.where(toi_hrs > 0, rel_agg["on_xgf"] / toi_hrs, 0)
rel_agg["on_xga_60"] = np.where(toi_hrs > 0, rel_agg["on_xga"] / toi_hrs, 0)
# Off-ice: need off-ice TOI estimate. OffIce stats are team totals when player is OFF.
# off_toi ≈ we don't have it directly, but we can compute per-60 from the raw counts
# using on_ice_rate as calibration: off_xgf_60 ≈ off_xgf / off_toi_hrs
# Approximate: team plays ~3400 min 5v5/season, player plays ~800, so off ≈ 2600
# Better: use the ratio. For now, use the percentage columns.
# Actually, simplest: rel = on_ice_rate - league_avg_rate (avoids needing off-ice TOI)
# Even better: use on-ice xGF% vs off-ice xGF% which is already available

# Simple approach: relative xGF/60 = on-ice xGF/60 - league average xGF/60
lg_xgf_60 = rel_agg.loc[toi_min >= 200, "on_xgf_60"].mean()
lg_xga_60 = rel_agg.loc[toi_min >= 200, "on_xga_60"].mean()
rel_agg["rel_xgf_60"] = rel_agg["on_xgf_60"] - lg_xgf_60
rel_agg["rel_xga_60"] = -(rel_agg["on_xga_60"] - lg_xga_60)  # flip: positive = good defense
rel_agg["toi_min"] = toi_min

# Scale relative rates to RAPM coefficient scale.
# RAPM xGF_O std ≈ 0.07, relative xGF/60 std ≈ 0.38. Scale factor ≈ 0.07/0.38 ≈ 0.18
# But we WANT the wider spread — the whole point is to un-compress star players.
# So don't rescale; instead blend at the xGF_O level with appropriate weights.
# The blend ratio determines how much of the wider spread we retain.
RAPM_WEIGHT = 0.5  # 50% RAPM (compressed but clean), 50% relative (wide but noisy)

season = season.merge(
    rel_agg[["player_id", "season", "rel_xgf_60", "rel_xga_60", "toi_min"]].rename(
        columns={"toi_min": "rel_toi_min"}),
    on=["player_id", "season"], how="left"
)

# Blend xGF_O and xGF_D with relative rates
rapm_xgf_o = season["xGF_O"].values.copy()
rapm_xgf_d = season["xGF_D"].values.copy()
rel_o = season["rel_xgf_60"].fillna(0).values
rel_d = season["rel_xga_60"].fillna(0).values

season["xGF_O"] = (RAPM_WEIGHT * rapm_xgf_o + (1 - RAPM_WEIGHT) * rel_o).round(4)
season["xGF_D"] = (RAPM_WEIGHT * rapm_xgf_d + (1 - RAPM_WEIGHT) * rel_d).round(4)

# Update BPR_O/D to reflect the hybrid (these feed into downstream composites)
# BPR = weighted sum of all metrics; we only changed xGF, so adjust BPR accordingly
# delta_O = new_xGF_O - old_xGF_O; BPR_O += delta_O * hand_coded_xGF_weight
# But with learned weights this gets complicated. Just recompute BPR from scratch.
# Actually, BPR is only used for display/diagnostics — gar.py recomputes xEV_O from
# individual metrics. So no need to update BPR here.

season = season.drop(columns=["rel_xgf_60", "rel_xga_60", "rel_toi_min"], errors="ignore")

qual_mask = season["toi_5v5"].fillna(0) >= 200
print(f"  Hybrid blend: RAPM weight={RAPM_WEIGHT}, "
      f"lg xGF/60={lg_xgf_60:.2f}, lg xGA/60={lg_xga_60:.2f}", file=sys.stderr)
print(f"  Post-hybrid xGF_O std: {season.loc[qual_mask, 'xGF_O'].std():.4f} "
      f"(was RAPM-only: {rapm_xgf_o[qual_mask.values].std():.4f})", file=sys.stderr)

# Diagnostic shifts
if "prior_O" in season.columns:
    season["rapm_shift_O"] = (season["BPR_O"] - season["prior_O"]).round(4)
    season["rapm_shift_D"] = (season["BPR_D"] - season["prior_D"]).round(4)

# Per-season PP/PK blend
t5s  = season["toi_5v5"].fillna(0).values
tpps = season["toi_pp"].fillna(0).values
tpks = season["toi_pk"].fillna(0).values
ttots = t5s + tpps + tpks

has_pps = ~season["PP_O"].isna().values
bpr5s   = season["BPR"].values
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

# Deployment-neutral
season["total_BPR_adj"] = np.where(
    has_pps,
    (W_ES * bpr5s + W_PP * pp_os + W_PK * pk_ds).round(4),
    bpr5s,
)

# ── Save per-season ───────────────────────────────────────────────────────
season.to_csv(OUT_SEASON, index=False)
print(f"\nPer-season: {len(season):,} player-seasons → {OUT_SEASON}", file=sys.stderr)

print("\nTop 20 (total_BPR_adj, 2022+ seasons):", file=sys.stderr)
recent = season[season["season"] >= 2022].sort_values("total_BPR_adj", ascending=False)
cols_season = ["player_name", "position", "season", "BPR", "BPR_se",
               "PP_O", "PK_D", "total_BPR_adj"]
cols_season = [c for c in cols_season if c in season.columns]
print(recent.head(20)[cols_season].to_string(index=False), file=sys.stderr)
