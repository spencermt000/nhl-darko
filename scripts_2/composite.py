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
RAPM_POOLED  = "data/v2_rapm_results.csv"
RAPM_SEASON  = "data/v2_rapm_by_season.csv"
PRIOR_FILE   = "data/v2_box_prior.csv"
PP_RAPM      = "data/pp_rapm.csv"
SKATERS_GAME = "data/skaters_by_game.csv"
OUT_POOLED   = "data/v2_final_ratings.csv"
OUT_SEASON   = "data/v2_final_ratings_by_season.csv"

# Replacement level for GAR (roughly 20th percentile BPR)
REPLACEMENT_LEVEL = -0.10

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

# GAR: Goals Above Replacement
# Pooled TOI (sum of 5v5 TOI across seasons, approximate)
pooled_toi = pooled["toi_5v5"].fillna(0).values
pooled["GAR"] = ((pooled["total_BPR_adj"] - REPLACEMENT_LEVEL) * pooled_toi / 60).round(1)

# ── Save pooled ───────────────────────────────────────────────────────────
pooled.to_csv(OUT_POOLED, index=False)
print(f"\nPooled: {len(pooled):,} players → {OUT_POOLED}", file=sys.stderr)

print("\nTop 20 (total_BPR_adj — deployment-neutral):", file=sys.stderr)
cols_show = ["player_name", "position", "BPR_O", "BPR_D", "BPR", "BPR_se",
             "PP_O", "PK_D", "total_BPR_adj", "GAR", "rapm_weight"]
cols_show = [c for c in cols_show if c in pooled.columns]
print(pooled.sort_values("total_BPR_adj", ascending=False).head(20)[cols_show].to_string(index=False),
      file=sys.stderr)

# ── Per-season composite ──────────────────────────────────────────────────
season = rapm_s.copy()
season = season.merge(sit_toi, on=["player_id", "season"], how="left")
season = season.merge(pp, on="player_id", how="left")

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

# Per-season GAR
season_toi = season["toi_5v5"].fillna(0).values
season["GAR"] = ((season["total_BPR_adj"] - REPLACEMENT_LEVEL) * season_toi / 60).round(1)

# ── Save per-season ───────────────────────────────────────────────────────
season.to_csv(OUT_SEASON, index=False)
print(f"\nPer-season: {len(season):,} player-seasons → {OUT_SEASON}", file=sys.stderr)

print("\nTop 20 (total_BPR_adj, 2022+ seasons):", file=sys.stderr)
recent = season[season["season"] >= 2022].sort_values("total_BPR_adj", ascending=False)
cols_season = ["player_name", "position", "season", "BPR", "BPR_se",
               "PP_O", "PK_D", "total_BPR_adj", "GAR"]
cols_season = [c for c in cols_season if c in season.columns]
print(recent.head(20)[cols_season].to_string(index=False), file=sys.stderr)
