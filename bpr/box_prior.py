"""
box_prior.py — Box score prior model for RAPM v2.

Builds per-season box score features (expanded from v1), fits models predicting
raw RAPM O and D coefficients, and calibrates prior SD using R² method.

The key Miya insight: prior_sd = std(rapm_coefs) * sqrt(1 - R²)
Higher R² → tighter prior → more box score weight in the Bayesian RAPM.
For NHL we cap R² at MAX_R2=0.45 (target ~35% box score weight).

Inputs:
  data/skaters_by_game.csv       Per-game box scores
  data/moneypuck_player_bio.csv  Player bio (position, birthdate)
  data/v2_penalties.csv          Penalty data (from build_dataset.py)
  data/rapm_results.csv          Pooled RAPM (training target for initial calibration)
                                 OR data/v2_rapm_raw.csv if available

Outputs:
  data/v2_box_prior.csv            Per-player-season priors (prior_O, prior_D)
  data/v2_prior_calibration.json   Global calibration constants
"""

import sys
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
SKATERS_FILE = Path("data/skaters_by_game.csv")
BIO_FILE     = Path("data/moneypuck_player_bio.csv")
PENALTY_FILE = Path("output/v2_penalties.csv")
OUT_PRIOR    = Path("output/v2_box_prior.csv")
OUT_CALIB    = Path("output/v2_prior_calibration.json")

# Use v2 raw RAPM if available, otherwise fall back to v1
RAPM_FILE = Path("output/v2_rapm_raw.csv")
if not RAPM_FILE.exists():
    RAPM_FILE = Path("output/rapm_results.csv")
    print(f"Using v1 RAPM as training target: {RAPM_FILE}", file=sys.stderr)

SITUATION   = "all"
MIN_TOI_SEC = 6_000   # ~100 min
MAX_R2      = 0.45     # cap to target ~35% box score weight for NHL

# ── Load & aggregate skaters_by_game ─────────────────────────────────────────
print("Loading skaters_by_game.csv...", file=sys.stderr)
sk = pd.read_csv(SKATERS_FILE, low_memory=False)
sk = sk[sk["situation"] == SITUATION].copy()
sk = sk[sk["position"] != "G"]

count_cols = [
    "icetime",
    "I_F_goals", "I_F_primaryAssists", "I_F_secondaryAssists",
    "I_F_shotsOnGoal", "I_F_takeaways", "I_F_giveaways",
    "shotsBlockedByPlayer", "I_F_hits",
    "I_F_xGoals",
    "OnIce_F_shotAttempts", "OnIce_A_shotAttempts",
    "OnIce_F_xGoals", "OnIce_A_xGoals",
    # NEW in v2:
    "I_F_oZoneShiftStarts", "I_F_dZoneShiftStarts", "I_F_neutralZoneShiftStarts",
    "I_F_highDangerShots", "I_F_mediumDangerShots", "I_F_lowDangerShots",
    "faceoffsWon", "faceoffsLost",
    "penalties", "penaltiesDrawn",
]
count_cols = [c for c in count_cols if c in sk.columns]

grp = sk.groupby(["playerId", "season", "name", "position"])[count_cols].sum().reset_index()
grp = grp.rename(columns={"playerId": "player_id", "name": "player_name"})
grp = grp[grp["icetime"] >= MIN_TOI_SEC].copy()
grp["toi_min"] = grp["icetime"] / 60
grp["player_id"] = grp["player_id"].astype(int)
print(f"  {len(grp):,} player-seasons after TOI filter", file=sys.stderr)

# ── Bio data ─────────────────────────────────────────────────────────────────
print("Loading bio data...", file=sys.stderr)
bio = pd.read_csv(BIO_FILE)
bio = bio.rename(columns={"playerId": "player_id"})
bio["player_id"] = bio["player_id"].astype(int)
bio["birthDate"] = pd.to_datetime(bio["birthDate"], errors="coerce")

def norm_pos(p):
    p = str(p).upper()
    return "F" if p in ("C", "L", "R", "LW", "RW", "F") else ("D" if p == "D" else "F")

bio["pos_norm"] = bio["position"].map(norm_pos)
grp = grp.merge(bio[["player_id", "birthDate", "pos_norm"]], on="player_id", how="left")
grp["age"] = grp["season"] - grp["birthDate"].dt.year
grp["age"] = grp["age"].clip(18, 45).fillna(26)
grp["age_sq"] = grp["age"] ** 2
grp["pos"] = grp["pos_norm"].fillna(grp["position"].map(norm_pos))

# ── Penalty data (per-season aggregation) ────────────────────────────────────
if PENALTY_FILE.exists():
    print("Loading penalty data...", file=sys.stderr)
    pen = pd.read_csv(PENALTY_FILE)
    pen["player_id"] = pen["player_id"].astype(int)

    # Penalties taken per player per season
    pen_taken = pen.groupby(["player_id", "season"]).size().reset_index(name="pen_count")

    # Penalties drawn per player per season
    pen_drawn = (
        pen[pen["drawn_by_id"].notna()]
        .assign(drawn_by_id=lambda x: x["drawn_by_id"].astype(int))
        .groupby(["drawn_by_id", "season"]).size()
        .reset_index(name="pen_drawn_count")
        .rename(columns={"drawn_by_id": "player_id"})
    )

    grp = grp.merge(pen_taken, on=["player_id", "season"], how="left")
    grp = grp.merge(pen_drawn, on=["player_id", "season"], how="left")
    grp["pen_count"] = grp["pen_count"].fillna(0)
    grp["pen_drawn_count"] = grp["pen_drawn_count"].fillna(0)
else:
    print("  No penalty file found, using skaters_by_game penalty columns", file=sys.stderr)
    grp["pen_count"] = grp.get("penalties", 0)
    grp["pen_drawn_count"] = grp.get("penaltiesDrawn", 0)

# ── Compute per-60 rates ─────────────────────────────────────────────────────
t60 = grp["toi_min"] / 60

grp["G60"]    = grp["I_F_goals"]               / t60
grp["A1_60"]  = grp["I_F_primaryAssists"]      / t60
grp["A2_60"]  = grp["I_F_secondaryAssists"]    / t60
grp["SOG60"]  = grp["I_F_shotsOnGoal"]         / t60
grp["TO60"]   = grp["I_F_takeaways"]           / t60
grp["GA60"]   = grp["I_F_giveaways"]           / t60
grp["BLK60"]  = grp["shotsBlockedByPlayer"]    / t60 if "shotsBlockedByPlayer" in grp else 0.0
grp["HIT60"]  = grp["I_F_hits"]               / t60 if "I_F_hits" in grp else 0.0
grp["xG60"]   = grp["I_F_xGoals"]             / t60 if "I_F_xGoals" in grp else grp["G60"]

# Corsi%
if "OnIce_F_shotAttempts" in grp.columns:
    total_cf = grp["OnIce_F_shotAttempts"] + grp["OnIce_A_shotAttempts"]
    grp["CF_pct"] = np.where(total_cf > 0, grp["OnIce_F_shotAttempts"] / total_cf * 100, 50.0)
else:
    grp["CF_pct"] = 50.0

# NEW v2 features
# Faceoff win %
fo_total = grp.get("faceoffsWon", 0) + grp.get("faceoffsLost", 0)
grp["FO_pct"] = np.where(fo_total > 0, grp.get("faceoffsWon", 0) / fo_total * 100, 50.0)

# Penalties per 60
grp["PEN60"]       = grp["pen_count"]       / t60
grp["PEN_DRAWN60"] = grp["pen_drawn_count"] / t60

# OZ start %
oz = grp.get("I_F_oZoneShiftStarts", 0)
dz = grp.get("I_F_dZoneShiftStarts", 0)
nz = grp.get("I_F_neutralZoneShiftStarts", 0)
shift_total = oz + dz + nz
grp["OZ_START_PCT"] = np.where(shift_total > 0, oz / shift_total * 100, 50.0)

# High-danger shot %
hd = grp.get("I_F_highDangerShots", 0)
md = grp.get("I_F_mediumDangerShots", 0)
ld = grp.get("I_F_lowDangerShots", 0)
shot_total = hd + md + ld
grp["HD_SHOT_PCT"] = np.where(shot_total > 0, hd / shot_total * 100, 33.0)

# On-ice xGoals share
if "OnIce_F_xGoals" in grp.columns and "OnIce_A_xGoals" in grp.columns:
    xg_total = grp["OnIce_F_xGoals"] + grp["OnIce_A_xGoals"]
    grp["OnIce_xGF_pct"] = np.where(xg_total > 0, grp["OnIce_F_xGoals"] / xg_total * 100, 50.0)
else:
    grp["OnIce_xGF_pct"] = 50.0

# ── Position-normalize ───────────────────────────────────────────────────────
rate_cols = [
    "G60", "A1_60", "A2_60", "SOG60", "TO60", "GA60", "BLK60", "HIT60", "xG60",
    "CF_pct", "FO_pct", "PEN60", "PEN_DRAWN60", "OZ_START_PCT", "HD_SHOT_PCT",
    "OnIce_xGF_pct",
]
for col in rate_cols:
    pos_mean = grp.groupby("pos")[col].transform("mean")
    grp[col + "_adj"] = (grp[col] - pos_mean).fillna(0)

# ── Load RAPM training target ────────────────────────────────────────────────
print(f"\nLoading RAPM training target: {RAPM_FILE}", file=sys.stderr)
rapm = pd.read_csv(RAPM_FILE)[["player_id", "BPR_O", "BPR_D"]]
rapm["player_id"] = rapm["player_id"].astype(int)
print(f"  {len(rapm):,} players", file=sys.stderr)

train = grp.merge(rapm, on="player_id", how="inner")
print(f"  {len(train):,} player-seasons matched to RAPM", file=sys.stderr)

# ── Feature sets ─────────────────────────────────────────────────────────────
off_feats = [
    "G60_adj", "A1_60_adj", "A2_60_adj", "SOG60_adj", "xG60_adj",
    "TO60_adj", "CF_pct_adj",
    "FO_pct_adj", "PEN_DRAWN60_adj", "OZ_START_PCT_adj", "HD_SHOT_PCT_adj",
    "age", "age_sq",
]
def_feats = [
    "BLK60_adj", "GA60_adj", "HIT60_adj", "CF_pct_adj", "TO60_adj",
    "PEN60_adj", "OZ_START_PCT_adj", "OnIce_xGF_pct_adj",
    "age", "age_sq",
]

# ── Fit models ───────────────────────────────────────────────────────────────
def fit_and_calibrate(X_df, y, name, max_r2):
    X = X_df.fillna(0).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=5, max_iter=5000, n_jobs=-1)
    model.fit(Xs, y)

    r2_train = model.score(Xs, y)

    # 5-fold CV R² for calibration
    cv_scores = cross_val_score(model, Xs, y, cv=5, scoring="r2")
    r2_cv = float(np.mean(cv_scores))

    # Cap at MAX_R2
    r2_eff = min(r2_cv, max_r2)

    sd_rapm = float(np.std(y))
    prior_sd = sd_rapm * np.sqrt(1 - r2_eff)

    coef_str = ", ".join(f"{f}={c:.3f}" for f, c in zip(X_df.columns, model.coef_))
    print(f"  {name}:", file=sys.stderr)
    print(f"    R² (train): {r2_train:.3f}", file=sys.stderr)
    print(f"    R² (CV):    {r2_cv:.3f}", file=sys.stderr)
    print(f"    R² (eff):   {r2_eff:.3f} (capped at {max_r2})", file=sys.stderr)
    print(f"    SD(RAPM):   {sd_rapm:.4f}", file=sys.stderr)
    print(f"    prior_sd:   {prior_sd:.4f}", file=sys.stderr)
    print(f"    Coefs: {coef_str}", file=sys.stderr)

    return model, scaler, {
        "r2_train": round(r2_train, 4),
        "r2_cv": round(r2_cv, 4),
        "r2_eff": round(r2_eff, 4),
        "sd_rapm": round(sd_rapm, 6),
        "prior_sd": round(prior_sd, 6),
    }

print("\nFitting offensive prior model (target: BPR_O)...", file=sys.stderr)
model_off, scaler_off, cal_O = fit_and_calibrate(
    train[off_feats], train["BPR_O"].values, "BPR_O", MAX_R2
)

print("\nFitting defensive prior model (target: BPR_D)...", file=sys.stderr)
model_def, scaler_def, cal_D = fit_and_calibrate(
    train[def_feats], train["BPR_D"].values, "BPR_D", MAX_R2
)

# ── Save calibration ────────────────────────────────────────────────────────
calibration = {
    "offense": cal_O,
    "defense": cal_D,
    "max_r2": MAX_R2,
}
with open(OUT_CALIB, "w") as f:
    json.dump(calibration, f, indent=2)
print(f"\nCalibration saved to {OUT_CALIB}", file=sys.stderr)

# ── Predict priors for ALL player-seasons ────────────────────────────────────
print("\nGenerating box score priors for all player-seasons...", file=sys.stderr)
grp["prior_O"] = model_off.predict(
    scaler_off.transform(grp[off_feats].fillna(0))
).round(4)
grp["prior_D"] = model_def.predict(
    scaler_def.transform(grp[def_feats].fillna(0))
).round(4)
grp["prior_BPR"] = (grp["prior_O"] + grp["prior_D"]).round(4)

# ── Output ───────────────────────────────────────────────────────────────────
out_cols = [
    "player_id", "season", "player_name", "pos", "toi_min", "age",
    "G60", "A1_60", "A2_60", "SOG60", "TO60", "GA60", "BLK60", "HIT60", "xG60",
    "CF_pct", "FO_pct", "PEN60", "PEN_DRAWN60", "OZ_START_PCT", "HD_SHOT_PCT",
    "OnIce_xGF_pct",
    "prior_O", "prior_D", "prior_BPR",
]
out = grp[out_cols].rename(columns={"pos": "position", "toi_min": "toi"})
out.to_csv(OUT_PRIOR, index=False)

print(f"\nWrote {len(out):,} player-seasons to {OUT_PRIOR}", file=sys.stderr)
print("\nTop 15 by prior_BPR:", file=sys.stderr)
print(
    out.sort_values("prior_BPR", ascending=False)
    .head(15)[["player_name", "position", "season", "toi", "prior_O", "prior_D", "prior_BPR"]]
    .to_string(index=False),
    file=sys.stderr,
)
