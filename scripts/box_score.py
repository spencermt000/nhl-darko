"""
box_score.py — Production-based player ratings from per-game MoneyPuck box scores.

Inputs:
  data/skaters_by_game.csv     MoneyPuck per-game skater stats (2007-2024)
  data/moneypuck_player_bio.csv  Player bio (birthdate, position)
  data/rapm_results.csv          Pooled RAPM (training target)

Steps:
  1. Filter to situation="all" and aggregate per-game rows to per-season totals
  2. Compute per-60 rates; position-normalize (F vs D have different baselines)
  3. Fit ElasticNet: box score rates → RAPM BPR_O and BPR_D
  4. Predict box_O / box_D for every player-season

Output:
  data/box_score_ratings.csv
    player_id, season, player_name, position, toi, age,
    G60, A1_60, A2_60, SOG60, TO60, GA60, BLK60,
    box_O, box_D, box_BPR
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
SKATERS_FILE = Path("data/skaters_by_game.csv")
BIO_FILE     = Path("data/moneypuck_player_bio.csv")
RAPM_FILE    = Path("data/rapm_results.csv")
OUT_FILE     = Path("data/box_score_ratings.csv")

SITUATION    = "all"       # use all-situation totals
MIN_TOI_SEC  = 6_000       # ~100 min — drop thin samples
PRIME_AGE    = 26          # used for age context (not filtering)

# ── Load & filter ─────────────────────────────────────────────────────────────
print("Loading skaters_by_game.csv...")
sk = pd.read_csv(SKATERS_FILE, low_memory=False)
print(f"  Raw: {len(sk):,} rows")

sk = sk[sk["situation"] == SITUATION].copy()
print(f"  After situation='{SITUATION}' filter: {len(sk):,} rows")

# Exclude goalies
sk = sk[sk["position"] != "G"]

# ── Aggregate to per-season ───────────────────────────────────────────────────
count_cols = [
    "icetime",
    "I_F_goals", "I_F_primaryAssists", "I_F_secondaryAssists",
    "I_F_shotsOnGoal", "I_F_takeaways", "I_F_giveaways",
    "shotsBlockedByPlayer", "I_F_hits",
    "I_F_xGoals",           # individual expected goals
    "OnIce_F_shotAttempts", "OnIce_A_shotAttempts",  # Corsi for/against
]
# Only keep columns that actually exist
count_cols = [c for c in count_cols if c in sk.columns]

grp = sk.groupby(["playerId", "season", "name", "position"])[count_cols].sum().reset_index()
grp = grp.rename(columns={"playerId": "player_id", "name": "player_name"})

# Filter by TOI
grp = grp[grp["icetime"] >= MIN_TOI_SEC].copy()
grp["toi_min"] = grp["icetime"] / 60
print(f"  After TOI filter (≥{MIN_TOI_SEC/60:.0f} min): {len(grp):,} player-seasons")

# ── Bio data: position + age ──────────────────────────────────────────────────
print("Loading bio data...")
bio = pd.read_csv(BIO_FILE)
bio = bio.rename(columns={"playerId": "player_id"})
bio["player_id"] = bio["player_id"].astype(int)
bio["birthDate"] = pd.to_datetime(bio["birthDate"], errors="coerce")

# Normalize position to F / D
def norm_pos(p):
    p = str(p).upper()
    return "F" if p in ("C", "L", "R", "LW", "RW", "F") else ("D" if p == "D" else "F")

bio["pos_norm"] = bio["position"].map(norm_pos)
grp["player_id"] = grp["player_id"].astype(int)
grp = grp.merge(bio[["player_id", "birthDate", "pos_norm"]], on="player_id", how="left")

# Age at start of season (season col = start year in MoneyPuck)
grp["age"] = grp["season"] - grp["birthDate"].dt.year
grp["age"] = grp["age"].clip(18, 45)

# Use bio position when available, fall back to skaters file position
grp["pos"] = grp["pos_norm"].fillna(grp["position"].map(norm_pos))

# ── Per-60 rates ──────────────────────────────────────────────────────────────
t60 = grp["toi_min"] / 60   # hours → per-60 divisor

grp["G60"]    = grp["I_F_goals"]               / t60
grp["A1_60"]  = grp["I_F_primaryAssists"]      / t60
grp["A2_60"]  = grp["I_F_secondaryAssists"]    / t60
grp["SOG60"]  = grp["I_F_shotsOnGoal"]         / t60
grp["TO60"]   = grp["I_F_takeaways"]           / t60
grp["GA60"]   = grp["I_F_giveaways"]           / t60
grp["BLK60"]  = grp["shotsBlockedByPlayer"]    / t60 if "shotsBlockedByPlayer" in grp else 0.0
grp["HIT60"]  = grp["I_F_hits"]               / t60 if "I_F_hits" in grp else 0.0
grp["xG60"]   = grp["I_F_xGoals"]             / t60 if "I_F_xGoals" in grp else grp["G60"]

# Corsi% (team possession proxy)
if "OnIce_F_shotAttempts" in grp.columns and "OnIce_A_shotAttempts" in grp.columns:
    total_cf = grp["OnIce_F_shotAttempts"] + grp["OnIce_A_shotAttempts"]
    grp["CF_pct"] = np.where(total_cf > 0, grp["OnIce_F_shotAttempts"] / total_cf * 100, 50.0)
else:
    grp["CF_pct"] = 50.0

# ── Position-normalize ────────────────────────────────────────────────────────
rate_cols = ["G60", "A1_60", "A2_60", "SOG60", "TO60", "GA60", "BLK60", "HIT60", "xG60", "CF_pct"]
for col in rate_cols:
    pos_mean = grp.groupby("pos")[col].transform("mean")
    grp[col + "_adj"] = (grp[col] - pos_mean).fillna(0)

adj_cols = [c + "_adj" for c in rate_cols]

# ── Load RAPM to train box score model ───────────────────────────────────────
print("\nLoading pooled RAPM (training target)...")
rapm = pd.read_csv(RAPM_FILE)[["player_id", "BPR_O", "BPR_D"]]
rapm["player_id"] = rapm["player_id"].astype(int)
print(f"  {len(rapm):,} players")

train = grp.merge(rapm, on="player_id", how="inner")
print(f"  {len(train):,} player-seasons matched to RAPM")

# ── Fit models ────────────────────────────────────────────────────────────────
off_feats = ["G60_adj", "A1_60_adj", "A2_60_adj", "SOG60_adj", "TO60_adj", "xG60_adj", "CF_pct_adj"]
def_feats  = ["BLK60_adj", "GA60_adj", "HIT60_adj", "CF_pct_adj"]

def fit_model(X_df, y, name):
    X = X_df.fillna(0).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=5, max_iter=5000, n_jobs=-1)
    model.fit(Xs, y)
    r2 = model.score(Xs, y)
    coef_str = ", ".join(f"{f}={c:.3f}" for f, c in zip(X_df.columns, model.coef_))
    print(f"  {name}: R²={r2:.3f} | {coef_str}")
    return model, scaler

print("\nFitting offensive box score model (target: BPR_O)...")
model_off, scaler_off = fit_model(train[off_feats], train["BPR_O"].values, "BPR_O")

print("Fitting defensive box score model (target: BPR_D)...")
model_def, scaler_def = fit_model(train[def_feats], train["BPR_D"].values, "BPR_D")

# ── Predict all player-seasons ────────────────────────────────────────────────
print("\nGenerating box score ratings for all player-seasons...")
grp["box_O"]   = model_off.predict(scaler_off.transform(grp[off_feats].fillna(0))).round(4)
grp["box_D"]   = model_def.predict(scaler_def.transform(grp[def_feats].fillna(0))).round(4)
grp["box_BPR"] = (grp["box_O"] + grp["box_D"]).round(4)

# ── Output ────────────────────────────────────────────────────────────────────
out = grp[[
    "player_id", "season", "player_name", "pos", "toi_min", "age",
    "G60", "A1_60", "A2_60", "SOG60", "TO60", "GA60", "BLK60",
    "box_O", "box_D", "box_BPR",
]].rename(columns={"pos": "position", "toi_min": "toi"})

out.to_csv(OUT_FILE, index=False)
print(f"\nWrote {len(out):,} player-seasons to {OUT_FILE}")
print("\nTop 15 by box_BPR:")
print(
    out.sort_values("box_BPR", ascending=False)
    .head(15)[["player_name", "position", "season", "toi", "G60", "A1_60", "box_O", "box_D", "box_BPR"]]
    .to_string(index=False)
)
