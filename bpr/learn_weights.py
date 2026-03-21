"""
learn_weights.py — Learn optimal BPR weights + individual finishing component.

Instead of hand-coded BPR weights (xGF=0.50, SOG=0.22, GF=0.15, TO=0.06, GA=-0.04),
learns optimal weights by predicting next-season on-ice impact from this season's
per-metric RAPM coefficients + individual finishing (goals - xGoals).

Individual finishing (iFinish) captures shooting skill OUTSIDE of RAPM, avoiding the
dilution problem where RAPM distributes finishing credit across all 5 on-ice skaters.

Inputs:
  output/v2_rapm_by_season.csv        Per-season per-metric RAPM coefficients
  data/skaters_by_game.csv            Individual shooting stats + on-ice impact targets

Outputs:
  output/learned_bpr_weights.json     Learned weights for GAR decomposition
  output/ifinish_by_season.csv        Per-player-season individual finishing stats
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)

MIN_TOI_SEASON = 200  # minutes 5v5 to qualify
MIN_SEASON = 2015

# ── Phase A: Compute iFinish per player-season ────────────────────────────────
print("Phase A: Computing individual finishing (iFinish)...", file=sys.stderr)

sbg = pd.read_csv("data/skaters_by_game.csv",
                   usecols=["playerId", "season", "situation", "icetime",
                            "I_F_goals", "I_F_xGoals", "I_F_shotsOnGoal",
                            "OnIce_F_xGoals", "OnIce_A_xGoals",
                            "OffIce_F_xGoals", "OffIce_A_xGoals",
                            "OnIce_F_goals", "OnIce_A_goals"])
sbg = sbg.rename(columns={"playerId": "player_id"})
ev = sbg[(sbg["situation"] == "5on5") & (sbg["season"] >= MIN_SEASON)].copy()
ev["toi_min"] = ev["icetime"] / 60.0

# Aggregate to player-season
ifinish = ev.groupby(["player_id", "season"]).agg(
    toi_min=("toi_min", "sum"),
    goals=("I_F_goals", "sum"),
    xgoals=("I_F_xGoals", "sum"),
    shots=("I_F_shotsOnGoal", "sum"),
    on_xgf=("OnIce_F_xGoals", "sum"),
    on_xga=("OnIce_A_xGoals", "sum"),
    off_xgf=("OffIce_F_xGoals", "sum"),
    off_xga=("OffIce_A_xGoals", "sum"),
    on_gf=("OnIce_F_goals", "sum"),
    on_ga=("OnIce_A_goals", "sum"),
).reset_index()

ifinish["iFinish_raw"] = ifinish["goals"] - ifinish["xgoals"]
toi_hrs = ifinish["toi_min"] / 60.0
ifinish["iFinish_per60"] = np.where(toi_hrs > 0, ifinish["iFinish_raw"] / toi_hrs, 0)

# Filter to qualified
ifinish_qual = ifinish[ifinish["toi_min"] >= MIN_TOI_SEASON].copy()
print(f"  {len(ifinish_qual):,} qualified player-seasons (≥{MIN_TOI_SEASON} min 5v5)", file=sys.stderr)
print(f"  iFinish_per60: mean={ifinish_qual['iFinish_per60'].mean():.3f}, "
      f"std={ifinish_qual['iFinish_per60'].std():.3f}", file=sys.stderr)


# ── Phase B: Shrink iFinish ──────────────────────────────────────────────────
print("\nPhase B: Calibrating iFinish shrinkage...", file=sys.stderr)

# Build YoY pairs
curr = ifinish_qual[["player_id", "season", "iFinish_per60", "shots"]].copy()
nxt = ifinish_qual[["player_id", "season", "iFinish_per60"]].copy()
nxt = nxt.rename(columns={"iFinish_per60": "next_iFinish_per60"})
nxt["season"] = nxt["season"] - 1  # shift so join gives "next season"

yoy = curr.merge(nxt, on=["player_id", "season"], how="inner")
print(f"  YoY pairs for iFinish: {len(yoy):,}", file=sys.stderr)

# YoY correlation
r_raw = yoy["iFinish_per60"].corr(yoy["next_iFinish_per60"])
print(f"  Raw iFinish YoY r = {r_raw:.3f}", file=sys.stderr)

# Bayesian shrinkage calibration
# k = mean_shots * (1 - r) / r  gives the statistically optimal k (~312)
# However, this over-shrinks for current-season valuation (keeps only ~2% of signal).
# We cap k at 80 to balance prediction vs current-season attribution:
#   - At 80, a player with ~120 shots keeps ~60% of their raw signal
#   - This better matches observed finishing impact in other public models
#   - The learned W_iFIN will naturally adjust to compensate for lighter shrinkage
K_MAX = 80
mean_shots = ifinish_qual["shots"].mean()
r_clipped = np.clip(r_raw, 0.05, 0.95)  # safety bounds
k_optimal = mean_shots * (1 - r_clipped) / r_clipped
k = min(k_optimal, K_MAX)
print(f"  Shrinkage k = {k:.1f} (optimal={k_optimal:.1f}, capped at {K_MAX}; mean shots = {mean_shots:.0f})", file=sys.stderr)

# Apply shrinkage to ALL player-seasons (not just qualified, for completeness)
ifinish["iFinish_shrunk"] = ifinish["iFinish_per60"] * ifinish["shots"] / (ifinish["shots"] + k)

# Verify shrinkage improves YoY correlation
curr_s = ifinish[ifinish["toi_min"] >= MIN_TOI_SEASON][["player_id", "season", "iFinish_shrunk", "shots"]].copy()
nxt_s = ifinish[ifinish["toi_min"] >= MIN_TOI_SEASON][["player_id", "season", "iFinish_shrunk"]].copy()
nxt_s = nxt_s.rename(columns={"iFinish_shrunk": "next_iFinish_shrunk"})
nxt_s["season"] = nxt_s["season"] - 1
yoy_s = curr_s.merge(nxt_s, on=["player_id", "season"], how="inner")
r_shrunk = yoy_s["iFinish_shrunk"].corr(yoy_s["next_iFinish_shrunk"])
print(f"  Shrunk iFinish YoY r = {r_shrunk:.3f}", file=sys.stderr)

# Save iFinish
ifinish_out = ifinish[["player_id", "season", "toi_min", "goals", "xgoals", "shots",
                        "iFinish_raw", "iFinish_per60", "iFinish_shrunk"]].copy()
ifinish_out.to_csv("output/ifinish_by_season.csv", index=False)
print(f"  Saved output/ifinish_by_season.csv ({len(ifinish_out):,} rows)", file=sys.stderr)


# ── Phase C: Build feature matrix + targets ──────────────────────────────────
print("\nPhase C: Building feature matrix + next-season targets...", file=sys.stderr)

# Load per-metric RAPM (use blended final ratings — same data gar.py consumes)
rapm = pd.read_csv("output/v2_final_ratings_by_season.csv")
rapm["player_id"] = rapm["player_id"].astype(int)

# Merge RAPM with iFinish
features = rapm.merge(
    ifinish[["player_id", "season", "iFinish_shrunk", "shots", "toi_min"]].rename(
        columns={"toi_min": "toi_sbg"}),
    on=["player_id", "season"],
    how="inner",
)

# Use RAPM toi_5v5 if available, else sbg toi
if "toi_5v5" in features.columns:
    features["toi_min"] = features["toi_5v5"]
else:
    features["toi_min"] = features["toi_sbg"]

# Filter qualified
features = features[features["toi_min"] >= MIN_TOI_SEASON].copy()
print(f"  Feature rows (RAPM + iFinish): {len(features):,}", file=sys.stderr)

# Compute next-season on-ice impact targets (same as composite_v4.py)
impact = ifinish[ifinish["toi_min"] >= MIN_TOI_SEASON].copy()
toi60 = impact["toi_min"] / 60.0
impact["oiXGF_60"] = impact["on_xgf"] / toi60
impact["oiXGA_60"] = impact["on_xga"] / toi60
impact["offXGF_60"] = impact["off_xgf"] / toi60
impact["offXGA_60"] = impact["off_xga"] / toi60
impact["relXGF_60"] = impact["oiXGF_60"] - impact["offXGF_60"]
impact["relXGA_60"] = impact["oiXGA_60"] - impact["offXGA_60"]

# Target: 50% raw on-ice + 50% relative (same blend as composite_v4.py)
impact["target_O"] = 0.5 * impact["oiXGF_60"] + 0.5 * impact["relXGF_60"]
impact["target_D"] = -(0.5 * impact["oiXGA_60"] + 0.5 * impact["relXGA_60"])

# Build YoY pairs: season N features → season N+1 target
target_next = impact[["player_id", "season", "target_O", "target_D"]].copy()
target_next = target_next.rename(columns={"target_O": "next_target_O", "target_D": "next_target_D"})
target_next["season"] = target_next["season"] - 1

pairs = features.merge(target_next, on=["player_id", "season"], how="inner")
pairs = pairs.dropna(subset=["next_target_O", "next_target_D"])

print(f"  YoY prediction pairs: {len(pairs):,}", file=sys.stderr)
print(f"  Seasons: {sorted(pairs['season'].unique())}", file=sys.stderr)


# ── Phase D: Learn weights via LOSO Ridge ────────────────────────────────────
print("\n── Phase D: Learning optimal weights (LOSO Ridge) ──", file=sys.stderr)

alphas = np.logspace(-1, 3, 20)
seasons = sorted(pairs["season"].unique())

O_FEATURES = ["xGF_O", "SOG_O", "GF_O", "TO_O", "GA_O", "iFinish_shrunk"]
D_FEATURES = ["xGF_D", "SOG_D", "GF_D", "TO_D", "GA_D"]


def train_loso(X_cols, y_col, label):
    """Train Ridge via leave-one-season-out CV. Returns coefficients and CV R²."""
    all_pred = np.full(len(pairs), np.nan)

    for held_out in seasons:
        train_mask = pairs["season"] != held_out
        test_mask = pairs["season"] == held_out

        X_train = pairs.loc[train_mask, X_cols].fillna(0).values
        y_train = pairs.loc[train_mask, y_col].values
        X_test = pairs.loc[test_mask, X_cols].fillna(0).values

        w = np.sqrt(pairs.loc[train_mask, "toi_min"].values / 800)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = RidgeCV(alphas=alphas, fit_intercept=True)
        model.fit(X_train_s, y_train, sample_weight=w)
        all_pred[test_mask.values] = model.predict(X_test_s)

    # CV R²
    valid = ~np.isnan(all_pred)
    y_actual = pairs[y_col].values[valid]
    y_pred = all_pred[valid]
    ss_res = np.sum((y_actual - y_pred) ** 2)
    ss_tot = np.sum((y_actual - y_actual.mean()) ** 2)
    cv_r2 = 1 - ss_res / ss_tot

    # Final model on all data
    X_all = pairs[X_cols].fillna(0).values
    y_all = pairs[y_col].values
    w_all = np.sqrt(pairs["toi_min"].values / 800)

    scaler_final = StandardScaler()
    X_all_s = scaler_final.fit_transform(X_all)
    model_final = RidgeCV(alphas=alphas, fit_intercept=True)
    model_final.fit(X_all_s, y_all, sample_weight=w_all)

    # Convert coefficients back to original scale
    coefs_orig = model_final.coef_ / scaler_final.scale_
    intercept_orig = model_final.intercept_ - np.sum(coefs_orig * scaler_final.mean_)

    print(f"\n  {label}:", file=sys.stderr)
    print(f"    LOSO CV R² = {cv_r2:.4f} (alpha={model_final.alpha_:.1f})", file=sys.stderr)
    print(f"    Weights (original scale):", file=sys.stderr)
    for feat, c in zip(X_cols, coefs_orig):
        print(f"      {feat:20s} {c:+.4f}", file=sys.stderr)
    print(f"      {'intercept':20s} {intercept_orig:+.4f}", file=sys.stderr)

    return dict(zip(X_cols, [round(float(c), 6) for c in coefs_orig])), round(float(cv_r2), 4)


# Also train baseline with hand-coded features (no iFinish) for comparison
O_FEATURES_BASELINE = ["xGF_O", "SOG_O", "GF_O", "TO_O", "GA_O"]

print("\n=== Baseline (no iFinish, learned weights) ===", file=sys.stderr)
baseline_O_weights, baseline_O_r2 = train_loso(O_FEATURES_BASELINE, "next_target_O", "Offense (baseline)")
baseline_D_weights, baseline_D_r2 = train_loso(D_FEATURES, "next_target_D", "Defense (baseline)")

print("\n=== With iFinish ===", file=sys.stderr)
learned_O_weights, learned_O_r2 = train_loso(O_FEATURES, "next_target_O", "Offense (with iFinish)")
learned_D_weights, learned_D_r2 = train_loso(D_FEATURES, "next_target_D", "Defense")

# ── Comparison ───────────────────────────────────────────────────────────────
print(f"\n{'='*60}", file=sys.stderr)
print("COMPARISON: Hand-coded vs Learned Weights", file=sys.stderr)
print(f"{'='*60}", file=sys.stderr)

HAND_CODED = {"xGF": 0.50, "SOG": 0.22, "GF": 0.15, "TO": 0.06, "GA": -0.04}

print(f"\n  {'Metric':<12s} {'Hand-coded':>12s} {'Learned(no iF)':>16s} {'Learned(+iF)':>16s}", file=sys.stderr)
print(f"  {'-'*58}", file=sys.stderr)
for metric in ["xGF", "SOG", "GF", "TO", "GA"]:
    hc = HAND_CODED[metric]
    bl = baseline_O_weights.get(f"{metric}_O", 0)
    lw = learned_O_weights.get(f"{metric}_O", 0)
    print(f"  {metric+'_O':<12s} {hc:>+12.4f} {bl:>+16.4f} {lw:>+16.4f}", file=sys.stderr)
if "iFinish_shrunk" in learned_O_weights:
    print(f"  {'iFinish':<12s} {'N/A':>12s} {'N/A':>16s} {learned_O_weights['iFinish_shrunk']:>+16.4f}", file=sys.stderr)

print(f"\n  Offense LOSO R²: baseline={baseline_O_r2:.4f}  +iFinish={learned_O_r2:.4f}  "
      f"delta={learned_O_r2 - baseline_O_r2:+.4f}", file=sys.stderr)
print(f"  Defense LOSO R²: {learned_D_r2:.4f}", file=sys.stderr)


# ── Save outputs ─────────────────────────────────────────────────────────────
output = {
    "offense": learned_O_weights,
    "defense": learned_D_weights,
    "cv_r2_O": learned_O_r2,
    "cv_r2_D": learned_D_r2,
    "cv_r2_O_baseline": baseline_O_r2,
    "ifinish_shrinkage_k": round(float(k), 1),
    "ifinish_yoy_r_raw": round(float(r_raw), 4),
    "ifinish_yoy_r_shrunk": round(float(r_shrunk), 4),
    "n_pairs": len(pairs),
    "seasons": [int(s) for s in sorted(pairs["season"].unique())],
}

with open("output/learned_bpr_weights.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\n  Saved output/learned_bpr_weights.json", file=sys.stderr)
print("\nDone.", file=sys.stderr)
