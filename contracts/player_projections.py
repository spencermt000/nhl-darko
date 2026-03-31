"""
Player Projections — Predict next-season stats from current performance.

Uses carry-forward ratings, age curves, box score stats, and advanced
metrics to project WAR, points, goals, and GAR components for next season.

Usage: python contracts/player_projections.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(BASE, "output")

# ── 1. Load data ─────────────────────────────────────────────────────────────

cf = pd.read_csv(os.path.join(OUTPUT, "v6_carry_forward.csv"))
dw = pd.read_csv(os.path.join(OUTPUT, "v5_daily_war.csv"))
sw = pd.read_csv(os.path.join(OUTPUT, "dashboard_skater_war.csv"))

print(f"Carry-forward: {len(cf)} player-seasons")
print(f"Daily WAR: {len(dw)} player-seasons")

# ── 2. Build YoY training pairs ─────────────────────────────────────────────

# Current season features
curr = cf.copy()

# Next season targets (from daily_war and box scores)
next_war = dw[["player_id", "season", "GP", "WAR", "WAR_82", "WAR_O", "WAR_D",
               "EV_O_GAR", "EV_D_GAR", "PP_GAR", "PK_GAR", "PEN_GAR"]].copy()
next_war = next_war.rename(columns={c: f"next_{c}" for c in next_war.columns if c not in ["player_id", "season"]})
next_war["season"] = next_war["season"] - 1  # align: season N features → season N+1 targets

next_box = sw[["player_id", "season", "goals", "points", "assists_1", "assists_2",
               "shots", "toi_all"]].copy()
next_box = next_box.rename(columns={c: f"next_{c}" for c in next_box.columns if c not in ["player_id", "season"]})
next_box["season"] = next_box["season"] - 1

# Merge
pairs = curr.merge(next_war, on=["player_id", "season"], how="inner")
pairs = pairs.merge(next_box, on=["player_id", "season"], how="left")

# Filter: need minimum GP in both seasons
pairs = pairs[(pairs["GP"] >= 20) & (pairs["next_GP"] >= 20)].copy()
print(f"YoY training pairs (min 20 GP both seasons): {len(pairs)}")

# ── 3. Features ─────────────────────────────────────────────────────────────

def build_features(df):
    feat = pd.DataFrame(index=df.index)

    # Carry-forward ratings (the best predictive signal)
    feat["cf_O"] = df["cf_O"]
    feat["cf_D"] = df["cf_D"]
    feat["cf_total"] = df["cf_total"]

    # Age (with quadratic for non-linear curve)
    feat["age"] = df["age"]
    feat["age_sq"] = df["age"] ** 2

    # Current season WAR components
    feat["WAR"] = df["WAR"]
    feat["WAR_82"] = df["WAR_82"]
    feat["WAR_O"] = df["WAR_O"]
    feat["WAR_D"] = df["WAR_D"]

    # GAR components
    feat["EV_O_GAR"] = df["EV_O_GAR"]
    feat["EV_D_GAR"] = df["EV_D_GAR"]
    feat["PP_GAR"] = df["PP_GAR"]
    feat["PK_GAR"] = df["PK_GAR"]

    # Component model ratings
    for c in ["composite_O", "composite_D", "GV_O", "GV_D", "OOI_O", "OOI_D"]:
        feat[c] = df.get(c, 0)

    # Special teams rates
    feat["PP_rate"] = df.get("PP_rate", 0)
    feat["PK_rate"] = df.get("PK_rate", 0)

    # Usage / role
    feat["GP"] = df["GP"]
    feat["toi_min"] = df.get("toi_min", 0)
    feat["sit_pp"] = df.get("sit_pp", 0)
    feat["sit_pk"] = df.get("sit_pk", 0)

    # Position
    feat["pos_D"] = (df["position"] == "D").astype(int)

    return feat


X = build_features(pairs).fillna(0)

# ── 4. Train models for each target ─────────────────────────────────────────

targets = {
    "WAR": "next_WAR",
    "WAR_82": "next_WAR_82",
    "Points": "next_points",
    "Goals": "next_goals",
    "GP": "next_GP",
    "EV_O_GAR": "next_EV_O_GAR",
    "EV_D_GAR": "next_EV_D_GAR",
    "PP_GAR": "next_PP_GAR",
    "PK_GAR": "next_PK_GAR",
}

models = {}
print(f"\n{'Target':15s} {'CV MAE':>8s} {'CV R²':>8s} {'Model':>8s}")
print("-" * 45)

# Use XGBoost for WAR targets (handles skewed distribution better)
# Use Ridge for counting stats (linear relationship, no over-shrinking issue)
xgb_targets = {"WAR", "WAR_82", "EV_O_GAR", "EV_D_GAR", "PP_GAR", "PK_GAR"}

for name, col in targets.items():
    if col not in pairs.columns:
        continue
    y = pairs[col].fillna(0)

    if name in xgb_targets:
        model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.08,
                             subsample=0.8, colsample_bytree=0.8, random_state=42)
        model_type = "XGB"
    else:
        model = Ridge(alpha=10.0)
        model_type = "Ridge"

    # Cross-val
    cv_mae = -cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error").mean()
    cv_r2 = cross_val_score(model, X, y, cv=5, scoring="r2").mean()

    # Fit on full data
    model.fit(X, y)
    models[name] = model

    print(f"{name:15s} {cv_mae:8.3f} {cv_r2:8.3f} {model_type:>8s}")

# ── 5. Project current season → next season ─────────────────────────────────

CURRENT_SEASON = cf["season"].max()
current = cf[cf["season"] == CURRENT_SEASON].copy()
current = current[current["GP"] >= 10].copy()  # at least 10 GP this season

print(f"\nProjecting {len(current)} players from {CURRENT_SEASON} → {CURRENT_SEASON + 1}")

X_proj = build_features(current).fillna(0)

projections = current[["player_id", "player_name", "position", "age", "GP", "WAR", "WAR_82"]].copy()
projections["age_next"] = projections["age"] + 1

# Add current box score stats
curr_box = sw[sw["season"] == CURRENT_SEASON][["player_id", "goals", "points"]].copy()
projections = projections.merge(curr_box, on="player_id", how="left")
projections = projections.rename(columns={"goals": "curr_goals", "points": "curr_points",
                                           "WAR": "curr_WAR", "WAR_82": "curr_WAR_82"})

# ── WAR projection: empirical retention + age curve + model adjustment ──
#
# Approach:
#   1. Start with WAR_82 * empirical retention rate (tiered by current WAR level)
#   2. Apply age curve adjustment (young players improve, old players decline)
#   3. Blend with XGBoost model prediction to capture carry-forward/context signals
#   4. Weight: 70% retention-based, 30% model-based (model provides shape, retention provides scale)

# Step 1: Empirical retention by WAR tier
def get_retention(war_val):
    """Empirical YoY WAR retention by tier."""
    if war_val >= 2.0: return 0.71
    if war_val >= 1.0: return 0.82
    if war_val >= 0.5: return 0.83
    if war_val >= 0.0: return 0.98
    return 1.05  # negative WAR players tend to regress up

# Step 2: Age curve modifier (from carry_forward.py age factors)
def age_modifier(age_next):
    """Multiplicative adjustment to retention based on aging."""
    if age_next <= 22: return 1.10   # young players improving
    if age_next <= 25: return 1.03   # still developing
    if age_next <= 30: return 1.00   # peak
    if age_next <= 33: return 0.97   # early decline
    if age_next <= 36: return 0.93   # mid decline
    return 0.88                       # late career

# Step 3: Compute retention-based projection for training data (for CI estimation)
retention_proj = []
for _, row in pairs.iterrows():
    ret = get_retention(row["WAR_82"])
    age_mod = age_modifier(int(row["age"]) + 1)
    retention_proj.append(row["WAR_82"] * ret * age_mod)
retention_proj = np.array(retention_proj)

# XGBoost model predictions
xgb_pred = models["WAR"].predict(X)

# Blend: 70% retention, 30% model
blended_train = 0.70 * retention_proj + 0.30 * xgb_pred

y_war = pairs["next_WAR"].fillna(0).values
war_residuals = y_war - blended_train
# Use absolute spread for symmetric CI (avoids bias from systematic over/under-prediction)
residual_spread = np.percentile(np.abs(war_residuals), 80)
war_ci_lo = -residual_spread
war_ci_hi = +residual_spread
blend_mae = np.mean(np.abs(war_residuals))
print(f"WAR projection (retention+model blend): MAE={blend_mae:.3f}, "
      f"80% CI: ±{residual_spread:.2f}")

# Step 4: Apply to current players
proj_war_retention = []
for _, row in projections.iterrows():
    ret = get_retention(row["curr_WAR_82"])
    age_mod = age_modifier(int(row["age"]) + 1)
    proj_war_retention.append(row["curr_WAR_82"] * ret * age_mod)
proj_war_retention = np.array(proj_war_retention)

xgb_proj = models["WAR"].predict(X_proj)
proj_war = 0.70 * proj_war_retention + 0.30 * xgb_proj

# WAR_82 (same approach)
proj_war82_retention = proj_war_retention  # WAR_82 is already pace-adjusted
xgb_proj82 = models["WAR_82"].predict(X_proj)
proj_war82 = 0.70 * proj_war82_retention + 0.30 * xgb_proj82

projections["proj_WAR"] = np.round(proj_war, 2)
projections["proj_WAR_82"] = np.round(proj_war82, 2)
projections["proj_WAR_lo"] = np.round(proj_war + war_ci_lo, 2)
projections["proj_WAR_hi"] = np.round(proj_war + war_ci_hi, 2)

# Use XGBoost/Ridge models for counting stats and GAR components
for name, model in models.items():
    if name in ("WAR", "WAR_82"):
        continue
    projections[f"proj_{name}"] = np.round(model.predict(X_proj), 2)

# Clip projections to reasonable ranges
projections["proj_GP"] = projections["proj_GP"].clip(0, 82).round(0).astype(int)
projections["proj_Goals"] = projections["proj_Goals"].clip(0, None)
projections["proj_Points"] = projections["proj_Points"].clip(0, None)

projections = projections.sort_values("proj_WAR", ascending=False)

# ── 6. Output ───────────────────────────────────────────────────────────────

print(f"\n{'='*90}")
print(f"TOP 40 PROJECTED PLAYERS — {CURRENT_SEASON + 1}-{str(CURRENT_SEASON + 2)[-2:]} SEASON")
print(f"{'='*90}\n")

print(f"{'Player':25s} {'Pos':>3s} {'Age':>3s} {'GP':>3s}  "
      f"{'cWAR':>5s} {'pWAR':>5s}  {'cPts':>4s} {'pPts':>4s}  "
      f"{'pGP':>3s} {'pGoals':>6s} {'pWAR/82':>7s}")
print("-" * 85)
for _, r in projections.head(40).iterrows():
    cp = int(r["curr_points"]) if pd.notna(r.get("curr_points")) else 0
    print(f"{r['player_name']:25s} {r['position']:>3s} {int(r['age_next']):3d} {int(r['GP']):3d}  "
          f"{r['curr_WAR']:5.2f} {r['proj_WAR']:5.2f}  "
          f"{cp:4d} {r['proj_Points']:4.0f}  "
          f"{r['proj_GP']:3.0f} {r['proj_Goals']:6.1f} {r['proj_WAR_82']:7.2f}")

# Biggest risers (projected WAR - current WAR)
projections["war_delta"] = projections["proj_WAR"] - projections["curr_WAR"]

print(f"\n{'='*90}")
print("BIGGEST PROJECTED RISERS")
print(f"{'='*90}\n")
risers = projections[projections["GP"] >= 40].nlargest(15, "war_delta")
for _, r in risers.iterrows():
    print(f"  {r['player_name']:25s} {r['curr_WAR']:+5.2f} → {r['proj_WAR']:+5.2f}  "
          f"({r['war_delta']:+5.2f})")

print(f"\n{'='*90}")
print("BIGGEST PROJECTED DECLINERS")
print(f"{'='*90}\n")
decliners = projections[projections["GP"] >= 40].nsmallest(15, "war_delta")
for _, r in decliners.iterrows():
    print(f"  {r['player_name']:25s} {r['curr_WAR']:+5.2f} → {r['proj_WAR']:+5.2f}  "
          f"({r['war_delta']:+5.2f})")

# Save
out_path = os.path.join(BASE, "contracts", "player_projections_2026.csv")
projections.to_csv(out_path, index=False)
print(f"\nSaved {len(projections)} projections to {out_path}")
