"""
Contract Prediction — Predict Cap % and Term from player stats.

Joins contract data to per-season player stats, builds features from
the most recent completed season before signing, and trains XGBoost
models to predict Cap % and Term.

ELCs and goalies are excluded (ELCs are formulaic, goalie stats live
in a separate pipeline).

Usage: python contracts/predict_contracts.py
"""

import os
import numpy as np
import pandas as pd
from unicodedata import normalize as _ucnorm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(BASE, "output")

# ── 1. Load data ─────────────────────────────────────────────────────────────

contracts = pd.read_csv(os.path.join(BASE, "contracts", "contracts.csv"))
contracts["cap_hit_num"] = (contracts["Cap Hit"]
                            .str.replace("$", "", regex=False)
                            .str.replace(",", "", regex=False)
                            .astype(float))
contracts["cap_pct"] = (contracts["Cap %"]
                        .str.replace("%", "", regex=False)
                        .astype(float))
contracts["term_yr"] = (contracts["Term"]
                        .str.replace("yr", "", regex=False)
                        .astype(int))
contracts["sign_date"] = pd.to_datetime(contracts["Date"], format="%d-%b-%y")

# Filter: standard contracts only, skaters only
contracts = contracts[(contracts["Level"] == "STD") & (contracts["POS"] != "G")].copy()
print(f"Contracts after filtering (STD, skaters): {len(contracts)}")

# Stats sources
skater_war = pd.read_csv(os.path.join(OUTPUT, "dashboard_skater_war.csv"))
bpr = pd.read_csv(os.path.join(OUTPUT, "v2_final_ratings_by_season.csv"))
daily_war = pd.read_csv(os.path.join(OUTPUT, "v5_daily_war.csv"))

# ── 2. Name matching ────────────────────────────────────────────────────────

_contract_name_fixes = {
    "Joshua Norris": "Josh Norris",
    "John-Jason Peterka": "JJ Peterka",
    "Matthew Beniers": "Matty Beniers",
    "Janis Jérôme Moser": "J.J. Moser",
    "Mike Matheson": "Michael Matheson",
    "Cameron York": "Cam York",
    "Artyom Zub": "Artem Zub",
    "Nicklaus Perbix": "Nick Perbix",
    "Joseph Veleno": "Joe Veleno",
    "Alexander Petrovic": "Alex Petrovic",
    "Pat Maroon": "Patrick Maroon",
    "Yegor Zamula": "Egor Zamula",
    "Nicolai Knyzhov": "Nikolai Knyzhov",
    "Fyodor Svechkov": "Fedor Svechkov",
    "Callan Foote": "Cal Foote",
    "Zachary Jones": "Zac Jones",
    "Joshua Dunne": "Josh Dunne",
    "Samuel Poulin": "Sam Poulin",
    "Matthew Stienburg": "Matt Stienburg",
    "Ronald Attard": "Ronnie Attard",
    "Danny O'Regan": "Daniel O'Regan",
    "Nikolay Prokhorkin": "Nikolai Prokhorkin",
    "Yegor Korshkov": "Egor Korshkov",
    "Matthew Savoie": "Matt Savoie",
    "Benjamin Kindel": "Ben Kindel",
    "Maxim Shabanov": "Max Shabanov",
    "Cameron Lund": "Cam Lund",
}


def _strip_accents(s):
    return _ucnorm("NFD", s).encode("ascii", "ignore").decode("ascii")


# Build player_name -> player_id map from stats
_name_to_id = (skater_war.dropna(subset=["player_name"])
               .drop_duplicates("player_name")
               .set_index("player_name")["player_id"].to_dict())
_name_lower = {n.lower(): n for n in _name_to_id}


def resolve_name(contract_name):
    """Resolve a contract player name to the stats player_name."""
    if contract_name in _contract_name_fixes:
        return _contract_name_fixes[contract_name]
    if contract_name in _name_to_id:
        return contract_name
    stripped = _strip_accents(contract_name)
    if stripped in _name_to_id:
        return stripped
    if stripped.lower() in _name_lower:
        return _name_lower[stripped.lower()]
    no_hyph = _strip_accents(contract_name.replace("-", " "))
    if no_hyph in _name_to_id:
        return no_hyph
    return None


contracts["stats_name"] = contracts["Player"].map(resolve_name)
contracts["player_id"] = contracts["stats_name"].map(_name_to_id)

matched = contracts["player_id"].notna().sum()
print(f"Matched to stats: {matched}/{len(contracts)} ({matched/len(contracts)*100:.1f}%)")

contracts = contracts.dropna(subset=["player_id"]).copy()
contracts["player_id"] = contracts["player_id"].astype(int)

# ── 3. Determine stats season for each contract ─────────────────────────────
# Use the most recent completed season before signing.
# Season code convention: season=2024 means the 2023-24 season (ends ~June 2024).
# July+ signings -> use season that just ended; Jan-June -> use prior season.
contracts["stats_season"] = contracts["sign_date"].apply(
    lambda d: d.year if d.month >= 7 else d.year - 1
)

# ── 4. Merge stats ──────────────────────────────────────────────────────────

# dashboard_skater_war: box score stats + GAR
sw_cols = ["player_id", "season", "GP", "goals", "assists_1", "assists_2", "points",
           "shots", "hits", "blocks", "takeaways", "giveaways",
           "toi_all", "toi_5v5", "toi_pp", "toi_pk",
           "EV_O_GAR", "EV_D_GAR", "PP_GAR", "PK_GAR", "PEN_GAR", "GAR", "WAR"]
sw = skater_war[[c for c in sw_cols if c in skater_war.columns]].copy()

df = contracts.merge(sw, left_on=["player_id", "stats_season"],
                     right_on=["player_id", "season"], how="inner")
print(f"After merging box score stats: {len(df)}")

# daily_war: WAR_82 (pace-adjusted)
dw_cols = ["player_id", "season", "WAR_82", "EV_O_rate", "EV_D_rate", "PP_rate", "PK_rate"]
dw = daily_war[[c for c in dw_cols if c in daily_war.columns]].copy()
df = df.merge(dw, left_on=["player_id", "stats_season"],
              right_on=["player_id", "season"], how="left", suffixes=("", "_dw"))

# BPR: per-season ratings
bpr_cols = ["player_id", "season", "BPR_O", "BPR_D", "BPR", "total_BPR", "PP_O", "PK_D"]
bp = bpr[[c for c in bpr_cols if c in bpr.columns]].copy()
df = df.merge(bp, left_on=["player_id", "stats_season"],
              right_on=["player_id", "season"], how="left", suffixes=("", "_bpr"))

print(f"After all merges: {len(df)}")

# ── 5. Also grab prior season stats for trend features ───────────────────────

sw_prior = sw.copy()
sw_prior = sw_prior.rename(columns={c: f"prior_{c}" for c in sw_prior.columns if c not in ["player_id", "season"]})
sw_prior["prior_season_key"] = sw_prior["season"] + 1  # season N stats -> used for season N+1 contracts
df = df.merge(sw_prior[["player_id", "prior_season_key", "prior_GP", "prior_WAR", "prior_GAR",
                         "prior_goals", "prior_points", "prior_toi_all"]],
              left_on=["player_id", "stats_season"],
              right_on=["player_id", "prior_season_key"], how="left")

# ── 6. Feature engineering ───────────────────────────────────────────────────

feat = pd.DataFrame(index=df.index)

# Contract metadata
feat["sign_age"] = df["Sign Age"]
feat["is_ufa"] = (df["Sign Status"] == "UFA").astype(int)
feat["pos_D"] = (df["POS"] == "D").astype(int)
feat["pos_C"] = (df["POS"] == "C").astype(int)

# Per-game rates (avoid division by zero)
gp = df["GP"].clip(lower=1)
feat["gp"] = df["GP"]
feat["goals_pg"] = df["goals"] / gp
feat["points_pg"] = df["points"] / gp
feat["shots_pg"] = df["shots"] / gp
feat["hits_pg"] = df["hits"] / gp
feat["blocks_pg"] = df["blocks"] / gp
feat["toi_pg"] = df["toi_all"] / gp
feat["toi_pp_pg"] = df["toi_pp"] / gp
feat["toi_pk_pg"] = df["toi_pk"] / gp

# WAR / GAR
feat["WAR"] = df["WAR"]
feat["GAR"] = df["GAR"]
feat["EV_O_GAR"] = df["EV_O_GAR"]
feat["EV_D_GAR"] = df["EV_D_GAR"]
feat["PP_GAR"] = df["PP_GAR"]
feat["PK_GAR"] = df["PK_GAR"]
feat["PEN_GAR"] = df["PEN_GAR"]
feat["WAR_82"] = df.get("WAR_82", pd.Series(dtype=float))

# BPR components
for c in ["BPR_O", "BPR_D", "BPR", "total_BPR", "PP_O", "PK_D"]:
    feat[c] = df.get(c, pd.Series(dtype=float))

# Rates from daily model
for c in ["EV_O_rate", "EV_D_rate", "PP_rate", "PK_rate"]:
    feat[c] = df.get(c, pd.Series(dtype=float))

# Trend features (current vs prior season)
feat["WAR_trend"] = df["WAR"] - df["prior_WAR"].fillna(df["WAR"])
feat["goals_trend"] = df["goals"] - df["prior_goals"].fillna(df["goals"])
feat["WAR_2yr_avg"] = (df["WAR"] + df["prior_WAR"].fillna(df["WAR"])) / 2

# Targets
targets = pd.DataFrame(index=df.index)
targets["cap_pct"] = df["cap_pct"]
targets["term_yr"] = df["term_yr"]
targets["cap_hit_num"] = df["cap_hit_num"]

# Keep metadata for reporting
meta = df[["Player", "POS", "Sign Age", "Sign Status", "sign_date",
           "stats_season", "Cap Hit", "Cap %", "Term"]].copy()

# Drop rows with missing features
valid = feat.notna().all(axis=1)
print(f"Rows with complete features: {valid.sum()}/{len(feat)}")

feat = feat[valid].copy()
targets = targets[valid].copy()
meta = meta[valid].copy()

print(f"\nFeature matrix: {feat.shape}")
print(f"Target distribution (Cap %):")
print(f"  Mean: {targets['cap_pct'].mean():.2f}%  Median: {targets['cap_pct'].median():.2f}%")
print(f"  Min: {targets['cap_pct'].min():.2f}%  Max: {targets['cap_pct'].max():.2f}%")

# ── 7. Train/test split (temporal) ──────────────────────────────────────────

split_date = pd.Timestamp("2024-07-01")
train_mask = meta["sign_date"] < split_date
test_mask = ~train_mask

X_train, X_test = feat[train_mask], feat[test_mask]
y_train_pct, y_test_pct = targets.loc[train_mask, "cap_pct"], targets.loc[test_mask, "cap_pct"]
y_train_term, y_test_term = targets.loc[train_mask, "term_yr"], targets.loc[test_mask, "term_yr"]
meta_test = meta[test_mask]

print(f"\nTrain: {len(X_train)} contracts (before {split_date.date()})")
print(f"Test:  {len(X_test)} contracts (after {split_date.date()})")

# ── 8. Model: Cap % ─────────────────────────────────────────────────────────

print("\n" + "="*70)
print("MODEL 1: Cap % Prediction")
print("="*70)

xgb_pct = XGBRegressor(
    n_estimators=300, max_depth=5, learning_rate=0.08,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
)
xgb_pct.fit(X_train, y_train_pct)

# Cross-val on training set
cv_scores = cross_val_score(xgb_pct, X_train, y_train_pct,
                            cv=5, scoring="neg_mean_absolute_error")
print(f"\n5-fold CV MAE (train): {-cv_scores.mean():.3f}% ± {cv_scores.std():.3f}%")

# Test set
pred_pct = xgb_pct.predict(X_test)
mae_pct = mean_absolute_error(y_test_pct, pred_pct)
r2_pct = r2_score(y_test_pct, pred_pct)
print(f"Test MAE: {mae_pct:.3f}%")
print(f"Test R²:  {r2_pct:.3f}")

# Feature importance
imp = pd.Series(xgb_pct.feature_importances_, index=feat.columns).sort_values(ascending=False)
print(f"\nTop 15 features (Cap %):")
for f, v in imp.head(15).items():
    print(f"  {f:20s} {v:.4f}")

# ── 9. Model: Term ──────────────────────────────────────────────────────────

print("\n" + "="*70)
print("MODEL 2: Term Prediction")
print("="*70)

xgb_term = XGBRegressor(
    n_estimators=300, max_depth=4, learning_rate=0.08,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
)
xgb_term.fit(X_train, y_train_term)

cv_term = cross_val_score(xgb_term, X_train, y_train_term,
                          cv=5, scoring="neg_mean_absolute_error")
print(f"\n5-fold CV MAE (train): {-cv_term.mean():.2f} yrs ± {cv_term.std():.2f}")

pred_term = xgb_term.predict(X_test)
mae_term = mean_absolute_error(y_test_term, pred_term)
r2_term = r2_score(y_test_term, pred_term)
print(f"Test MAE: {mae_term:.2f} yrs")
print(f"Test R²:  {r2_term:.3f}")

imp_term = pd.Series(xgb_term.feature_importances_, index=feat.columns).sort_values(ascending=False)
print(f"\nTop 15 features (Term):")
for f, v in imp_term.head(15).items():
    print(f"  {f:20s} {v:.4f}")

# ── 10. Results table ────────────────────────────────────────────────────────

print("\n" + "="*70)
print("TEST SET: Top 30 Contracts by Cap Hit")
print("="*70)

results = meta_test.copy()
results["pred_cap_pct"] = pred_pct
results["actual_cap_pct"] = y_test_pct.values
results["residual_pct"] = pred_pct - y_test_pct.values
results["pred_term"] = np.round(pred_term, 1)
results["actual_term"] = y_test_term.values
results = results.sort_values("actual_cap_pct", ascending=False)

print(f"\n{'Player':25s} {'Pos':>3s} {'Age':>3s} {'Status':>6s} "
      f"{'Actual':>7s} {'Pred':>7s} {'Diff':>7s}  "
      f"{'Term':>4s} {'PTerm':>5s}")
print("-" * 85)
for _, r in results.head(30).iterrows():
    print(f"{r['Player']:25s} {r['POS']:>3s} {int(r['Sign Age']):3d} {r['Sign Status']:>6s} "
          f"{r['actual_cap_pct']:6.2f}% {r['pred_cap_pct']:6.2f}% {r['residual_pct']:+6.2f}%  "
          f"{r['actual_term']:3d}yr {r['pred_term']:4.1f}yr")

# ── 11. Value analysis ──────────────────────────────────────────────────────

print("\n" + "="*70)
print("BEST VALUE CONTRACTS (model says should cost more)")
print("="*70)

# Negative residual = actual is lower than predicted
results_sorted = results.sort_values("residual_pct", ascending=False)
print(f"\n{'Player':25s} {'Pos':>3s} {'Actual':>7s} {'Pred':>7s} {'Diff':>7s} {'Cap Hit':>12s}")
print("-" * 70)
for _, r in results_sorted.head(15).iterrows():
    print(f"{r['Player']:25s} {r['POS']:>3s} {r['actual_cap_pct']:6.2f}% "
          f"{r['pred_cap_pct']:6.2f}% {r['residual_pct']:+6.2f}% {r['Cap Hit']:>12s}")

print(f"\n{'WORST VALUE CONTRACTS (model says should cost less)':70s}")
print("-" * 70)
for _, r in results_sorted.tail(15).iterrows():
    print(f"{r['Player']:25s} {r['POS']:>3s} {r['actual_cap_pct']:6.2f}% "
          f"{r['pred_cap_pct']:6.2f}% {r['residual_pct']:+6.2f}% {r['Cap Hit']:>12s}")

# ── 12. Save predictions ────────────────────────────────────────────────────

# Predict on full dataset
all_pred_pct = xgb_pct.predict(feat)
all_pred_term = xgb_term.predict(feat)

out = meta.copy()
out["actual_cap_pct"] = targets["cap_pct"].values
out["pred_cap_pct"] = np.round(all_pred_pct, 3)
out["residual_pct"] = np.round(all_pred_pct - targets["cap_pct"].values, 3)
out["actual_term"] = targets["term_yr"].values
out["pred_term"] = np.round(all_pred_term, 1)
out["cap_hit"] = targets["cap_hit_num"].values

out_path = os.path.join(BASE, "contracts", "contract_predictions.csv")
out.to_csv(out_path, index=False)
print(f"\nSaved predictions to {out_path}")
print(f"Total: {len(out)} contracts with predictions")
