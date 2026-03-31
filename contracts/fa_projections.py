"""
Free Agent Projections — Predict next contracts for upcoming UFAs/RFAs.

Identifies players whose contracts expire after the current season,
pulls their latest stats, and uses the trained contract prediction
model to project their next deal (Cap %, Term, AAV).

Usage: python contracts/fa_projections.py
"""

import os
import numpy as np
import pandas as pd
from unicodedata import normalize as _ucnorm
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(BASE, "output")

# Projected 2026-27 salary cap
PROJECTED_CAP_2027 = 104_500_000

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

skater_war = pd.read_csv(os.path.join(OUTPUT, "dashboard_skater_war.csv"))
daily_war = pd.read_csv(os.path.join(OUTPUT, "v5_daily_war.csv"))
bpr = pd.read_csv(os.path.join(OUTPUT, "v2_final_ratings_by_season.csv"))

# ── 2. Identify expiring contracts ──────────────────────────────────────────

contracts["first_season"] = contracts["sign_date"].apply(
    lambda d: d.year + 1 if d.month >= 7 else d.year
)
contracts["expiry_season"] = contracts["first_season"] + contracts["term_yr"] - 1

# Use each player's LATEST contract to determine their status
# (players who signed extensions already have a newer contract superseding the old one)
contracts = contracts.sort_values("sign_date").drop_duplicates("Player", keep="last")

# Players whose latest contract expires after 2025-26 season (become FA summer 2026)
expiring = contracts[contracts["expiry_season"] == 2026].copy()
expiring = expiring[expiring["POS"] != "G"]  # skaters only

print(f"Expiring skater contracts (summer 2026): {len(expiring)}")

# ── 3. Name matching ────────────────────────────────────────────────────────

_contract_name_fixes = {
    "Joshua Norris": "Josh Norris", "John-Jason Peterka": "JJ Peterka",
    "Matthew Beniers": "Matty Beniers", "Janis Jérôme Moser": "J.J. Moser",
    "Mike Matheson": "Michael Matheson", "Cameron York": "Cam York",
    "Artyom Zub": "Artem Zub", "Nicklaus Perbix": "Nick Perbix",
    "Joseph Veleno": "Joe Veleno", "Alexander Petrovic": "Alex Petrovic",
    "Pat Maroon": "Patrick Maroon", "Yegor Zamula": "Egor Zamula",
    "Nicolai Knyzhov": "Nikolai Knyzhov", "Fyodor Svechkov": "Fedor Svechkov",
    "Callan Foote": "Cal Foote", "Zachary Jones": "Zac Jones",
    "Joshua Dunne": "Josh Dunne", "Samuel Poulin": "Sam Poulin",
    "Matthew Stienburg": "Matt Stienburg", "Ronald Attard": "Ronnie Attard",
    "Danny O'Regan": "Daniel O'Regan", "Nikolay Prokhorkin": "Nikolai Prokhorkin",
    "Yegor Korshkov": "Egor Korshkov", "Matthew Savoie": "Matt Savoie",
    "Benjamin Kindel": "Ben Kindel", "Maxim Shabanov": "Max Shabanov",
    "Cameron Lund": "Cam Lund",
}


def _strip_accents(s):
    return _ucnorm("NFD", s).encode("ascii", "ignore").decode("ascii")


_name_to_id = (skater_war.dropna(subset=["player_name"])
               .drop_duplicates("player_name")
               .set_index("player_name")["player_id"].to_dict())
_name_lower = {n.lower(): n for n in _name_to_id}


def resolve_name(contract_name):
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


expiring["stats_name"] = expiring["Player"].map(resolve_name)
expiring["player_id"] = expiring["stats_name"].map(_name_to_id)
expiring = expiring.dropna(subset=["player_id"]).copy()
expiring["player_id"] = expiring["player_id"].astype(int)

print(f"Matched to stats: {len(expiring)}")

# ── 4. Train model on historical contracts ──────────────────────────────────
# Retrain from scratch so we use the full dataset (same approach as predict_contracts.py)

# Filter training data: STD contracts, skaters, signed before this season
train_contracts = contracts[
    (contracts["Level"] == "STD") &
    (contracts["POS"] != "G")
].copy()
train_contracts["stats_name"] = train_contracts["Player"].map(resolve_name)
train_contracts["player_id"] = train_contracts["stats_name"].map(_name_to_id)
train_contracts = train_contracts.dropna(subset=["player_id"]).copy()
train_contracts["player_id"] = train_contracts["player_id"].astype(int)

# Stats season: last completed season before signing
train_contracts["stats_season"] = train_contracts["sign_date"].apply(
    lambda d: d.year if d.month >= 7 else d.year - 1
)

# Merge stats
sw_cols = ["player_id", "season", "GP", "goals", "assists_1", "assists_2", "points",
           "shots", "hits", "blocks", "takeaways", "giveaways",
           "toi_all", "toi_5v5", "toi_pp", "toi_pk",
           "EV_O_GAR", "EV_D_GAR", "PP_GAR", "PK_GAR", "PEN_GAR", "GAR", "WAR"]
sw = skater_war[[c for c in sw_cols if c in skater_war.columns]].copy()

train = train_contracts.merge(sw, left_on=["player_id", "stats_season"],
                              right_on=["player_id", "season"], how="inner")

dw_cols = ["player_id", "season", "WAR_82", "EV_O_rate", "EV_D_rate", "PP_rate", "PK_rate"]
dw = daily_war[[c for c in dw_cols if c in daily_war.columns]].copy()
train = train.merge(dw, left_on=["player_id", "stats_season"],
                    right_on=["player_id", "season"], how="left", suffixes=("", "_dw"))

bpr_cols = ["player_id", "season", "BPR_O", "BPR_D", "BPR", "total_BPR", "PP_O", "PK_D"]
bp = bpr[[c for c in bpr_cols if c in bpr.columns]].copy()
train = train.merge(bp, left_on=["player_id", "stats_season"],
                    right_on=["player_id", "season"], how="left", suffixes=("", "_bpr"))

# Prior season for trend
sw_prior = sw.rename(columns={c: f"prior_{c}" for c in sw.columns if c not in ["player_id", "season"]})
sw_prior["prior_season_key"] = sw_prior["season"] + 1
train = train.merge(sw_prior[["player_id", "prior_season_key", "prior_GP", "prior_WAR",
                               "prior_GAR", "prior_goals", "prior_points", "prior_toi_all"]],
                    left_on=["player_id", "stats_season"],
                    right_on=["player_id", "prior_season_key"], how="left")


def build_features(df):
    feat = pd.DataFrame(index=df.index)
    feat["sign_age"] = df["Sign Age"]
    feat["is_ufa"] = (df["Sign Status"] == "UFA").astype(int) if "Sign Status" in df else 0
    feat["pos_D"] = (df["POS"] == "D").astype(int)
    feat["pos_C"] = (df["POS"] == "C").astype(int)

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

    feat["WAR"] = df["WAR"]
    feat["GAR"] = df["GAR"]
    feat["EV_O_GAR"] = df["EV_O_GAR"]
    feat["EV_D_GAR"] = df["EV_D_GAR"]
    feat["PP_GAR"] = df["PP_GAR"]
    feat["PK_GAR"] = df["PK_GAR"]
    feat["PEN_GAR"] = df["PEN_GAR"]
    feat["WAR_82"] = df.get("WAR_82", pd.Series(dtype=float))

    for c in ["BPR_O", "BPR_D", "BPR", "total_BPR", "PP_O", "PK_D"]:
        feat[c] = df.get(c, pd.Series(dtype=float))

    for c in ["EV_O_rate", "EV_D_rate", "PP_rate", "PK_rate"]:
        feat[c] = df.get(c, pd.Series(dtype=float))

    feat["WAR_trend"] = df["WAR"] - df["prior_WAR"].fillna(df["WAR"])
    feat["goals_trend"] = df["goals"] - df["prior_goals"].fillna(df["goals"])
    feat["WAR_2yr_avg"] = (df["WAR"] + df["prior_WAR"].fillna(df["WAR"])) / 2

    return feat


train_feat = build_features(train)
train_feat = train_feat.fillna(0)
valid = train_feat.notna().all(axis=1)
train_feat = train_feat[valid]
train_pct = train.loc[valid, "cap_pct"]
train_term = train.loc[valid, "term_yr"]

print(f"Training samples: {len(train_feat)}")

# Train models
xgb_pct = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.08,
                        subsample=0.8, colsample_bytree=0.8, random_state=42)
xgb_pct.fit(train_feat, train_pct)

xgb_term = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.08,
                         subsample=0.8, colsample_bytree=0.8, random_state=42)
xgb_term.fit(train_feat, train_term)

# ── 5. Build features for expiring players ──────────────────────────────────
# Use their 2025 season (current) stats + 2024 prior season

CURRENT_SEASON = 2025

fa_stats = expiring.merge(sw, left_on=["player_id"],
                          right_on=["player_id"], how="inner")
# Keep only current season stats
fa_stats = fa_stats[fa_stats["season"] == CURRENT_SEASON].copy()

# Merge daily WAR
fa_stats = fa_stats.merge(dw, left_on=["player_id"],
                          right_on=["player_id"], how="left", suffixes=("", "_dw"))
fa_stats = fa_stats[fa_stats["season_dw"] == CURRENT_SEASON].copy() if "season_dw" in fa_stats.columns else fa_stats

# Merge BPR
fa_stats = fa_stats.merge(bp, left_on=["player_id"],
                          right_on=["player_id"], how="left", suffixes=("", "_bpr"))
# Keep BPR for current or most recent season
if "season_bpr" in fa_stats.columns:
    fa_stats = fa_stats.sort_values("season_bpr", ascending=False).drop_duplicates("player_id", keep="first")

# Prior season (2024)
fa_stats = fa_stats.merge(sw_prior[["player_id", "prior_season_key", "prior_GP", "prior_WAR",
                                     "prior_GAR", "prior_goals", "prior_points", "prior_toi_all"]],
                          left_on=["player_id"],
                          right_on=["player_id"], how="left")
if "prior_season_key" in fa_stats.columns:
    fa_stats = fa_stats[fa_stats["prior_season_key"] == CURRENT_SEASON].copy()

# Estimate signing age: current age + 1 (signing happens in summer)
# Use Sign Age from their current contract + years elapsed
fa_stats["Sign Age"] = expiring.set_index("player_id").loc[
    fa_stats["player_id"].values, "Sign Age"
].values + expiring.set_index("player_id").loc[
    fa_stats["player_id"].values, "term_yr"
].values
fa_stats["Sign Status"] = expiring.set_index("player_id").loc[
    fa_stats["player_id"].values, "Sign Expiry"
].values
fa_stats["POS"] = expiring.set_index("player_id").loc[
    fa_stats["player_id"].values, "POS"
].values

print(f"FAs with current season stats: {len(fa_stats)}")

# Build features
fa_feat = build_features(fa_stats)
fa_feat = fa_feat.fillna(0)

# ── 6. Predict with confidence intervals ────────────────────────────────────

pred_pct = xgb_pct.predict(fa_feat)
pred_term = xgb_term.predict(fa_feat)

# Compute confidence intervals from cross-validation residuals
from sklearn.model_selection import cross_val_predict
cv_pred_pct = cross_val_predict(
    XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.08,
                 subsample=0.8, colsample_bytree=0.8, random_state=42),
    train_feat, train_pct, cv=5)
cv_pred_term = cross_val_predict(
    XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.08,
                 subsample=0.8, colsample_bytree=0.8, random_state=42),
    train_feat, train_term, cv=5)
residuals_pct = train_pct.values - cv_pred_pct
residuals_term = train_term.values - cv_pred_term

# 80% confidence interval (10th to 90th percentile of CV residuals)
pct_lo = np.percentile(residuals_pct, 10)
pct_hi = np.percentile(residuals_pct, 90)
term_lo = np.percentile(residuals_term, 10)
term_hi = np.percentile(residuals_term, 90)

print(f"Cap % residual 80% CI: [{pct_lo:+.2f}%, {pct_hi:+.2f}%]")
print(f"Term residual 80% CI: [{term_lo:+.1f}yr, {term_hi:+.1f}yr]")

# Convert to dollars using projected cap
pred_aav = pred_pct / 100 * PROJECTED_CAP_2027
aav_lo = (pred_pct + pct_lo) / 100 * PROJECTED_CAP_2027
aav_hi = (pred_pct + pct_hi) / 100 * PROJECTED_CAP_2027

term_pred_clipped = np.clip(np.round(pred_term), 1, 8).astype(int)
term_lo_clipped = np.clip(np.round(pred_term + term_lo), 1, 8).astype(int)
term_hi_clipped = np.clip(np.round(pred_term + term_hi), 1, 8).astype(int)

# Build results
results = pd.DataFrame({
    "Player": expiring.set_index("player_id").loc[fa_stats["player_id"].values, "Player"].values,
    "stats_name": expiring.set_index("player_id").loc[fa_stats["player_id"].values, "stats_name"].values,
    "POS": fa_stats["POS"].values,
    "Age": fa_stats["Sign Age"].values.astype(int),
    "Status": fa_stats["Sign Status"].values,
    "Current_Cap_Hit": expiring.set_index("player_id").loc[
        fa_stats["player_id"].values, "Cap Hit"
    ].values,
    "Current_Cap_Num": expiring.set_index("player_id").loc[
        fa_stats["player_id"].values, "cap_hit_num"
    ].values,
    "GP": fa_stats["GP"].values.astype(int),
    "WAR": fa_stats["WAR"].values.round(2),
    "WAR_82": fa_stats.get("WAR_82", pd.Series(0, index=fa_stats.index)).values.round(2),
    "Points": fa_stats["points"].values.astype(int),
    "Pred_Cap_Pct": np.round(pred_pct, 2),
    "Pred_AAV": np.round(pred_aav, -4).astype(int),
    "AAV_Lo": np.round(np.maximum(aav_lo, 775000), -4).astype(int),
    "AAV_Hi": np.round(aav_hi, -4).astype(int),
    "Pred_Term": term_pred_clipped,
    "Term_Lo": term_lo_clipped,
    "Term_Hi": term_hi_clipped,
})

# Add delta (predicted vs current)
results["AAV_Delta"] = results["Pred_AAV"] - results["Current_Cap_Num"]

results = results.sort_values("Pred_AAV", ascending=False)

# ── 7. Output ───────────────────────────────────────────────────────────────

print(f"\n{'='*90}")
print(f"2026 FREE AGENT CONTRACT PROJECTIONS (Cap: ${PROJECTED_CAP_2027/1e6:.1f}M)")
print(f"{'='*90}")

print(f"\n{'Player':25s} {'Pos':>3s} {'Age':>3s} {'Sts':>3s} {'GP':>3s} {'WAR':>5s} {'Pts':>3s} "
      f"{'Current':>12s}  {'Projected':>12s} {'Term':>4s} {'Delta':>12s}")
print("-" * 100)

for _, r in results.head(50).iterrows():
    delta_str = f"{'+'if r['AAV_Delta']>0 else ''}{r['AAV_Delta']:,.0f}"
    print(f"{r['Player']:25s} {r['POS']:>3s} {r['Age']:3d} {r['Status']:>3s} {r['GP']:3d} "
          f"{r['WAR']:5.2f} {r['Points']:3d} "
          f"${r['Current_Cap_Num']:>11,.0f}  ${r['Pred_AAV']:>11,.0f} {r['Pred_Term']:3d}yr "
          f"{delta_str:>12s}")

# UFA-specific leaderboard
print(f"\n{'='*90}")
print(f"TOP UFA TARGETS")
print(f"{'='*90}\n")

ufas = results[results["Status"] == "UFA"].head(30)
print(f"{'Player':25s} {'Pos':>3s} {'Age':>3s} {'GP':>3s} {'WAR':>5s} {'Pts':>3s} "
      f"{'Projected AAV':>14s} {'Term':>4s}")
print("-" * 80)
for _, r in ufas.iterrows():
    print(f"{r['Player']:25s} {r['POS']:>3s} {r['Age']:3d} {r['GP']:3d} "
          f"{r['WAR']:5.2f} {r['Points']:3d} "
          f"${r['Pred_AAV']:>13,.0f} {r['Pred_Term']:3d}yr")

# RFA-specific leaderboard
print(f"\n{'='*90}")
print(f"TOP RFA TARGETS")
print(f"{'='*90}\n")

rfas = results[results["Status"] == "RFA"].head(20)
print(f"{'Player':25s} {'Pos':>3s} {'Age':>3s} {'GP':>3s} {'WAR':>5s} {'Pts':>3s} "
      f"{'Projected AAV':>14s} {'Term':>4s}")
print("-" * 80)
for _, r in rfas.iterrows():
    print(f"{r['Player']:25s} {r['POS']:>3s} {r['Age']:3d} {r['GP']:3d} "
          f"{r['WAR']:5.2f} {r['Points']:3d} "
          f"${r['Pred_AAV']:>13,.0f} {r['Pred_Term']:3d}yr")

# Save
out_path = os.path.join(BASE, "contracts", "fa_projections_2026.csv")
results.to_csv(out_path, index=False)
print(f"\nSaved {len(results)} FA projections to {out_path}")
