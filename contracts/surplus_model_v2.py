"""
Surplus Value Model v2 — Per-season multi-metric valuation.

For each player-season, predicts what Cap % their stats are worth using
a Ridge model trained on all player-seasons with known cap hits.
Surplus = predicted Cap % - actual Cap %.

No contract signing logic — just: given these stats, what should you cost?

Usage: python contracts/surplus_model_v2.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(BASE, "output")

SALARY_CAP = {
    2015: 71_400_000, 2016: 73_000_000, 2017: 75_000_000, 2018: 79_500_000,
    2019: 81_500_000, 2020: 81_500_000, 2021: 81_500_000, 2022: 82_500_000,
    2023: 83_500_000, 2024: 88_000_000, 2025: 95_500_000,
}

# ── 1. Load data ─────────────────────────────────────────────────────────────

daily_war = pd.read_csv(os.path.join(OUTPUT, "v5_daily_war.csv"))
skater_war = pd.read_csv(os.path.join(OUTPUT, "dashboard_skater_war.csv"))
bpr = pd.read_csv(os.path.join(OUTPUT, "v2_final_ratings_by_season.csv"))

# Surplus v1 has per-season cap hits already expanded
surplus_v1 = pd.read_csv(os.path.join(BASE, "contracts", "surplus_values.csv"))

print(f"Player-seasons with cap data: {len(surplus_v1)}")

# ── 2. Build the dataset ────────────────────────────────────────────────────
# Join all stats onto the surplus_v1 rows (which already have player_id, season, cap_hit)

df = surplus_v1[["player_id", "player_name", "position", "season", "GP",
                 "WAR", "WAR_82", "cap_hit", "contract_type", "sign_status",
                 "sign_age", "draft_year", "draft_round"]].copy()

# Box score stats
sw_cols = ["player_id", "season", "goals", "assists_1", "assists_2", "points",
           "shots", "hits", "blocks", "takeaways", "giveaways",
           "toi_all", "toi_5v5", "toi_pp", "toi_pk",
           "EV_O_GAR", "EV_D_GAR", "PP_GAR", "PK_GAR", "PEN_GAR", "FO_GAR", "GAR"]
sw = skater_war[[c for c in sw_cols if c in skater_war.columns]].copy()
df = df.merge(sw, on=["player_id", "season"], how="left")

# Daily WAR rates
dw_cols = ["player_id", "season", "EV_O_rate", "EV_D_rate", "PP_rate", "PK_rate", "PEN_rate",
           "WAR_O", "WAR_D"]
dw = daily_war[[c for c in dw_cols if c in daily_war.columns]].copy()
df = df.merge(dw, on=["player_id", "season"], how="left")

# BPR
bpr_cols = ["player_id", "season", "BPR_O", "BPR_D", "BPR", "total_BPR", "PP_O", "PK_D"]
bp = bpr[[c for c in bpr_cols if c in bpr.columns]].copy()
df = df.merge(bp, on=["player_id", "season"], how="left")

# Target: actual cap % for this season
df["salary_cap"] = df["season"].map(SALARY_CAP)
df["actual_cap_pct"] = df["cap_hit"] / df["salary_cap"] * 100

# Drop rows missing key stats
df = df.dropna(subset=["GP", "WAR", "goals"]).copy()
print(f"Player-seasons with full stats: {len(df)}")

# ── 3. Features ─────────────────────────────────────────────────────────────

def build_features(df):
    feat = pd.DataFrame(index=df.index)
    gp = df["GP"].clip(lower=1)

    # ── Player metadata ──
    feat["age"] = df["sign_age"]
    feat["age_sq"] = df["sign_age"] ** 2  # captures non-linear age curve
    feat["pos_D"] = (df["position"] == "D").astype(int)
    feat["is_elc"] = (df["contract_type"] == "ELC").astype(int)
    feat["is_ufa"] = (df["sign_status"] == "UFA").astype(int)

    # Draft pedigree (1st round = 1, undrafted = 8)
    feat["draft_round"] = df["draft_round"].fillna(8).clip(upper=8)
    feat["is_first_round"] = (df["draft_round"] == 1).astype(int)

    # Career stage: seasons since draft year (proxy for experience)
    feat["pro_years"] = (df["season"] - df["draft_year"].fillna(df["season"] - 3)).clip(lower=0)

    # ── Basic counting rates ──
    feat["gp"] = df["GP"]
    feat["goals_pg"] = df["goals"] / gp
    feat["assists_pg"] = (df.get("assists_1", 0) + df.get("assists_2", 0)) / gp
    feat["points_pg"] = df["points"] / gp
    feat["shots_pg"] = df["shots"] / gp
    feat["hits_pg"] = df["hits"] / gp
    feat["blocks_pg"] = df["blocks"] / gp

    # ── Ice time ──
    feat["toi_pg"] = df["toi_all"] / gp
    feat["toi_pp_pg"] = df["toi_pp"] / gp
    feat["toi_pk_pg"] = df["toi_pk"] / gp
    feat["pp_share"] = df["toi_pp"] / df["toi_all"].clip(lower=1)  # PP usage rate

    # ── WAR components ──
    feat["WAR"] = df["WAR"]
    feat["WAR_82"] = df["WAR_82"]
    feat["WAR_O"] = df.get("WAR_O", 0)
    feat["WAR_D"] = df.get("WAR_D", 0)

    # ── GAR components ──
    feat["EV_O_GAR"] = df["EV_O_GAR"]
    feat["EV_D_GAR"] = df["EV_D_GAR"]
    feat["PP_GAR"] = df["PP_GAR"]
    feat["PK_GAR"] = df["PK_GAR"]
    feat["PEN_GAR"] = df["PEN_GAR"]
    feat["GAR"] = df["GAR"]

    # ── Per-60 rates ──
    for c in ["EV_O_rate", "EV_D_rate", "PP_rate", "PK_rate"]:
        feat[c] = df.get(c, 0)

    # ── BPR ──
    for c in ["BPR_O", "BPR_D", "BPR", "total_BPR", "PP_O", "PK_D"]:
        feat[c] = df.get(c, 0)

    return feat


feat = build_features(df).fillna(0)
target = df["actual_cap_pct"]

print(f"Feature matrix: {feat.shape}")
print(f"Target (Cap %) mean: {target.mean():.2f}%, median: {target.median():.2f}%")

# ── 4. Train model ──────────────────────────────────────────────────────────

model = Ridge(alpha=10.0)
model.fit(feat, target)

pred = model.predict(feat)
mae = mean_absolute_error(target, pred)
r2 = r2_score(target, pred)
print(f"\nModel MAE: {mae:.3f}%")
print(f"Model R²: {r2:.3f}")

# Feature coefficients
coefs = pd.Series(model.coef_, index=feat.columns)
coefs_sorted = coefs.abs().sort_values(ascending=False)
print(f"\nFeature coefficients (Cap % per unit):")
for f in coefs_sorted.head(20).index:
    print(f"  {f:20s} {coefs[f]:+.4f}")

# ── 5. Compute surplus ─────────────────────────────────────────────────────

df["pred_cap_pct"] = pred
df["surplus_pct"] = df["pred_cap_pct"] - df["actual_cap_pct"]
df["pred_market_value"] = df["pred_cap_pct"] / 100 * df["salary_cap"]
df["surplus_value"] = df["surplus_pct"] / 100 * df["salary_cap"]

print(f"\n{'='*70}")
print("SURPLUS BY CONTRACT TYPE (v2)")
print(f"{'='*70}")

for ctype in ["ELC", "STD"]:
    sub = df[df["contract_type"] == ctype]
    if len(sub):
        print(f"\n  {ctype} ({len(sub)} player-seasons):")
        print(f"    Mean surplus: {sub['surplus_pct'].mean():+.2f}% of cap (${sub['surplus_value'].mean():+,.0f})")

for status in ["RFA", "UFA"]:
    sub = df[df["sign_status"] == status]
    if len(sub):
        print(f"  {status}: mean surplus {sub['surplus_pct'].mean():+.2f}% (${sub['surplus_value'].mean():+,.0f})")

# ── 6. Leaderboards ────────────────────────────────────────────────────────

print(f"\n{'='*70}")
print("TOP 30 SURPLUS SEASONS (v2)")
print(f"{'='*70}\n")

top = df.nlargest(30, "surplus_value")
print(f"{'Player':25s} {'Szn':>7s} {'WAR':>5s} {'Pts':>4s} {'Cap Hit':>12s} {'Pred Value':>12s} {'Surplus':>12s}")
print("-" * 90)
for _, r in top.iterrows():
    print(f"{r['player_name']:25s} {int(r['season'])}-{str(int(r['season'])+1)[-2:]:>2s} "
          f"{r['WAR']:5.2f} {int(r['points']):4d} {r['cap_hit']:>12,.0f} "
          f"{r['pred_market_value']:>12,.0f} {r['surplus_value']:>+12,.0f}")

print(f"\n{'='*70}")
print("WORST 30 SURPLUS SEASONS (v2)")
print(f"{'='*70}\n")

bottom = df.nsmallest(30, "surplus_value")
print(f"{'Player':25s} {'Szn':>7s} {'WAR':>5s} {'Pts':>4s} {'Cap Hit':>12s} {'Pred Value':>12s} {'Surplus':>12s}")
print("-" * 90)
for _, r in bottom.iterrows():
    print(f"{r['player_name']:25s} {int(r['season'])}-{str(int(r['season'])+1)[-2:]:>2s} "
          f"{r['WAR']:5.2f} {int(r['points']):4d} {r['cap_hit']:>12,.0f} "
          f"{r['pred_market_value']:>12,.0f} {r['surplus_value']:>+12,.0f}")

# Career totals
career = (df.groupby(["player_id", "player_name"])
          .agg(total_surplus=("surplus_value", "sum"),
               total_war=("WAR", "sum"),
               total_cap=("cap_hit", "sum"),
               seasons=("season", "count"),
               avg_surplus_pct=("surplus_pct", "mean"))
          .reset_index()
          .sort_values("total_surplus", ascending=False))

print(f"\n{'='*70}")
print("TOP 30 CAREER SURPLUS (v2)")
print(f"{'='*70}\n")

print(f"{'Player':25s} {'Seasons':>7s} {'WAR':>6s} {'Total Cap':>12s} {'Total Surplus':>14s} {'Avg S%':>7s}")
print("-" * 80)
for _, r in career.head(30).iterrows():
    print(f"{r['player_name']:25s} {int(r['seasons']):>7d} {r['total_war']:>6.1f} "
          f"{r['total_cap']:>12,.0f} {r['total_surplus']:>+14,.0f} {r['avg_surplus_pct']:>+6.2f}%")

# ── 7. Save ─────────────────────────────────────────────────────────────────

out_cols = ["player_id", "player_name", "position", "season", "GP",
            "WAR", "WAR_82", "goals", "points",
            "EV_O_GAR", "EV_D_GAR", "PP_GAR", "PK_GAR", "PEN_GAR",
            "cap_hit", "actual_cap_pct", "salary_cap",
            "pred_cap_pct", "pred_market_value", "surplus_pct", "surplus_value",
            "contract_type", "sign_status"]
out = df[[c for c in out_cols if c in df.columns]].copy()
out = out.sort_values(["season", "surplus_value"], ascending=[True, False])

out_path = os.path.join(BASE, "contracts", "surplus_values_v2.csv")
out.to_csv(out_path, index=False)
print(f"\nSaved {len(out)} player-season surplus values to {out_path}")

career_path = os.path.join(BASE, "contracts", "career_surplus_v2.csv")
career.to_csv(career_path, index=False)
print(f"Saved {len(career)} career surplus totals to {career_path}")
