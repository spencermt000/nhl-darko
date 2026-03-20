"""
carry_forward.py — DARKO-style carry-forward with learned age curve + Bayesian blending.

Two key improvements over the naive version:
  1. Learns the age retention curve from data (separate O/D curves) instead
     of hand-tuned factors.
  2. Proper Bayesian precision-weighted blend: prior uncertainty is modeled
     explicitly and current-season uncertainty scales with 1/sqrt(TOI).

Signal source: composite_O/D from composite_v4.py (per-season predictive composite).
These are used because they're already on a meaningful scale (per-60 goals above average)
and combine GV, OOI, and RAPM optimally.

Architecture:
  For each player-season:
    prior_mean = age_retention(age, side) × last_season_posterior_mean
    prior_se   = inflated from last season's posterior_se
    curr_mean  = this season's composite_O/D
    curr_se    = base_se / sqrt(TOI / reference_TOI)  (shrinks with ice time)
    posterior  = precision-weighted blend of prior and current

Inputs:
  output/v5_composite_player_seasons.csv   Per-season composite (signal)
  output/v5_daily_ratings.csv              Per-game ratings (for GP counts)
  data/moneypuck_player_bio.csv            Birth dates for age curve

Outputs:
  output/v6_carry_forward.csv              Per-season carry-forward ratings
  output/v6_age_curve.csv                  Learned age retention curves
"""

import sys
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# ── Config ───────────────────────────────────────────────────────────────────

GOALS_TO_WINS = 6.0
RL_PERCENTILE = 17
MIN_TOI_SEASON = 100   # for WAR qualification (minutes)
MIN_TOI_CURVE = 400    # for learning the age curve (minutes)

# Current-season SE model: SE = BASE_SE / sqrt(TOI / REF_TOI)
# At REF_TOI minutes, SE = BASE_SE. At 2x REF_TOI, SE = BASE_SE/sqrt(2).
BASE_SE_O = 0.30       # offensive uncertainty at reference TOI
BASE_SE_D = 0.15       # defensive uncertainty at reference TOI (defense is more stable)
REF_TOI = 800.0        # minutes — roughly a full season of 5v5 TOI

# Prior SE model
PRIOR_SE_INFLATION = 1.20   # SE grows each year of carry-forward
MIN_PRIOR_SE_O = 0.08       # floor on prior uncertainty
MIN_PRIOR_SE_D = 0.04

# Rookie prior: league average (0) with high uncertainty
ROOKIE_PRIOR_SE_O = 0.40
ROOKIE_PRIOR_SE_D = 0.20


# ── 1. Load data ─────────────────────────────────────────────────────────────
print("Loading data...", file=sys.stderr)

comp = pd.read_csv("output/v5_composite_player_seasons.csv")
comp["player_id"] = comp["player_id"].astype(int)
print(f"  Composite: {len(comp):,} player-seasons", file=sys.stderr)

# Daily ratings: just need GP per player-season
daily = pd.read_csv("output/v5_daily_ratings.csv", usecols=["player_id", "season", "game_id"])
daily["player_id"] = daily["player_id"].astype(int)
daily_gp = daily.groupby(["player_id", "season"]).size().reset_index(name="daily_GP")
print(f"  Daily GP records: {len(daily_gp):,}", file=sys.stderr)

bio = pd.read_csv("data/moneypuck_player_bio.csv")
bio = bio.rename(columns={"playerId": "player_id"})
bio["player_id"] = bio["player_id"].astype(int)
bio["birth_year"] = pd.to_datetime(bio["birthDate"]).dt.year
birth_year_map = bio.set_index("player_id")["birth_year"].to_dict()

# Per-season PP/PK from GAR
try:
    gar = pd.read_csv("output/v2_gar_by_season.csv",
                       usecols=["player_id", "season", "PP_GAR", "PK_GAR",
                                "toi_pp", "toi_pk"])
    gar["player_id"] = gar["player_id"].astype(int)
    gar["PP_rate"] = np.where(gar["toi_pp"] > 10,
                               gar["PP_GAR"] / (gar["toi_pp"] / 60), 0.0)
    gar["PK_rate"] = np.where(gar["toi_pk"] > 10,
                               gar["PK_GAR"] / (gar["toi_pk"] / 60), 0.0)
    has_per_season_st = True
    print(f"  Per-season PP/PK: {len(gar):,} rows", file=sys.stderr)
except Exception as e:
    print(f"  Per-season PP/PK not available: {e}", file=sys.stderr)
    has_per_season_st = False


# ── 2. Merge and prepare ────────────────────────────────────────────────────
print("\nPreparing master dataset...", file=sys.stderr)

master = comp.merge(daily_gp, on=["player_id", "season"], how="left")
master["daily_GP"] = master["daily_GP"].fillna(0).astype(int)
master["birth_year"] = master["player_id"].map(birth_year_map)
master["age"] = master["season"] - master["birth_year"] + 1

print(f"  Master: {len(master):,} player-seasons", file=sys.stderr)


# ── 3. Learn age retention curves ────────────────────────────────────────────
print("\nLearning age retention curves from data...", file=sys.stderr)

# Build YoY pairs
curr = master[["player_id", "season", "age", "composite_O", "composite_D", "toi_min"]].copy()
nxt = master[["player_id", "season", "composite_O", "composite_D", "toi_min"]].copy()
nxt["season"] = nxt["season"] - 1
nxt = nxt.rename(columns={
    "composite_O": "next_O", "composite_D": "next_D", "toi_min": "next_toi"
})

pairs = curr.merge(nxt, on=["player_id", "season"], how="inner")
pairs = pairs[(pairs["toi_min"] >= MIN_TOI_CURVE) & (pairs["next_toi"] >= MIN_TOI_CURVE)]
print(f"  YoY pairs for curve learning: {len(pairs):,}", file=sys.stderr)


def learn_retention_curve(pairs, curr_col, next_col, label):
    """Learn age-dependent retention: next ≈ retention(age) × current."""
    raw_betas = []
    for age in range(19, 43):
        ap = pairs[pairs["age"] == age]
        if len(ap) < 15:
            continue
        x = ap[curr_col].values
        y = ap[next_col].values
        x2 = np.sum(x * x)
        beta = np.sum(x * y) / x2 if x2 > 0 else 0.5
        raw_betas.append({"age": age, "retention": np.clip(beta, 0.1, 1.0), "n": len(ap)})

    bdf = pd.DataFrame(raw_betas)
    if len(bdf) < 5:
        print(f"  {label}: insufficient data, using flat 0.7", file=sys.stderr)
        return lambda age: 0.7, bdf

    # Fit smooth curve: retention = a + b*(age-27) + c*(age-27)²
    ages = bdf["age"].values
    betas = bdf["retention"].values
    weights = np.sqrt(bdf["n"].values)
    age_centered = ages - 27.0

    def quad_model(age_c, a, b, c):
        return a + b * age_c + c * age_c ** 2

    try:
        popt, _ = curve_fit(quad_model, age_centered, betas,
                            p0=[0.75, 0.0, -0.001], sigma=1.0/weights)
        a, b, c = popt
    except RuntimeError:
        a, b, c = np.median(betas), 0.0, 0.0

    def retention_fn(age):
        ac = age - 27.0
        ret = a + b * ac + c * ac ** 2
        return float(np.clip(ret, 0.15, 0.98))

    print(f"\n  {label}: retention = {a:.4f} + {b:.5f}*(age-27) + {c:.6f}*(age-27)²",
          file=sys.stderr)
    print(f"  {'Age':>4s} {'Raw':>6s} {'Fit':>6s} {'N':>5s}", file=sys.stderr)
    for _, r in bdf.iterrows():
        print(f"  {r['age']:4.0f} {r['retention']:6.3f} {retention_fn(r['age']):6.3f} {r['n']:5.0f}",
              file=sys.stderr)

    bdf["fitted"] = bdf["age"].apply(retention_fn)
    return retention_fn, bdf


ret_O_fn, curve_O = learn_retention_curve(pairs, "composite_O", "next_O", "OFFENSIVE")
ret_D_fn, curve_D = learn_retention_curve(pairs, "composite_D", "next_D", "DEFENSIVE")

# Save learned curves
curve_O["side"] = "O"
curve_D["side"] = "D"
pd.concat([curve_O, curve_D], ignore_index=True).to_csv("output/v6_age_curve.csv", index=False)
print(f"\n  Saved → output/v6_age_curve.csv", file=sys.stderr)


# ── 4. Build carry-forward ratings (Bayesian) ───────────────────────────────
print("\n\nComputing Bayesian carry-forward ratings...", file=sys.stderr)

seasons = sorted(master["season"].unique())
print(f"  Seasons: {seasons}", file=sys.stderr)

# Store: player_id → {O_mean, O_se, D_mean, D_se}
prev_posterior = {}

all_rows = []

for szn in seasons:
    szn_df = master[master["season"] == szn].copy()
    n_with_prior = 0

    for _, row in szn_df.iterrows():
        pid = int(row["player_id"])
        age = row["age"] if not np.isnan(row.get("age", np.nan)) else 27
        toi = row.get("toi_min", 0) or 0

        # Current-season signal
        curr_O = row["composite_O"]
        curr_D = row["composite_D"]
        if np.isnan(curr_O) or np.isnan(curr_D):
            continue

        # Current-season SE: scales with 1/sqrt(TOI)
        toi_factor = np.sqrt(max(toi, 50) / REF_TOI)
        curr_O_se = BASE_SE_O / toi_factor
        curr_D_se = BASE_SE_D / toi_factor

        # Get or create prior
        if pid in prev_posterior:
            prev = prev_posterior[pid]
            retain_O = ret_O_fn(age)
            retain_D = ret_D_fn(age)

            # Prior: retained fraction, regressed toward zero
            prior_O = retain_O * prev["O_mean"]
            prior_D = retain_D * prev["D_mean"]

            # Prior SE: inflated from posterior SE + additional uncertainty from aging
            aging_unc_O = abs(prev["O_mean"]) * (1 - retain_O) * 0.5
            aging_unc_D = abs(prev["D_mean"]) * (1 - retain_D) * 0.3
            prior_O_se = max(prev["O_se"] * PRIOR_SE_INFLATION + aging_unc_O, MIN_PRIOR_SE_O)
            prior_D_se = max(prev["D_se"] * PRIOR_SE_INFLATION + aging_unc_D, MIN_PRIOR_SE_D)

            n_with_prior += 1
        else:
            prior_O = 0.0
            prior_D = 0.0
            prior_O_se = ROOKIE_PRIOR_SE_O
            prior_D_se = ROOKIE_PRIOR_SE_D

        # Bayesian update: precision-weighted blend
        prior_prec_O = 1.0 / (prior_O_se ** 2)
        prior_prec_D = 1.0 / (prior_D_se ** 2)
        curr_prec_O = 1.0 / (curr_O_se ** 2)
        curr_prec_D = 1.0 / (curr_D_se ** 2)

        total_prec_O = prior_prec_O + curr_prec_O
        total_prec_D = prior_prec_D + curr_prec_D

        cf_O = (prior_prec_O * prior_O + curr_prec_O * curr_O) / total_prec_O
        cf_D = (prior_prec_D * prior_D + curr_prec_D * curr_D) / total_prec_D
        cf_O_se = np.sqrt(1.0 / total_prec_O)
        cf_D_se = np.sqrt(1.0 / total_prec_D)

        # Data weight (for diagnostics)
        w_O = curr_prec_O / total_prec_O
        w_D = curr_prec_D / total_prec_D

        # Store posterior
        prev_posterior[pid] = {
            "O_mean": cf_O, "O_se": cf_O_se,
            "D_mean": cf_D, "D_se": cf_D_se,
        }

        out = {
            "player_id": pid,
            "player_name": row["player_name"],
            "position": row["position"],
            "season": szn,
            "GP": row.get("GP", 0),
            "toi_min": toi,
            "daily_GP": row.get("daily_GP", 0),
            "age": round(age, 0),
            # Current season signal
            "curr_O": round(curr_O, 4),
            "curr_D": round(curr_D, 4),
            "curr_O_se": round(curr_O_se, 4),
            "curr_D_se": round(curr_D_se, 4),
            # Prior
            "prior_O": round(prior_O, 4),
            "prior_D": round(prior_D, 4),
            "prior_O_se": round(prior_O_se, 4),
            "prior_D_se": round(prior_D_se, 4),
            # Carry-forward posterior
            "cf_O": round(cf_O, 4),
            "cf_D": round(cf_D, 4),
            "cf_total": round(cf_O + cf_D, 4),
            "cf_O_se": round(cf_O_se, 4),
            "cf_D_se": round(cf_D_se, 4),
            # Diagnostics
            "data_weight_O": round(w_O, 3),
            "data_weight_D": round(w_D, 3),
            "retain_O": round(ret_O_fn(age), 3),
            "retain_D": round(ret_D_fn(age), 3),
        }

        # Component metrics for reference
        for col in ["composite_O", "composite_D", "PV_O", "PV_D", "IV_O", "IV_D",
                     "GV_O", "GV_D", "OOI_O", "OOI_D", "RAPM_O", "RAPM_D"]:
            if col in row.index and not pd.isna(row[col]):
                out[col] = row[col]

        all_rows.append(out)

    print(f"  {szn}: {len(szn_df)} players, {n_with_prior} with carry-forward prior",
          file=sys.stderr)

cf = pd.DataFrame(all_rows)

# Merge per-season PP/PK
if has_per_season_st:
    cf = cf.merge(gar[["player_id", "season", "PP_rate", "PK_rate"]],
                  on=["player_id", "season"], how="left")
    cf["PP_rate"] = cf["PP_rate"].fillna(0.0).round(4)
    cf["PK_rate"] = cf["PK_rate"].fillna(0.0).round(4)

print(f"\nCarry-forward: {len(cf):,} player-seasons", file=sys.stderr)


# ── 5. Compute WAR ──────────────────────────────────────────────────────────
print("\nComputing WAR...", file=sys.stderr)

sit = pd.read_csv("data/skaters_by_game.csv",
                   usecols=["playerId", "season", "situation", "icetime"])
sit = sit.rename(columns={"playerId": "player_id"})
sit["player_id"] = sit["player_id"].astype(int)
sit_toi = (
    sit[sit["situation"].isin(["5on5", "5on4", "4on5"])]
    .groupby(["player_id", "season", "situation"])["icetime"]
    .sum().unstack("situation", fill_value=0).reset_index()
)
sit_toi.columns.name = None
rename_map = {"5on5": "sit_5v5", "5on4": "sit_pp", "4on5": "sit_pk"}
sit_toi = sit_toi.rename(columns=rename_map)
for col in ["sit_5v5", "sit_pp", "sit_pk"]:
    if col not in sit_toi.columns:
        sit_toi[col] = 0
    sit_toi[col] = (sit_toi[col] / 60).round(1)

cf = cf.merge(sit_toi, on=["player_id", "season"], how="left")
for col in ["sit_5v5", "sit_pp", "sit_pk"]:
    cf[col] = cf[col].fillna(0)

cf["EV_O_GAR"] = (cf["cf_O"] * cf["sit_5v5"] / 60).round(2)
cf["EV_D_GAR"] = (cf["cf_D"] * cf["sit_5v5"] / 60).round(2)
if "PP_rate" in cf.columns:
    cf["PP_GAR"] = (cf["PP_rate"] * cf["sit_pp"] / 60).round(2)
    cf["PK_GAR"] = (cf["PK_rate"] * cf["sit_pk"] / 60).round(2)
else:
    cf["PP_GAR"] = 0.0
    cf["PK_GAR"] = 0.0

cf["GAR_O"] = (cf["EV_O_GAR"] + cf["PP_GAR"]).round(2)
cf["GAR_D"] = (cf["EV_D_GAR"] + cf["PK_GAR"]).round(2)

total_toi = (cf["sit_5v5"] + cf["sit_pp"] + cf["sit_pk"]).values
qualified = cf["sit_5v5"] >= MIN_TOI_SEASON

for pos_label, pos_fn in [("F", lambda p: p.isin(["C", "L", "R"])),
                           ("D", lambda p: p == "D")]:
    pos_mask = pos_fn(cf["position"])
    q = qualified & pos_mask
    if q.sum() == 0:
        continue
    per60_O = cf.loc[q, "GAR_O"].values / (total_toi[q.values] / 60)
    per60_D = cf.loc[q, "GAR_D"].values / (total_toi[q.values] / 60)
    rl_O = float(np.percentile(per60_O[np.isfinite(per60_O)], RL_PERCENTILE))
    rl_D = float(np.percentile(per60_D[np.isfinite(per60_D)], RL_PERCENTILE))
    print(f"  Replacement level {pos_label}: O={rl_O:.4f}, D={rl_D:.4f}", file=sys.stderr)

    cf.loc[pos_mask, "WAR_O"] = (
        (cf.loc[pos_mask, "GAR_O"] - rl_O * total_toi[pos_mask.values] / 60) / GOALS_TO_WINS
    ).round(2)
    cf.loc[pos_mask, "WAR_D"] = (
        (cf.loc[pos_mask, "GAR_D"] - rl_D * total_toi[pos_mask.values] / 60) / GOALS_TO_WINS
    ).round(2)

cf["WAR"] = (cf["WAR_O"].fillna(0) + cf["WAR_D"].fillna(0)).round(2)
cf["WAR_82"] = (cf["WAR"] * 82 / cf["GP"].clip(lower=1)).round(2)


# ── 6. Save ──────────────────────────────────────────────────────────────────
print("\n── Saving ──", file=sys.stderr)

cf = cf.sort_values(["season", "player_name"])
cf.to_csv("output/v6_carry_forward.csv", index=False)
print(f"  {len(cf):,} rows → output/v6_carry_forward.csv", file=sys.stderr)


# ── 7. Diagnostics ──────────────────────────────────────────────────────────
print("\n── Diagnostics ──", file=sys.stderr)

has_prior = cf[cf["prior_O"] != 0]
print(f"  Players with prior: {len(has_prior):,} / {len(cf):,}", file=sys.stderr)
print(f"  Mean data_weight_O: {cf['data_weight_O'].mean():.3f}", file=sys.stderr)
print(f"  Mean data_weight_D: {cf['data_weight_D'].mean():.3f}", file=sys.stderr)
print(f"  Mean |cf - raw| O: {(cf['cf_O'] - cf['curr_O']).abs().mean():.4f}", file=sys.stderr)
print(f"  Mean |cf - raw| D: {(cf['cf_D'] - cf['curr_D']).abs().mean():.4f}", file=sys.stderr)

# Data weight by TOI bucket
print("\n  Data weight by TOI:", file=sys.stderr)
for lo, hi in [(0,200),(200,500),(500,800),(800,1200),(1200,2000)]:
    sub = cf[(cf['toi_min']>=lo)&(cf['toi_min']<hi)]
    if len(sub):
        print(f"    TOI {lo:4d}-{hi:4d}: wO={sub['data_weight_O'].mean():.3f} "
              f"wD={sub['data_weight_D'].mean():.3f} (n={len(sub)})", file=sys.stderr)


# ── 8. Leaderboards ─────────────────────────────────────────────────────────
print("\n── Leaderboards ──", file=sys.stderr)

key_players = [
    "Connor McDavid", "Nikita Kucherov", "Auston Matthews", "Nathan MacKinnon",
    "Kirill Kaprizov", "Cale Makar", "Leon Draisaitl", "David Pastrnak",
    "Matthew Tkachuk", "Sidney Crosby", "Adam Fox", "Charlie McAvoy",
]

for szn in sorted(cf["season"].unique())[-2:]:
    s = cf[cf["season"] == szn].sort_values("cf_total", ascending=False).reset_index(drop=True)
    s["rank"] = s.index + 1
    nhl = f"{szn}-{str(szn+1)[-2:]}"

    print(f"\n{'='*130}", file=sys.stderr)
    print(f"  Top 15 ({nhl})", file=sys.stderr)
    print(f"{'='*130}", file=sys.stderr)
    print(f"  {'Rk':>3s} {'Name':25s} {'Pos':3s} {'Age':>3s} {'GP':>3s} {'TOI':>5s} "
          f"{'wO':>5s} {'curr':>8s} {'prior':>8s} {'cf_O':>7s} {'cf_D':>7s} "
          f"{'cf':>7s} {'WAR':>5s}", file=sys.stderr)
    print(f"  {'-'*120}", file=sys.stderr)

    for i, (_, r) in enumerate(s.head(15).iterrows()):
        print(f"  {i+1:3d} {r['player_name']:25s} {r['position']:3s} {r['age']:3.0f} "
              f"{r['GP']:3.0f} {r['toi_min']:5.0f} {r['data_weight_O']:5.2f} "
              f"{r['curr_O']+r['curr_D']:+8.4f} {r['prior_O']+r['prior_D']:+8.4f} "
              f"{r['cf_O']:+7.4f} {r['cf_D']:+7.4f} {r['cf_total']:+7.4f} "
              f"{r.get('WAR', 0):5.2f}", file=sys.stderr)

    print(f"\n  Key Players ({nhl}):", file=sys.stderr)
    for name in key_players:
        row = s[s["player_name"] == name]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        print(f"  {r['rank']:3.0f} {name:25s} age={r['age']:.0f} GP={r['GP']:.0f} "
              f"wO={r['data_weight_O']:.2f} "
              f"curr={r['curr_O']+r['curr_D']:+.4f} prior={r['prior_O']+r['prior_D']:+.4f} "
              f"→ cf={r['cf_total']:+.4f} WAR={r.get('WAR', 0):.2f}",
              file=sys.stderr)

print("\nDone.", file=sys.stderr)
