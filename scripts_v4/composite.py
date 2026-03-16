"""
composite.py — Two-layer predictive composite metric.

Two predictive models, each answering a different question:

  Layer 1: "Production Value" (PV)
    Predicts next-season individual production (GV).
    GV dominates — production IS stable and predictive.
    This is where Kucherov/Kaprizov get credit.

  Layer 2: "Impact Value" (IV)
    Predicts next-season blended on-ice results.
    OOI/RAPM dominate — on-ice context predicts future impact.
    This is where system-drivers (Tkachuk, McAvoy) get credit.

  Final composite = λ * PV + (1-λ) * IV, with λ=0.5 default.

RAPM is precision-shrunk before blending (noisy RAPM → attenuated).
Output is position-centered, WAR uses position-specific replacement levels.

Inputs:
  data/v4_bpm_player_seasons.csv   GV, OOI, RAPM per player-season
  data/skaters_by_game.csv          For next-season on-ice targets
  data/pp_rapm.csv                  PP/PK special teams

Outputs:
  data/v5_composite_player_seasons.csv  Per-player-season composite + components
  data/v5_season_war.csv                Season WAR leaderboard
"""

import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

# ── Config ───────────────────────────────────────────────────────────────────
LAMBDA = 0.5          # production vs impact blend (0=pure impact, 1=pure production)
GOALS_TO_WINS = 6.0
RL_PERCENTILE = 17
MIN_TOI_SEASON = 100  # minimum 5v5 TOI for WAR qualification (minutes)

FEATURES = ["GV_O", "GV_D", "OOI_O", "OOI_D",
            "RAPM_O_shrunk", "RAPM_D_shrunk", "ozPct", "dzPct", "isD"]
O_FEATURES = ["GV_O", "OOI_O", "RAPM_O_shrunk", "ozPct", "dzPct", "isD"]
D_FEATURES = ["GV_D", "OOI_D", "RAPM_D_shrunk", "ozPct", "dzPct", "isD"]


# ── 1. Load & prepare ────────────────────────────────────────────────────────
print("Loading data...", file=sys.stderr)

bpm = pd.read_csv("data/v4_bpm_player_seasons.csv")
bpm["player_id"] = bpm["player_id"].astype(int)
bpm["isD"] = (bpm["position"] == "D").astype(float)
print(f"  BPM player-seasons: {len(bpm):,}", file=sys.stderr)

# Shrink RAPM by precision
pop_var_O = bpm["RAPM_O"].var()
pop_var_D = bpm["RAPM_D"].var()

for side, pv in [("O", pop_var_O), ("D", pop_var_D)]:
    se = bpm[f"RAPM_{side}_se"].clip(lower=0.01)
    precision = 1.0 / (se ** 2)
    prior_precision = 1.0 / pv
    bpm[f"RAPM_{side}_shrunk"] = bpm[f"RAPM_{side}"] * precision / (precision + prior_precision)

print(f"  RAPM_O pop_var={pop_var_O:.4f}, RAPM_D pop_var={pop_var_D:.4f}", file=sys.stderr)


# ── 2. Compute next-season targets ───────────────────────────────────────────
print("\nComputing next-season targets...", file=sys.stderr)

# Impact target: from skaters_by_game.csv
sbg = pd.read_csv("data/skaters_by_game.csv",
                   usecols=["playerId", "season", "situation", "icetime",
                            "OnIce_F_xGoals", "OnIce_A_xGoals",
                            "OffIce_F_xGoals", "OffIce_A_xGoals"])
sbg = sbg.rename(columns={"playerId": "player_id"})
ev = sbg[(sbg["situation"] == "5on5") & (sbg["season"] >= 2015)].copy()
ev["toi_min"] = ev["icetime"] / 60.0
ev = ev[ev["toi_min"] >= 2.0]

# Aggregate to player-season
impact = ev.groupby(["player_id", "season"]).agg({
    "toi_min": "sum",
    "OnIce_F_xGoals": "sum",
    "OnIce_A_xGoals": "sum",
    "OffIce_F_xGoals": "sum",
    "OffIce_A_xGoals": "sum",
}).reset_index()

toi60 = impact["toi_min"] / 60.0
impact["oiXGF_60"] = impact["OnIce_F_xGoals"] / toi60
impact["oiXGA_60"] = impact["OnIce_A_xGoals"] / toi60
impact["offXGF_60"] = impact["OffIce_F_xGoals"] / toi60
impact["offXGA_60"] = impact["OffIce_A_xGoals"] / toi60
impact["relXGF_60"] = impact["oiXGF_60"] - impact["offXGF_60"]
impact["relXGA_60"] = impact["oiXGA_60"] - impact["offXGA_60"]

# Impact target: 50% raw + 50% relative
impact["impact_O"] = 0.5 * impact["oiXGF_60"] + 0.5 * impact["relXGF_60"]
impact["impact_D"] = -(0.5 * impact["oiXGA_60"] + 0.5 * impact["relXGA_60"])  # positive = good D

# Filter to qualified
impact = impact[impact["toi_min"] >= MIN_TOI_SEASON]
impact["player_id"] = impact["player_id"].astype(int)

print(f"  Impact targets: {len(impact):,} player-seasons", file=sys.stderr)


# ── 3. Build year-over-year pairs ────────────────────────────────────────────
print("\nBuilding year-over-year prediction pairs...", file=sys.stderr)

# Production target (Layer 1): next-season GV
bpm_next = bpm[["player_id", "season", "GV_O", "GV_D"]].copy()
bpm_next = bpm_next.rename(columns={"GV_O": "next_GV_O", "GV_D": "next_GV_D"})
bpm_next["season"] = bpm_next["season"] - 1  # shift back so join gives "next season"

# Impact target (Layer 2): next-season impact
impact_next = impact[["player_id", "season", "impact_O", "impact_D"]].copy()
impact_next = impact_next.rename(columns={"impact_O": "next_impact_O", "impact_D": "next_impact_D"})
impact_next["season"] = impact_next["season"] - 1

# Join current features to next-season targets
pairs = bpm.merge(bpm_next, on=["player_id", "season"], how="inner")
pairs = pairs.merge(impact_next, on=["player_id", "season"], how="inner")

print(f"  Year-over-year pairs: {len(pairs):,}", file=sys.stderr)
print(f"  Seasons: {sorted(pairs['season'].unique())}", file=sys.stderr)


# ── 4. Train both layers via LOSO CV ─────────────────────────────────────────
print("\n── Training predictive models (LOSO CV) ──", file=sys.stderr)

alphas = np.logspace(-1, 3, 20)
seasons = sorted(pairs["season"].unique())


def train_loso(X_cols, y_col, label):
    """Train Ridge via leave-one-season-out CV. Returns final model and CV R²."""
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

    # Print
    print(f"\n  {label}:", file=sys.stderr)
    print(f"    LOSO CV R² = {cv_r2:.3f} (alpha={model_final.alpha_:.1f})", file=sys.stderr)

    coefs_orig = model_final.coef_ / scaler_final.scale_
    intercept_orig = model_final.intercept_ - np.sum(coefs_orig * scaler_final.mean_)

    print(f"    Weights (original scale):", file=sys.stderr)
    for feat, c in zip(X_cols, coefs_orig):
        print(f"      {feat:20s} {c:+.4f}", file=sys.stderr)

    return model_final, scaler_final, coefs_orig, intercept_orig, cv_r2


# Layer 1: Production Value
model_PV_O, scaler_PV_O, coefs_PV_O, int_PV_O, r2_PV_O = train_loso(
    O_FEATURES, "next_GV_O", "PV_O (production → next-year GV_O)")
model_PV_D, scaler_PV_D, coefs_PV_D, int_PV_D, r2_PV_D = train_loso(
    D_FEATURES, "next_GV_D", "PV_D (production → next-year GV_D)")

# Layer 2: Impact Value
model_IV_O, scaler_IV_O, coefs_IV_O, int_IV_O, r2_IV_O = train_loso(
    O_FEATURES, "next_impact_O", "IV_O (impact → next-year on-ice)")
model_IV_D, scaler_IV_D, coefs_IV_D, int_IV_D, r2_IV_D = train_loso(
    D_FEATURES, "next_impact_D", "IV_D (impact → next-year on-ice)")


# ── 5. Apply models to all player-seasons ─────────────────────────────────────
print("\n── Applying models ──", file=sys.stderr)

X_O = bpm[O_FEATURES].fillna(0).values
X_D = bpm[D_FEATURES].fillna(0).values

bpm["PV_O"] = X_O @ coefs_PV_O + int_PV_O
bpm["PV_D"] = X_D @ coefs_PV_D + int_PV_D
bpm["IV_O"] = X_O @ coefs_IV_O + int_IV_O
bpm["IV_D"] = X_D @ coefs_IV_D + int_IV_D

# Composite blend
bpm["composite_O_raw"] = LAMBDA * bpm["PV_O"] + (1 - LAMBDA) * bpm["IV_O"]
bpm["composite_D_raw"] = LAMBDA * bpm["PV_D"] + (1 - LAMBDA) * bpm["IV_D"]

print(f"  PV_O: mean={bpm['PV_O'].mean():.4f}, std={bpm['PV_O'].std():.4f}", file=sys.stderr)
print(f"  IV_O: mean={bpm['IV_O'].mean():.4f}, std={bpm['IV_O'].std():.4f}", file=sys.stderr)
print(f"  PV_D: mean={bpm['PV_D'].mean():.4f}, std={bpm['PV_D'].std():.4f}", file=sys.stderr)
print(f"  IV_D: mean={bpm['IV_D'].mean():.4f}, std={bpm['IV_D'].std():.4f}", file=sys.stderr)


# ── 6. Position centering ────────────────────────────────────────────────────
print("\n── Position centering ──", file=sys.stderr)

for side in ["O", "D"]:
    col = f"composite_{side}_raw"
    for pos in ["F", "D"]:
        if pos == "F":
            mask = bpm["position"].isin(["C", "L", "R"])
        else:
            mask = bpm["position"] == "D"
        pos_mean = bpm.loc[mask, col].mean()
        bpm.loc[mask, f"composite_{side}"] = bpm.loc[mask, col] - pos_mean
        print(f"  {side} {pos}: mean={pos_mean:.4f} ({mask.sum()} players), centered to 0", file=sys.stderr)

bpm["composite"] = bpm["composite_O"] + bpm["composite_D"]


# ── 7. Convert to GAR/WAR ────────────────────────────────────────────────────
print("\n── Computing WAR ──", file=sys.stderr)

# TOI by situation
sit = pd.read_csv("data/skaters_by_game.csv",
                   usecols=["playerId", "season", "situation", "icetime"])
sit = sit.rename(columns={"playerId": "player_id"})
sit_toi = (
    sit[sit["situation"].isin(["5on5", "5on4", "4on5"])]
    .groupby(["player_id", "season", "situation"])["icetime"]
    .sum().unstack("situation", fill_value=0).reset_index()
)
sit_toi.columns.name = None
rename_map = {"5on5": "toi_5v5", "5on4": "toi_pp", "4on5": "toi_pk"}
sit_toi = sit_toi.rename(columns=rename_map)
for col in ["toi_5v5", "toi_pp", "toi_pk"]:
    if col not in sit_toi.columns:
        sit_toi[col] = 0
    sit_toi[col] = (sit_toi[col] / 60).round(1)
sit_toi["player_id"] = sit_toi["player_id"].astype(int)

war = bpm.merge(sit_toi, on=["player_id", "season"], how="left")

# PP_O and PK_D already in bpm from v4_bpm_player_seasons.csv
if "PP_O" not in war.columns:
    pp_rapm = pd.read_csv("data/pp_rapm.csv")[["player_id", "PP_O", "PK_D"]]
    pp_rapm["player_id"] = pp_rapm["player_id"].astype(int)
    war = war.merge(pp_rapm, on="player_id", how="left")

toi_5v5 = war["toi_5v5"].fillna(0).values
toi_pp = war["toi_pp"].fillna(0).values
toi_pk = war["toi_pk"].fillna(0).values
total_toi = toi_5v5 + toi_pp + toi_pk

war["EV_O_GAR"] = (war["composite_O"] * toi_5v5 / 60).round(2)
war["EV_D_GAR"] = (war["composite_D"] * toi_5v5 / 60).round(2)
war["PP_GAR"] = (war["PP_O"].fillna(0) * toi_pp / 60).round(2)
war["PK_GAR"] = (war["PK_D"].fillna(0) * toi_pk / 60).round(2)

war["GAR_O"] = (war["EV_O_GAR"] + war["PP_GAR"]).round(2)
war["GAR_D"] = (war["EV_D_GAR"] + war["PK_GAR"]).round(2)

# Position-specific replacement level
qualified = toi_5v5 >= MIN_TOI_SEASON

for pos_group, pos_mask_fn in [("F", lambda p: p.isin(["C", "L", "R"])),
                                ("D", lambda p: p == "D")]:
    pos_mask = pos_mask_fn(war["position"]) & qualified
    if pos_mask.sum() == 0:
        continue

    per60_O = war.loc[pos_mask, "GAR_O"].values / (total_toi[pos_mask] / 60)
    per60_D = war.loc[pos_mask, "GAR_D"].values / (total_toi[pos_mask] / 60)
    rl_O = float(np.percentile(per60_O[np.isfinite(per60_O)], RL_PERCENTILE))
    rl_D = float(np.percentile(per60_D[np.isfinite(per60_D)], RL_PERCENTILE))

    print(f"  Replacement level {pos_group}: O={rl_O:.4f}, D={rl_D:.4f} per 60", file=sys.stderr)

    war.loc[pos_mask_fn(war["position"]), "WAR_O"] = (
        (war.loc[pos_mask_fn(war["position"]), "GAR_O"] -
         rl_O * total_toi[pos_mask_fn(war["position"]).values] / 60) / GOALS_TO_WINS
    ).round(2)
    war.loc[pos_mask_fn(war["position"]), "WAR_D"] = (
        (war.loc[pos_mask_fn(war["position"]), "GAR_D"] -
         rl_D * total_toi[pos_mask_fn(war["position"]).values] / 60) / GOALS_TO_WINS
    ).round(2)

war["WAR"] = (war["WAR_O"].fillna(0) + war["WAR_D"].fillna(0)).round(2)

# Pace-adjusted: WAR per 82 games
war["GP"] = war["GP"].astype(int)
war["WAR_82"] = (war["WAR"] * 82 / war["GP"].clip(lower=1)).round(2)
war["WAR_O_82"] = (war["WAR_O"] * 82 / war["GP"].clip(lower=1)).round(2)
war["WAR_D_82"] = (war["WAR_D"] * 82 / war["GP"].clip(lower=1)).round(2)


# ── 8. Output ─────────────────────────────────────────────────────────────────
print("\n── Saving outputs ──", file=sys.stderr)

# Composite player-seasons
comp_cols = [
    "player_id", "player_name", "position", "season", "GP", "toi_min",
    "toi_pp_min", "toi_pk_min",
    "composite_O", "composite_D", "composite",
    "PV_O", "PV_D", "IV_O", "IV_D",
    "GV_O", "GV_D", "OOI_O", "OOI_D",
    "RAPM_O", "RAPM_D", "RAPM_O_se", "RAPM_D_se",
    "PP_O", "PK_D", "ozPct", "dzPct",
]
comp_cols = [c for c in comp_cols if c in bpm.columns]
comp_out = bpm[comp_cols].copy()
for c in ["composite_O", "composite_D", "composite", "PV_O", "PV_D", "IV_O", "IV_D"]:
    if c in comp_out.columns:
        comp_out[c] = comp_out[c].round(4)
comp_out = comp_out.sort_values(["season", "player_name"])
comp_out.to_csv("data/v5_composite_player_seasons.csv", index=False)
print(f"  {len(comp_out):,} rows → data/v5_composite_player_seasons.csv", file=sys.stderr)

# Season WAR
war_cols = [
    "player_id", "player_name", "position", "season", "GP",
    "toi_5v5", "toi_pp", "toi_pk",
    "composite_O", "composite_D",
    "PV_O", "PV_D", "IV_O", "IV_D",
    "EV_O_GAR", "EV_D_GAR", "PP_GAR", "PK_GAR",
    "GAR_O", "GAR_D", "WAR_O", "WAR_D", "WAR",
    "WAR_O_82", "WAR_D_82", "WAR_82",
]
war_cols = [c for c in war_cols if c in war.columns]
war_out = war[war_cols].copy()
for c in ["composite_O", "composite_D", "PV_O", "PV_D", "IV_O", "IV_D"]:
    if c in war_out.columns:
        war_out[c] = war_out[c].round(4)
war_out = war_out.sort_values(["season", "WAR"], ascending=[True, False])
war_out.to_csv("data/v5_season_war.csv", index=False)
print(f"  {len(war_out):,} rows → data/v5_season_war.csv", file=sys.stderr)


# ── 9. Leaderboards ──────────────────────────────────────────────────────────
print("\n── Leaderboards ──", file=sys.stderr)

key_players = [
    "Connor McDavid", "Nikita Kucherov", "Auston Matthews", "Nathan MacKinnon",
    "Kirill Kaprizov", "Cale Makar", "Leon Draisaitl", "David Pastrnak",
    "Matthew Tkachuk", "Sidney Crosby", "Adam Fox", "Charlie McAvoy",
    "Mikko Rantanen", "Jack Hughes", "Miro Heiskanen", "Aleksander Barkov",
    "Sam Reinhart", "Mark Stone",
]

for szn in sorted(war_out["season"].unique())[-2:]:
    s = war_out[war_out["season"] == szn].copy().reset_index(drop=True)
    s["rank"] = s.index + 1
    nhl = f"{szn}-{str(szn+1)[-2:]}"

    print(f"\n{'='*110}", file=sys.stderr)
    print(f"  Top 25 WAR ({nhl}) [λ={LAMBDA}]", file=sys.stderr)
    print(f"{'='*110}", file=sys.stderr)
    s = s.sort_values("WAR_82", ascending=False).reset_index(drop=True)
    s["rank"] = s.index + 1
    show_cols = ["player_name", "position", "GP",
                 "composite_O", "composite_D",
                 "WAR_O", "WAR_D", "WAR", "WAR_82"]
    show_cols = [c for c in show_cols if c in s.columns]
    print(s.head(25)[show_cols].to_string(index=False), file=sys.stderr)

    print(f"\n  Key Players ({nhl}):", file=sys.stderr)
    print(f"  {'Rk':>3s} {'Name':25s} {'Pos':3s} {'GP':>3s} {'comp_O':>7s} {'comp_D':>7s} "
          f"{'WAR_O':>6s} {'WAR_D':>6s} {'WAR':>5s} {'WAR/82':>6s}",
          file=sys.stderr)
    print(f"  {'-'*90}", file=sys.stderr)
    for name in key_players:
        row = s[s["player_name"] == name]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        print(f"  {r['rank']:3.0f} {name:25s} {r['position']:3s} {r['GP']:3.0f} "
              f"{r.get('composite_O', 0):+7.4f} {r.get('composite_D', 0):+7.4f} "
              f"{r['WAR_O']:+6.2f} {r['WAR_D']:+6.2f} {r['WAR']:5.2f} {r['WAR_82']:6.2f}",
              file=sys.stderr)

# Compare λ values
print(f"\n\n── λ sensitivity (2024-25, top players) ──", file=sys.stderr)
latest = war_out["season"].max()
sl = war_out[war_out["season"] == latest].copy()

for lam in [0.3, 0.5, 0.7]:
    sl[f"comp_O_{lam}"] = lam * sl["PV_O"] + (1 - lam) * sl["IV_O"]
    sl[f"comp_D_{lam}"] = lam * sl["PV_D"] + (1 - lam) * sl["IV_D"]
    sl[f"comp_{lam}"] = sl[f"comp_O_{lam}"] + sl[f"comp_D_{lam}"]

print(f"\n  {'Name':25s} {'comp(λ=.3)':>10s} {'comp(λ=.5)':>10s} {'comp(λ=.7)':>10s}", file=sys.stderr)
for name in key_players:
    row = sl[sl["player_name"] == name]
    if len(row) == 0:
        continue
    r = row.iloc[0]
    print(f"  {name:25s} {r.get('comp_0.3', 0):+10.4f} {r.get('comp_0.5', 0):+10.4f} {r.get('comp_0.7', 0):+10.4f}",
          file=sys.stderr)

print("\n\nDone.", file=sys.stderr)
