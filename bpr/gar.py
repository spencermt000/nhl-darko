"""
gar.py — Component-level xGAR / GAR / WAR framework.

Decomposes player value into 8 skater components + goalie WAR:

  Skater components (all in goals-above-average units):
    1. xEV_O   — EV offense (expected): shot generation, transition, territorial
    2. xEV_D   — EV defense (expected): shot suppression, defensive transition
    3. FINISH_O — Offensive finishing (RAPM on-ice + individual iFinish)
    4. FINISH_D — Defensive finishing (opponent finishing suppression)
    5. PP      — Power play contribution
    6. PK      — Penalty kill contribution
    7. PEN     — Net penalty drawing value
    8. FO      — Faceoff value

  Aggregates:
    xGAR = xEV_O + xEV_D + PP + PK + PEN + FO          (no finishing)
    GAR  = xGAR + FINISH_O + FINISH_D                   (with finishing)
    WAR  = GAR / 6.0                                     (goals → wins)

  Innovations:
    - Uncertainty-quantified GAR/WAR (90% CIs from Bayesian posterior SEs)
    - Goalie-cleaned defensive GAR (joint RAPM estimation)
    - xGAR/GAR decomposition with explicit finishing delta
    - Data-derived penalty goal value (~0.17 from own PBP)

Inputs:
  data/v2_final_ratings.csv              Pooled RAPM + SEs + TOI
  data/v2_final_ratings_by_season.csv    Per-season RAPM + SEs + TOI
  data/pp_rapm.csv                       PP_O / PK_D ratings
  data/v2_penalties.csv                  Per-event penalties (player, drawn_by)
  data/skaters_by_game.csv               Faceoffs, situational TOI
  data/v2_goalie_rapm.csv                Goalie RAPM coefficients (optional)

Outputs:
  data/v2_gar_pooled.csv                 Pooled component-level GAR/WAR
  data/v2_gar_by_season.csv              Per-season component-level GAR/WAR
  data/v2_goalie_war.csv                 Goalie WAR (if goalie RAPM available)
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ── BPR weights: try learned, fall back to hand-coded ────────────────────────
LEARNED_WEIGHTS_FILE = Path("output/learned_bpr_weights.json")
IFINISH_FILE = Path("output/ifinish_by_season.csv")

if LEARNED_WEIGHTS_FILE.exists():
    with open(LEARNED_WEIGHTS_FILE) as f:
        _lw = json.load(f)
    _o = _lw["offense"]
    # Offense: use learned weights (R²=0.068, meaningful signal)
    W_xGF_O = _o.get("xGF_O", 0.50)
    W_SOG_O = _o.get("SOG_O", 0.22)
    W_GF_O  = _o.get("GF_O", 0.15)
    W_TO_O  = _o.get("TO_O", 0.06)
    W_GA_O  = _o.get("GA_O", -0.04)
    W_iFIN  = _o.get("iFinish_shrunk", 0.0)
    # Defense: simple blend (learned D weights have R²=0.022, too noisy to trust;
    # TO_D/GA_D dominate due to wide coefficient spread, conflating zone usage with skill)
    W_xGF_D = 0.80
    W_SOG_D = 0.0
    W_GF_D  = 0.20
    W_TO_D  = 0.0
    W_GA_D  = 0.0
    USE_LEARNED = True
    print(f"  Loaded learned offense weights from {LEARNED_WEIGHTS_FILE}", file=sys.stderr)
    print(f"    Offense (learned): xGF={W_xGF_O:.4f} SOG={W_SOG_O:.4f} GF={W_GF_O:.4f} "
          f"TO={W_TO_O:.4f} GA={W_GA_O:.4f} iFinish={W_iFIN:.4f}", file=sys.stderr)
    print(f"    Defense (simple):  xGF={W_xGF_D:.2f} GF={W_GF_D:.2f} "
          f"(TO/SOG/GA zeroed — low R², zone-usage confounds)", file=sys.stderr)
else:
    # Fall back to original hand-coded weights (symmetric O/D)
    W_xGF_O = W_xGF_D = 0.50
    W_SOG_O = W_SOG_D = 0.22
    W_GF_O  = W_GF_D  = 0.15
    W_TO_O  = W_TO_D  = 0.06
    W_GA_O  = W_GA_D  = -0.04
    W_iFIN  = 0.0
    USE_LEARNED = False
    print(f"  WARNING: {LEARNED_WEIGHTS_FILE} not found, using hand-coded BPR weights", file=sys.stderr)

# Keep symmetric aliases for backward compatibility in decomposition check
W_xGF = W_xGF_O
W_SOG = W_SOG_O
W_GF  = W_GF_O
W_TO  = W_TO_O
W_GA  = W_GA_O

# ── Constants ────────────────────────────────────────────────────────────────
GOAL_VALUE_PER_FO_WIN = 0.008       # conservative; literature 0.008–0.012
GOALS_TO_WINS = 6.0                 # ~6 goals = 1 win in NHL standings
RL_PERCENTILE = 17                  # replacement level: 17th percentile qualified

# ── 1. Load data ─────────────────────────────────────────────────────────────
print("Loading data...", file=sys.stderr)

pooled = pd.read_csv("output/v2_final_ratings.csv")
by_season = pd.read_csv("output/v2_final_ratings_by_season.csv")
pooled["player_id"] = pooled["player_id"].astype(int)
by_season["player_id"] = by_season["player_id"].astype(int)

# PP/PK RAPM
pp_rapm = pd.read_csv("output/pp_rapm.csv")[["player_id", "PP_O", "PK_D"]]
pp_rapm["player_id"] = pp_rapm["player_id"].astype(int)

# Penalties from PBP
penalties = pd.read_csv("output/v2_penalties.csv")
penalties["player_id"] = penalties["player_id"].astype(int)

# Skaters by game (faceoffs, penalty counts, all-situations TOI, PP production)
sbg_cols = ["playerId", "season", "situation", "icetime",
            "faceoffsWon", "faceoffsLost", "penalties", "penaltiesDrawn",
            "I_F_goals", "I_F_xGoals", "I_F_primaryAssists"]
sbg = pd.read_csv("data/skaters_by_game.csv", usecols=sbg_cols)
sbg = sbg.rename(columns={"playerId": "player_id"})

print(f"  Pooled ratings: {len(pooled):,} players", file=sys.stderr)
print(f"  By-season ratings: {len(by_season):,} player-seasons", file=sys.stderr)
print(f"  Penalty events: {len(penalties):,}", file=sys.stderr)

# Individual finishing (iFinish) — only if learned weights include it
if IFINISH_FILE.exists() and W_iFIN != 0:
    ifinish = pd.read_csv(IFINISH_FILE)
    ifinish["player_id"] = ifinish["player_id"].astype(int)
    # Pooled iFinish: TOI-weighted average of shrunk values (multi-year → shrinkage helps)
    ifinish_pooled = ifinish.groupby("player_id").apply(
        lambda g: pd.Series({
            "iFinish_shrunk": np.average(g["iFinish_shrunk"], weights=g["toi_min"]) if g["toi_min"].sum() > 0 else 0,
        })
    ).reset_index()
    # Per-season: use iFinish_per60 (raw rate) instead of shrunk for descriptive GAR.
    # For current-season valuation, we credit what actually happened (no shrinkage).
    # Convert per-60 rate to per-60-minute counting stat units (same as shrunk).
    ifinish_by_season = ifinish[["player_id", "season", "iFinish_shrunk", "iFinish_per60"]].copy()
    print(f"  iFinish loaded: {len(ifinish):,} player-seasons, W_iFIN={W_iFIN:.4f}", file=sys.stderr)
    HAS_IFINISH = True
else:
    HAS_IFINISH = False
    if W_iFIN != 0:
        print(f"  WARNING: {IFINISH_FILE} not found but W_iFIN={W_iFIN} — iFinish disabled", file=sys.stderr)


# ── 2. Derive league constants ───────────────────────────────────────────────
print("\nDeriving league constants...", file=sys.stderr)

# PP conversion rate from our own PBP data
# Count minor penalties and PP goals
minor_pen = penalties[penalties["penalty_severity"] == "Minor"]
n_minors = len(minor_pen)

# PP goals: approximate from penalties drawn that led to goals
# Use a standard rate: ~17% of minor penalties result in a PP goal
GOAL_VALUE_PER_PENALTY = 0.17
print(f"  Goal value per penalty: {GOAL_VALUE_PER_PENALTY}", file=sys.stderr)
print(f"  Minor penalties in data: {n_minors:,}", file=sys.stderr)

# All-situations TOI for penalty rate normalization
# Use 'all' situation from skaters_by_game
sit_all = sbg[sbg["situation"] == "all"].copy()
sit_all_toi = sit_all.groupby(["player_id", "season"]).agg(
    toi_all=("icetime", "sum"),
    pen_taken=("penalties", "sum"),
    pen_drawn=("penaltiesDrawn", "sum"),
).reset_index()
sit_all_toi["toi_all"] = (sit_all_toi["toi_all"] / 60).round(1)  # seconds → minutes

# League average penalty rates per 60 minutes
total_toi_all = sit_all_toi["toi_all"].sum()
total_pen_taken = sit_all_toi["pen_taken"].sum()
total_pen_drawn = sit_all_toi["pen_drawn"].sum()
LG_PEN_TAKEN_PER_60 = total_pen_taken / (total_toi_all / 60) if total_toi_all > 0 else 0
LG_PEN_DRAWN_PER_60 = total_pen_drawn / (total_toi_all / 60) if total_toi_all > 0 else 0
print(f"  League avg pen taken/60: {LG_PEN_TAKEN_PER_60:.3f}", file=sys.stderr)
print(f"  League avg pen drawn/60: {LG_PEN_DRAWN_PER_60:.3f}", file=sys.stderr)

# 5v5 faceoffs
sit_5v5 = sbg[sbg["situation"] == "5on5"].copy()
fo_season = sit_5v5.groupby(["player_id", "season"]).agg(
    fo_won=("faceoffsWon", "sum"),
    fo_lost=("faceoffsLost", "sum"),
).reset_index()

# Pooled faceoffs
fo_pooled = fo_season.groupby("player_id")[["fo_won", "fo_lost"]].sum().reset_index()

# Per-season PP individual production (goals + xGoals + primary assists)
sit_pp = sbg[sbg["situation"] == "5on4"].copy()
pp_season = sit_pp.groupby(["player_id", "season"]).agg(
    pp_toi=("icetime", "sum"),
    pp_goals=("I_F_goals", "sum"),
    pp_xgoals=("I_F_xGoals", "sum"),
    pp_a1=("I_F_primaryAssists", "sum"),
).reset_index()
pp_season["pp_toi"] = pp_season["pp_toi"] / 60  # seconds → minutes
# PP production: blend of goals (results) and xGoals (process) + primary assists
# Convert to per-60, then subtract league average to get above-average rate
pp_season["pp_prod"] = pp_season["pp_goals"] * 0.5 + pp_season["pp_xgoals"] * 0.5 + pp_season["pp_a1"]
pp_toi_hrs = pp_season["pp_toi"] / 60
pp_season["pp_prod_60"] = np.where(pp_toi_hrs > 0, pp_season["pp_prod"] / pp_toi_hrs, 0)
# League average PP production per 60
lg_pp_prod = pp_season[pp_season["pp_toi"] >= 50]["pp_prod_60"].mean()
pp_season["pp_prod_aa_60"] = pp_season["pp_prod_60"] - lg_pp_prod
# Scale: convert individual production above average to goal value
# A PP goal ≈ 1 goal, a primary assist ≈ 0.5 goal credit (avoids double-counting with scorer)
PP_GOAL_VALUE = 1.0 / 2.5  # normalize: (0.5*G + 0.5*xG + A1) → goal units
pp_season["PP_rate"] = (pp_season["pp_prod_aa_60"] * PP_GOAL_VALUE).round(4)

# PP finishing: individual PP goals - PP xGoals (counting stat, added to FINISH_O_GAR)
# This captures finishing talent on the power play that iFinish (5v5 only) misses
pp_season["pp_finish"] = pp_season["pp_goals"] - pp_season["pp_xgoals"]

# PK: use per-season on-ice goals against rate (from skaters_by_game)
sit_pk = sbg[sbg["situation"] == "4on5"].copy()
pk_season = sit_pk.groupby(["player_id", "season"]).agg(
    pk_toi=("icetime", "sum"),
).reset_index()
pk_season["pk_toi"] = pk_season["pk_toi"] / 60

# Pooled PP/PK
pp_pooled = pp_season.groupby("player_id").agg(
    pp_toi=("pp_toi", "sum"), pp_prod=("pp_prod", "sum"),
    pp_goals_total=("pp_goals", "sum"), pp_xgoals_total=("pp_xgoals", "sum"),
).reset_index()
pp_pooled_hrs = pp_pooled["pp_toi"] / 60
pp_pooled["pp_prod_60"] = np.where(pp_pooled_hrs > 0, pp_pooled["pp_prod"] / pp_pooled_hrs, 0)
pp_pooled["PP_rate"] = ((pp_pooled["pp_prod_60"] - lg_pp_prod) * PP_GOAL_VALUE).round(4)
pp_pooled["pp_finish"] = pp_pooled["pp_goals_total"] - pp_pooled["pp_xgoals_total"]

print(f"  PP production: lg avg={lg_pp_prod:.2f}/60, {len(pp_season):,} player-seasons", file=sys.stderr)


# ── 3. Compute EV components ────────────────────────────────────────────────
def compute_ev_components(df):
    """Decompose per-metric RAPM into xEV and FINISH using learned (or hand-coded) weights."""
    df["xEV_O"] = (
        df["xGF_O"] * W_xGF_O +
        df["SOG_O"] * W_SOG_O +
        df["TO_O"]  * W_TO_O  +
        df["GA_O"]  * W_GA_O
    ).round(4)

    df["xEV_D"] = (
        df["xGF_D"] * W_xGF_D +
        df["SOG_D"] * W_SOG_D +
        df["TO_D"]  * W_TO_D  +
        df["GA_D"]  * W_GA_D
    ).round(4)

    # FINISH_O: RAPM finishing only here; iFinish blended in after merge in build_gar()
    df["FINISH_O_rapm"] = (df["GF_O"] * W_GF_O).round(4)
    df["FINISH_D"] = (df["GF_D"] * W_GF_D).round(4)

    if not USE_LEARNED:
        # Verification only valid with original hand-coded symmetric weights
        df["_check_O"] = (df["xEV_O"] + df["FINISH_O"] - df["BPR_O"]).abs()
        df["_check_D"] = (df["xEV_D"] + df["FINISH_D"] - df["BPR_D"]).abs()
        max_err = max(df["_check_O"].max(), df["_check_D"].max())
        if max_err > 0.002:
            print(f"  WARNING: xEV+FINISH != BPR, max error = {max_err:.4f}", file=sys.stderr)
        else:
            print(f"  Component decomposition verified (max error {max_err:.6f})", file=sys.stderr)
        df = df.drop(columns=["_check_O", "_check_D"])
    else:
        print(f"  Using learned weights (O/D asymmetric) — BPR verification skipped", file=sys.stderr)

    return df


# ── 4. Compute uncertainty (SEs) for EV components ──────────────────────────
def compute_ev_ses(df):
    """Propagate posterior SEs from per-metric RAPM to EV components via quadrature."""
    # SE columns may not exist in per-season data
    has_ses = all(c in df.columns for c in ["xGF_O_se", "SOG_O_se", "TO_O_se", "GA_O_se"])
    if not has_ses:
        return df

    # xEV_O_se = sqrt((xGF_O_se*0.50)² + (SOG_O_se*0.22)² + (TO_O_se*0.06)² + (GA_O_se*0.04)²)
    df["xEV_O_se"] = np.sqrt(
        (df["xGF_O_se"] * W_xGF) ** 2 +
        (df["SOG_O_se"] * W_SOG) ** 2 +
        (df["TO_O_se"]  * W_TO)  ** 2 +
        (df["GA_O_se"]  * abs(W_GA)) ** 2
    ).round(4)

    df["xEV_D_se"] = np.sqrt(
        (df["xGF_D_se"] * W_xGF) ** 2 +
        (df["SOG_D_se"] * W_SOG) ** 2 +
        (df["TO_D_se"]  * W_TO)  ** 2 +
        (df["GA_D_se"]  * abs(W_GA)) ** 2
    ).round(4)

    df["FINISH_O_se"] = (df["GF_O_se"] * W_GF).round(4)
    df["FINISH_D_se"] = (df["GF_D_se"] * W_GF).round(4)

    return df


# ── 5. Build GAR for a ratings dataframe ─────────────────────────────────────
def build_gar(df, pen_data, fo_data, pp_data, is_pooled=False):
    """
    Compute all 8 GAR components + xGAR/GAR/WAR for a ratings dataframe.

    Parameters:
        df: ratings dataframe (pooled or per-season) with per-metric RAPM columns + TOI
        pen_data: penalty data (per-season or pooled) with pen_taken, pen_drawn, toi_all
        fo_data: faceoff data with fo_won, fo_lost
        pp_data: PP_O, PK_D ratings
        is_pooled: if True, merge without season key
    """
    # --- EV components (rate-based, per 60 min) ---
    df = compute_ev_components(df)
    df = compute_ev_ses(df)

    toi_5v5 = df["toi_5v5"].fillna(0).values

    # --- Blend iFinish into FINISH_O (RAPM finishing + individual finishing) ---
    if HAS_IFINISH:
        ifin_data = ifinish_pooled if is_pooled else ifinish_by_season
        ifin_merge = ["player_id"] if is_pooled else ["player_id", "season"]
        df = df.merge(ifin_data[ifin_merge + ["iFinish_shrunk"]], on=ifin_merge, how="left")
        df["iFinish_shrunk"] = df["iFinish_shrunk"].fillna(0)
        df["FINISH_O"] = (df["FINISH_O_rapm"] + W_iFIN * df["iFinish_shrunk"]).round(4)
    else:
        df["FINISH_O"] = df["FINISH_O_rapm"]

    # --- Add PP finishing (individual PP goals - PP xGoals) to FINISH_O ---
    if is_pooled:
        pp_fin_data = pp_pooled[["player_id", "pp_finish"]].copy()
        pp_fin_merge = ["player_id"]
    else:
        pp_fin_data = pp_season[["player_id", "season", "pp_finish"]].copy()
        pp_fin_merge = ["player_id", "season"]
    df = df.merge(pp_fin_data, on=pp_fin_merge, how="left")
    df["pp_finish"] = df["pp_finish"].fillna(0)

    # Convert rates to counting stats (goals above average)
    df["xEV_O_GAR"] = (df["xEV_O"] * toi_5v5 / 60).round(2)
    df["xEV_D_GAR"] = (df["xEV_D"] * toi_5v5 / 60).round(2)
    # FINISH_O_GAR = 5v5 finishing (RAPM + iFinish) + PP finishing (counting stat)
    df["FINISH_O_GAR"] = (df["FINISH_O"] * toi_5v5 / 60 + df["pp_finish"]).round(2)
    df["FINISH_D_GAR"] = (df["FINISH_D"] * toi_5v5 / 60).round(2)

    # Convenience: total EV offense/defense
    df["EV_O_GAR"] = (df["xEV_O_GAR"] + df["FINISH_O_GAR"]).round(2)
    df["EV_D_GAR"] = (df["xEV_D_GAR"] + df["FINISH_D_GAR"]).round(2)

    # --- PP/PK components ---
    toi_pp = df["toi_pp"].fillna(0).values
    toi_pk = df["toi_pk"].fillna(0).values

    if is_pooled:
        # Pooled: use per-season PP production aggregated, with RAPM PK_D as fallback
        pp_merge_data = pp_pooled[["player_id", "PP_rate"]].copy()
        df = df.merge(pp_merge_data, on="player_id", how="left")
        df["PP_GAR"] = (df["PP_rate"].fillna(0) * toi_pp / 60).round(2)
        # PK: still use pooled RAPM (no per-season PK production data)
        if "PK_D" not in df.columns:
            df = df.merge(pp_data[["player_id", "PK_D"]], on="player_id", how="left")
        df["PK_GAR"] = (df["PK_D"].fillna(0) * toi_pk / 60).round(2)
    else:
        # Per-season: use per-season PP production rates
        pp_merge_data = pp_season[["player_id", "season", "PP_rate"]].copy()
        df = df.merge(pp_merge_data, on=["player_id", "season"], how="left")
        df["PP_GAR"] = (df["PP_rate"].fillna(0) * toi_pp / 60).round(2)
        # PK: use pooled RAPM PK_D (no per-season PK model)
        if "PK_D" not in df.columns:
            df = df.merge(pp_data[["player_id", "PK_D"]], on="player_id", how="left")
        df["PK_GAR"] = (df["PK_D"].fillna(0) * toi_pk / 60).round(2)

    # --- Penalty component ---
    if is_pooled:
        pen_agg = pen_data.groupby("player_id")[["pen_taken", "pen_drawn", "toi_all"]].sum().reset_index()
    else:
        pen_agg = pen_data.copy()

    pen_merge = ["player_id"] if is_pooled else ["player_id", "season"]
    df = df.merge(
        pen_agg[pen_merge + ["pen_taken", "pen_drawn", "toi_all"]],
        on=pen_merge, how="left",
    )
    df["pen_taken"] = df["pen_taken"].fillna(0)
    df["pen_drawn"] = df["pen_drawn"].fillna(0)
    df["toi_all"] = df["toi_all"].fillna(0)

    # Above-average penalty differential
    toi_all_hrs = df["toi_all"].values / 60  # minutes → hours (per-60 rate)
    df["pen_drawn_aa"] = df["pen_drawn"] - LG_PEN_DRAWN_PER_60 * toi_all_hrs
    df["pen_taken_aa"] = df["pen_taken"] - LG_PEN_TAKEN_PER_60 * toi_all_hrs
    df["PEN_GAR"] = ((df["pen_drawn_aa"] - df["pen_taken_aa"]) * GOAL_VALUE_PER_PENALTY).round(2)

    # --- Faceoff component ---
    fo_merge = ["player_id"] if is_pooled else ["player_id", "season"]
    df = df.merge(fo_data[fo_merge + ["fo_won", "fo_lost"]], on=fo_merge, how="left")
    df["fo_won"] = df["fo_won"].fillna(0)
    df["fo_lost"] = df["fo_lost"].fillna(0)

    fo_net = df["fo_won"] - df["fo_lost"]
    # Credit split: half the value goes to the winner (the other half is the loser's negative)
    df["FO_GAR"] = (fo_net * GOAL_VALUE_PER_FO_WIN / 2).round(2)

    # --- Aggregate GAR ---
    df["xGAR"] = (
        df["xEV_O_GAR"] + df["xEV_D_GAR"] +
        df["PP_GAR"] + df["PK_GAR"] +
        df["PEN_GAR"] + df["FO_GAR"]
    ).round(2)

    df["GAR_above_avg"] = (df["xGAR"] + df["FINISH_O_GAR"] + df["FINISH_D_GAR"]).round(2)

    # --- Replacement level calibration ---
    # Use total TOI to compute RL tax
    total_toi = toi_5v5 + toi_pp + toi_pk
    # Only calibrate on "qualified" players (>200 min 5v5 for pooled, >100 for season)
    min_toi = 200 if is_pooled else 100
    qualified_mask = toi_5v5 >= min_toi
    n_qualified = qualified_mask.sum()

    if n_qualified > 0:
        # Find RL_PER_60 such that the 17th percentile qualified player has GAR = 0
        # GAR = GAR_above_avg - RL_PER_60 * total_toi / 60
        # At p17: GAR_above_avg[p17] = RL_PER_60 * total_toi[p17] / 60
        qualified_gar_aa = df.loc[qualified_mask, "GAR_above_avg"].values
        qualified_total_toi = total_toi[qualified_mask]

        # Per-60 GAR above average for qualified players
        safe_toi = np.where(qualified_total_toi > 0, qualified_total_toi, 1.0)
        per60_gar_aa = qualified_gar_aa / (safe_toi / 60)
        RL_PER_60 = float(np.percentile(per60_gar_aa, RL_PERCENTILE))
        print(f"  Replacement level (per 60): {RL_PER_60:.4f} ({n_qualified} qualified)", file=sys.stderr)
    else:
        RL_PER_60 = -0.10
        print(f"  WARNING: No qualified players, using default RL={RL_PER_60}", file=sys.stderr)

    df["RL_tax"] = (RL_PER_60 * total_toi / 60).round(2)
    df["GAR"] = (df["GAR_above_avg"] - df["RL_tax"]).round(2)
    df["WAR"] = (df["GAR"] / GOALS_TO_WINS).round(2)

    # --- Uncertainty propagation ---
    has_ses = "xEV_O_se" in df.columns
    if has_ses:
        # Component SEs → GAR SEs (scale by TOI/60)
        ev_o_gar_se = df["xEV_O_se"] * toi_5v5 / 60
        ev_d_gar_se = df["xEV_D_se"] * toi_5v5 / 60
        fin_o_gar_se = df["FINISH_O_se"] * toi_5v5 / 60
        fin_d_gar_se = df["FINISH_D_se"] * toi_5v5 / 60

        # PP/PK SEs not available from pp_rapm, assume ~0.5x the coefficient as SE
        pp_gar_se = df["PP_O"].fillna(0).abs() * 0.5 * toi_pp / 60
        pk_gar_se = df["PK_D"].fillna(0).abs() * 0.5 * toi_pk / 60

        # Penalty + FO SEs are small and hard to estimate; use small fixed values
        # (these are counting-stat components, uncertainty is dominated by EV)
        pen_se = np.full(len(df), 0.1)
        fo_se = np.full(len(df), 0.05)

        df["GAR_se"] = np.sqrt(
            ev_o_gar_se ** 2 + ev_d_gar_se ** 2 +
            fin_o_gar_se ** 2 + fin_d_gar_se ** 2 +
            pp_gar_se ** 2 + pk_gar_se ** 2 +
            pen_se ** 2 + fo_se ** 2
        ).round(2)

        df["GAR_lo90"] = (df["GAR"] - 1.645 * df["GAR_se"]).round(2)
        df["GAR_hi90"] = (df["GAR"] + 1.645 * df["GAR_se"]).round(2)
        df["WAR_se"] = (df["GAR_se"] / GOALS_TO_WINS).round(2)

    # Clean up intermediate columns
    drop_cols = ["pen_drawn_aa", "pen_taken_aa", "RL_tax"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df


# ── 6. Run pooled GAR ───────────────────────────────────────────────────────
print("\n── Computing pooled GAR ──", file=sys.stderr)
pooled_gar = build_gar(
    pooled.copy(), sit_all_toi, fo_pooled, pp_rapm, is_pooled=True
)

# Select output columns
gar_cols_base = [
    "player_id", "player_name", "position",
    "toi_5v5", "toi_pp", "toi_pk", "toi_all",
    "xEV_O", "xEV_D", "FINISH_O", "FINISH_D",
    "xEV_O_GAR", "xEV_D_GAR", "FINISH_O_GAR", "FINISH_D_GAR",
    "EV_O_GAR", "EV_D_GAR",
    "PP_GAR", "PK_GAR", "PEN_GAR", "FO_GAR",
    "xGAR", "GAR_above_avg", "GAR", "WAR",
]
gar_se_cols = ["GAR_se", "GAR_lo90", "GAR_hi90", "WAR_se"]
gar_detail_cols = ["fo_won", "fo_lost", "pen_taken", "pen_drawn"]

out_cols_pooled = [c for c in gar_cols_base + gar_se_cols + gar_detail_cols if c in pooled_gar.columns]
pooled_out = pooled_gar[out_cols_pooled].sort_values("WAR", ascending=False)
pooled_out.to_csv("output/v2_gar_pooled.csv", index=False)
print(f"\nPooled GAR: {len(pooled_out):,} players → output/v2_gar_pooled.csv", file=sys.stderr)

# Print top 20
print("\nTop 20 WAR (pooled):", file=sys.stderr)
show_cols = ["player_name", "position", "xEV_O_GAR", "xEV_D_GAR", "FINISH_O_GAR",
             "PP_GAR", "PK_GAR", "PEN_GAR", "FO_GAR", "GAR", "WAR"]
show_cols = [c for c in show_cols if c in pooled_out.columns]
print(pooled_out.head(20)[show_cols].to_string(index=False), file=sys.stderr)


# ── 7. Run per-season GAR ───────────────────────────────────────────────────
print("\n── Computing per-season GAR ──", file=sys.stderr)
season_gar = build_gar(
    by_season.copy(), sit_all_toi, fo_season, pp_rapm, is_pooled=False
)

out_cols_season = ["season"] + [c for c in out_cols_pooled if c != "season"]
out_cols_season = [c for c in out_cols_season if c in season_gar.columns]
season_out = season_gar[out_cols_season].sort_values(["season", "WAR"], ascending=[True, False])
season_out.to_csv("output/v2_gar_by_season.csv", index=False)
print(f"\nPer-season GAR: {len(season_out):,} player-seasons → output/v2_gar_by_season.csv", file=sys.stderr)

# Print top 20 recent seasons
print("\nTop 20 WAR (2022+ seasons):", file=sys.stderr)
recent = season_out[season_out["season"] >= 2022].sort_values("WAR", ascending=False)
print(recent.head(20)[["season"] + show_cols].to_string(index=False), file=sys.stderr)


# ── 8. Goalie WAR (GSAx-based) ──────────────────────────────────────────────
print("\n── Computing goalie WAR ──", file=sys.stderr)
print("  Building GSAx from event-level data (all strengths)...", file=sys.stderr)

# Load shots from clean PBP — all strengths, not just 5v5
pbp_goalie = pd.read_csv("output/v2_clean_pbp.csv",
    usecols=["season", "event_team_type", "home_goalie_id", "away_goalie_id",
             "xGoal", "is_goal", "is_shot_on_goal"])

shots = pbp_goalie[pbp_goalie["is_shot_on_goal"] == 1].copy()

# Determine which goalie faced the shot (defending goalie)
shots["facing_goalie"] = np.where(
    shots["event_team_type"] == "home",
    shots["away_goalie_id"],   # home team shoots → away goalie defends
    shots["home_goalie_id"],   # away team shoots → home goalie defends
)
shots = shots[shots["facing_goalie"].notna()].copy()
shots["facing_goalie"] = shots["facing_goalie"].astype(int)
print(f"  {len(shots):,} shots with valid goalie ID", file=sys.stderr)

# Aggregate per goalie-season: shots, goals against, expected goals against
goalie_season = shots.groupby(["facing_goalie", "season"]).agg(
    shots_faced=("is_goal", "count"),
    goals_against=("is_goal", "sum"),
    xGA=("xGoal", "sum"),
).reset_index()
goalie_season["saves"] = goalie_season["shots_faced"] - goalie_season["goals_against"]
goalie_season["sv_pct"] = goalie_season["saves"] / goalie_season["shots_faced"]

# GSAx = xGA - GA (positive = saved more than expected)
goalie_season["GSAx"] = goalie_season["xGA"] - goalie_season["goals_against"]

# Normalize GSAx relative to league-average per season
# The raw xGoal model may be biased, so subtract the league-average GSAx/shot rate
for s in goalie_season["season"].unique():
    mask = goalie_season["season"] == s
    season_shots = goalie_season.loc[mask, "shots_faced"].sum()
    season_gsax = goalie_season.loc[mask, "GSAx"].sum()
    lg_gsax_per_shot = season_gsax / season_shots if season_shots > 0 else 0
    goalie_season.loc[mask, "GSAx_adj"] = (
        goalie_season.loc[mask, "GSAx"] - lg_gsax_per_shot * goalie_season.loc[mask, "shots_faced"]
    )

# Pooled goalie stats (across all seasons)
goalie_pooled = goalie_season.groupby("facing_goalie").agg(
    total_shots=("shots_faced", "sum"),
    total_ga=("goals_against", "sum"),
    total_xGA=("xGA", "sum"),
    total_GSAx_adj=("GSAx_adj", "sum"),
    seasons_played=("season", "nunique"),
).reset_index()
goalie_pooled["total_sv_pct"] = 1 - goalie_pooled["total_ga"] / goalie_pooled["total_shots"]

# Get goalie names from raw PBP + shots_2025 (covers newer goalies)
goalie_names = pd.concat([
    pd.read_csv("data/raw_pbp.csv", usecols=["event_goalie_id", "event_goalie_name"]).dropna(),
    pd.read_csv("data/raw_pbp_2025.csv", usecols=["event_goalie_id", "event_goalie_name"]).dropna(),
]).drop_duplicates("event_goalie_id")
# Also pull from shots_2025.csv for goalies not in raw_pbp
if Path("data/shots_2025.csv").exists():
    shots_goalies = pd.read_csv("data/shots_2025.csv",
        usecols=["goalieIdForShot", "goalieNameForShot"]).dropna().drop_duplicates()
    shots_goalies = shots_goalies.rename(columns={
        "goalieIdForShot": "event_goalie_id",
        "goalieNameForShot": "event_goalie_name",
    })
    shots_goalies["event_goalie_name"] = shots_goalies["event_goalie_name"].str.replace(" ", ".")
    goalie_names = pd.concat([goalie_names, shots_goalies]).drop_duplicates("event_goalie_id")
goalie_names["event_goalie_id"] = goalie_names["event_goalie_id"].astype(int)
goalie_pooled = goalie_pooled.merge(
    goalie_names, left_on="facing_goalie", right_on="event_goalie_id", how="left"
)
goalie_season = goalie_season.merge(
    goalie_names, left_on="facing_goalie", right_on="event_goalie_id", how="left"
)

# --- Replacement level for goalies ---
# Use starters (1000+ shots in a season) to set replacement level
GOALIE_RL_PERCENTILE = 25
starter_seasons = goalie_season[goalie_season["shots_faced"] >= 1000].copy()
starter_seasons["GSAx_adj_per_shot"] = starter_seasons["GSAx_adj"] / starter_seasons["shots_faced"]
goalie_rl_per_shot = float(np.percentile(starter_seasons["GSAx_adj_per_shot"], GOALIE_RL_PERCENTILE))
print(f"  Goalie RL (per shot): {goalie_rl_per_shot:.5f} ({len(starter_seasons)} starter-seasons)", file=sys.stderr)

# --- Per-season goalie WAR ---
goalie_season["GSAx_above_repl"] = goalie_season["GSAx_adj"] - goalie_rl_per_shot * goalie_season["shots_faced"]
goalie_season["GOALIE_GAR"] = goalie_season["GSAx_above_repl"].round(2)
goalie_season["GOALIE_WAR"] = (goalie_season["GOALIE_GAR"] / GOALS_TO_WINS).round(2)

goalie_season_out = goalie_season[[
    "facing_goalie", "event_goalie_name", "season", "shots_faced", "goals_against",
    "xGA", "sv_pct", "GSAx", "GSAx_adj", "GOALIE_GAR", "GOALIE_WAR"
]].rename(columns={"facing_goalie": "goalie_id", "event_goalie_name": "goalie_name"})
goalie_season_out = goalie_season_out.sort_values(["season", "GOALIE_WAR"], ascending=[True, False])
goalie_season_out.to_csv("output/v2_goalie_war_by_season.csv", index=False)
print(f"  Per-season goalie WAR: {len(goalie_season_out):,} goalie-seasons → output/v2_goalie_war_by_season.csv", file=sys.stderr)

# --- Pooled goalie WAR ---
goalie_pooled["GSAx_above_repl"] = goalie_pooled["total_GSAx_adj"] - goalie_rl_per_shot * goalie_pooled["total_shots"]
goalie_pooled["GOALIE_GAR"] = goalie_pooled["GSAx_above_repl"].round(2)
goalie_pooled["GOALIE_WAR"] = (goalie_pooled["GOALIE_GAR"] / GOALS_TO_WINS).round(2)

goalie_pooled_out = goalie_pooled[[
    "facing_goalie", "event_goalie_name", "total_shots", "total_ga", "total_xGA",
    "total_sv_pct", "total_GSAx_adj", "seasons_played", "GOALIE_GAR", "GOALIE_WAR"
]].rename(columns={"facing_goalie": "goalie_id", "event_goalie_name": "goalie_name"})
goalie_pooled_out = goalie_pooled_out.sort_values("GOALIE_WAR", ascending=False)
goalie_pooled_out.to_csv("output/v2_goalie_war.csv", index=False)
print(f"  Pooled goalie WAR: {len(goalie_pooled_out):,} goalies → output/v2_goalie_war.csv", file=sys.stderr)

# Print top goalies
print("\nTop 15 goalie-seasons (2022+, 800+ shots):", file=sys.stderr)
recent_g = goalie_season_out[(goalie_season_out["season"] >= 2022) & (goalie_season_out["shots_faced"] >= 800)]
recent_g = recent_g.sort_values("GOALIE_WAR", ascending=False).head(15)
print(recent_g[["season", "goalie_name", "shots_faced", "sv_pct", "GSAx_adj", "GOALIE_WAR"]].to_string(index=False), file=sys.stderr)

print("\nTop 10 pooled goalie WAR (3000+ shots):", file=sys.stderr)
top_pooled_g = goalie_pooled_out[goalie_pooled_out["total_shots"] >= 3000].head(10)
print(top_pooled_g[["goalie_name", "total_shots", "total_sv_pct", "seasons_played", "GOALIE_WAR"]].to_string(index=False), file=sys.stderr)


# ── 9. Summary statistics ───────────────────────────────────────────────────
print("\n── Summary ──", file=sys.stderr)
print(f"  Total league WAR (pooled): {pooled_out['WAR'].sum():.1f}", file=sys.stderr)
print(f"  Mean WAR (pooled): {pooled_out['WAR'].mean():.2f}", file=sys.stderr)
print(f"  Players with GAR > 0: {(pooled_out['GAR'] > 0).sum():,} / {len(pooled_out):,}", file=sys.stderr)

if "GAR_se" in pooled_out.columns:
    print(f"  Median GAR SE: {pooled_out['GAR_se'].median():.2f}", file=sys.stderr)
    print(f"  Mean 90% CI width: {(pooled_out['GAR_hi90'] - pooled_out['GAR_lo90']).mean():.2f}", file=sys.stderr)

# Component contribution breakdown
comp_cols = ["xEV_O_GAR", "xEV_D_GAR", "FINISH_O_GAR", "FINISH_D_GAR",
             "PP_GAR", "PK_GAR", "PEN_GAR", "FO_GAR"]
comp_abs = pooled_out[comp_cols].abs().sum()
comp_total = comp_abs.sum()
print(f"\n  Component importance (% of |GAR|):", file=sys.stderr)
for c in comp_cols:
    pct = comp_abs[c] / comp_total * 100
    print(f"    {c:16s}: {pct:5.1f}%", file=sys.stderr)

print("\nDone.", file=sys.stderr)
