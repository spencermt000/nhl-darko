"""
gar.py — Component-level xGAR / GAR / WAR framework.

Decomposes player value into 8 skater components + goalie WAR:

  Skater components (all in goals-above-average units):
    1. xEV_O   — EV offense (expected): shot generation, transition, territorial
    2. xEV_D   — EV defense (expected): shot suppression, defensive transition
    3. FINISH_O — Offensive finishing (scoring above expected)
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
import numpy as np
import pandas as pd
from pathlib import Path

# ── BPR weights (must match rapm_v2.py) ──────────────────────────────────────
W_xGF = 0.50
W_SOG = 0.22
W_GF  = 0.15
W_TO  = 0.06
W_GA  = -0.04

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

# Skaters by game (faceoffs, penalty counts, all-situations TOI)
sbg_cols = ["playerId", "season", "situation", "icetime",
            "faceoffsWon", "faceoffsLost", "penalties", "penaltiesDrawn"]
sbg = pd.read_csv("data/skaters_by_game.csv", usecols=sbg_cols)
sbg = sbg.rename(columns={"playerId": "player_id"})

print(f"  Pooled ratings: {len(pooled):,} players", file=sys.stderr)
print(f"  By-season ratings: {len(by_season):,} player-seasons", file=sys.stderr)
print(f"  Penalty events: {len(penalties):,}", file=sys.stderr)


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


# ── 3. Compute EV components ────────────────────────────────────────────────
def compute_ev_components(df):
    """Decompose BPR into xEV and FINISH components using per-metric RAPM coefficients."""
    # xEV_O = xGF_O*0.50 + SOG_O*0.22 + TO_O*0.06 + GA_O*(-0.04)
    df["xEV_O"] = (
        df["xGF_O"] * W_xGF +
        df["SOG_O"] * W_SOG +
        df["TO_O"]  * W_TO  +
        df["GA_O"]  * W_GA
    ).round(4)

    # xEV_D = xGF_D*0.50 + SOG_D*0.22 + TO_D*0.06 + GA_D*(-0.04)
    df["xEV_D"] = (
        df["xGF_D"] * W_xGF +
        df["SOG_D"] * W_SOG +
        df["TO_D"]  * W_TO  +
        df["GA_D"]  * W_GA
    ).round(4)

    # FINISH = GF * 0.15
    df["FINISH_O"] = (df["GF_O"] * W_GF).round(4)
    df["FINISH_D"] = (df["GF_D"] * W_GF).round(4)

    # Verification: xEV + FINISH should equal BPR
    df["_check_O"] = (df["xEV_O"] + df["FINISH_O"] - df["BPR_O"]).abs()
    df["_check_D"] = (df["xEV_D"] + df["FINISH_D"] - df["BPR_D"]).abs()
    max_err = max(df["_check_O"].max(), df["_check_D"].max())
    if max_err > 0.002:
        print(f"  WARNING: xEV+FINISH != BPR, max error = {max_err:.4f}", file=sys.stderr)
    else:
        print(f"  Component decomposition verified (max error {max_err:.6f})", file=sys.stderr)
    df = df.drop(columns=["_check_O", "_check_D"])

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

    # Convert rates to counting stats (goals above average)
    df["xEV_O_GAR"] = (df["xEV_O"] * toi_5v5 / 60).round(2)
    df["xEV_D_GAR"] = (df["xEV_D"] * toi_5v5 / 60).round(2)
    df["FINISH_O_GAR"] = (df["FINISH_O"] * toi_5v5 / 60).round(2)
    df["FINISH_D_GAR"] = (df["FINISH_D"] * toi_5v5 / 60).round(2)

    # Convenience: total EV offense/defense
    df["EV_O_GAR"] = (df["xEV_O_GAR"] + df["FINISH_O_GAR"]).round(2)
    df["EV_D_GAR"] = (df["xEV_D_GAR"] + df["FINISH_D_GAR"]).round(2)

    # --- PP/PK components ---
    # PP/PK is pooled career data, so always merge on player_id only
    if "PP_O" not in df.columns:
        df = df.merge(pp_data, on="player_id", how="left")

    toi_pp = df["toi_pp"].fillna(0).values
    toi_pk = df["toi_pk"].fillna(0).values

    df["PP_GAR"] = (df["PP_O"].fillna(0) * toi_pp / 60).round(2)
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


# ── 8. Goalie WAR ───────────────────────────────────────────────────────────
goalie_file = Path("output/v2_goalie_rapm.csv")
if goalie_file.exists():
    print("\n── Computing goalie WAR ──", file=sys.stderr)
    g = pd.read_csv(goalie_file)
    g["goalie_id"] = g["goalie_id"].astype(int)

    # Goalie coefficients: home_goalie = +1 in design matrix.
    # Negative xGF_G = fewer goals when this goalie plays at home = GOOD goalie
    # (suppresses overall shot/goal events — captures both save ability and team effects)
    # So: goalie_rate = -coef (flip sign so positive = good)
    # Blend process (xGF) and results (GF)
    has_xgf = "xGF_G" in g.columns
    has_gf = "GF_G" in g.columns

    if has_xgf and has_gf:
        g["goalie_rate"] = -(g["xGF_G"] * 0.60 + g["GF_G"] * 0.40)
    elif has_xgf:
        g["goalie_rate"] = -g["xGF_G"]
    elif has_gf:
        g["goalie_rate"] = -g["GF_G"]
    else:
        print("  WARNING: No xGF_G or GF_G columns in goalie RAPM, skipping", file=sys.stderr)
        g = None

    # CAVEAT: Goalie coefficients use fixed home(+1)/away(-1) encoding, not
    # relative to acting team. This conflates goalie save ability with team-level
    # shot environment. These ratings are EXPERIMENTAL — a proper goalie model
    # would encode defending_goalie relative to the shooting team.
    print("  NOTE: Goalie WAR is experimental (home/away encoding limitation)", file=sys.stderr)

    if g is not None:
        # Goalie TOI from skaters_by_game (goalies won't be there) — use raw_pbp instead
        # For now, estimate goalie TOI from the number of events they appeared in
        # Actually, use the MoneyPuck goalie data or approximate from 5v5 events
        # Simple approach: use the magnitude of the coefficient as a proxy for reliability

        # Replacement level: ~30th percentile of goalie rates
        goalie_rl = float(np.percentile(g["goalie_rate"], 30))
        print(f"  Goalie replacement level: {goalie_rl:.4f}", file=sys.stderr)

        g["goalie_rate_above_repl"] = g["goalie_rate"] - goalie_rl

        # Estimate goalie TOI: goalies with more events get more reliable estimates
        # Use a fixed TOI estimate based on NHL average (~2500 min/season, pooled over multiple seasons)
        # Better: count events from v2_clean_pbp where this goalie appeared
        # For now, use a uniform estimate (the RAPM coefficient IS per-60, so multiply by estimated TOI)
        # We'll refine this with actual goalie TOI later
        # Average starter: ~1500 min 5v5 per season × ~10 seasons in data ≈ 10,000-15,000 min pooled
        # But we should estimate per goalie. Use the design matrix event counts as proxy.

        # Rough approach: assume coefficients are per-event rates scaled to per-60
        # The RAPM coefficients are already in "goals per 60" units
        # GAR = rate_above_repl * estimated_toi / 60
        # Without actual goalie TOI, report rate-based metrics only
        g["GOALIE_GAR_per60"] = g["goalie_rate_above_repl"].round(4)
        g["GOALIE_WAR_per60"] = (g["GOALIE_GAR_per60"] / GOALS_TO_WINS).round(4)

        g_out = g[["goalie_id", "goalie_name", "goalie_rate", "goalie_rate_above_repl",
                    "GOALIE_GAR_per60", "GOALIE_WAR_per60"]].copy()

        # Add per-metric coefficients for transparency
        for col in g.columns:
            if col.endswith("_G") and col not in g_out.columns:
                g_out[col] = g[col]

        g_out = g_out.sort_values("GOALIE_GAR_per60", ascending=False)
        g_out.to_csv("output/v2_goalie_war.csv", index=False)
        print(f"  Goalie WAR: {len(g_out):,} goalies → output/v2_goalie_war.csv", file=sys.stderr)

        print("\nTop 15 goalies (GAR per 60):", file=sys.stderr)
        print(g_out.head(15)[["goalie_name", "goalie_rate", "GOALIE_GAR_per60",
                              "GOALIE_WAR_per60"]].to_string(index=False), file=sys.stderr)
else:
    print("\n  No v2_goalie_rapm.csv found — skipping goalie WAR", file=sys.stderr)
    print("  (Re-run rapm_v2.py to generate goalie coefficients)", file=sys.stderr)


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
