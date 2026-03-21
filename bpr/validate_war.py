#!/usr/bin/env python3
"""
validate_war.py — Validation checks for the WAR model.

Three validation tests:
  1. Team Wins Prediction: aggregate player WAR per team-season, correlate with
     actual team points/win%. Show R^2.
  2. YoY Player WAR Stability: for players with 200+ TOI in consecutive seasons,
     compute year-over-year WAR correlation. Also test component-level stability.
  3. Top Player Reasonableness: distribution stats, F vs D comparison, top-10 lists.

Prints results to stdout. Does not write any files.
"""

import sys
import os
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAR_PATH = os.path.join(ROOT, "output", "v2_gar_by_season.csv")
SKATERS_PATH = os.path.join(ROOT, "data", "skaters_by_game.csv")
SKATERS_2025_PATH = os.path.join(ROOT, "data", "skaters_by_game2025.csv")

# ── helpers ──────────────────────────────────────────────────────────────────

def header(title):
    width = 72
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def subheader(title):
    print(f"\n--- {title} ---")


# ── load data ────────────────────────────────────────────────────────────────

def load_gar():
    df = pd.read_csv(GAR_PATH)
    # normalise column names
    df.columns = [c.strip() for c in df.columns]
    return df


def load_skaters():
    """Load skaters_by_game (both pre-2025 and 2025 files) and concatenate."""
    frames = []
    for path in [SKATERS_PATH, SKATERS_2025_PATH]:
        if os.path.exists(path):
            frames.append(pd.read_csv(path, low_memory=False))
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    return df


# ── 1. Team wins prediction ─────────────────────────────────────────────────

def team_wins_prediction(gar: pd.DataFrame, skaters: pd.DataFrame):
    header("1. TEAM WINS PREDICTION (team-aggregated WAR vs team points)")

    # --- derive team standings from on-ice data ---
    # Use the 'all' situation rows (total game stats) to find game outcomes.
    # For each game, find one player per team with situation=='all' and get
    # OnIce_F_goals / OnIce_A_goals totals — but that's on-ice while they play.
    # Better approach: for each (gameId, playerTeam), take the max-icetime player
    # in situation 'all', sum their on-ice goals for/against as a proxy for the
    # full game score. This is imperfect but workable.

    # Actually, we can derive game-level GF/GA more directly:
    # For each game-team, the player with iceTimeRank==1 in situation 'all'
    # played the most minutes and their on-ice goals ~ team game totals.

    all_sit = skaters[skaters["situation"] == "all"].copy()
    all_sit["icetime"] = pd.to_numeric(all_sit["icetime"], errors="coerce")
    all_sit["OnIce_F_goals"] = pd.to_numeric(all_sit["OnIce_F_goals"], errors="coerce")
    all_sit["OnIce_A_goals"] = pd.to_numeric(all_sit["OnIce_A_goals"], errors="coerce")

    # For each game-team, take the player with highest icetime
    idx = all_sit.groupby(["gameId", "playerTeam"])["icetime"].idxmax()
    game_team = all_sit.loc[idx, ["gameId", "season", "playerTeam", "OnIce_F_goals", "OnIce_A_goals"]].copy()
    game_team = game_team.dropna(subset=["OnIce_F_goals", "OnIce_A_goals"])

    # Determine win/loss/OTL
    # Regulation/OT win: GF > GA
    game_team["win"] = (game_team["OnIce_F_goals"] > game_team["OnIce_A_goals"]).astype(int)
    game_team["loss"] = (game_team["OnIce_F_goals"] < game_team["OnIce_A_goals"]).astype(int)

    # Aggregate to team-season
    team_season = game_team.groupby(["season", "playerTeam"]).agg(
        games=("win", "count"),
        wins=("win", "sum"),
        gf=("OnIce_F_goals", "sum"),
        ga=("OnIce_A_goals", "sum"),
    ).reset_index()

    # Approximate points: 2*wins + (games - wins - losses where GF<GA might be OTL)
    # Simple: use points = 2*wins + loser_points_estimate; or just use win%
    team_season["win_pct"] = team_season["wins"] / team_season["games"]
    team_season["pts_approx"] = team_season["win_pct"] * 2 * team_season["games"]  # rough proxy

    # Filter to seasons in GAR data and reasonable game counts
    gar_seasons = set(gar["season"].unique())
    team_season = team_season[team_season["season"].isin(gar_seasons)]
    team_season = team_season[team_season["games"] >= 20]

    # --- aggregate WAR per team-season ---
    # Need to map players to teams. Use skaters_by_game to get primary team.
    player_team = (
        all_sit.groupby(["playerId", "season", "playerTeam"])["icetime"]
        .sum()
        .reset_index()
    )
    # Primary team = team where player had most icetime that season
    idx2 = player_team.groupby(["playerId", "season"])["icetime"].idxmax()
    player_primary = player_team.loc[idx2, ["playerId", "season", "playerTeam"]].copy()
    player_primary = player_primary.rename(columns={"playerId": "player_id", "playerTeam": "team"})

    # Merge with GAR
    gar_with_team = gar.merge(player_primary, on=["player_id", "season"], how="left")
    gar_with_team = gar_with_team.dropna(subset=["team"])

    team_war = (
        gar_with_team.groupby(["season", "team"])
        .agg(
            total_WAR=("WAR", "sum"),
            total_GAR=("GAR", "sum"),
            n_players=("player_id", "nunique"),
        )
        .reset_index()
    )

    # Merge with standings
    merged = team_war.merge(
        team_season,
        left_on=["season", "team"],
        right_on=["season", "playerTeam"],
        how="inner",
    )

    if len(merged) < 10:
        print(f"\n  WARNING: only {len(merged)} team-seasons matched. Results may be unreliable.")

    # Correlation
    r_war, p_war = stats.pearsonr(merged["total_WAR"], merged["win_pct"])
    r_gar, p_gar = stats.pearsonr(merged["total_GAR"], merged["win_pct"])

    subheader("Overall correlation (all seasons pooled)")
    print(f"  Team-seasons matched:  {len(merged)}")
    print(f"  WAR vs win%:  r = {r_war:.3f},  R^2 = {r_war**2:.3f},  p = {p_war:.2e}")
    print(f"  GAR vs win%:  r = {r_gar:.3f},  R^2 = {r_gar**2:.3f},  p = {p_gar:.2e}")

    # Per-season breakdown
    subheader("Per-season R^2 (WAR vs win%)")
    for season in sorted(merged["season"].unique()):
        sub = merged[merged["season"] == season]
        if len(sub) >= 10:
            r, _ = stats.pearsonr(sub["total_WAR"], sub["win_pct"])
            print(f"  {season}:  R^2 = {r**2:.3f}  (n={len(sub)} teams)")
        else:
            print(f"  {season}:  n={len(sub)} teams — too few for reliable R^2")

    # Sanity: show a few top/bottom teams
    subheader("Sample: highest and lowest WAR teams")
    top = merged.nlargest(5, "total_WAR")[["season", "team", "total_WAR", "win_pct", "games"]]
    bot = merged.nsmallest(5, "total_WAR")[["season", "team", "total_WAR", "win_pct", "games"]]
    print("\n  Top 5 team-seasons by WAR:")
    for _, row in top.iterrows():
        print(f"    {int(row.season)} {row.team:>4s}  WAR={row.total_WAR:6.1f}  win%={row.win_pct:.3f}  GP={int(row.games)}")
    print("\n  Bottom 5 team-seasons by WAR:")
    for _, row in bot.iterrows():
        print(f"    {int(row.season)} {row.team:>4s}  WAR={row.total_WAR:6.1f}  win%={row.win_pct:.3f}  GP={int(row.games)}")


# ── 2. YoY stability ────────────────────────────────────────────────────────

def yoy_stability(gar: pd.DataFrame):
    header("2. YEAR-OVER-YEAR PLAYER WAR STABILITY")

    # Filter to players with 200+ TOI (5v5) each season
    qualified = gar[gar["toi_5v5"] >= 200].copy()

    # Build consecutive-season pairs
    pairs = qualified.merge(
        qualified,
        on=["player_id", "player_name", "position"],
        suffixes=("_y1", "_y2"),
    )
    pairs = pairs[pairs["season_y2"] == pairs["season_y1"] + 1]

    if len(pairs) < 20:
        print(f"\n  Only {len(pairs)} player-season pairs found with 200+ TOI — too few.")
        return

    # Component columns to test
    components = [
        ("WAR", "WAR"),
        ("GAR", "GAR"),
        ("xEV_O_GAR", "xEV_O"),
        ("xEV_D_GAR", "xEV_D"),
        ("FINISH_O_GAR", "FINISH_O"),
        ("FINISH_D_GAR", "FINISH_D"),
        ("PP_GAR", "PP"),
        ("PK_GAR", "PK"),
        ("PEN_GAR", "PEN"),
        ("FO_GAR", "FO"),
    ]

    subheader(f"YoY correlations (n={len(pairs)} consecutive player-seasons, 200+ 5v5 TOI)")
    print(f"  {'Metric':<14s} {'r':>6s} {'R^2':>6s} {'p-value':>10s}")
    print(f"  {'-'*14} {'-'*6} {'-'*6} {'-'*10}")
    for col, label in components:
        col_y1 = f"{col}_y1"
        col_y2 = f"{col}_y2"
        if col_y1 in pairs.columns and col_y2 in pairs.columns:
            valid = pairs[[col_y1, col_y2]].dropna()
            if len(valid) > 10:
                r, p = stats.pearsonr(valid[col_y1], valid[col_y2])
                print(f"  {label:<14s} {r:6.3f} {r**2:6.3f} {p:10.2e}")

    # Higher TOI threshold
    for min_toi in [400, 600]:
        q2 = gar[gar["toi_5v5"] >= min_toi]
        p2 = q2.merge(q2, on=["player_id", "player_name", "position"], suffixes=("_y1", "_y2"))
        p2 = p2[p2["season_y2"] == p2["season_y1"] + 1]
        if len(p2) > 20:
            r, _ = stats.pearsonr(p2["WAR_y1"], p2["WAR_y2"])
            print(f"\n  WAR stability at {min_toi}+ 5v5 TOI:  r = {r:.3f},  R^2 = {r**2:.3f}  (n={len(p2)})")

    # Forward vs Defense stability
    subheader("YoY WAR stability by position")
    for pos in ["F", "D"]:
        pos_pairs = pairs[pairs["position"] == pos]
        if len(pos_pairs) > 20:
            r, _ = stats.pearsonr(pos_pairs["WAR_y1"], pos_pairs["WAR_y2"])
            print(f"  {pos}:  r = {r:.3f},  R^2 = {r**2:.3f}  (n={len(pos_pairs)})")


# ── 3. Top-player reasonableness ────────────────────────────────────────────

def top_player_reasonableness(gar: pd.DataFrame):
    header("3. TOP PLAYER REASONABLENESS")

    # Filter to meaningful sample (200+ 5v5 TOI)
    qualified = gar[gar["toi_5v5"] >= 200].copy()

    subheader("WAR distribution (players with 200+ 5v5 TOI)")
    print(f"  n = {len(qualified)}")
    print(f"  Mean   = {qualified['WAR'].mean():.2f}")
    print(f"  Median = {qualified['WAR'].median():.2f}")
    print(f"  Std    = {qualified['WAR'].std():.2f}")
    print(f"  Min    = {qualified['WAR'].min():.2f}")
    print(f"  Max    = {qualified['WAR'].max():.2f}")

    pcts = [10, 25, 50, 75, 90, 95, 99]
    vals = np.percentile(qualified["WAR"], pcts)
    print(f"\n  Percentiles:")
    for p, v in zip(pcts, vals):
        print(f"    {p:3d}th:  {v:6.2f}")

    # Forward vs Defense
    subheader("Forwards vs Defensemen")
    for pos in ["F", "D"]:
        sub = qualified[qualified["position"] == pos]
        print(f"  {pos}: n={len(sub):>5d},  mean WAR={sub['WAR'].mean():.2f},  "
              f"median={sub['WAR'].median():.2f},  std={sub['WAR'].std():.2f}")

    # Component means by position
    comp_cols = ["xEV_O_GAR", "xEV_D_GAR", "FINISH_O_GAR", "FINISH_D_GAR",
                 "PP_GAR", "PK_GAR", "PEN_GAR", "FO_GAR"]
    subheader("Mean GAR component by position (200+ 5v5 TOI)")
    print(f"  {'Component':<14s} {'F':>8s} {'D':>8s} {'diff':>8s}")
    print(f"  {'-'*14} {'-'*8} {'-'*8} {'-'*8}")
    for col in comp_cols:
        f_mean = qualified.loc[qualified["position"] == "F", col].mean()
        d_mean = qualified.loc[qualified["position"] == "D", col].mean()
        label = col.replace("_GAR", "")
        print(f"  {label:<14s} {f_mean:8.2f} {d_mean:8.2f} {f_mean - d_mean:8.2f}")

    # Top 10 per recent season
    recent_seasons = sorted(qualified["season"].unique())[-3:]
    subheader("Top 10 WAR by season (recent seasons)")
    for season in recent_seasons:
        sub = qualified[qualified["season"] == season].nlargest(10, "WAR")
        print(f"\n  {int(season)} season:")
        print(f"  {'Rank':>4s}  {'Player':<25s} {'Pos':>3s} {'WAR':>6s} {'GAR':>7s} "
              f"{'xEV_O':>6s} {'xEV_D':>6s} {'FIN_O':>6s} {'FIN_D':>6s} {'PP':>5s}")
        for i, (_, row) in enumerate(sub.iterrows(), 1):
            name = row["player_name"].replace(".", " ")
            print(f"  {i:4d}  {name:<25s} {row['position']:>3s} {row['WAR']:6.2f} "
                  f"{row['GAR']:7.2f} {row['xEV_O_GAR']:6.2f} {row['xEV_D_GAR']:6.2f} "
                  f"{row['FINISH_O_GAR']:6.2f} {row['FINISH_D_GAR']:6.2f} {row['PP_GAR']:5.2f}")

    # Bottom 10 (worst players)
    subheader("Bottom 10 WAR (most recent full season)")
    last_full = recent_seasons[-2] if len(recent_seasons) > 1 else recent_seasons[-1]
    bot = qualified[qualified["season"] == last_full].nsmallest(10, "WAR")
    print(f"\n  {int(last_full)} season:")
    for i, (_, row) in enumerate(bot.iterrows(), 1):
        name = row["player_name"].replace(".", " ")
        print(f"  {i:4d}  {name:<25s} {row['position']:>3s} {row['WAR']:6.2f}  GAR={row['GAR']:7.2f}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading GAR/WAR data...")
    gar = load_gar()
    print(f"  {len(gar)} player-seasons loaded from {GAR_PATH}")
    print(f"  Seasons: {sorted(gar['season'].unique())}")

    print("\nLoading skaters_by_game data (this may take a moment)...")
    skaters = load_skaters()
    print(f"  {len(skaters)} rows loaded")

    team_wins_prediction(gar, skaters)
    yoy_stability(gar)
    top_player_reasonableness(gar)

    header("VALIDATION COMPLETE")


if __name__ == "__main__":
    main()
