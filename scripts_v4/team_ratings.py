"""
team_ratings.py — Team-level NHL ratings from player metrics.

Two approaches compared head-to-head:
  1. Roster Aggregation: TOI-weighted sum of pre-game player daily ratings + goalie RAPM
  2. XGBoost Team-Game Model: roster features + schedule + rolling team stats → goal differential

Inputs:
  data/v5_daily_ratings.csv        Per-game player ratings (476K rows)
  data/v2_clean_pbp.csv            Play-by-play for game outcomes
  data/skaters_by_game.csv         Team/TOI mapping per player-game
  data/v2_goalie_rapm.csv          Career-pooled goalie ratings

Outputs:
  data/v6_team_game_ratings.csv    Per team-game ratings + predictions
  data/v6_team_season_ratings.csv  Season-averaged team ratings
"""

import sys
import warnings
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from scipy.special import expit  # logistic sigmoid

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Config ────────────────────────────────────────────────────────────────────
RATING_COLS = ["EV_O", "EV_D", "PP", "PK", "PEN"]
GAR_COLS = ["EV_O_gar", "EV_D_gar", "PP_gar", "PK_gar", "PEN_gar"]
SE_COLS = ["EV_O_se", "EV_D_se", "PP_se", "PK_se", "PEN_se"]

# Prior-season metrics to join (from season-level files, lagged by 1 year)
PRIOR_COMPOSITE_COLS = ["composite_O", "composite_D", "PV_O", "PV_D",
                        "GV_O", "GV_D", "OOI_O", "OOI_D"]
PRIOR_BPR_COLS = ["BPR_O", "BPR_D", "BPR", "total_BPR", "total_BPR_adj"]
PRIOR_GAR_COLS = ["xEV_O", "xEV_D", "FINISH_O", "FINISH_D", "FO_GAR",
                  "WAR"]
TRAIN_SEASONS = list(range(2015, 2022))   # 2015-2021 (daily ratings convention)
TEST_SEASONS = [2022, 2023]               # 2022-23 + 2023-24 NHL seasons

# Typical per-game TOI split (minutes) for weighting components
TOI_5V5_PER_GAME = 50.0
TOI_PP_PER_GAME = 4.0
TOI_PK_PER_GAME = 4.0

ROLLING_WINDOW = 10  # games for team rolling stats


# ── Step 1: Game-Outcome Table ────────────────────────────────────────────────

def build_game_outcomes():
    """Build one row per game with goals, xG, and goalie IDs."""
    print("  Loading PBP...")
    pbp = pd.read_csv("data/v2_clean_pbp.csv", usecols=[
        "game_id", "season", "event_team_type", "period", "is_goal", "xGoal",
        "home_goalie_id", "away_goalie_id",
    ])

    # Goal differential: exclude shootout (period 5)
    reg = pbp[pbp["period"] <= 4]
    goals = (
        reg[reg["is_goal"] == True]
        .groupby(["game_id", "season", "event_team_type"])
        .size()
        .unstack(fill_value=0)
    )
    goals.columns = [f"{c}_goals" for c in goals.columns]
    if "home_goals" not in goals.columns:
        goals["home_goals"] = 0
    if "away_goals" not in goals.columns:
        goals["away_goals"] = 0

    # xG: only 5v5 events have meaningful xGoal, but sum all for completeness
    xg = (
        reg.groupby(["game_id", "event_team_type"])["xGoal"]
        .sum()
        .unstack(fill_value=0)
    )
    xg.columns = [f"{c}_xG" for c in xg.columns]

    games = goals.join(xg, how="left").reset_index()
    games["goal_diff"] = games["home_goals"] - games["away_goals"]

    # Actual winner (including SO — use all periods)
    all_goals = (
        pbp[pbp["is_goal"] == True]
        .groupby(["game_id", "event_team_type"])
        .size()
        .unstack(fill_value=0)
    )
    all_goals.columns = [f"{c}_goals_all" for c in all_goals.columns]
    if "home_goals_all" not in all_goals.columns:
        all_goals["home_goals_all"] = 0
    if "away_goals_all" not in all_goals.columns:
        all_goals["away_goals_all"] = 0
    games = games.join(all_goals, on="game_id", how="left")
    games["home_win"] = (games["home_goals_all"] >= games.get("away_goals_all", 0)).astype(int)

    # Goalie IDs: mode per game
    goalie_home = (
        pbp.dropna(subset=["home_goalie_id"])
        .groupby("game_id")["home_goalie_id"]
        .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else np.nan)
    )
    goalie_away = (
        pbp.dropna(subset=["away_goalie_id"])
        .groupby("game_id")["away_goalie_id"]
        .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else np.nan)
    )
    games = games.join(goalie_home, on="game_id").join(goalie_away, on="game_id")

    return games


# ── Step 2: Pre-Game Roster Features ─────────────────────────────────────────

def build_pregame_ratings():
    """Join daily ratings with team/TOI info, shift to get pre-game values.
    Also joins prior-season metrics (BPR, composite, GAR) for talent priors."""
    print("  Loading daily ratings...")
    dr = pd.read_csv("data/v5_daily_ratings.csv")

    print("  Loading skaters_by_game for team mapping...")
    sk = pd.read_csv("data/skaters_by_game.csv", usecols=[
        "playerId", "gameId", "playerTeam", "home_or_away", "situation", "icetime",
    ])
    sk = sk[sk["situation"] == "all"].rename(columns={
        "playerId": "player_id", "gameId": "game_id",
    })
    sk = sk[["player_id", "game_id", "playerTeam", "home_or_away", "icetime"]]

    # Merge daily ratings with team info
    merged = dr.merge(sk, on=["player_id", "game_id"], how="inner")

    # Shift daily ratings by 1 game per player to get pre-game values (leakage fix)
    merged = merged.sort_values(["player_id", "game_date"])
    shift_cols = RATING_COLS + GAR_COLS + SE_COLS
    for col in shift_cols:
        merged[col] = merged.groupby("player_id")[col].shift(1)
    merged[shift_cols] = merged[shift_cols].fillna(0)

    # ── Join prior-season metrics (lagged 1 year → no leakage) ────────────
    print("  Loading prior-season metrics (composite, BPR, GAR)...")

    # Composite: per-season ensemble blend
    comp = pd.read_csv("data/v5_composite_player_seasons.csv",
                        usecols=["player_id", "season"] + PRIOR_COMPOSITE_COLS)
    comp = comp.rename(columns={c: f"prior_{c}" for c in PRIOR_COMPOSITE_COLS})
    comp["season"] = comp["season"] + 1  # lag: season N metrics → available for season N+1 games
    merged = merged.merge(comp, on=["player_id", "season"], how="left")

    # BPR: Bayesian RAPM blend
    bpr = pd.read_csv("data/v2_final_ratings_by_season.csv",
                       usecols=["player_id", "season"] + PRIOR_BPR_COLS)
    bpr = bpr.rename(columns={c: f"prior_{c}" for c in PRIOR_BPR_COLS})
    bpr["season"] = bpr["season"] + 1
    merged = merged.merge(bpr, on=["player_id", "season"], how="left")

    # GAR components: skill vs luck decomposition
    gar = pd.read_csv("data/v2_gar_by_season.csv",
                       usecols=["player_id", "season"] + PRIOR_GAR_COLS)
    gar = gar.rename(columns={c: f"prior_{c}" for c in PRIOR_GAR_COLS})
    gar["season"] = gar["season"] + 1
    merged = merged.merge(gar, on=["player_id", "season"], how="left")

    # Fill NaN priors with 0 (rookies / first season in data)
    all_prior_cols = ([f"prior_{c}" for c in PRIOR_COMPOSITE_COLS]
                      + [f"prior_{c}" for c in PRIOR_BPR_COLS]
                      + [f"prior_{c}" for c in PRIOR_GAR_COLS])
    merged[all_prior_cols] = merged[all_prior_cols].fillna(0)

    print(f"  Prior-season features: {len(all_prior_cols)} columns joined")

    return merged


# ── Step 3: Aggregate to Team-Game Features ──────────────────────────────────

def aggregate_team_features(roster):
    """Compute TOI-weighted team-level features per game."""

    def weighted_mean(df, col, weight_col):
        w = df[weight_col].clip(lower=0)
        if w.sum() == 0:
            return 0.0
        return np.average(df[col], weights=w)

    # All prior-season columns to aggregate
    all_prior_cols = ([f"prior_{c}" for c in PRIOR_COMPOSITE_COLS]
                      + [f"prior_{c}" for c in PRIOR_BPR_COLS]
                      + [f"prior_{c}" for c in PRIOR_GAR_COLS])

    def agg_team(grp):
        out = {}
        # ── Daily ratings (current form) ──────────────────────────────────
        # 5v5 ratings weighted by 5v5 TOI
        for col in ["EV_O", "EV_D"]:
            out[f"roster_{col}"] = weighted_mean(grp, col, "toi_5v5")
        # PP weighted by PP TOI
        pp_players = grp[grp["toi_pp"] > 0]
        out["roster_PP"] = weighted_mean(pp_players, "PP", "toi_pp") if len(pp_players) > 0 else 0
        # PK weighted by PK TOI
        pk_players = grp[grp["toi_pk"] > 0]
        out["roster_PK"] = weighted_mean(pk_players, "PK", "toi_pk") if len(pk_players) > 0 else 0
        # Penalties weighted by total TOI
        out["roster_PEN"] = weighted_mean(grp, "PEN", "toi_all")
        # GAR sums (goals above average per game)
        for col in GAR_COLS:
            out[f"roster_{col}_sum"] = grp[col].sum()
        out["roster_GAR_total"] = sum(out[f"roster_{c}_sum"] for c in GAR_COLS)
        # Depth: std of EV_O/D
        out["roster_EV_O_std"] = grp["EV_O"].std() if len(grp) > 1 else 0
        out["roster_EV_D_std"] = grp["EV_D"].std() if len(grp) > 1 else 0
        # Top-6 forward avg
        fwds = grp[grp["position"].isin(["C", "L", "R"])]
        if len(fwds) >= 6:
            top6 = fwds.nlargest(6, "toi_5v5")
            out["roster_EV_O_top6"] = top6["EV_O"].mean()
        else:
            out["roster_EV_O_top6"] = fwds["EV_O"].mean() if len(fwds) > 0 else 0
        # Average uncertainty
        out["roster_avg_se"] = grp[["EV_O_se", "EV_D_se"]].mean().mean()

        # ── Prior-season metrics (talent baseline) ────────────────────────
        # TOI-weighted mean of each prior-season metric
        for col in all_prior_cols:
            out[f"roster_{col}"] = weighted_mean(grp, col, "toi_5v5")

        # Metadata
        out["home_or_away"] = grp["home_or_away"].iloc[0]
        out["season"] = grp["season"].iloc[0]
        out["game_date"] = grp["game_date"].iloc[0]
        out["n_skaters"] = len(grp)
        return pd.Series(out)

    print("  Aggregating roster features per team-game...")
    team_feats = roster.groupby(["game_id", "playerTeam"]).apply(agg_team, include_groups=False).reset_index()
    team_feats = team_feats.rename(columns={"playerTeam": "team"})

    return team_feats


def add_goalie_features(team_feats, games):
    """Add goalie RAPM to team-game features."""
    print("  Adding goalie features...")
    goalie = pd.read_csv("data/v2_goalie_rapm.csv")
    goalie_map = goalie.set_index("goalie_id")["GA_G"].to_dict()

    # Map goalie to each team-game row
    def get_goalie_rating(row):
        if row["home_or_away"] == "HOME":
            gid = games.loc[games["game_id"] == row["game_id"], "home_goalie_id"]
        else:
            gid = games.loc[games["game_id"] == row["game_id"], "away_goalie_id"]
        if len(gid) == 0 or pd.isna(gid.iloc[0]):
            return 0.0
        return goalie_map.get(int(gid.iloc[0]), 0.0)

    # More efficient: build a lookup
    games_idx = games.set_index("game_id")
    goalie_ratings = []
    for _, row in team_feats.iterrows():
        gid = row["game_id"]
        if gid not in games_idx.index:
            goalie_ratings.append(0.0)
            continue
        game = games_idx.loc[gid]
        if isinstance(game, pd.DataFrame):
            game = game.iloc[0]
        col = "home_goalie_id" if row["home_or_away"] == "HOME" else "away_goalie_id"
        goalie_id = game[col]
        goalie_ratings.append(goalie_map.get(int(goalie_id), 0.0) if pd.notna(goalie_id) else 0.0)

    team_feats["goalie_GA_G"] = goalie_ratings
    return team_feats


def add_schedule_features(team_feats):
    """Add rest days, back-to-back flags."""
    print("  Adding schedule features...")
    team_feats["game_date"] = pd.to_datetime(team_feats["game_date"])
    team_feats = team_feats.sort_values(["team", "game_date"])

    team_feats["prev_game_date"] = team_feats.groupby("team")["game_date"].shift(1)
    team_feats["rest_days"] = (team_feats["game_date"] - team_feats["prev_game_date"]).dt.days
    team_feats["rest_days"] = team_feats["rest_days"].fillna(7).clip(upper=7)
    team_feats["is_b2b"] = (team_feats["rest_days"] == 1).astype(int)
    team_feats = team_feats.drop(columns=["prev_game_date"])

    return team_feats


def add_rolling_team_stats(team_feats, games):
    """Add rolling team-level xGF%, PP%, PK% from PBP outcomes."""
    print("  Computing rolling team stats...")

    # Build per-team-game outcome stats from the games table
    # We need team-level xG data — use the games table which has home/away xG
    # Reshape to per-team rows
    home = games[["game_id", "season"]].copy()
    home["xGF"] = games.get("home_xG", 0)
    home["xGA"] = games.get("away_xG", 0)
    home["GF"] = games["home_goals"]
    home["GA"] = games["away_goals"]

    away = games[["game_id", "season"]].copy()
    away["xGF"] = games.get("away_xG", 0)
    away["xGA"] = games.get("home_xG", 0)
    away["GF"] = games["away_goals"]
    away["GA"] = games["home_goals"]

    # Get team names from team_feats
    home_teams = team_feats[team_feats["home_or_away"] == "HOME"][["game_id", "team"]].drop_duplicates()
    away_teams = team_feats[team_feats["home_or_away"] == "AWAY"][["game_id", "team"]].drop_duplicates()

    home = home.merge(home_teams, on="game_id", how="inner")
    away = away.merge(away_teams, on="game_id", how="inner")

    team_outcomes = pd.concat([home, away], ignore_index=True)
    team_outcomes = team_outcomes.merge(
        team_feats[["game_id", "team", "game_date"]].drop_duplicates(),
        on=["game_id", "team"],
        how="left",
    )
    team_outcomes = team_outcomes.sort_values(["team", "game_date"])

    # Rolling stats (strictly prior games via shift)
    for col in ["xGF", "xGA", "GF", "GA"]:
        team_outcomes[f"{col}_roll"] = (
            team_outcomes.groupby("team")[col]
            .transform(lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=3).mean())
        )

    team_outcomes["team_xGF_pct"] = team_outcomes["xGF_roll"] / (
        team_outcomes["xGF_roll"] + team_outcomes["xGA_roll"]
    )
    team_outcomes["team_GF_pct"] = team_outcomes["GF_roll"] / (
        team_outcomes["GF_roll"] + team_outcomes["GA_roll"]
    )
    team_outcomes[["team_xGF_pct", "team_GF_pct"]] = team_outcomes[
        ["team_xGF_pct", "team_GF_pct"]
    ].fillna(0.5)

    # Merge back
    rolling_cols = ["game_id", "team", "team_xGF_pct", "team_GF_pct"]
    team_feats = team_feats.merge(
        team_outcomes[rolling_cols].drop_duplicates(),
        on=["game_id", "team"],
        how="left",
    )
    team_feats[["team_xGF_pct", "team_GF_pct"]] = team_feats[
        ["team_xGF_pct", "team_GF_pct"]
    ].fillna(0.5)

    return team_feats


# ── Step 4: Build Matchup Table ──────────────────────────────────────────────

def build_matchup_table(team_feats, games):
    """Combine home and away team features into one row per game."""
    print("  Building matchup table...")
    home = team_feats[team_feats["home_or_away"] == "HOME"].copy()
    away = team_feats[team_feats["home_or_away"] == "AWAY"].copy()

    # Rename columns with prefix
    feat_cols = [c for c in team_feats.columns if c.startswith("roster_") or c.startswith("goalie_") or c.startswith("team_")]
    sched_cols = ["rest_days", "is_b2b"]

    home_rename = {c: f"home_{c}" for c in feat_cols + sched_cols}
    away_rename = {c: f"away_{c}" for c in feat_cols + sched_cols}

    home = home.rename(columns=home_rename)
    away = away.rename(columns=away_rename)

    matchup = home[["game_id", "season", "game_date", "team"] + list(home_rename.values())].merge(
        away[["game_id", "team"] + list(away_rename.values())],
        on="game_id",
        suffixes=("", "_away"),
    )
    matchup = matchup.rename(columns={"team": "home_team", "team_away": "away_team"})

    # Add game outcomes
    games_sub = games[["game_id", "home_goals", "away_goals", "goal_diff", "home_win"]].copy()
    # handle xG columns if they exist
    for col in ["home_xG", "away_xG"]:
        if col in games.columns:
            games_sub[col] = games[col]
    matchup = matchup.merge(games_sub, on="game_id", how="inner")

    # Difference features for XGBoost
    for col in feat_cols:
        matchup[f"diff_{col}"] = matchup[f"home_{col}"] - matchup[f"away_{col}"]
    matchup["rest_advantage"] = matchup["home_rest_days"] - matchup["away_rest_days"]

    return matchup


# ── Model 1: Pure Roster Aggregation ─────────────────────────────────────────

def pure_roster_agg_predict(matchup, train_mask):
    """Pure roster aggregation: sum player GARs, no fitted weights."""
    print("\n=== Model 1: Pure Roster Aggregation (no fitting) ===")
    from sklearn.linear_model import LogisticRegression

    # Team strength = sum of player GARs (already in goals-per-game units)
    for prefix in ["home", "away"]:
        matchup[f"{prefix}_strength"] = matchup[f"{prefix}_roster_GAR_total"]

    # Predicted GD = home_GAR - away_GAR + home-ice advantage (mean GD on train)
    train_idx = matchup.index[train_mask]
    raw_diff = matchup.loc[train_idx, "home_strength"] - matchup.loc[train_idx, "away_strength"]
    hia = matchup.loc[train_idx, "goal_diff"].mean()  # just overall home advantage
    print(f"  Home-ice advantage (mean home GD): {hia:.3f}")

    matchup["pred_gd_pure"] = (
        matchup["home_strength"] - matchup["away_strength"] + hia
    )

    # Win prob via logistic on training data
    lr = LogisticRegression()
    lr.fit(matchup.loc[train_idx, ["pred_gd_pure"]], matchup.loc[train_idx, "home_win"])
    matchup["pred_win_pure"] = lr.predict_proba(matchup[["pred_gd_pure"]])[:, 1]

    return matchup


# ── Model 2: Linear Regression (player-only features) ───────────────────────

def linear_player_predict(matchup, train_mask):
    """Linear regression on player GAR components — calibrated but player-only."""
    print("\n=== Model 2: Linear Regression (player features only) ===")
    from sklearn.linear_model import LinearRegression, LogisticRegression

    train_idx = matchup.index[train_mask]

    # Features: per-component GAR sums + goalie rating
    agg_features = [
        f"{p}_{c}" for p in ["home", "away"]
        for c in ["roster_EV_O_gar_sum", "roster_EV_D_gar_sum",
                   "roster_PP_gar_sum", "roster_PK_gar_sum", "roster_PEN_gar_sum",
                   "goalie_GA_G"]
    ]
    agg_features = [c for c in agg_features if c in matchup.columns]

    lr_gd = LinearRegression()
    lr_gd.fit(matchup.loc[train_idx, agg_features], matchup.loc[train_idx, "goal_diff"])
    matchup["pred_gd_linear"] = lr_gd.predict(matchup[agg_features])

    print(f"  Features: {len(agg_features)}")
    print(f"  Train R²: {lr_gd.score(matchup.loc[train_idx, agg_features], matchup.loc[train_idx, 'goal_diff']):.4f}")
    print(f"  Coefficients:")
    for feat, coef in zip(agg_features, lr_gd.coef_):
        print(f"    {feat:40s} {coef:>8.3f}")
    print(f"    {'intercept':40s} {lr_gd.intercept_:>8.3f}")

    # Win probability
    lr_win = LogisticRegression()
    lr_win.fit(matchup.loc[train_idx, ["pred_gd_linear"]], matchup.loc[train_idx, "home_win"])
    matchup["pred_win_linear"] = lr_win.predict_proba(matchup[["pred_gd_linear"]])[:, 1]

    return matchup


# ── Model 3: XGBoost (player + team features) ───────────────────────────────

def xgboost_predict(matchup, train_mask, test_mask):
    """XGBoost: player features + team rolling stats + schedule → goal diff."""
    print("\n=== Model 3: XGBoost (player + team features) ===")
    from sklearn.linear_model import LogisticRegression

    feature_cols = [c for c in matchup.columns if c.startswith("diff_")
                    or c.startswith("home_roster_") or c.startswith("away_roster_")
                    or c.startswith("home_goalie_") or c.startswith("away_goalie_")
                    or c.startswith("home_team_") or c.startswith("away_team_")
                    or c in ["home_rest_days", "away_rest_days", "home_is_b2b",
                             "away_is_b2b", "rest_advantage"]]

    X_train = matchup.loc[train_mask, feature_cols].values
    y_train = matchup.loc[train_mask, "goal_diff"].values
    X_test = matchup.loc[test_mask, feature_cols].values

    print(f"  Features: {len(feature_cols)}")
    print(f"  Train: {len(X_train)} games, Test: {len(X_test)} games")

    model = XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        early_stopping_rounds=50,
    )

    # Use last season of training as validation for early stopping
    train_data = matchup[train_mask].copy()
    max_train_season = train_data["season"].max()
    val_mask_inner = train_data["season"] == max_train_season
    X_tr = train_data.loc[~val_mask_inner, feature_cols].values
    y_tr = train_data.loc[~val_mask_inner, "goal_diff"].values
    X_val = train_data.loc[val_mask_inner, feature_cols].values
    y_val = train_data.loc[val_mask_inner, "goal_diff"].values

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    print(f"  Best iteration: {model.best_iteration}")

    # Predict on all data
    X_all = matchup[feature_cols].values
    matchup["pred_gd_xgb"] = model.predict(X_all)

    # Win probability
    lr = LogisticRegression()
    lr.fit(matchup.loc[train_mask, ["pred_gd_xgb"]], matchup.loc[train_mask, "home_win"])
    matchup["pred_win_xgb"] = lr.predict_proba(matchup[["pred_gd_xgb"]])[:, 1]

    # Feature importance
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    print("\n  Top 15 features by importance:")
    for feat, imp in importances.nlargest(15).items():
        print(f"    {feat:40s} {imp:.4f}")

    return matchup, model


# ── Step 7: Evaluation ───────────────────────────────────────────────────────

def evaluate(matchup, test_mask):
    """Compare both approaches on test set."""
    from sklearn.metrics import (
        accuracy_score, log_loss, brier_score_loss, mean_squared_error, r2_score,
    )

    test = matchup[test_mask].copy()
    print(f"\n{'='*70}")
    print(f"EVALUATION — Test set: {len(test)} games "
          f"(seasons {test['season'].min()}-{test['season'].max()})")
    print(f"{'='*70}")

    results = {}

    # Baseline: home always wins
    results["Home-always"] = {
        "Accuracy": accuracy_score(test["home_win"], np.ones(len(test))),
        "LogLoss": None,
        "Brier": None,
        "RMSE_GD": None,
        "R2_GD": None,
    }

    # Baseline: season points %
    # Approximate with rolling GF% as proxy
    test["naive_win_prob"] = test["home_team_GF_pct"].clip(0.3, 0.7)
    results["Rolling GF%"] = {
        "Accuracy": accuracy_score(test["home_win"], (test["naive_win_prob"] > 0.5).astype(int)),
        "LogLoss": log_loss(test["home_win"], test["naive_win_prob"]),
        "Brier": brier_score_loss(test["home_win"], test["naive_win_prob"]),
        "RMSE_GD": None,
        "R2_GD": None,
    }

    # Model 1: Pure Roster Aggregation
    results["Pure GAR Sum"] = {
        "Accuracy": accuracy_score(test["home_win"], (test["pred_win_pure"] > 0.5).astype(int)),
        "LogLoss": log_loss(test["home_win"], test["pred_win_pure"].clip(0.01, 0.99)),
        "Brier": brier_score_loss(test["home_win"], test["pred_win_pure"]),
        "RMSE_GD": np.sqrt(mean_squared_error(test["goal_diff"], test["pred_gd_pure"])),
        "R2_GD": r2_score(test["goal_diff"], test["pred_gd_pure"]),
    }

    # Model 2: Linear (player-only)
    results["Linear (player)"] = {
        "Accuracy": accuracy_score(test["home_win"], (test["pred_win_linear"] > 0.5).astype(int)),
        "LogLoss": log_loss(test["home_win"], test["pred_win_linear"].clip(0.01, 0.99)),
        "Brier": brier_score_loss(test["home_win"], test["pred_win_linear"]),
        "RMSE_GD": np.sqrt(mean_squared_error(test["goal_diff"], test["pred_gd_linear"])),
        "R2_GD": r2_score(test["goal_diff"], test["pred_gd_linear"]),
    }

    # Model 3: XGBoost (player + team)
    results["XGBoost (p+t)"] = {
        "Accuracy": accuracy_score(test["home_win"], (test["pred_win_xgb"] > 0.5).astype(int)),
        "LogLoss": log_loss(test["home_win"], test["pred_win_xgb"].clip(0.01, 0.99)),
        "Brier": brier_score_loss(test["home_win"], test["pred_win_xgb"]),
        "RMSE_GD": np.sqrt(mean_squared_error(test["goal_diff"], test["pred_gd_xgb"])),
        "R2_GD": r2_score(test["goal_diff"], test["pred_gd_xgb"]),
    }

    # Print table
    print(f"\n{'Model':<20s} {'Accuracy':>10s} {'LogLoss':>10s} {'Brier':>10s} {'RMSE(GD)':>10s} {'R²(GD)':>10s}")
    print("-" * 72)
    for name, metrics in results.items():
        acc = f"{metrics['Accuracy']:.3f}" if metrics["Accuracy"] is not None else "--"
        ll = f"{metrics['LogLoss']:.4f}" if metrics["LogLoss"] is not None else "--"
        bs = f"{metrics['Brier']:.4f}" if metrics["Brier"] is not None else "--"
        rmse = f"{metrics['RMSE_GD']:.3f}" if metrics["RMSE_GD"] is not None else "--"
        r2 = f"{metrics['R2_GD']:.4f}" if metrics["R2_GD"] is not None else "--"
        print(f"{name:<20s} {acc:>10s} {ll:>10s} {bs:>10s} {rmse:>10s} {r2:>10s}")

    # Calibration
    print("\nCalibration (XGBoost predicted win prob → actual win rate):")
    test["prob_bin"] = pd.cut(test["pred_win_xgb"], bins=10)
    cal = test.groupby("prob_bin", observed=True).agg(
        pred_prob=("pred_win_xgb", "mean"),
        actual_rate=("home_win", "mean"),
        n=("home_win", "count"),
    )
    for _, row in cal.iterrows():
        bar = "█" * int(row["actual_rate"] * 30)
        print(f"  pred={row['pred_prob']:.2f}  actual={row['actual_rate']:.2f}  n={int(row['n']):>4d}  {bar}")


# ── Step 8: Season Ratings ───────────────────────────────────────────────────

def build_season_ratings(matchup):
    """Average team ratings per season for leaderboard."""
    # Combine home and away into one per-team view
    home = matchup[["game_id", "season", "home_team", "home_roster_EV_O", "home_roster_EV_D",
                     "home_roster_PP", "home_roster_PK", "home_roster_PEN",
                     "home_goalie_GA_G", "home_strength",
                     "pred_gd_linear", "pred_gd_xgb", "goal_diff", "home_win"]].copy()
    home = home.rename(columns={
        "home_team": "team",
        "home_roster_EV_O": "roster_EV_O", "home_roster_EV_D": "roster_EV_D",
        "home_roster_PP": "roster_PP", "home_roster_PK": "roster_PK",
        "home_roster_PEN": "roster_PEN",
        "home_goalie_GA_G": "goalie_GA_G", "home_strength": "team_strength",
    })
    home["team_gd"] = home["goal_diff"]
    home["team_win"] = home["home_win"]

    away = matchup[["game_id", "season", "away_team", "away_roster_EV_O", "away_roster_EV_D",
                     "away_roster_PP", "away_roster_PK", "away_roster_PEN",
                     "away_goalie_GA_G", "away_strength",
                     "pred_gd_linear", "pred_gd_xgb", "goal_diff", "home_win"]].copy()
    away = away.rename(columns={
        "away_team": "team",
        "away_roster_EV_O": "roster_EV_O", "away_roster_EV_D": "roster_EV_D",
        "away_roster_PP": "roster_PP", "away_roster_PK": "roster_PK",
        "away_roster_PEN": "roster_PEN",
        "away_goalie_GA_G": "goalie_GA_G", "away_strength": "team_strength",
    })
    away["team_gd"] = -away["goal_diff"]
    away["team_win"] = 1 - away["home_win"]

    combined = pd.concat([home, away], ignore_index=True)

    season_ratings = combined.groupby(["season", "team"]).agg(
        GP=("game_id", "count"),
        roster_EV_O=("roster_EV_O", "mean"),
        roster_EV_D=("roster_EV_D", "mean"),
        roster_PP=("roster_PP", "mean"),
        roster_PK=("roster_PK", "mean"),
        roster_PEN=("roster_PEN", "mean"),
        goalie_GA_G=("goalie_GA_G", "mean"),
        team_strength=("team_strength", "mean"),
        actual_GD=("team_gd", "sum"),
        actual_wins=("team_win", "sum"),
    ).reset_index()

    season_ratings["win_pct"] = season_ratings["actual_wins"] / season_ratings["GP"]
    season_ratings = season_ratings.sort_values(["season", "team_strength"], ascending=[True, False])

    return season_ratings


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Step 1: Building game-outcome table...")
    games = build_game_outcomes()
    print(f"  {len(games)} games")

    print("\nStep 2: Building pre-game roster features...")
    roster = build_pregame_ratings()
    print(f"  {len(roster)} player-game rows with team info")

    print("\nStep 3: Aggregating to team-game features...")
    team_feats = aggregate_team_features(roster)
    team_feats = add_goalie_features(team_feats, games)
    team_feats = add_schedule_features(team_feats)
    team_feats = add_rolling_team_stats(team_feats, games)
    print(f"  {len(team_feats)} team-game rows")

    print("\nStep 4: Building matchup table...")
    matchup = build_matchup_table(team_feats, games)
    print(f"  {len(matchup)} game matchups")

    # Train/test split
    train_mask = matchup["season"].isin(TRAIN_SEASONS)
    test_mask = matchup["season"].isin(TEST_SEASONS)
    print(f"\n  Train: {train_mask.sum()} games ({TRAIN_SEASONS[0]}-{TRAIN_SEASONS[-1]})")
    print(f"  Test:  {test_mask.sum()} games ({TEST_SEASONS})")

    print("\nStep 5: Model 1 — Pure Roster Aggregation...")
    matchup = pure_roster_agg_predict(matchup, train_mask)

    print("\nStep 6a: Model 2 — Linear Regression (player-only)...")
    matchup = linear_player_predict(matchup, train_mask)

    print("\nStep 6b: Model 3 — XGBoost (player + team)...")
    matchup, model = xgboost_predict(matchup, train_mask, test_mask)

    print("\nStep 7: Evaluation...")
    evaluate(matchup, test_mask)

    print("\nStep 8: Building season ratings...")
    season_ratings = build_season_ratings(matchup)

    # Show leaderboard for last full season
    full_seasons = season_ratings[season_ratings["GP"] >= 70]
    if len(full_seasons) > 0:
        latest = full_seasons["season"].max()
    else:
        latest = season_ratings["season"].max()
    print(f"\n  Team Strength Leaderboard — {latest}-{latest+1} season:")
    top = season_ratings[season_ratings["season"] == latest].head(15)
    print(f"  {'Team':<5s} {'Strength':>10s} {'Win%':>8s} {'GD':>6s} {'GP':>5s}")
    for _, row in top.iterrows():
        print(f"  {row['team']:<5s} {row['team_strength']:>10.3f} {row['win_pct']:>8.3f} {int(row['actual_GD']):>6d} {int(row['GP']):>5d}")

    # Save outputs
    out_cols = ["game_id", "season", "game_date", "home_team", "away_team",
                "home_roster_EV_O", "home_roster_EV_D", "home_roster_PP", "home_roster_PK",
                "away_roster_EV_O", "away_roster_EV_D", "away_roster_PP", "away_roster_PK",
                "home_goalie_GA_G", "away_goalie_GA_G",
                "home_strength", "away_strength",
                "pred_gd_pure", "pred_win_pure",
                "pred_gd_linear", "pred_win_linear",
                "pred_gd_xgb", "pred_win_xgb",
                "home_goals", "away_goals", "goal_diff", "home_win"]
    # Only include columns that exist
    out_cols = [c for c in out_cols if c in matchup.columns]
    matchup[out_cols].to_csv("data/v6_team_game_ratings.csv", index=False)
    print(f"\n  Saved data/v6_team_game_ratings.csv ({len(matchup)} rows)")

    season_ratings.to_csv("data/v6_team_season_ratings.csv", index=False)
    print(f"  Saved data/v6_team_season_ratings.csv ({len(season_ratings)} rows)")


if __name__ == "__main__":
    main()
