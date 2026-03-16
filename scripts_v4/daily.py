"""
daily.py — DARKO-style daily ratings with 5 Bayesian-smoothed components.

Each game updates 5 independent components via sequential Bayesian smoothing
with exponential decay (halflife=30 games). Low-GP players regress to prior;
high-GP players are data-driven.

Components (all per-60 goals above average):
  EV_O   — 5v5 offensive impact (from XGBoost EPM predictions)
  EV_D   — 5v5 defensive impact (from XGBoost EPM predictions)
  PP     — power play production: (ixG + 0.7*A1) / (toi_pp/60) - league_avg
  PK     — penalty kill defense: -(oiXGA / (toi_pk/60) - league_avg)
  PEN    — penalty drawing/taking: (drawn - taken) * 0.17 / (toi/60) - league_avg

Season GAR accumulated game-by-game (smoothed rate × toi / 60 each game).

Inputs:
  data/v4_epm_raw_per_game.csv       XGBoost per-game predictions (EV_O/D raw)
  data/skaters_by_game.csv           Per-game box scores (PP/PK/penalty stats)
  data/v5_composite_player_seasons.csv  Composite O/D for EV priors
  data/pp_rapm.csv                   PP_O, PK_D career RAPM for PP/PK priors

Outputs:
  data/v5_daily_ratings.csv          Per-player-game smoothed 5-component ratings
  data/v5_daily_war.csv              Per-player-season aggregated WAR
"""

import sys
import numpy as np
import pandas as pd

# ── Config ───────────────────────────────────────────────────────────────────
DECAY_HALFLIFE = 30          # games
GAME_EVIDENCE_SCALE = 1.5    # evidence weight per 15-min 5v5 game
OFFSEASON_DECAY = 0.85       # prior carry across seasons
GOALS_TO_WINS = 6.0
RL_PERCENTILE = 17
GOAL_VALUE_PER_PENALTY = 0.17

# Prior SEs (how uncertain we are before any data)
PRIOR_SE = {
    "EV_O": 0.08,
    "EV_D": 0.05,
    "PP":   0.15,
    "PK":   0.10,
    "PEN":  0.05,
}

# Reference TOI for evidence weighting (minutes)
# PP/PK: 2 min reference (a full PP/PK game gives similar evidence to 15 min 5v5)
REF_TOI = {
    "EV_O": 15.0,
    "EV_D": 15.0,
    "PP":   2.0,
    "PK":   2.0,
    "PEN":  15.0,
}

# EV raw values are per-10-min xGI; convert to per-60: × 6.0
EV_TO_PER60 = 6.0


# ── 1. Load data ─────────────────────────────────────────────────────────────
print("Loading data...", file=sys.stderr)

# EV raw per-game predictions from epm.py
epm = pd.read_csv("data/v4_epm_raw_per_game.csv")
epm = epm.rename(columns={"player_id": "player_id", "game_id": "game_id"})
epm["player_id"] = epm["player_id"].astype(int)
print(f"  EPM raw: {len(epm):,} rows", file=sys.stderr)

# Skaters by game — load PP (5on4), PK (4on5), and all-situations rows
sbg_cols = [
    "playerId", "name", "gameId", "season", "gameDate", "position", "situation",
    "icetime",
    "I_F_xGoals", "I_F_primaryAssists",
    "OnIce_A_xGoals",
    "penalties", "penaltiesDrawn",
]
sbg = pd.read_csv("data/skaters_by_game.csv", usecols=sbg_cols)
sbg = sbg.rename(columns={"playerId": "player_id", "name": "player_name",
                           "gameId": "game_id", "gameDate": "game_date"})
sbg["player_id"] = sbg["player_id"].astype(int)
sbg["toi_sec"] = sbg["icetime"].fillna(0)
sbg["toi_min"] = sbg["toi_sec"] / 60.0

# Convert gameDate from int (20090102) to string date
sbg["game_date"] = pd.to_datetime(sbg["game_date"].astype(str), format="%Y%m%d")
print(f"  Skaters by game: {len(sbg):,} rows", file=sys.stderr)

# Composite player-seasons (for EV priors)
composite = pd.read_csv("data/v5_composite_player_seasons.csv",
                         usecols=["player_id", "season", "composite_O", "composite_D"])
composite["player_id"] = composite["player_id"].astype(int)

# PP/PK RAPM (career priors)
pp_rapm = pd.read_csv("data/pp_rapm.csv", usecols=["player_id", "PP_O", "PK_D"])
pp_rapm["player_id"] = pp_rapm["player_id"].astype(int)
pp_prior = pp_rapm.set_index("player_id")["PP_O"].to_dict()
pk_prior = pp_rapm.set_index("player_id")["PK_D"].to_dict()


# ── 2. Compute league averages per season ────────────────────────────────────
print("\nComputing league averages...", file=sys.stderr)

pp_rows = sbg[sbg["situation"] == "5on4"].copy()
pk_rows = sbg[sbg["situation"] == "4on5"].copy()
all_rows = sbg[sbg["situation"] == "all"].copy()

league_avg = {}
for szn in sorted(sbg["season"].unique()):
    pp_s = pp_rows[pp_rows["season"] == szn]
    pk_s = pk_rows[pk_rows["season"] == szn]
    all_s = all_rows[all_rows["season"] == szn]

    # PP: (ixG + 0.7*A1) per 60 minutes of PP time
    pp_production = (pp_s["I_F_xGoals"].fillna(0) + 0.7 * pp_s["I_F_primaryAssists"].fillna(0)).sum()
    pp_minutes = pp_s["toi_min"].sum()
    avg_pp = (pp_production / pp_minutes * 60) if pp_minutes > 0 else 0.0

    # PK: on-ice xGA per 60 minutes of PK time
    pk_xga = pk_s["OnIce_A_xGoals"].fillna(0).sum()
    pk_minutes = pk_s["toi_min"].sum()
    avg_pk_xga = (pk_xga / pk_minutes * 60) if pk_minutes > 0 else 0.0

    # PEN: (drawn - taken) * 0.17 per 60 minutes all-sit
    pen_drawn = all_s["penaltiesDrawn"].fillna(0).sum()
    pen_taken = all_s["penalties"].fillna(0).sum()
    all_minutes = all_s["toi_min"].sum()
    avg_pen = ((pen_drawn - pen_taken) * GOAL_VALUE_PER_PENALTY / all_minutes * 60) if all_minutes > 0 else 0.0

    league_avg[szn] = {"PP": avg_pp, "PK_xGA": avg_pk_xga, "PEN": avg_pen}
    print(f"  {szn}: PP_rate={avg_pp:.3f}/60  PK_xGA={avg_pk_xga:.3f}/60  PEN={avg_pen:.4f}/60",
          file=sys.stderr)


# ── 3. Compute per-game raw signals ──────────────────────────────────────────
print("\nComputing per-game raw signals...", file=sys.stderr)

# 3a. EV: convert xGI per-10 to per-60
epm["EV_O_raw"] = epm["xGI_O_raw"] * EV_TO_PER60
epm["EV_D_raw"] = epm["xGI_D_raw"] * EV_TO_PER60
epm["game_date"] = pd.to_datetime(epm["game_date"])

# 3b. PP raw: per player-game from 5on4 situation
pp_game = pp_rows.groupby(["player_id", "game_id", "season"]).agg(
    toi_pp_min=("toi_min", "sum"),
    ixG_pp=("I_F_xGoals", "sum"),
    A1_pp=("I_F_primaryAssists", "sum"),
).reset_index()

pp_game["PP_production"] = pp_game["ixG_pp"].fillna(0) + 0.7 * pp_game["A1_pp"].fillna(0)
pp_game["PP_raw"] = np.where(
    pp_game["toi_pp_min"] > 0,
    pp_game["PP_production"] / (pp_game["toi_pp_min"] / 60) - pp_game["season"].map(lambda s: league_avg[s]["PP"]),
    np.nan,
)

# 3c. PK raw: per player-game from 4on5 situation
pk_game = pk_rows.groupby(["player_id", "game_id", "season"]).agg(
    toi_pk_min=("toi_min", "sum"),
    oiXGA_pk=("OnIce_A_xGoals", "sum"),
).reset_index()

pk_game["PK_raw"] = np.where(
    pk_game["toi_pk_min"] > 0,
    -(pk_game["oiXGA_pk"].fillna(0) / (pk_game["toi_pk_min"] / 60) - pk_game["season"].map(lambda s: league_avg[s]["PK_xGA"])),
    np.nan,
)

# 3d. PEN raw: per player-game from all-situations
pen_game = all_rows.groupby(["player_id", "game_id", "season"]).agg(
    toi_all_min=("toi_min", "sum"),
    pen_drawn=("penaltiesDrawn", "sum"),
    pen_taken=("penalties", "sum"),
).reset_index()

pen_game["PEN_raw"] = np.where(
    pen_game["toi_all_min"] > 0,
    (pen_game["pen_drawn"].fillna(0) - pen_game["pen_taken"].fillna(0)) * GOAL_VALUE_PER_PENALTY
    / (pen_game["toi_all_min"] / 60) - pen_game["season"].map(lambda s: league_avg[s]["PEN"]),
    np.nan,
)

# 3e. TOI from 5v5 situation
ev_toi = sbg[sbg["situation"] == "5on5"].groupby(["player_id", "game_id"]).agg(
    toi_5v5_min=("toi_min", "sum"),
).reset_index()

# 3f. Merge everything onto EV base
games = epm[["player_id", "player_name", "position", "game_id", "season",
             "game_date", "toi_min", "EV_O_raw", "EV_D_raw"]].copy()
games = games.rename(columns={"toi_min": "toi_epm_min"})

games = games.merge(ev_toi, on=["player_id", "game_id"], how="left")
games = games.merge(pp_game[["player_id", "game_id", "toi_pp_min", "PP_raw"]],
                    on=["player_id", "game_id"], how="left")
games = games.merge(pk_game[["player_id", "game_id", "toi_pk_min", "PK_raw"]],
                    on=["player_id", "game_id"], how="left")
games = games.merge(pen_game[["player_id", "game_id", "toi_all_min", "PEN_raw"]],
                    on=["player_id", "game_id"], how="left")

# Fill missing TOI
for col in ["toi_5v5_min", "toi_pp_min", "toi_pk_min", "toi_all_min"]:
    games[col] = games[col].fillna(0)

games = games.sort_values(["player_id", "game_date", "game_id"]).reset_index(drop=True)
print(f"  Unified games: {len(games):,} rows", file=sys.stderr)
print(f"  PP data: {games['PP_raw'].notna().sum():,} game-rows with PP TOI", file=sys.stderr)
print(f"  PK data: {games['PK_raw'].notna().sum():,} game-rows with PK TOI", file=sys.stderr)


# ── 4. Load priors ───────────────────────────────────────────────────────────
print("\nLoading priors...", file=sys.stderr)

# EV priors: previous season's composite × offseason decay
# Build a dict: (player_id, season) → (EV_O_prior, EV_D_prior)
ev_prior_map = {}
for _, row in composite.iterrows():
    pid = int(row["player_id"])
    next_szn = row["season"] + 1
    ev_prior_map[(pid, next_szn)] = (
        row["composite_O"] * OFFSEASON_DECAY,
        row["composite_D"] * OFFSEASON_DECAY,
    )

print(f"  EV priors: {len(ev_prior_map):,} player-season entries", file=sys.stderr)
print(f"  PP priors: {len(pp_prior):,} players", file=sys.stderr)
print(f"  PK priors: {len(pk_prior):,} players", file=sys.stderr)


# ── 5. Bayesian smoothing loop ───────────────────────────────────────────────
print("\nRunning Bayesian smoothing (5 components)...", file=sys.stderr)

decay_rate = np.log(2) / DECAY_HALFLIFE
prior_prec = {comp: 1.0 / (se ** 2) for comp, se in PRIOR_SE.items()}

COMPONENTS = ["EV_O", "EV_D", "PP", "PK", "PEN"]
RAW_COL = {"EV_O": "EV_O_raw", "EV_D": "EV_D_raw", "PP": "PP_raw", "PK": "PK_raw", "PEN": "PEN_raw"}
TOI_COL = {"EV_O": "toi_5v5_min", "EV_D": "toi_5v5_min", "PP": "toi_pp_min", "PK": "toi_pk_min", "PEN": "toi_all_min"}

player_ids = games["player_id"].unique()
smoothed = []

for pi, pid in enumerate(player_ids):
    if pi % 500 == 0 and pi > 0:
        print(f"  smoothed {pi:,} / {len(player_ids):,} players...", file=sys.stderr)

    pdf = games[games["player_id"] == pid].reset_index(drop=True)
    pname = pdf.iloc[0]["player_name"]
    pos = pdf.iloc[0]["position"]

    # Initialize per-component state
    acc_weight = {c: 0.0 for c in COMPONENTS}
    acc_sum = {c: 0.0 for c in COMPONENTS}
    current_season = None
    priors = {}

    for gi in range(len(pdf)):
        row = pdf.iloc[gi]
        szn = row["season"]

        # Season boundary: carry posteriors as new priors
        if szn != current_season:
            if current_season is not None:
                # Compute end-of-previous-season posteriors → new priors with decay
                for c in COMPONENTS:
                    tp = prior_prec[c] + acc_weight[c]
                    posterior = (prior_prec[c] * priors[c] + acc_sum[c]) / tp if tp > 0 else priors[c]
                    priors[c] = posterior * OFFSEASON_DECAY
                acc_weight = {c: 0.0 for c in COMPONENTS}
                acc_sum = {c: 0.0 for c in COMPONENTS}
            else:
                # First season for this player: use external priors
                ev_pr = ev_prior_map.get((pid, szn), (0.0, 0.0))
                priors = {
                    "EV_O": ev_pr[0],
                    "EV_D": ev_pr[1],
                    "PP":   pp_prior.get(pid, 0.0),
                    "PK":   pk_prior.get(pid, 0.0),
                    "PEN":  0.0,
                }
            current_season = szn

        # Decay based on date gap
        if gi > 0:
            date_gap = (row["game_date"] - pdf.iloc[gi - 1]["game_date"]).days
            game_gap = max(date_gap / 2.0, 1.0)
            decay = np.exp(-decay_rate * game_gap)
        else:
            decay = 1.0

        for c in COMPONENTS:
            acc_weight[c] *= decay
            acc_sum[c] *= decay

        # Update each component where we have data
        for c in COMPONENTS:
            raw_val = row[RAW_COL[c]]
            toi = row[TOI_COL[c]]
            if pd.notna(raw_val) and toi > 0:
                ev_w = GAME_EVIDENCE_SCALE * (toi / REF_TOI[c])
                acc_weight[c] += ev_w
                acc_sum[c] += ev_w * raw_val

        # Compute posteriors and SEs
        post = {}
        se = {}
        for c in COMPONENTS:
            tp = prior_prec[c] + acc_weight[c]
            post[c] = (prior_prec[c] * priors[c] + acc_sum[c]) / tp
            se[c] = np.sqrt(1.0 / tp)

        # Per-game GAR contributions
        toi_5v5 = row["toi_5v5_min"]
        toi_pp = row["toi_pp_min"]
        toi_pk = row["toi_pk_min"]
        toi_all = row["toi_all_min"]

        smoothed.append({
            "player_id": pid,
            "player_name": pname,
            "position": pos,
            "game_id": row["game_id"],
            "season": szn,
            "game_date": row["game_date"],
            "game_number": gi + 1,
            "toi_5v5": round(toi_5v5, 1),
            "toi_pp": round(toi_pp, 1),
            "toi_pk": round(toi_pk, 1),
            "toi_all": round(toi_all, 1),
            # Smoothed rates (per-60)
            "EV_O": round(post["EV_O"], 4),
            "EV_D": round(post["EV_D"], 4),
            "PP": round(post["PP"], 4),
            "PK": round(post["PK"], 4),
            "PEN": round(post["PEN"], 4),
            # SEs
            "EV_O_se": round(se["EV_O"], 4),
            "EV_D_se": round(se["EV_D"], 4),
            "PP_se": round(se["PP"], 4),
            "PK_se": round(se["PK"], 4),
            "PEN_se": round(se["PEN"], 4),
            # Per-game GAR
            "EV_O_gar": round(post["EV_O"] * toi_5v5 / 60, 4),
            "EV_D_gar": round(post["EV_D"] * toi_5v5 / 60, 4),
            "PP_gar": round(post["PP"] * toi_pp / 60, 4),
            "PK_gar": round(post["PK"] * toi_pk / 60, 4),
            "PEN_gar": round(post["PEN"] * toi_all / 60, 4),
        })

daily = pd.DataFrame(smoothed)
daily.to_csv("data/v5_daily_ratings.csv", index=False)
print(f"\n  {len(daily):,} rows → data/v5_daily_ratings.csv", file=sys.stderr)


# ── 6. Season aggregation ────────────────────────────────────────────────────
print("\nAggregating season WAR...", file=sys.stderr)

# Sum per-game GARs by player-season
season_gar = daily.groupby(["player_id", "player_name", "position", "season"]).agg(
    GP=("game_id", "size"),
    toi_5v5=("toi_5v5", "sum"),
    toi_pp=("toi_pp", "sum"),
    toi_pk=("toi_pk", "sum"),
    toi_all=("toi_all", "sum"),
    EV_O_GAR=("EV_O_gar", "sum"),
    EV_D_GAR=("EV_D_gar", "sum"),
    PP_GAR=("PP_gar", "sum"),
    PK_GAR=("PK_gar", "sum"),
    PEN_GAR=("PEN_gar", "sum"),
).reset_index()

# End-of-season ratings (last game)
last_game = (
    daily.sort_values(["player_id", "season", "game_date"])
    .groupby(["player_id", "season"])[["EV_O", "EV_D", "PP", "PK", "PEN"]]
    .last()
    .reset_index()
    .rename(columns={"EV_O": "EV_O_rate", "EV_D": "EV_D_rate",
                      "PP": "PP_rate", "PK": "PK_rate", "PEN": "PEN_rate"})
)

war = season_gar.merge(last_game, on=["player_id", "season"], how="left")

# Round GARs
for c in ["EV_O_GAR", "EV_D_GAR", "PP_GAR", "PK_GAR", "PEN_GAR"]:
    war[c] = war[c].round(2)

# Composite GAR
war["GAR_O"] = (war["EV_O_GAR"] + war["PP_GAR"] + 0.5 * war["PEN_GAR"]).round(2)
war["GAR_D"] = (war["EV_D_GAR"] + war["PK_GAR"] + 0.5 * war["PEN_GAR"]).round(2)

# Position-specific replacement level
total_toi = (war["toi_5v5"] + war["toi_pp"] + war["toi_pk"]).values
qualified = war["toi_5v5"] >= 100

is_fwd = war["position"].isin(["C", "L", "R"])
is_def = war["position"] == "D"

for pos_label, pos_mask in [("F", is_fwd), ("D", is_def)]:
    q = qualified & pos_mask
    if q.sum() == 0:
        continue
    per60_O = war.loc[q, "GAR_O"].values / (total_toi[q.values] / 60)
    per60_D = war.loc[q, "GAR_D"].values / (total_toi[q.values] / 60)
    rl_O = float(np.percentile(per60_O, RL_PERCENTILE))
    rl_D = float(np.percentile(per60_D, RL_PERCENTILE))
    print(f"  Replacement level {pos_label}: O={rl_O:.4f}, D={rl_D:.4f} per 60", file=sys.stderr)

    war.loc[pos_mask, "WAR_O"] = (
        (war.loc[pos_mask, "GAR_O"] - rl_O * total_toi[pos_mask.values] / 60) / GOALS_TO_WINS
    ).round(2)
    war.loc[pos_mask, "WAR_D"] = (
        (war.loc[pos_mask, "GAR_D"] - rl_D * total_toi[pos_mask.values] / 60) / GOALS_TO_WINS
    ).round(2)

war["WAR"] = (war["WAR_O"].fillna(0) + war["WAR_D"].fillna(0)).round(2)
war["WAR_82"] = (war["WAR"] * 82 / war["GP"].clip(lower=1)).round(2)
war["WAR_O_82"] = (war["WAR_O"] * 82 / war["GP"].clip(lower=1)).round(2)
war["WAR_D_82"] = (war["WAR_D"] * 82 / war["GP"].clip(lower=1)).round(2)

# Round rates
for c in ["EV_O_rate", "EV_D_rate", "PP_rate", "PK_rate", "PEN_rate"]:
    if c in war.columns:
        war[c] = war[c].round(4)
for c in ["toi_5v5", "toi_pp", "toi_pk", "toi_all"]:
    war[c] = war[c].round(1)


# ── 7. Output + Leaderboards ─────────────────────────────────────────────────
print("\n── Saving outputs ──", file=sys.stderr)

war_cols = [
    "player_id", "player_name", "position", "season", "GP",
    "toi_5v5", "toi_pp", "toi_pk",
    "EV_O_rate", "EV_D_rate", "PP_rate", "PK_rate", "PEN_rate",
    "EV_O_GAR", "EV_D_GAR", "PP_GAR", "PK_GAR", "PEN_GAR",
    "GAR_O", "GAR_D", "WAR_O", "WAR_D", "WAR",
    "WAR_O_82", "WAR_D_82", "WAR_82",
]
war_cols = [c for c in war_cols if c in war.columns]
war_out = war[war_cols].sort_values(["season", "WAR"], ascending=[True, False])
war_out.to_csv("data/v5_daily_war.csv", index=False)
print(f"  {len(war_out):,} player-seasons → data/v5_daily_war.csv", file=sys.stderr)


# ── Leaderboards ──
print("\n── Leaderboards ──", file=sys.stderr)

key_players = [
    "Connor McDavid", "Nikita Kucherov", "Auston Matthews", "Nathan MacKinnon",
    "Kirill Kaprizov", "Cale Makar", "Leon Draisaitl", "David Pastrnak",
    "Matthew Tkachuk", "Sidney Crosby", "Adam Fox", "Charlie McAvoy",
    "Mikko Rantanen", "Jack Hughes", "Miro Heiskanen", "Aleksander Barkov",
    "Sam Reinhart", "Mark Stone", "Brady Tkachuk", "Jack Quinn",
]

for szn in sorted(war_out["season"].unique())[-2:]:
    s = war_out[war_out["season"] == szn].copy()
    nhl = f"{szn}-{str(szn+1)[-2:]}"

    print(f"\n{'='*130}", file=sys.stderr)
    print(f"  Top 25 WAR/82 ({nhl})", file=sys.stderr)
    print(f"{'='*130}", file=sys.stderr)
    s = s.sort_values("WAR_82", ascending=False).reset_index(drop=True)
    s["rank"] = s.index + 1

    show_cols = ["rank", "player_name", "position", "GP",
                 "EV_O_GAR", "EV_D_GAR", "PP_GAR", "PK_GAR", "PEN_GAR",
                 "WAR_O", "WAR_D", "WAR", "WAR_82"]
    show_cols = [c for c in show_cols if c in s.columns]
    print(s.head(25)[show_cols].to_string(index=False), file=sys.stderr)

    # Key player breakdown
    print(f"\n  Key Players ({nhl}) — 5-component breakdown:", file=sys.stderr)
    print(f"  {'Rk':>3s} {'Name':25s} {'Pos':3s} {'GP':>3s} "
          f"{'EV_O':>6s} {'EV_D':>6s} {'PP':>6s} {'PK':>6s} {'PEN':>6s} "
          f"{'WAR_O':>6s} {'WAR_D':>6s} {'WAR':>5s} {'W/82':>5s}",
          file=sys.stderr)
    print(f"  {'-'*110}", file=sys.stderr)
    for name in key_players:
        row = s[s["player_name"] == name]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        print(f"  {r['rank']:3.0f} {name:25s} {r['position']:3s} {r['GP']:3.0f} "
              f"{r['EV_O_GAR']:+6.2f} {r['EV_D_GAR']:+6.2f} {r['PP_GAR']:+6.2f} "
              f"{r['PK_GAR']:+6.2f} {r['PEN_GAR']:+6.2f} "
              f"{r['WAR_O']:+6.2f} {r['WAR_D']:+6.2f} {r['WAR']:5.2f} {r['WAR_82']:5.2f}",
              file=sys.stderr)

# Verify small sample regression: Jack Quinn
print("\n── Small sample check ──", file=sys.stderr)
latest_szn = war_out["season"].max()
latest = war_out[war_out["season"] == latest_szn].sort_values("WAR", ascending=False).reset_index(drop=True)
latest["rank"] = latest.index + 1
quinn = latest[latest["player_name"] == "Jack Quinn"]
if len(quinn):
    q = quinn.iloc[0]
    print(f"  Jack Quinn: rank={q['rank']:.0f}, GP={q['GP']:.0f}, WAR={q['WAR']:.2f}, WAR_82={q['WAR_82']:.2f}", file=sys.stderr)
    print(f"    EV_O_GAR={q['EV_O_GAR']:.2f}  EV_D_GAR={q['EV_D_GAR']:.2f}  PP_GAR={q['PP_GAR']:.2f}  "
          f"PK_GAR={q['PK_GAR']:.2f}  PEN_GAR={q['PEN_GAR']:.2f}", file=sys.stderr)

print("\nDone.", file=sys.stderr)
