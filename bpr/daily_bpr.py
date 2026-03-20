"""
daily_bpr.py — Game-level BPR with Bayesian smoothing (DARKO-style Daily Plus Minus).

Uses per-game stats from MoneyPuck's skaters_by_game.csv (individual + on-ice),
converts to a game-level BPR observation, then applies Bayesian smoothing with
the career RAPM as prior.

Game-level BPR = blend of:
  - On-ice impact: (OnIce_F - OnIce_A) differentials for xGF, GF, SOG, TO, GA
  - Individual production: I_F stats (xGoals, goals, shots, takeaways, giveaways)

The blend ensures elite producers (Kucherov, McDavid) get credit for individual
output while still capturing defensive/suppression value from on-ice differentials.

Bayesian smoothing:
  posterior_t = (prior_precision * prior + evidence_precision * evidence) / total
  - Prior = career RAPM (pooled BPR_O, BPR_D)
  - Evidence = exponentially-decayed weighted average of game observations
  - After a full season (~82 games), evidence carries ~60-70% weight

Inputs:
  data/skaters_by_game.csv           Per-game individual + on-ice stats
  data/v2_final_ratings.csv          Pooled career RAPM (prior)
  data/pp_rapm.csv                   PP_O / PK_D ratings

Outputs:
  data/v3_daily_bpr.csv              Per-player-game smoothed BPR
  data/v3_season_war.csv             Season-aggregated WAR from daily ratings
"""

import sys
import numpy as np
import pandas as pd

# ── Config ───────────────────────────────────────────────────────────────────
# BPR weights (must match rapm_v2.py / gar.py)
W_xGF = 0.50
W_GF  = 0.15
W_SOG = 0.22
W_TO  = 0.06
W_GA  = -0.04

# Blend: how much weight to give individual stats vs on-ice differentials
# Higher = more individual production, lower = more on-ice context
INDIVIDUAL_WEIGHT = 0.40
ONICE_WEIGHT = 0.60

# Smoothing parameters
DECAY_HALFLIFE = 25        # games — half-life for exponential decay
GAME_EVIDENCE_SCALE = 1.00 # precision per game (at avg TOI)
                           # With prior_se=0.35 → precision=8.2, a full season
                           # (~82 games) gives ~82 precision → 91% data / 9% prior
                           # This lets a single strong season clearly show through
PRIOR_SE_FLOOR = 0.35      # min prior SE — allows season variation
DEFAULT_PRIOR_SE = 0.35    # prior SE for players not in pooled RAPM

# GAR conversion
GOALS_TO_WINS = 6.0
RL_PERCENTILE = 17

# ── 1. Load data ─────────────────────────────────────────────────────────────
print("Loading data...", file=sys.stderr)

# Career RAPM (prior)
career = pd.read_csv("output/v2_final_ratings.csv")
career["player_id"] = career["player_id"].astype(int)
career_prior = career.set_index("player_id")[
    ["BPR_O", "BPR_D", "BPR_O_se", "BPR_D_se", "player_name", "position"]
].to_dict("index")
print(f"  Career priors: {len(career_prior):,} players", file=sys.stderr)

# Skaters by game — load 5v5 situation with relevant columns
sbg_cols = [
    "playerId", "name", "gameId", "season", "gameDate", "position", "situation",
    "icetime",
    # On-ice for/against
    "OnIce_F_xGoals", "OnIce_A_xGoals",
    "OnIce_F_goals", "OnIce_A_goals",
    "OnIce_F_shotsOnGoal", "OnIce_A_shotsOnGoal",
    # Individual
    "I_F_xGoals", "I_F_goals", "I_F_shotsOnGoal",
    "I_F_primaryAssists", "I_F_secondaryAssists",
    "I_F_xOnGoal",
    # Turnovers
    "I_F_takeaways", "I_F_giveaways",
    # Penalties
    "penalties", "penaltiesDrawn",
    "faceoffsWon", "faceoffsLost",
]

print("Loading skaters_by_game.csv...", file=sys.stderr)
sbg = pd.read_csv("data/skaters_by_game.csv", usecols=sbg_cols)
sbg = sbg.rename(columns={"playerId": "player_id", "gameId": "game_id", "name": "player_name"})

# Filter to 5v5 and seasons in our RAPM data (2014+)
sbg_5v5 = sbg[
    (sbg["situation"] == "5on5") &
    (sbg["season"] >= 2014)
].copy()
sbg_5v5["game_date"] = pd.to_datetime(sbg_5v5["gameDate"].astype(str), format="%Y%m%d")
sbg_5v5["toi_min"] = sbg_5v5["icetime"] / 60.0  # seconds → minutes

print(f"  5v5 player-games: {len(sbg_5v5):,}", file=sys.stderr)
print(f"  Seasons: {sorted(sbg_5v5['season'].unique())}", file=sys.stderr)


# ── 2. Compute per-game raw BPR ─────────────────────────────────────────────
print("\nComputing per-game raw BPR...", file=sys.stderr)

# On-ice differential BPR (per 60 min)
toi_safe = sbg_5v5["toi_min"].clip(lower=1.0).values
scale60 = 60.0 / toi_safe

# On-ice differentials (per 60)
oi_xgf_diff = (sbg_5v5["OnIce_F_xGoals"].fillna(0) - sbg_5v5["OnIce_A_xGoals"].fillna(0)).values * scale60
oi_gf_diff  = (sbg_5v5["OnIce_F_goals"].fillna(0) - sbg_5v5["OnIce_A_goals"].fillna(0)).values * scale60
oi_sog_diff = (sbg_5v5["OnIce_F_shotsOnGoal"].fillna(0) - sbg_5v5["OnIce_A_shotsOnGoal"].fillna(0)).values * scale60

# On-ice takeaway/giveaway diffs aren't directly available, use individual
oi_to_diff = sbg_5v5["I_F_takeaways"].fillna(0).values * scale60  # individual takeaways as proxy
oi_ga_diff = -sbg_5v5["I_F_giveaways"].fillna(0).values * scale60  # giveaways are negative

# On-ice BPR: use differentials
onice_bpr = (
    oi_xgf_diff * W_xGF +
    oi_gf_diff  * W_GF  +
    oi_sog_diff * W_SOG +
    oi_to_diff  * W_TO  +
    oi_ga_diff  * W_GA
)

# Individual BPR: based on individual production per 60
# For offense: individual xGoals, goals, shots (all positive contributions)
ind_xgf = sbg_5v5["I_F_xGoals"].fillna(0).values * scale60
ind_gf  = sbg_5v5["I_F_goals"].fillna(0).values * scale60
ind_sog = sbg_5v5["I_F_shotsOnGoal"].fillna(0).values * scale60
ind_to  = sbg_5v5["I_F_takeaways"].fillna(0).values * scale60
ind_ga  = -sbg_5v5["I_F_giveaways"].fillna(0).values * scale60  # giveaways hurt

# Individual stats are all "for" — we need to center them around league average
# so that average players get ~0 BPR, not positive
# League average per 60: compute from data
lg_avg_xgf = sbg_5v5["I_F_xGoals"].fillna(0).sum() / (sbg_5v5["toi_min"].sum() / 60)
lg_avg_gf  = sbg_5v5["I_F_goals"].fillna(0).sum() / (sbg_5v5["toi_min"].sum() / 60)
lg_avg_sog = sbg_5v5["I_F_shotsOnGoal"].fillna(0).sum() / (sbg_5v5["toi_min"].sum() / 60)
lg_avg_to  = sbg_5v5["I_F_takeaways"].fillna(0).sum() / (sbg_5v5["toi_min"].sum() / 60)
lg_avg_ga  = sbg_5v5["I_F_giveaways"].fillna(0).sum() / (sbg_5v5["toi_min"].sum() / 60)

print(f"  League avg per 60: xGF={lg_avg_xgf:.3f} GF={lg_avg_gf:.3f} SOG={lg_avg_sog:.3f} TO={lg_avg_to:.3f} GA={lg_avg_ga:.3f}", file=sys.stderr)

ind_bpr = (
    (ind_xgf - lg_avg_xgf) * W_xGF +
    (ind_gf  - lg_avg_gf)  * W_GF  +
    (ind_sog - lg_avg_sog)  * W_SOG +
    (ind_to  - lg_avg_to)   * W_TO  +
    (-sbg_5v5["I_F_giveaways"].fillna(0).values * scale60 + lg_avg_ga) * W_GA
)

# Blended raw BPR
raw_bpr = ONICE_WEIGHT * onice_bpr + INDIVIDUAL_WEIGHT * ind_bpr

# Split into O/D: offense = individual production above avg, defense = on-ice suppression
# This gives offensive players credit for production, defensive players for suppression
raw_bpr_o = (
    INDIVIDUAL_WEIGHT * (
        (ind_xgf - lg_avg_xgf) * W_xGF +
        (ind_gf  - lg_avg_gf)  * W_GF  +
        (ind_sog - lg_avg_sog)  * W_SOG
    ) +
    ONICE_WEIGHT * (
        oi_xgf_diff * W_xGF * 0.5 +
        oi_gf_diff  * W_GF  * 0.5 +
        oi_sog_diff * W_SOG * 0.5
    )
)

raw_bpr_d = raw_bpr - raw_bpr_o

sbg_5v5["bpr_o_raw"] = np.round(raw_bpr_o, 4)
sbg_5v5["bpr_d_raw"] = np.round(raw_bpr_d, 4)
sbg_5v5["bpr_raw"] = np.round(raw_bpr, 4)

print(f"  Raw BPR: mean={raw_bpr.mean():.4f}, std={raw_bpr.std():.4f}", file=sys.stderr)
print(f"  Raw BPR O: mean={raw_bpr_o.mean():.4f}, std={raw_bpr_o.std():.4f}", file=sys.stderr)


# ── 3. Bayesian smoothing per player ────────────────────────────────────────
print("\nApplying Bayesian smoothing...", file=sys.stderr)

decay_rate = np.log(2) / DECAY_HALFLIFE

# Sort by player and date
sbg_5v5 = sbg_5v5.sort_values(["player_id", "game_date", "game_id"]).reset_index(drop=True)

smoothed_records = []
player_ids = sbg_5v5["player_id"].unique()

for pi, pid in enumerate(player_ids):
    if pi % 500 == 0 and pi > 0:
        print(f"  smoothed {pi:,} / {len(player_ids):,} players...", file=sys.stderr)

    pdf = sbg_5v5[sbg_5v5["player_id"] == pid].reset_index(drop=True)

    # Get career prior
    if pid in career_prior:
        cp = career_prior[pid]
        prior_o = cp["BPR_O"]
        prior_d = cp["BPR_D"]
        prior_se_o = max(cp["BPR_O_se"] if not np.isnan(cp["BPR_O_se"]) else DEFAULT_PRIOR_SE, PRIOR_SE_FLOOR)
        prior_se_d = max(cp["BPR_D_se"] if not np.isnan(cp["BPR_D_se"]) else DEFAULT_PRIOR_SE, PRIOR_SE_FLOOR)
        pname = cp["player_name"]
        pos = cp["position"]
    else:
        prior_o = 0.0
        prior_d = 0.0
        prior_se_o = DEFAULT_PRIOR_SE
        prior_se_d = DEFAULT_PRIOR_SE
        pname = pdf.iloc[0]["player_name"] if len(pdf) > 0 else "?"
        pos = pdf.iloc[0]["position"] if len(pdf) > 0 else "?"

    prior_prec_o = 1.0 / (prior_se_o ** 2)
    prior_prec_d = 1.0 / (prior_se_d ** 2)

    n_games = len(pdf)
    sum_w = 0.0
    sum_wo = 0.0
    sum_wd = 0.0

    for gi in range(n_games):
        row = pdf.iloc[gi]

        # Decay existing evidence
        if gi > 0:
            date_gap = (row["game_date"] - pdf.iloc[gi - 1]["game_date"]).days
            game_gap = max(date_gap / 2.0, 1.0)
            decay_factor = np.exp(-decay_rate * game_gap)
        else:
            decay_factor = 1.0

        sum_w *= decay_factor
        sum_wo *= decay_factor
        sum_wd *= decay_factor

        # Evidence weight: proportional to TOI (more ice time = more reliable)
        toi = row["toi_min"]
        evidence_w = GAME_EVIDENCE_SCALE * (toi / 15.0)  # 15 min = typical 5v5 TOI per game

        sum_w += evidence_w
        sum_wo += evidence_w * row["bpr_o_raw"]
        sum_wd += evidence_w * row["bpr_d_raw"]

        # Posterior
        total_prec_o = prior_prec_o + sum_w
        total_prec_d = prior_prec_d + sum_w

        post_o = (prior_prec_o * prior_o + sum_wo) / total_prec_o
        post_d = (prior_prec_d * prior_d + sum_wd) / total_prec_d
        post_bpr = post_o + post_d

        post_se_o = np.sqrt(1.0 / total_prec_o)
        post_se_d = np.sqrt(1.0 / total_prec_d)
        post_se = np.sqrt(post_se_o ** 2 + post_se_d ** 2)

        smoothed_records.append({
            "player_id": pid,
            "player_name": pname,
            "position": pos,
            "game_id": row["game_id"],
            "season": row["season"],
            "game_date": row["game_date"],
            "game_number": gi + 1,
            "toi_min": round(toi, 1),
            "bpr_o_raw": row["bpr_o_raw"],
            "bpr_d_raw": row["bpr_d_raw"],
            "bpr_raw": row["bpr_raw"],
            "BPR_O": round(post_o, 4),
            "BPR_D": round(post_d, 4),
            "BPR": round(post_bpr, 4),
            "BPR_se": round(post_se, 4),
        })

print(f"  {len(smoothed_records):,} smoothed records", file=sys.stderr)

daily = pd.DataFrame(smoothed_records)
daily.to_csv("output/v3_daily_bpr.csv", index=False)
print(f"\nWrote {len(daily):,} rows to output/v3_daily_bpr.csv", file=sys.stderr)


# ── 4. Season-aggregated WAR ────────────────────────────────────────────────
print("\nComputing season-aggregated WAR...", file=sys.stderr)

# For each player-season, take their LAST game's smoothed BPR as the season rating
season_last = (
    daily.sort_values(["player_id", "season", "game_date"])
    .groupby(["player_id", "season"])
    .last()
    .reset_index()
)

# Per-season TOI from skaters_by_game
sit = pd.read_csv("data/skaters_by_game.csv", usecols=["playerId", "season", "situation", "icetime"])
sit = sit.rename(columns={"playerId": "player_id"})
sit_toi = (
    sit[sit["situation"].isin(["5on5", "5on4", "4on5"])]
    .groupby(["player_id", "season", "situation"])["icetime"]
    .sum()
    .unstack("situation", fill_value=0)
    .reset_index()
)
sit_toi.columns.name = None
sit_toi = sit_toi.rename(columns={"5on5": "toi_5v5", "5on4": "toi_pp", "4on5": "toi_pk"})
for col in ["toi_5v5", "toi_pp", "toi_pk"]:
    if col in sit_toi.columns:
        sit_toi[col] = (sit_toi[col] / 60).round(1)
sit_toi["player_id"] = sit_toi["player_id"].astype(int)

# PP/PK ratings
pp_rapm = pd.read_csv("output/pp_rapm.csv")[["player_id", "PP_O", "PK_D"]]
pp_rapm["player_id"] = pp_rapm["player_id"].astype(int)

season_war = season_last.merge(sit_toi, on=["player_id", "season"], how="left")
season_war = season_war.merge(pp_rapm, on="player_id", how="left")

toi_5v5 = season_war["toi_5v5"].fillna(0).values
toi_pp = season_war["toi_pp"].fillna(0).values
toi_pk = season_war["toi_pk"].fillna(0).values

# EV GAR
season_war["EV_O_GAR"] = (season_war["BPR_O"] * toi_5v5 / 60).round(2)
season_war["EV_D_GAR"] = (season_war["BPR_D"] * toi_5v5 / 60).round(2)
season_war["EV_GAR"] = (season_war["BPR"] * toi_5v5 / 60).round(2)

# PP/PK GAR
season_war["PP_GAR"] = (season_war["PP_O"].fillna(0) * toi_pp / 60).round(2)
season_war["PK_GAR"] = (season_war["PK_D"].fillna(0) * toi_pk / 60).round(2)

season_war["GAR_above_avg"] = (
    season_war["EV_GAR"] + season_war["PP_GAR"] + season_war["PK_GAR"]
).round(2)

# Replacement level
total_toi = toi_5v5 + toi_pp + toi_pk
min_toi = 100
qualified = toi_5v5 >= min_toi
n_qual = qualified.sum()

if n_qual > 0:
    safe_toi = np.where(total_toi > 0, total_toi, 1.0)
    per60 = season_war.loc[qualified, "GAR_above_avg"].values / (total_toi[qualified] / 60)
    rl_per_60 = float(np.percentile(per60, RL_PERCENTILE))
    print(f"  Replacement level: {rl_per_60:.4f} per 60 ({n_qual} qualified)", file=sys.stderr)
else:
    rl_per_60 = -0.10

season_war["GAR"] = (season_war["GAR_above_avg"] - rl_per_60 * total_toi / 60).round(2)
season_war["WAR"] = (season_war["GAR"] / GOALS_TO_WINS).round(2)

# Uncertainty
season_war["GAR_se"] = (season_war["BPR_se"] * toi_5v5 / 60).round(2)
season_war["WAR_se"] = (season_war["GAR_se"] / GOALS_TO_WINS).round(2)

# Games played per season
gp = daily.groupby(["player_id", "season"]).size().reset_index(name="GP")
season_war = season_war.merge(gp, on=["player_id", "season"], how="left")

# Select output
out_cols = [
    "player_id", "player_name", "position", "season", "GP",
    "toi_5v5", "toi_pp", "toi_pk",
    "BPR_O", "BPR_D", "BPR", "BPR_se",
    "EV_O_GAR", "EV_D_GAR", "EV_GAR", "PP_GAR", "PK_GAR",
    "GAR", "WAR", "GAR_se", "WAR_se",
]
out_cols = [c for c in out_cols if c in season_war.columns]
season_out = season_war[out_cols].sort_values(["season", "WAR"], ascending=[True, False])
season_out.to_csv("output/v3_season_war.csv", index=False)
print(f"\nWrote {len(season_out):,} player-seasons to output/v3_season_war.csv", file=sys.stderr)

# Print leaderboards for last two complete seasons
for szn in [2021, 2022]:
    s = season_out[season_out["season"] == szn].sort_values("WAR", ascending=False)
    nhl_str = f"{szn}-{str(szn+1)[-2:]}"
    print(f"\nTop 20 WAR ({nhl_str}):", file=sys.stderr)
    show = ["player_name", "position", "GP", "BPR_O", "BPR_D", "BPR", "EV_GAR", "PP_GAR", "PK_GAR", "GAR", "WAR"]
    show = [c for c in show if c in s.columns]
    print(s.head(20)[show].to_string(index=False), file=sys.stderr)

print("\nDone.", file=sys.stderr)
