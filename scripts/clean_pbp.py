import pandas as pd
import sys

# ── 1. Load raw PBP, keep only what we need ───────────────────────────────────
KEEP_EVENTS = {"SHOT", "GOAL", "MISSED_SHOT", "BLOCKED_SHOT", "GIVEAWAY", "TAKEAWAY"}

pbp_cols = [
    "game_id", "season", "event_id", "event_type", "event_team", "event_team_type",
    "period", "period_seconds", "home_score", "away_score",
    "strength_state", "event_player_1_id", "x", "y",
]

print("Loading raw_pbp.csv...", file=sys.stderr)
pbp = pd.read_csv(
    "data/raw_pbp.csv",
    usecols=pbp_cols,
    dtype={"game_id": "int64", "event_id": "int64", "event_player_1_id": "Int64"},
)
pbp = pbp[pbp["event_type"].isin(KEEP_EVENTS)].copy()
print(f"  {len(pbp):,} rows after filtering to RAPM events", file=sys.stderr)

# ── 2. Merge on-ice lineups via as-of join ────────────────────────────────────
# rapm_dataset contains lineup state at CHANGE events; we need the last change
# before each shot/goal/giveaway/takeaway within the same game.
print("Loading rapm_dataset.csv...", file=sys.stderr)
lineups = pd.read_csv(
    "data/rapm_dataset.csv",
    dtype={"event_id": "int64"},
)
# Derive game_id: event_id format is {game_id}{event_num_4digits}
lineups["game_id"] = lineups["event_id"] // 10000

# merge_asof requires both sorted by the key
pbp = pbp.sort_values(["game_id", "event_id"]).reset_index(drop=True)
lineups = lineups.sort_values("event_id").reset_index(drop=True)

pbp = pd.merge_asof(
    pbp,
    lineups,
    on="event_id",
    by="game_id",
    direction="backward",
)
print(f"  {pbp['home_on_ice'].notna().sum():,} rows with lineup data", file=sys.stderr)

# ── 3. Load MoneyPuck shots for xG ────────────────────────────────────────────
print("Loading shots_2007-2024.csv...", file=sys.stderr)
shot_cols = [
    "season", "game_id", "period", "time",
    "shooterPlayerId", "xGoal", "shotWasOnGoal", "shotType", "shotDistance",
    "shotOnEmptyNet",
]
mp = pd.read_csv("data/shots_2007-2024.csv", usecols=shot_cols)

# Reconstruct full NHL game_id:
# MoneyPuck uses start-year labeling (season=2023 = 2023-24 season)
# PBP uses end-year labeling (season=2024 = 2023-24 season)
# MoneyPuck game_id=20074 + season=2023 → 2023020074 (PBP game_id format)
mp["full_game_id"] = (
    mp["season"].astype(str) + mp["game_id"].apply(lambda x: f"{int(x):06d}")
).astype("int64")
mp["shooterPlayerId"] = mp["shooterPlayerId"].astype("Int64")
# MoneyPuck `time` is cumulative game seconds; convert to per-period seconds
# Period 1: time = period_seconds; Period 2: period_seconds = time - 1200; etc.
mp["period_seconds_calc"] = mp["time"] - (mp["period"] - 1) * 1200

# Rename to match pbp column names for joining
mp = mp.rename(columns={
    "full_game_id": "_game_id",
    "period_seconds_calc": "_period_seconds",
    "shooterPlayerId": "_player_id",
}).drop(columns=["season", "game_id", "time"])

# Filter pbp to shot-type events for xG join
shot_mask = pbp["event_type"].isin({"SHOT", "GOAL", "MISSED_SHOT", "BLOCKED_SHOT"})
pbp_shots = pbp[shot_mask].copy()

xg = pbp_shots.merge(
    mp.rename(columns={
        "_game_id": "game_id",
        "_period_seconds": "period_seconds",
        "_player_id": "event_player_1_id",
    }),
    on=["game_id", "period", "period_seconds", "event_player_1_id"],
    how="left",
)

matched = xg["xGoal"].notna().sum()
print(f"  xGoal matched (with player): {matched:,} / {len(xg):,}", file=sys.stderr)

# Fallback: join on game+period+time only for unblocked shots
# (BLOCKED_SHOT events use the blocker as player_1, not the shooter; MoneyPuck
#  only tracks unblocked attempts so blocked shots will always have null xGoal)
unmatched_mask = xg["xGoal"].isna() & (xg["event_type"] != "BLOCKED_SHOT")
if unmatched_mask.sum() > 0:
    mp_noplay = mp.rename(columns={
        "_game_id": "game_id",
        "_period_seconds": "period_seconds",
    }).drop(columns=["_player_id"])
    # Drop duplicate rows to avoid fan-out on fallback
    mp_noplay = mp_noplay.drop_duplicates(subset=["game_id", "period", "period_seconds"])
    fallback = xg.loc[unmatched_mask, ["game_id", "period", "period_seconds"]].merge(
        mp_noplay,
        on=["game_id", "period", "period_seconds"],
        how="left",
    )
    for col in ["xGoal", "shotWasOnGoal", "shotType", "shotDistance", "shotOnEmptyNet"]:
        xg.loc[unmatched_mask, col] = fallback[col].values
    print(f"  xGoal matched (fallback):   {xg['xGoal'].notna().sum():,} / {len(xg):,}", file=sys.stderr)

# Bring xG columns back into the full pbp dataframe
xg_cols = ["xGoal", "shotWasOnGoal", "shotType", "shotDistance", "shotOnEmptyNet"]
for col in xg_cols:
    pbp[col] = pd.NA

pbp = pbp.set_index("event_id")
xg_sub = xg.set_index("event_id")[xg_cols]
# Deduplicate in case of any duplicate event_ids
xg_sub = xg_sub[~xg_sub.index.duplicated(keep="first")]
pbp.loc[xg_sub.index, xg_cols] = xg_sub
pbp = pbp.reset_index()

# ── 4. Derive indicator columns ───────────────────────────────────────────────
pbp["is_goal"]         = pbp["event_type"] == "GOAL"
pbp["is_shot_on_goal"] = pbp["shotWasOnGoal"] == 1
pbp["is_giveaway"]     = pbp["event_type"] == "GIVEAWAY"
pbp["is_takeaway"]     = pbp["event_type"] == "TAKEAWAY"
pbp["is_5v5"]          = pbp["strength"] == "5v5"
pbp["score_diff"]      = pbp["home_score"] - pbp["away_score"]

# ── 5. Select final columns and write output ──────────────────────────────────
out_cols = [
    "game_id", "season", "event_id", "event_type", "event_team", "event_team_type",
    "period", "period_seconds", "score_diff",
    "home_on_ice", "away_on_ice", "strength", "is_5v5",
    "is_goal", "is_shot_on_goal", "xGoal",
    "is_giveaway", "is_takeaway",
    "shotType", "shotDistance", "shotOnEmptyNet",
    "x", "y",
]
out = pbp[out_cols]
out.to_csv("data/clean_pbp.csv", index=False)

print(f"\nWrote {len(out):,} rows to data/clean_pbp.csv", file=sys.stderr)
print(f"  5v5 rows:      {out['is_5v5'].sum():,}", file=sys.stderr)
print(f"  goals:         {out['is_goal'].sum():,}", file=sys.stderr)
print(f"  shots on goal: {out['is_shot_on_goal'].sum():,}", file=sys.stderr)
print(f"  xGoal matched: {out['xGoal'].notna().sum():,}", file=sys.stderr)
print(f"  giveaways:     {out['is_giveaway'].sum():,}", file=sys.stderr)
print(f"  takeaways:     {out['is_takeaway'].sum():,}", file=sys.stderr)
