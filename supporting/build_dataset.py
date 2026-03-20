"""
build_dataset.py — Enriched PBP dataset for RAPM v2.

Extends v1's clean_pbp.py with:
  - Zone start classification (OZ/DZ/NZ from faceoff x-coordinates)
  - Goalie IDs per event (home_goalie_id, away_goalie_id)
  - Period context (carried through)
  - Rest differential (from MoneyPuck shots file)

Outputs:
  output/v2_clean_pbp.csv     Enriched event-level data
  output/v2_penalties.csv     Per-penalty-event data (player, severity, drawn_by)
"""

import sys
import numpy as np
import pandas as pd

# ── 1. Load raw PBP ──────────────────────────────────────────────────────────
# Keep more event types than v1: add FACEOFF for zone start tracking
KEEP_EVENTS = {"SHOT", "GOAL", "MISSED_SHOT", "BLOCKED_SHOT", "GIVEAWAY", "TAKEAWAY"}
FACEOFF_EVENTS = {"FACEOFF"}
PENALTY_EVENTS = {"PENALTY"}
ALL_NEEDED = KEEP_EVENTS | FACEOFF_EVENTS | PENALTY_EVENTS

pbp_cols = [
    "game_id", "season", "event_id", "event_type", "event_team", "event_team_type",
    "period", "period_seconds", "home_score", "away_score",
    "strength_state", "strength",
    "event_player_1_id", "event_player_1_name",
    "event_player_2_id",
    "x", "y", "x_fixed", "y_fixed",
    "home_goalie", "away_goalie",
    "penalty_severity", "penalty_minutes",
]

print("Loading raw_pbp.csv...", file=sys.stderr)
pbp = pd.read_csv(
    "data/raw_pbp.csv",
    usecols=pbp_cols,
    dtype={"game_id": "int64", "event_id": "int64",
           "event_player_1_id": "Int64", "event_player_2_id": "Int64"},
)
pbp = pbp[pbp["event_type"].isin(ALL_NEEDED)].copy()
print(f"  {len(pbp):,} rows after filtering", file=sys.stderr)

# ── 2. Build goalie name → ID mapping ────────────────────────────────────────
print("Building goalie name→ID mapping...", file=sys.stderr)
raw_goalie = pd.read_csv(
    "data/raw_pbp.csv",
    usecols=["event_goalie_name", "event_goalie_id"],
)
raw_goalie = raw_goalie.dropna().drop_duplicates()
raw_goalie["event_goalie_id"] = raw_goalie["event_goalie_id"].astype(int)
goalie_name_to_id = (
    raw_goalie.groupby("event_goalie_name")["event_goalie_id"]
    .first().to_dict()
)
print(f"  {len(goalie_name_to_id)} goalie name→ID mappings", file=sys.stderr)

# Map home_goalie and away_goalie names to IDs
pbp["home_goalie_id"] = pbp["home_goalie"].map(goalie_name_to_id)
pbp["away_goalie_id"] = pbp["away_goalie"].map(goalie_name_to_id)

goalie_mapped = pbp["home_goalie_id"].notna().sum()
print(f"  home_goalie_id mapped: {goalie_mapped:,} / {len(pbp):,}", file=sys.stderr)

# ── 3. Extract penalty data ──────────────────────────────────────────────────
print("Extracting penalty data...", file=sys.stderr)
pen = pbp[pbp["event_type"] == "PENALTY"].copy()
pen_out = pd.DataFrame({
    "game_id": pen["game_id"],
    "season": pen["season"],
    "player_id": pen["event_player_1_id"],
    "player_name": pen["event_player_1_name"],
    "penalty_severity": pen["penalty_severity"],
    "penalty_minutes": pen["penalty_minutes"],
    "drawn_by_id": pen["event_player_2_id"],
})
pen_out = pen_out.dropna(subset=["player_id"])
pen_out["player_id"] = pen_out["player_id"].astype(int)
pen_out.to_csv("output/v2_penalties.csv", index=False)
print(f"  {len(pen_out):,} penalty events → output/v2_penalties.csv", file=sys.stderr)

# ── 4. Zone start classification from faceoffs ──────────────────────────────
# x_fixed > 25 = home offensive zone (right side of rink)
# x_fixed < -25 = away offensive zone (left side of rink)
# |x_fixed| <= 25 = neutral zone
print("Classifying zone starts from faceoffs...", file=sys.stderr)

# Get faceoff events with zone classification
faceoffs = pbp[pbp["event_type"] == "FACEOFF"][
    ["game_id", "period", "event_id", "period_seconds", "x_fixed", "event_team_type"]
].copy()

# Zone relative to home team: positive x_fixed = home offensive zone
# OZ for home = x_fixed > 25; DZ for home = x_fixed < -25
faceoffs["fo_zone_home"] = np.where(
    faceoffs["x_fixed"] > 25, "OZ",
    np.where(faceoffs["x_fixed"] < -25, "DZ", "NZ")
)

# Now filter to just the RAPM events (shots, goals, giveaways, takeaways)
rapm_events = pbp[pbp["event_type"].isin(KEEP_EVENTS)].copy()

# For each RAPM event, find the most recent faceoff in the same game+period
rapm_events = rapm_events.sort_values(["game_id", "period", "event_id"]).reset_index(drop=True)
faceoffs = faceoffs.sort_values(["game_id", "period", "event_id"]).reset_index(drop=True)

# Merge_asof: find the most recent faceoff before each event
rapm_events = pd.merge_asof(
    rapm_events,
    faceoffs[["game_id", "period", "event_id", "period_seconds", "fo_zone_home"]].rename(
        columns={"event_id": "fo_event_id", "period_seconds": "fo_period_seconds"}
    ),
    left_on="event_id",
    right_on="fo_event_id",
    by=["game_id", "period"],
    direction="backward",
)

# Compute seconds since faceoff
rapm_events["seconds_since_faceoff"] = (
    rapm_events["period_seconds"] - rapm_events["fo_period_seconds"]
).clip(lower=0)

# Zone start relative to the acting team
# If home is acting: OZ for home = OZ for acting team
# If away is acting: OZ for home = DZ for acting team (flip)
home_acting = rapm_events["event_team_type"] == "home"
zone_home = rapm_events["fo_zone_home"]

rapm_events["zone_start"] = np.where(
    rapm_events["seconds_since_faceoff"] > 60, "NZ",  # zone context decayed
    np.where(
        home_acting,
        zone_home,  # home acting: use home perspective directly
        np.where(zone_home == "OZ", "DZ", np.where(zone_home == "DZ", "OZ", "NZ"))  # flip for away
    )
)

# Fill missing zone starts (no faceoff found) with NZ
rapm_events["zone_start"] = rapm_events["zone_start"].fillna("NZ")

print(f"  Zone start distribution:", file=sys.stderr)
print(f"    {(rapm_events['zone_start']=='OZ').sum():,} OZ", file=sys.stderr)
print(f"    {(rapm_events['zone_start']=='DZ').sum():,} DZ", file=sys.stderr)
print(f"    {(rapm_events['zone_start']=='NZ').sum():,} NZ", file=sys.stderr)

# ── 5. Merge on-ice lineups via merge_asof ───────────────────────────────────
print("Loading rapm_dataset.csv for lineups...", file=sys.stderr)
lineups = pd.read_csv("output/rapm_dataset.csv", dtype={"event_id": "int64"})
lineups = lineups.drop(columns=["strength"], errors="ignore")  # avoid suffix conflict
lineups["game_id"] = lineups["event_id"] // 10000

rapm_events = rapm_events.sort_values(["game_id", "event_id"]).reset_index(drop=True)
lineups = lineups.sort_values("event_id").reset_index(drop=True)

rapm_events = pd.merge_asof(
    rapm_events,
    lineups,
    on="event_id",
    by="game_id",
    direction="backward",
)
print(f"  {rapm_events['home_on_ice'].notna().sum():,} rows with lineup data", file=sys.stderr)

# ── 6. Load MoneyPuck shots for xGoal + rest differential ────────────────────
print("Loading shots_2007-2024.csv...", file=sys.stderr)
shot_cols = [
    "season", "game_id", "period", "time",
    "shooterPlayerId", "xGoal", "shotWasOnGoal", "shotType", "shotDistance",
    "shotOnEmptyNet",
    "averageRestDifference",
]
mp = pd.read_csv("data/shots_2007-2024.csv", usecols=shot_cols)

# Reconstruct full game_id (MoneyPuck uses start-year labeling)
mp["full_game_id"] = (
    mp["season"].astype(str) + mp["game_id"].apply(lambda x: f"{int(x):06d}")
).astype("int64")
mp["shooterPlayerId"] = mp["shooterPlayerId"].astype("Int64")
mp["period_seconds_calc"] = mp["time"] - (mp["period"] - 1) * 1200

mp = mp.rename(columns={
    "full_game_id": "_game_id",
    "period_seconds_calc": "_period_seconds",
    "shooterPlayerId": "_player_id",
}).drop(columns=["season", "game_id", "time"])

# Join xGoal + rest_differential on game+period+time+player
shot_mask = rapm_events["event_type"].isin({"SHOT", "GOAL", "MISSED_SHOT", "BLOCKED_SHOT"})
pbp_shots = rapm_events[shot_mask].copy()

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

# Fallback: join without player for unmatched non-blocked shots
unmatched_mask = xg["xGoal"].isna() & (xg["event_type"] != "BLOCKED_SHOT")
if unmatched_mask.sum() > 0:
    mp_noplay = mp.rename(columns={
        "_game_id": "game_id",
        "_period_seconds": "period_seconds",
    }).drop(columns=["_player_id"])
    mp_noplay = mp_noplay.drop_duplicates(subset=["game_id", "period", "period_seconds"])
    fallback = xg.loc[unmatched_mask, ["game_id", "period", "period_seconds"]].merge(
        mp_noplay, on=["game_id", "period", "period_seconds"], how="left",
    )
    for col in ["xGoal", "shotWasOnGoal", "shotType", "shotDistance", "shotOnEmptyNet", "averageRestDifference"]:
        xg.loc[unmatched_mask, col] = fallback[col].values
    print(f"  xGoal matched (fallback):   {xg['xGoal'].notna().sum():,} / {len(xg):,}", file=sys.stderr)

# Bring xGoal + rest columns back into main dataframe
xg_cols = ["xGoal", "shotWasOnGoal", "shotType", "shotDistance", "shotOnEmptyNet", "averageRestDifference"]
for col in xg_cols:
    rapm_events[col] = pd.NA

rapm_events = rapm_events.set_index("event_id")
xg_sub = xg.set_index("event_id")[xg_cols]
xg_sub = xg_sub[~xg_sub.index.duplicated(keep="first")]
rapm_events.loc[xg_sub.index, xg_cols] = xg_sub
rapm_events = rapm_events.reset_index()

# Rename for clarity
rapm_events = rapm_events.rename(columns={"averageRestDifference": "rest_differential"})

# ── 7. Derive indicator columns ──────────────────────────────────────────────
rapm_events["is_goal"]         = rapm_events["event_type"] == "GOAL"
rapm_events["is_shot_on_goal"] = rapm_events["shotWasOnGoal"] == 1
rapm_events["is_giveaway"]     = rapm_events["event_type"] == "GIVEAWAY"
rapm_events["is_takeaway"]     = rapm_events["event_type"] == "TAKEAWAY"
rapm_events["is_5v5"]          = rapm_events["strength_state"] == "5v5"
rapm_events["score_diff"]      = rapm_events["home_score"] - rapm_events["away_score"]

# ── 8. Select final columns and write output ─────────────────────────────────
out_cols = [
    "game_id", "season", "event_id", "event_type", "event_team", "event_team_type",
    "period", "period_seconds", "score_diff",
    "home_on_ice", "away_on_ice", "strength_state", "strength", "is_5v5",
    "home_goalie_id", "away_goalie_id",
    "is_goal", "is_shot_on_goal", "xGoal",
    "is_giveaway", "is_takeaway",
    "shotType", "shotDistance", "shotOnEmptyNet",
    "x", "y",
    "zone_start", "seconds_since_faceoff",
    "rest_differential",
]
out = rapm_events[out_cols]
out.to_csv("output/v2_clean_pbp.csv", index=False)

print(f"\nWrote {len(out):,} rows to output/v2_clean_pbp.csv", file=sys.stderr)
print(f"  5v5 rows:          {out['is_5v5'].sum():,}", file=sys.stderr)
print(f"  goals:             {out['is_goal'].sum():,}", file=sys.stderr)
print(f"  shots on goal:     {out['is_shot_on_goal'].sum():,}", file=sys.stderr)
print(f"  xGoal matched:     {out['xGoal'].notna().sum():,}", file=sys.stderr)
print(f"  giveaways:         {out['is_giveaway'].sum():,}", file=sys.stderr)
print(f"  takeaways:         {out['is_takeaway'].sum():,}", file=sys.stderr)
print(f"  rest_diff matched: {out['rest_differential'].notna().sum():,}", file=sys.stderr)
print(f"  home_goalie_id:    {out['home_goalie_id'].notna().sum():,}", file=sys.stderr)
