"""
integrate_2025.py — Integrate 2025-26 scraped data into the pipeline.

Steps:
  1. Append raw_pbp_2025.csv to raw_pbp.csv (or read both)
  2. Build rapm_dataset from raw_data_2025.csv (different format than old data)
  3. Append shots_2025.csv to shots data
  4. Merge skaters_by_game2025.csv into skaters_by_game.csv

Outputs:
  output/rapm_dataset_2025.csv  — Lineup data in the format build_dataset.py expects
"""

import sys
import numpy as np
import pandas as pd

# ── 1. Build RAPM dataset from scraped lineup data ──────────────────────────
print("Building RAPM dataset from raw_data_2025.csv...", file=sys.stderr)

lineup = pd.read_csv("data/raw_data_2025.csv")
print(f"  Loaded {len(lineup):,} lineup rows", file=sys.stderr)

# Our scraped data has full on-ice state at each shift boundary.
# We need to produce: event_id, home_on_ice (space-separated IDs),
# away_on_ice (space-separated IDs), strength

# We need player name → player ID mapping. The scraped data has names in
# home_on_1..home_on_6 and IDs mixed into ids_on. But we also have the
# roster from PBP data. Let's build it from the skaters_by_game file.

print("  Loading player ID mapping from skaters_by_game2025...", file=sys.stderr)
sbg = pd.read_csv("data/skaters_by_game2025.csv",
                   usecols=["playerId", "name", "gameId"])
name_to_id = sbg.drop_duplicates("name").set_index("name")["playerId"].to_dict()

# Also load bio for backup mapping
bio = pd.read_csv("data/moneypuck_player_bio.csv")
bio = bio.rename(columns={"playerId": "player_id"})
for _, row in bio.iterrows():
    name = str(row.get("playerName", ""))
    pid = row["player_id"]
    if name and pid:
        # MoneyPuck uses "First Last" format
        dotted = name.replace(" ", ".")
        if dotted not in name_to_id:
            name_to_id[dotted] = int(pid)

# Build lookup for both dotted and spaced name formats
name_to_id_dotted = {}
for name, pid in name_to_id.items():
    name_to_id_dotted[name] = pid
    # "First Last" → "First.Last"
    name_to_id_dotted[name.replace(" ", ".")] = pid

print(f"  Name→ID mappings: {len(name_to_id_dotted):,} (including dotted variants)", file=sys.stderr)

away_cols = [f"away_on_{i}" for i in range(1, 7)]
home_cols = [f"home_on_{i}" for i in range(1, 7)]

results = []
unmapped_names = set()


def lookup_pid(name):
    """Look up player ID by name, trying dotted and spaced formats."""
    pid = name_to_id_dotted.get(name)
    if pid:
        return pid
    # Try converting dots to spaces
    spaced = name.replace(".", " ")
    pid = name_to_id_dotted.get(spaced)
    if pid:
        return pid
    # Try converting spaces to dots
    dotted = name.replace(" ", ".")
    pid = name_to_id_dotted.get(dotted)
    return pid


for _, row in lineup.iterrows():
    # Collect home skater IDs from name columns
    home_ids = []
    for col in home_cols:
        name = str(row[col]).strip()
        if name and name != "nan":
            pid = lookup_pid(name)
            if pid:
                home_ids.append(str(int(pid)))
            else:
                unmapped_names.add(name)

    # Collect away skater IDs
    away_ids = []
    for col in away_cols:
        name = str(row[col]).strip()
        if name and name != "nan":
            pid = lookup_pid(name)
            if pid:
                away_ids.append(str(int(pid)))
            else:
                unmapped_names.add(name)

    # Goalie IDs
    home_goalie_name = str(row.get("home_goalie", "")).strip()
    away_goalie_name = str(row.get("away_goalie", "")).strip()
    home_goalie_id = name_to_id.get(home_goalie_name)
    away_goalie_id = name_to_id.get(away_goalie_name)

    # Strength = number of skaters per side (excluding goalies)
    strength = f"{len(home_ids)}v{len(away_ids)}"

    # Build on-ice strings (space-separated, like the old format uses after processing)
    home_on_ice = ", ".join(sorted(home_ids))
    away_on_ice = ", ".join(sorted(away_ids))

    results.append({
        "event_id": row["event_id"],
        "home_on_ice": home_on_ice,
        "away_on_ice": away_on_ice,
        "strength": strength,
        "home_goalie_id": int(home_goalie_id) if home_goalie_id else np.nan,
        "away_goalie_id": int(away_goalie_id) if away_goalie_id else np.nan,
    })

rapm_df = pd.DataFrame(results)
rapm_df.to_csv("output/rapm_dataset_2025.csv", index=False)
print(f"  Wrote {len(rapm_df):,} rows → output/rapm_dataset_2025.csv", file=sys.stderr)
print(f"  Strength distribution:", file=sys.stderr)
print(f"    {rapm_df['strength'].value_counts().head(5).to_string()}", file=sys.stderr)

if unmapped_names:
    print(f"\n  WARNING: {len(unmapped_names)} unmapped player names", file=sys.stderr)
    print(f"  Samples: {sorted(unmapped_names)[:10]}", file=sys.stderr)


# ── 2. Merge skaters_by_game ────────────────────────────────────────────────
print("\nChecking skaters_by_game merge...", file=sys.stderr)
sbg_old = pd.read_csv("data/skaters_by_game.csv", nrows=1)
sbg_new = pd.read_csv("data/skaters_by_game2025.csv", nrows=1)

old_cols = set(sbg_old.columns)
new_cols = set(sbg_new.columns)

missing_in_new = old_cols - new_cols
missing_in_old = new_cols - old_cols

if missing_in_new:
    print(f"  Columns in old but not new: {missing_in_new}", file=sys.stderr)
if missing_in_old:
    print(f"  Columns in new but not old: {missing_in_old}", file=sys.stderr)
if not missing_in_new and not missing_in_old:
    print(f"  Column schemas match perfectly ({len(old_cols)} columns)", file=sys.stderr)


# ── 3. Summary ──────────────────────────────────────────────────────────────
print("\n" + "="*60, file=sys.stderr)
print("2025-26 data integration summary:", file=sys.stderr)
print(f"  RAPM lineup dataset: output/rapm_dataset_2025.csv ({len(rapm_df):,} rows)", file=sys.stderr)
print(f"  PBP events:          data/raw_pbp_2025.csv", file=sys.stderr)
print(f"  Shots with xGoal:    data/shots_2025.csv", file=sys.stderr)
print(f"  Skaters by game:     data/skaters_by_game2025.csv", file=sys.stderr)
print(f"\nTo run the pipeline with 2025 data, update build_dataset.py to:", file=sys.stderr)
print(f"  1. Read raw_pbp.csv + raw_pbp_2025.csv (concat)", file=sys.stderr)
print(f"  2. Read rapm_dataset.csv + rapm_dataset_2025.csv (concat)", file=sys.stderr)
print(f"  3. Read shots_2007-2024.csv + shots_2025.csv (concat)", file=sys.stderr)
print(f"  4. Read skaters_by_game.csv + skaters_by_game2025.csv (concat)", file=sys.stderr)
print("="*60, file=sys.stderr)
