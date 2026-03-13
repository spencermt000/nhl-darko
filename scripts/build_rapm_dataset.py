import pandas as pd
import sys

df = pd.read_csv("raw_data.csv")

away_skater_cols = [f"away_on_{i}" for i in range(1, 7)]
home_skater_cols = [f"home_on_{i}" for i in range(1, 7)]

home_ids = set()
away_ids = set()
current_game = None
results = []

for _, row in df.iterrows():
    game_id = row["game_id"]
    event_id = row["event_id"]
    team_type = row["event_team_type"]

    # Reset on-ice state when game changes
    if game_id != current_game:
        home_ids = set()
        away_ids = set()
        current_game = game_id

    # Parse ids_on
    raw_on = str(row["ids_on"]).strip()
    if raw_on not in ("0", "nan", "None", ""):
        ids_on = {x.strip() for x in raw_on.split(",")}
    else:
        ids_on = set()

    # Parse ids_off
    raw_off = str(row["ids_off"]).strip()
    if raw_off not in ("0", "nan", "None", ""):
        ids_off = {x.strip() for x in raw_off.split(",")}
    else:
        ids_off = set()

    # Apply to correct side
    if team_type == "home":
        home_ids -= ids_off
        home_ids |= ids_on
    elif team_type == "away":
        away_ids -= ids_off
        away_ids |= ids_on

    # Strength from skater columns (excludes goalies)
    home_skaters = sum(1 for c in home_skater_cols if str(row[c]) not in ("NA", "nan"))
    away_skaters = sum(1 for c in away_skater_cols if str(row[c]) not in ("NA", "nan"))
    strength = f"{home_skaters}v{away_skaters}"

    results.append({
        "event_id": event_id,
        "home_on_ice": ", ".join(sorted(home_ids)),
        "away_on_ice": ", ".join(sorted(away_ids)),
        "strength": strength,
    })

out = pd.DataFrame(results)
out.to_csv("rapm_dataset.csv", index=False)
print(f"Wrote {len(out)} rows to rapm_dataset.csv", file=sys.stderr)
