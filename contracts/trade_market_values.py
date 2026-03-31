"""
Trade Market Values — Learn asset values from historical trades.

Uses our existing models (surplus v2, draft pick chart) as priors,
then adjusts based on what the actual trade market reveals.

Approach:
  1. Start with model-based values for all assets (surplus v2 for players,
     draft chart for picks)
  2. For each historical trade, compute the imbalance (side1 value - side2 value)
  3. Use these imbalances to calibrate: if the market consistently trades
     a type of asset for more than our model says, adjust upward

Usage: python contracts/trade_market_values.py
"""

import os
import re
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(BASE, "output")

sys.path.insert(0, BASE)
from contracts.contract_utils import load_stats_names, resolve_name
from contracts.trade_evaluator import value_player, value_draft_pick

# ── 1. Load trades ──────────────────────────────────────────────────────────

trades = pd.read_csv(os.path.join(BASE, "data", "trades.csv"))
print(f"Trades loaded: {len(trades)}")

# ── 2. Normalize assets ─────────────────────────────────────────────────────

def normalize_pick_str(raw):
    """Convert messy pick text to standardized 'YEAR Nth' format."""
    raw = re.sub(r"\s+on\s+\d{4}-\d{2}-\d{2}$", "", raw.strip())
    raw = re.sub(r"\s*\([^)]*\)", "", raw)
    raw = re.sub(r"\s+", " ", raw).strip()

    round_words = {"first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th",
                   "fifth": "5th", "sixth": "6th", "seventh": "7th"}
    m = re.match(r"(\d{4})\s+(first|second|third|fourth|fifth|sixth|seventh)\s+round", raw, re.IGNORECASE)
    if m:
        return f"{m.group(1)} {round_words[m.group(2).lower()]}"

    # Conditional picks — find any round word and year
    for word, short in round_words.items():
        if word in raw.lower():
            year_m = re.search(r"(\d{4})", raw)
            if year_m:
                return f"{year_m.group(1)} {short}"
    return None


def normalize_player_str(raw):
    """Clean a player name from trade text."""
    raw = re.sub(r"^rights to\s+", "", raw.strip())
    raw = re.sub(r"\s+on\s+\d{4}-\d{2}-\d{2}$", "", raw)
    raw = re.sub(r"\s*\([^)]*\)", "", raw)
    return raw.strip()


def parse_trade_side(assets_str, trade_season):
    """Parse a trade side into valued assets."""
    if pd.isna(assets_str):
        return []
    assets = []
    for raw in assets_str.split(";"):
        raw = raw.strip()
        if not raw or raw in ("cash", "future considerations"):
            continue

        # Try as pick
        pick_norm = normalize_pick_str(raw)
        if pick_norm:
            val, desc = value_draft_pick(pick_norm, trade_season)
            assets.append({"raw": raw, "normalized": pick_norm, "type": "pick", "value": val})
            continue

        # Try as player
        player = normalize_player_str(raw)
        val, desc = value_player(player, trade_season)
        if val != 0 or "Unknown" not in desc:
            assets.append({"raw": raw, "normalized": player, "type": "player", "value": val})

    return assets


# ── 3. Evaluate all historical trades ───────────────────────────────────────

print("Evaluating all trades...")

trade_results = []
for _, trade in trades.iterrows():
    season = trade["draft_year"]

    side1 = parse_trade_side(trade["team_1_sends"], season)
    side2 = parse_trade_side(trade["team_2_sends"], season)

    if not side1 and not side2:
        continue

    val1 = sum(a["value"] for a in side1)
    val2 = sum(a["value"] for a in side2)
    imbalance = val1 - val2

    trade_results.append({
        "trade_id": trade["trade_id"],
        "year": season,
        "team_1": trade["team_1"],
        "team_2": trade["team_2"],
        "side_1_assets": len(side1),
        "side_2_assets": len(side2),
        "side_1_value": val1,
        "side_2_value": val2,
        "imbalance": imbalance,
        "abs_imbalance": abs(imbalance),
        "side_1_picks": sum(1 for a in side1 if a["type"] == "pick"),
        "side_1_players": sum(1 for a in side1 if a["type"] == "player"),
        "side_2_picks": sum(1 for a in side2 if a["type"] == "pick"),
        "side_2_players": sum(1 for a in side2 if a["type"] == "player"),
    })

results = pd.DataFrame(trade_results)
print(f"Evaluated trades: {len(results)}")

# ── 4. Analysis ─────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print("TRADE MARKET CALIBRATION")
print(f"{'='*60}")

print(f"\nOverall imbalance stats:")
print(f"  Mean imbalance: ${results['imbalance'].mean():+,.0f}")
print(f"  Median imbalance: ${results['imbalance'].median():+,.0f}")
print(f"  Mean |imbalance|: ${results['abs_imbalance'].mean():,.0f}")
print(f"  Std: ${results['imbalance'].std():,.0f}")

# If mean imbalance is consistently positive, it means team_1 (the team
# that initiated the pick-related trade) is systematically overpaying
# If negative, they're getting bargains

# Breakdown by trade type
print(f"\nBy trade structure:")
for desc, mask in [
    ("Player for picks", (results["side_1_players"] > 0) & (results["side_2_picks"] > 0) & (results["side_1_picks"] == 0)),
    ("Picks for player", (results["side_1_picks"] > 0) & (results["side_2_players"] > 0) & (results["side_1_players"] == 0)),
    ("Player for player", (results["side_1_players"] > 0) & (results["side_2_players"] > 0)),
    ("Picks for picks", (results["side_1_picks"] > 0) & (results["side_2_picks"] > 0) & (results["side_1_players"] == 0) & (results["side_2_players"] == 0)),
]:
    sub = results[mask]
    if len(sub) >= 5:
        print(f"  {desc:25s} n={len(sub):3d}  mean imbalance=${sub['imbalance'].mean():>+10,.0f}  "
              f"median=${sub['imbalance'].median():>+10,.0f}")

# ── 5. Learn empirical pick discount rate ───────────────────────────────────
# Look at trades where a current-year pick is traded for a future pick
# of the same round. The value ratio reveals the market discount rate.

print(f"\n{'='*60}")
print("EMPIRICAL PICK DISCOUNT RATE")
print(f"{'='*60}")

# Find pick-for-pick trades where rounds are similar
pick_discount_obs = []
for _, trade in trades.iterrows():
    season = trade["draft_year"]
    side1 = parse_trade_side(trade["team_1_sends"], season)
    side2 = parse_trade_side(trade["team_2_sends"], season)

    s1_picks = [a for a in side1 if a["type"] == "pick"]
    s2_picks = [a for a in side2 if a["type"] == "pick"]

    # For each pick, note its year offset from the trade season
    for picks_list in [s1_picks, s2_picks]:
        for p in picks_list:
            norm = p["normalized"]
            if norm:
                parts = norm.split()
                if len(parts) == 2:
                    try:
                        pick_year = int(parts[0])
                        years_out = pick_year - season
                        if 0 <= years_out <= 5:
                            pick_discount_obs.append({
                                "trade_season": season,
                                "pick_year": pick_year,
                                "years_out": years_out,
                                "round": parts[1],
                                "value": p["value"],
                            })
                    except ValueError:
                        pass

if pick_discount_obs:
    pdo = pd.DataFrame(pick_discount_obs)
    print(f"\nPick observations by years out:")
    for yo in sorted(pdo["years_out"].unique()):
        sub = pdo[pdo["years_out"] == yo]
        avg_val = sub["value"].mean()
        count = len(sub)
        print(f"  {yo}yr out: n={count:3d}, mean value=${avg_val:>10,.0f}")

    # Compare: value of Y+0 picks vs Y+1, Y+2 etc. to get discount
    y0 = pdo[pdo["years_out"] == 0]["value"].mean()
    for yo in [1, 2, 3]:
        yn = pdo[pdo["years_out"] == yo]["value"].mean()
        if y0 > 0 and yn > 0:
            implied_discount = 1 - (yn / y0) ** (1 / yo)
            print(f"  Implied annual discount (Y+0 vs Y+{yo}): {implied_discount:.1%}")

# ── 6. Most lopsided trades ─────────────────────────────────────────────────

print(f"\n{'='*60}")
print("MOST LOPSIDED TRADES (by model)")
print(f"{'='*60}")

print(f"\nTeam 1 wins big (model says team 1 sent more value):")
for _, r in results.nlargest(10, "imbalance").iterrows():
    print(f"  {r['team_1']:15s} ↔ {r['team_2']:15s} ({r['year']})  "
          f"imbalance: ${r['imbalance']:>+10,.0f}")

print(f"\nTeam 2 wins big:")
for _, r in results.nsmallest(10, "imbalance").iterrows():
    print(f"  {r['team_1']:15s} ↔ {r['team_2']:15s} ({r['year']})  "
          f"imbalance: ${r['imbalance']:>+10,.0f}")

# ── 6. Current player trade values ──────────────────────────────────────────

print(f"\n{'='*60}")
print("CURRENT PLAYER TRADE VALUES")
print(f"{'='*60}\n")

top_players = [
    "Connor McDavid", "Nathan MacKinnon", "Nikita Kucherov", "Leon Draisaitl",
    "Jason Robertson", "Connor Bedard", "Auston Matthews", "Cale Makar",
    "Quinn Hughes", "Matthew Tkachuk", "Sam Reinhart", "Kirill Kaprizov",
    "David Pastrnak", "Mika Zibanejad", "Brady Tkachuk", "Jack Hughes",
]

print(f"{'Player':25s} {'Trade Value':>12s} {'Details'}")
print("-" * 75)
for name in top_players:
    val, desc = value_player(name, 2025)
    print(f"{name:25s} ${val:>11,.0f}  {desc}")

print(f"\n{'Pick':25s} {'Trade Value':>12s}")
print("-" * 40)
for pick in ["2025 1st", "2025 2nd", "2025 3rd", "2026 1st", "2026 2nd", "2027 1st"]:
    val, desc = value_draft_pick(pick, 2025)
    print(f"{pick:25s} ${val:>11,.0f}")

# ── 7. Save ─────────────────────────────────────────────────────────────────

results.to_csv(os.path.join(BASE, "contracts", "trade_evaluations.csv"), index=False)
print(f"\nSaved {len(results)} trade evaluations to contracts/trade_evaluations.csv")
