"""
Trade Evaluator — Value every tradeable asset in dollars.

Player valuation uses the surplus v2 approach: a Ridge model trained on
all player-seasons predicts what Cap % a player's stats are worth, then
converts to dollars. Trade value = NPV of (predicted market value - actual
cap cost) over remaining contract + RFA extension years.

Draft pick valuation uses expected surplus from draft_pick_value_chart.csv.

Usage: python contracts/trade_evaluator.py
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(BASE, "output")

sys.path.insert(0, BASE)
from contracts.contract_utils import load_stats_names, resolve_name

CAP_GROWTH_RATE = 0.04

SALARY_CAP = {
    2015: 71_400_000, 2016: 73_000_000, 2017: 75_000_000, 2018: 79_500_000,
    2019: 81_500_000, 2020: 81_500_000, 2021: 81_500_000, 2022: 82_500_000,
    2023: 83_500_000, 2024: 88_000_000, 2025: 95_500_000,
    2026: 104_500_000, 2027: 110_000_000, 2028: 115_000_000,
    2029: 120_000_000, 2030: 125_000_000,
}

# ── 1. Load data ────────────────────────────────────────────────────────────

# Surplus v2: has pred_market_value (what a player's stats are worth) per season
surplus_v2 = pd.read_csv(os.path.join(BASE, "contracts", "surplus_values_v2.csv"))

# Draft pick values
dpv = pd.read_csv(os.path.join(BASE, "contracts", "draft_pick_value_chart.csv"))

# Active contracts
ac = pd.read_csv(os.path.join(BASE, "contracts", "active_contracts_by_season.csv"))

# Age data
cf = pd.read_csv(os.path.join(OUTPUT, "v6_carry_forward.csv"),
                 usecols=["player_id", "season", "age"])

# Name mapping
name_to_id = load_stats_names()
stats_names = set(name_to_id.keys())
stats_lower = {n.lower(): n for n in stats_names}

# ── 2. Draft pick valuation ─────────────────────────────────────────────────
# Use expected PRODUCTION value (mean_war_7yr * $/WAR proxy), not surplus.
# Surplus goes negative for late picks because cap cost > production,
# but trade value should reflect the production upside.

# Convert mean WAR over 7 years to a dollar value using the surplus model's
# predicted market value per WAR (from the Ridge model, ~$5-8M per WAR season)
MARKET_VALUE_PER_WAR_SEASON = 6_000_000  # conservative, from surplus v2 model

pick_production = {}
for _, row in dpv.iterrows():
    lo, hi = row["pick_lo"], row["pick_hi"]
    # Expected total production value = mean_war_7yr * value_per_war * nhl_rate
    prod_value = row["mean_war_7yr"] * MARKET_VALUE_PER_WAR_SEASON * row["nhl_rate"]
    for pick in range(int(lo), int(hi) + 1):
        pick_production[pick] = prod_value

ROUND_MID_PICK = {1: 16, 2: 48, 3: 80, 4: 112, 5: 144, 6: 176, 7: 208}
round_value = {}
for rnd, mid in ROUND_MID_PICK.items():
    round_value[rnd] = pick_production.get(mid, 0)

# Empirical discount rate for future picks (learned from 604 historical trades)
# Y+1 picks trade at 17% discount, Y+2 at 13%/yr, Y+3 at 14%/yr
# Use ~14% as the blended rate — much higher than cap growth (4%)
# because future picks carry uncertainty about team record + draft quality
PICK_ANNUAL_DISCOUNT = 0.14


def value_draft_pick(pick_str, trade_season):
    """Value a draft pick like '2024 1st' based on expected production."""
    parts = pick_str.strip().split()
    if len(parts) != 2:
        return 0, f"Can't parse: {pick_str}"
    try:
        year = int(parts[0])
    except ValueError:
        return 0, f"Bad year: {parts[0]}"

    round_map = {"1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5, "6th": 6, "7th": 7}
    rnd = round_map.get(parts[1], 0)
    if rnd == 0:
        return 0, f"Bad round: {parts[1]}"

    base_value = round_value.get(rnd, 0)
    years_out = max(0, year - trade_season)
    discounted = base_value / ((1 + PICK_ANNUAL_DISCOUNT) ** years_out)

    return discounted, f"Round {rnd} ({year}, {years_out}yr out)"


# ── 3. Player valuation (surplus v2 approach) ───────────────────────────────

AGE_MODIFIER = {}
for age in range(18, 45):
    if age <= 22: AGE_MODIFIER[age] = 1.10
    elif age <= 25: AGE_MODIFIER[age] = 1.03
    elif age <= 30: AGE_MODIFIER[age] = 1.00
    elif age <= 33: AGE_MODIFIER[age] = 0.97
    elif age <= 36: AGE_MODIFIER[age] = 0.93
    else: AGE_MODIFIER[age] = 0.88


def value_player(player_name, trade_season):
    """Value a player using surplus v2 market valuation.

    Uses pred_market_value (what the player's stats are worth on the open market)
    minus their actual cap hit, summed over remaining contract years + RFA extensions.
    """
    resolved = resolve_name(player_name, stats_names, stats_lower)
    if resolved is None:
        if player_name in stats_names:
            resolved = player_name
        else:
            return 0, f"Unknown player: {player_name}"

    pid = name_to_id.get(resolved)
    if pid is None:
        return 0, f"No ID for: {resolved}"

    # Get current season's market value from surplus v2
    sv = surplus_v2[surplus_v2["player_id"] == pid].sort_values("season", ascending=False)
    if len(sv) == 0:
        return 0, f"No surplus data: {resolved}"

    curr = sv.iloc[0]
    curr_market_value = curr.get("pred_market_value", curr.get("market_value", 0))
    curr_cap_pct = curr.get("pred_cap_pct", 0)
    war_82 = curr.get("WAR_82", curr.get("WAR", 0))

    # Get age
    cf_row = cf[(cf["player_id"] == pid) & (cf["season"] == curr["season"])]
    age = int(cf_row.iloc[0]["age"]) if len(cf_row) else 27

    # Get remaining contract
    player_contract = ac[(ac["player_id"] == pid) & (ac["season"] >= trade_season)]
    if len(player_contract) == 0:
        if age <= 27:
            # Pending RFA: 3 years of surplus at ~40% of market value
            total = 0
            for yr in range(3):
                age_mod = AGE_MODIFIER.get(age + yr, 0.95)
                proj_market = curr_market_value * age_mod * (0.95 ** yr)
                surplus = proj_market * 0.40  # RFA discount
                total += surplus / ((1 + CAP_GROWTH_RATE) ** yr)
            return total, f"{resolved}: mkt=${curr_market_value:,.0f}, age={age}, pending RFA ~3yr"
        return 0, f"{resolved}: UFA/no contract"

    player_contract = player_contract.sort_values("season")
    remaining_seasons = list(player_contract["season"].unique())

    # Extend for young RFA players
    last_season = remaining_seasons[-1]
    if age + (last_season - trade_season) <= 27:
        for extra in range(1, 4):
            ext = last_season + extra
            if ext not in remaining_seasons:
                remaining_seasons.append(ext)

    # NPV calculation
    total_npv = 0
    for i, szn in enumerate(remaining_seasons):
        future_age = age + (szn - trade_season)
        age_mod = AGE_MODIFIER.get(future_age, 0.88)

        # Project market value (what stats are worth)
        proj_market = curr_market_value * age_mod * (0.95 ** i)

        # Cap cost
        contract_row = player_contract[player_contract["season"] == szn]
        if len(contract_row):
            cap_cost = contract_row.iloc[0]["cap_hit"]
        else:
            # RFA extension: estimate at ~70% of market value
            cap_cost = proj_market * 0.70

        surplus = proj_market - cap_cost
        discount = (1 + CAP_GROWTH_RATE) ** (szn - trade_season)
        total_npv += surplus / discount

    yrs = len(remaining_seasons)
    cap_hit = player_contract.iloc[0]["cap_hit"]

    return total_npv, f"{resolved}: mkt=${curr_market_value:,.0f}, age={age}, {yrs}yr @ ${cap_hit:,.0f}"


def value_player_production(player_name, trade_season):
    """Value a player using raw predicted market value (no cap cost subtracted).

    Same approach as picks: what is the player's expected production worth,
    regardless of what they're paid.
    """
    resolved = resolve_name(player_name, stats_names, stats_lower)
    if resolved is None:
        if player_name in stats_names:
            resolved = player_name
        else:
            return 0, f"Unknown player: {player_name}"

    pid = name_to_id.get(resolved)
    if pid is None:
        return 0, f"No ID for: {resolved}"

    sv = surplus_v2[surplus_v2["player_id"] == pid].sort_values("season", ascending=False)
    if len(sv) == 0:
        return 0, f"No surplus data: {resolved}"

    curr = sv.iloc[0]
    curr_market_value = curr.get("pred_market_value", curr.get("market_value", 0))

    cf_row = cf[(cf["player_id"] == pid) & (cf["season"] == curr["season"])]
    age = int(cf_row.iloc[0]["age"]) if len(cf_row) else 27

    player_contract = ac[(ac["player_id"] == pid) & (ac["season"] >= trade_season)]
    if len(player_contract) == 0:
        remaining = 3 if age <= 27 else 1
    else:
        remaining_seasons = list(player_contract["season"].unique())
        last_season = remaining_seasons[-1]
        if age + (last_season - trade_season) <= 27:
            for extra in range(1, 4):
                ext = last_season + extra
                if ext not in remaining_seasons:
                    remaining_seasons.append(ext)
        remaining = len(remaining_seasons)

    prod_npv = 0
    for i in range(remaining):
        future_age = age + i
        age_mod = AGE_MODIFIER.get(future_age, 0.88)
        proj_market = curr_market_value * age_mod * (0.95 ** i)
        discount = (1 + CAP_GROWTH_RATE) ** i
        prod_npv += proj_market / discount

    return prod_npv, f"{resolved}: mkt=${curr_market_value:,.0f}, age={age}, {remaining}yr ctrl"


# ── 4. Evaluate a trade ────────────────────────────────────────────────────

def evaluate_trade(team_1_sends, team_2_sends, trade_season):
    """Evaluate a trade by valuing all assets on each side."""
    def value_assets(assets_str):
        assets = [a.strip() for a in assets_str.split(";")]
        total = 0
        details = []
        for asset in assets:
            if not asset:
                continue
            parts = asset.split()
            if len(parts) == 2 and parts[1] in ("1st", "2nd", "3rd", "4th", "5th", "6th", "7th"):
                val, desc = value_draft_pick(asset, trade_season)
            else:
                val, desc = value_player(asset, trade_season)
            total += val
            details.append({"asset": asset, "value": val, "description": desc})
        return total, details

    t1_val, t1_details = value_assets(team_1_sends)
    t2_val, t2_details = value_assets(team_2_sends)

    return {
        "team_1_value": t1_val,
        "team_2_value": t2_val,
        "net": t2_val - t1_val,
        "team_1_details": t1_details,
        "team_2_details": t2_details,
    }


# ── 5. Standalone ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick test
    for name in ["Jason Robertson", "Connor Bedard", "Connor McDavid", "Nikita Kucherov"]:
        val, desc = value_player(name, 2025)
        print(f"{name:25s} ${val:>12,.0f}  {desc}")

    print()
    for pick in ["2025 1st", "2025 2nd", "2026 1st", "2026 3rd"]:
        val, desc = value_draft_pick(pick, 2025)
        print(f"{pick:25s} ${val:>12,.0f}  {desc}")
