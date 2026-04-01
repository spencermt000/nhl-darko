"""
Contract Utilities — Shared name matching, contract expansion, and season lookup.

Single source of truth for contract-to-player resolution used by:
  - dashboard/streamlit_app.py
  - contracts/surplus_model.py, surplus_model_v2.py
  - contracts/predict_contracts.py, fa_projections.py

Usage:
  from contracts.contract_utils import (
      resolve_name, build_active_contracts, get_season_lookup
  )

  # Or run standalone to generate active_contracts_by_season.csv:
  python contracts/contract_utils.py
"""

import os
import numpy as np
import pandas as pd
from unicodedata import normalize as _ucnorm

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(BASE, "output")

# ── Canonical name fixes ────────────────────────────────────────────────────
# Contract name → stats pipeline name (consolidated from all scripts)

CONTRACT_NAME_FIXES = {
    # Nickname differences
    "Joshua Norris": "Josh Norris",
    "John-Jason Peterka": "JJ Peterka",
    "Matthew Beniers": "Matty Beniers",
    "Cameron York": "Cam York",
    "Nicklaus Perbix": "Nick Perbix",
    "Joseph Veleno": "Joe Veleno",
    "Alexander Petrovic": "Alex Petrovic",
    "Pat Maroon": "Patrick Maroon",
    "Mike Matheson": "Michael Matheson",
    "Callan Foote": "Cal Foote",
    "Zachary Jones": "Zac Jones",
    "Joshua Dunne": "Josh Dunne",
    "Samuel Poulin": "Sam Poulin",
    "Matthew Stienburg": "Matt Stienburg",
    "Ronald Attard": "Ronnie Attard",
    "Danny O'Regan": "Daniel O'Regan",
    "Matthew Savoie": "Matt Savoie",
    "Benjamin Kindel": "Ben Kindel",
    "Maxim Shabanov": "Max Shabanov",
    "Cameron Lund": "Cam Lund",
    # Data entry differences
    "Oskar Back": "Oskar Bck",
    # Transliteration differences
    "Janis Jérôme Moser": "J.J. Moser",
    "Artyom Zub": "Artem Zub",
    "Yegor Zamula": "Egor Zamula",
    "Nicolai Knyzhov": "Nikolai Knyzhov",
    "Fyodor Svechkov": "Fedor Svechkov",
    "Nikolay Prokhorkin": "Nikolai Prokhorkin",
    "Yegor Korshkov": "Egor Korshkov",
}

SALARY_CAP = {
    2015: 71_400_000, 2016: 73_000_000, 2017: 75_000_000, 2018: 79_500_000,
    2019: 81_500_000, 2020: 81_500_000, 2021: 81_500_000, 2022: 82_500_000,
    2023: 83_500_000, 2024: 88_000_000, 2025: 95_500_000,
}


def strip_accents(s):
    """Remove diacritics (Slafkovský → Slafkovsky)."""
    return _ucnorm("NFD", s).encode("ascii", "ignore").decode("ascii")


def resolve_name(contract_name, stats_names, stats_lower=None):
    """Resolve a contract player name to the stats pipeline name.

    5-step cascade:
      1. Manual fix (nickname/transliteration)
      2. Exact match
      3. Accent stripping (Slafkovský → Slafkovsky)
      4. Case-insensitive (DeBrincat vs Debrincat)
      5. Hyphen removal (Charles-Alexis → Charles Alexis)

    Args:
        contract_name: Player name from contracts.csv
        stats_names: set of player names from the stats pipeline
        stats_lower: optional precomputed {lowercase: original} dict

    Returns:
        Resolved stats name, or None if no match
    """
    # 1. Manual fix
    if contract_name in CONTRACT_NAME_FIXES:
        fixed = CONTRACT_NAME_FIXES[contract_name]
        if fixed in stats_names:
            return fixed

    # 2. Exact match
    if contract_name in stats_names:
        return contract_name

    # 3. Accent stripping
    stripped = strip_accents(contract_name)
    if stripped in stats_names:
        return stripped

    # 4. Case-insensitive
    if stats_lower is None:
        stats_lower = {n.lower(): n for n in stats_names}
    if stripped.lower() in stats_lower:
        return stats_lower[stripped.lower()]

    # 5. Hyphen removal
    no_hyph = strip_accents(contract_name.replace("-", " "))
    if no_hyph in stats_names:
        return no_hyph

    return None


def load_contracts():
    """Load and parse contracts.csv with numeric fields."""
    contracts = pd.read_csv(os.path.join(BASE, "contracts", "contracts.csv"))
    contracts["cap_hit_num"] = (contracts["Cap Hit"]
                                .str.replace("$", "", regex=False)
                                .str.replace(",", "", regex=False)
                                .astype(float))
    contracts["cap_pct_num"] = (contracts["Cap %"]
                                .str.replace("%", "", regex=False)
                                .astype(float))
    contracts["term_yr"] = (contracts["Term"]
                            .str.replace("yr", "", regex=False)
                            .astype(int))
    contracts["sign_date"] = pd.to_datetime(contracts["Date"], format="%d-%b-%y")
    return contracts


def load_stats_names():
    """Load the most complete player name→id mapping from daily_war + skaters_by_game."""
    name_to_id = {}

    # Primary source: v5_daily_war (most complete for recent seasons)
    dw_path = os.path.join(OUTPUT, "v5_daily_war.csv")
    if os.path.exists(dw_path):
        dw = pd.read_csv(dw_path, usecols=["player_id", "player_name"])
        dw = dw.dropna(subset=["player_name"]).drop_duplicates("player_name")
        for _, row in dw.iterrows():
            name_to_id[row["player_name"]] = int(row["player_id"])

    # Supplement with dashboard_skater_war (has team info, some older players)
    sw_path = os.path.join(OUTPUT, "dashboard_skater_war.csv")
    if os.path.exists(sw_path):
        sw = pd.read_csv(sw_path, usecols=["player_id", "player_name"])
        sw = sw.dropna(subset=["player_name"]).drop_duplicates("player_name")
        for _, row in sw.iterrows():
            if row["player_name"] not in name_to_id:
                name_to_id[row["player_name"]] = int(row["player_id"])

    # Supplement with goalie WAR data
    for goalie_file in ["output/dashboard_goalie_war.csv", "output/v2_goalie_war_by_season.csv"]:
        gpath = os.path.join(BASE, goalie_file)
        if os.path.exists(gpath):
            _gw = pd.read_csv(gpath)
            id_col = "goalie_id" if "goalie_id" in _gw.columns else "player_id"
            name_col = "goalie_name" if "goalie_name" in _gw.columns else "player_name"
            if id_col in _gw.columns and name_col in _gw.columns:
                _gw = _gw.dropna(subset=[id_col, name_col]).drop_duplicates(name_col)
                for _, _row in _gw.iterrows():
                    if _row[name_col] not in name_to_id:
                        name_to_id[_row[name_col]] = int(_row[id_col])

    # Supplement with skaters_by_game (catches callups not in daily)
    for sbg_file in ["data/skaters_by_game.csv", "data/skaters_by_game2025.csv"]:
        sbg_path = os.path.join(BASE, sbg_file)
        if os.path.exists(sbg_path):
            sbg = pd.read_csv(sbg_path, usecols=["playerId", "name"], low_memory=False)
            sbg = sbg.dropna(subset=["playerId", "name"]).drop_duplicates("name")
            for _, row in sbg.iterrows():
                if row["name"] not in name_to_id:
                    name_to_id[row["name"]] = int(row["playerId"])

    return name_to_id


def build_active_contracts(contracts=None, name_to_id=None):
    """Expand all contracts into per-season rows and deduplicate.

    Returns a DataFrame with one row per (player_id, season) showing the
    active contract for that season. Handles extensions correctly by keeping
    the most recently signed contract for any overlapping season.
    """
    if contracts is None:
        contracts = load_contracts()
    if name_to_id is None:
        name_to_id = load_stats_names()

    stats_names = set(name_to_id.keys())
    stats_lower = {n.lower(): n for n in stats_names}

    # Resolve names and assign player_ids
    contracts["stats_name"] = contracts["Player"].map(
        lambda n: resolve_name(n, stats_names, stats_lower))
    contracts["player_id"] = contracts["stats_name"].map(name_to_id)

    # Filter to matched contracts only
    matched = contracts.dropna(subset=["player_id"]).copy()
    matched["player_id"] = matched["player_id"].astype(int)

    # Expand into per-season rows
    rows = []
    for _, c in matched.iterrows():
        sign_date = c["sign_date"]
        # First season: July+ → next season, Jan-June → current season
        if sign_date.month >= 7:
            first_season = sign_date.year + 1
        else:
            first_season = sign_date.year

        for yr in range(c["term_yr"]):
            season = first_season + yr
            if season not in SALARY_CAP:
                continue
            rows.append({
                "player_id": c["player_id"],
                "player_name": c["stats_name"],
                "season": season,
                "cap_hit": c["cap_hit_num"],
                "cap_pct": c["cap_pct_num"],
                "contract_type": c["Level"],
                "sign_status": c["Sign Status"],
                "sign_age": c["Sign Age"],
                "sign_date": sign_date,
                "position": c["POS"],
                "draft_year": c.get("Draft Year", np.nan),
                "draft_round": c.get("Draft Round", np.nan),
            })

    result = pd.DataFrame(rows)

    # Deduplicate: if a player has overlapping contracts for the same season
    # (e.g., extension signed mid-season), keep the most recent signing
    result = (result
              .sort_values("sign_date", ascending=False)
              .drop_duplicates(["player_id", "season"], keep="first"))

    result = result.sort_values(["season", "player_name"]).reset_index(drop=True)
    return result


def get_season_lookup(active_contracts, season):
    """Get a {player_name: contract_row} dict for a specific season."""
    szn = active_contracts[active_contracts["season"] == season]
    return {row["player_name"]: row for _, row in szn.iterrows()}


# ── Standalone: generate active_contracts_by_season.csv ──────────────────────

if __name__ == "__main__":
    print("Loading contracts and stats names...")
    contracts = load_contracts()
    name_to_id = load_stats_names()
    print(f"  Contracts: {len(contracts)}")
    print(f"  Stats names: {len(name_to_id)}")

    print("\nBuilding active contracts by season...")
    active = build_active_contracts(contracts, name_to_id)
    print(f"  Active contract-season rows: {len(active)}")

    # Coverage report
    dw = pd.read_csv(os.path.join(OUTPUT, "v5_daily_war.csv"),
                     usecols=["player_id", "player_name", "season", "GP"])
    for szn in sorted(active["season"].unique()):
        dw_szn = dw[dw["season"] == szn]
        active_szn = active[active["season"] == szn]
        matched_ids = set(active_szn["player_id"]) & set(dw_szn["player_id"])
        total = len(dw_szn)
        pct = len(matched_ids) / total * 100 if total else 0
        print(f"  {szn}-{str(szn+1)[-2:]}: {len(matched_ids)}/{total} players matched ({pct:.0f}%)")

    # Spot checks
    print("\nSpot checks:")
    for name, expected_szn, expected_cap in [
        ("Thomas Harley", 2025, 4_000_000),
        ("Connor McDavid", 2025, 12_500_000),
        ("Nikita Kucherov", 2025, 9_500_000),
    ]:
        row = active[(active["player_name"] == name) & (active["season"] == expected_szn)]
        if len(row):
            actual = row.iloc[0]["cap_hit"]
            status = "OK" if actual == expected_cap else f"WRONG (got ${actual:,.0f})"
            print(f"  {name} {expected_szn}: ${actual:,.0f} — {status}")
        else:
            print(f"  {name} {expected_szn}: NOT FOUND")

    # Check for previously unmatched players
    print("\nPreviously unmatched players (2025 season):")
    for name in ["Macklin Celebrini", "Josh Doan", "Ben Kindel", "Oskar Back", "Alex Petrovic"]:
        row = active[(active["player_name"] == name) & (active["season"] == 2025)]
        if len(row):
            print(f"  {name}: ${row.iloc[0]['cap_hit']:,.0f}")
        else:
            print(f"  {name}: still missing")

    # Save
    out_path = os.path.join(BASE, "contracts", "active_contracts_by_season.csv")
    active.drop(columns=["sign_date"], errors="ignore").to_csv(out_path, index=False)
    print(f"\nSaved {len(active)} rows to {out_path}")
