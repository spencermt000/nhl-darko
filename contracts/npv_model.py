"""
Contract NPV Model — Net Present Value of NHL contracts.

Two NPV calculations per contract:
  1. Cost NPV: Present value of future cap hits, discounted by salary cap growth
     (a $10M hit is "cheaper" in future years as the cap grows)
  2. Production NPV: Present value of future WAR production, discounted by
     an empirical age curve (older players produce less)

Contract NPV = Production NPV - Cost NPV
Positive = contract expected to provide surplus value over its life
Negative = contract expected to be underwater

Usage: python contracts/npv_model.py
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(BASE, "output")

SALARY_CAP = {
    2015: 71_400_000, 2016: 73_000_000, 2017: 75_000_000, 2018: 79_500_000,
    2019: 81_500_000, 2020: 81_500_000, 2021: 81_500_000, 2022: 82_500_000,
    2023: 83_500_000, 2024: 88_000_000, 2025: 95_500_000,
}

# Forward cap projections (estimated)
PROJECTED_CAPS = {2026: 104_500_000, 2027: 110_000_000, 2028: 115_000_000,
                  2029: 120_000_000, 2030: 125_000_000, 2031: 130_000_000,
                  2032: 135_000_000, 2033: 140_000_000}
ALL_CAPS = {**SALARY_CAP, **PROJECTED_CAPS}

# Cap growth discount rate (historical average excluding COVID flat years)
CAP_GROWTH_RATE = 0.04  # ~4% annual cap growth

# $/WAR conversion (from surplus model v1)
DOLLARS_PER_WAR = 17_500_000

# ── 1. Build empirical age curve ────────────────────────────────────────────

print("Building empirical age curve...")

cf = pd.read_csv(os.path.join(OUTPUT, "v6_carry_forward.csv"),
                 usecols=["player_id", "season", "age", "WAR_82", "GP"])
cf = cf[cf["GP"] >= 30].copy()  # qualified players only

# WAR_82 by age (smoothed)
age_war = cf.groupby("age")["WAR_82"].agg(["mean", "count"]).reset_index()
age_war = age_war[age_war["count"] >= 20]  # need enough sample
age_war.columns = ["age", "mean_war82", "n"]

# Normalize to peak (age 25-27 average)
peak_war = age_war[(age_war["age"] >= 25) & (age_war["age"] <= 27)]["mean_war82"].mean()
age_war["pct_of_peak"] = age_war["mean_war82"] / peak_war

# Build lookup: age -> fraction of peak WAR expected
age_curve = {}
for _, r in age_war.iterrows():
    age_curve[int(r["age"])] = max(0, r["pct_of_peak"])

# Fill gaps and extend
for age in range(18, 45):
    if age not in age_curve:
        if age < min(age_curve.keys()):
            age_curve[age] = age_curve[min(age_curve.keys())]
        elif age > max(age_curve.keys()):
            age_curve[age] = max(0, age_curve[max(age_curve.keys())] * 0.85)

print("Age curve (WAR_82 as % of peak):")
for age in sorted(age_curve.keys()):
    if 19 <= age <= 40:
        bar = "#" * int(age_curve[age] * 40)
        print(f"  {age:2d}: {age_curve[age]:5.2f}  {bar}")

# ── 2. Load contract data ──────────────────────────────────────────────────

print("\nLoading contracts...")

from unicodedata import normalize as _ucnorm

contracts = pd.read_csv(os.path.join(BASE, "contracts", "contracts.csv"))
contracts["cap_hit_num"] = (contracts["Cap Hit"]
                            .str.replace("$", "", regex=False)
                            .str.replace(",", "", regex=False)
                            .astype(float))
contracts["term_yr"] = (contracts["Term"]
                        .str.replace("yr", "", regex=False)
                        .astype(int))
contracts["sign_date"] = pd.to_datetime(contracts["Date"], format="%d-%b-%y")
contracts = contracts[contracts["POS"] != "G"].copy()

# Use latest contract per player
contracts = contracts.sort_values("sign_date").drop_duplicates("Player", keep="last")

# Get player current WAR_82 from daily_war
daily_war = pd.read_csv(os.path.join(OUTPUT, "v5_daily_war.csv"))
latest_season = daily_war["season"].max()
current_war = daily_war[daily_war["season"] == latest_season][["player_id", "player_name", "WAR_82", "WAR", "GP", "position"]].copy()

# Name matching
skater_war = pd.read_csv(os.path.join(OUTPUT, "dashboard_skater_war.csv"))
_name_to_id = (skater_war.dropna(subset=["player_name"])
               .drop_duplicates("player_name")
               .set_index("player_name")["player_id"].to_dict())

_contract_name_fixes = {
    "Joshua Norris": "Josh Norris", "John-Jason Peterka": "JJ Peterka",
    "Matthew Beniers": "Matty Beniers", "Janis Jérôme Moser": "J.J. Moser",
    "Mike Matheson": "Michael Matheson", "Cameron York": "Cam York",
    "Artyom Zub": "Artem Zub", "Nicklaus Perbix": "Nick Perbix",
    "Joseph Veleno": "Joe Veleno", "Alex Debrincat": "Alex DeBrincat",
    "Dylan Demelo": "Dylan DeMelo",
}
_name_lower = {n.lower(): n for n in _name_to_id}


def resolve_name(name):
    if name in _contract_name_fixes:
        return _contract_name_fixes[name]
    if name in _name_to_id:
        return name
    s = _ucnorm("NFD", name).encode("ascii", "ignore").decode("ascii")
    if s in _name_to_id:
        return s
    if s.lower() in _name_lower:
        return _name_lower[s.lower()]
    return None


contracts["stats_name"] = contracts["Player"].map(resolve_name)
contracts["player_id"] = contracts["stats_name"].map(_name_to_id)
contracts = contracts.dropna(subset=["player_id"]).copy()
contracts["player_id"] = contracts["player_id"].astype(int)

# Merge current WAR
contracts = contracts.merge(current_war[["player_id", "WAR_82", "WAR", "GP"]],
                            on="player_id", how="left")

# Calculate contract timeline
contracts["first_season"] = contracts["sign_date"].apply(
    lambda d: d.year + 1 if d.month >= 7 else d.year
)
contracts["expiry_season"] = contracts["first_season"] + contracts["term_yr"] - 1

# Current season
CURRENT_SEASON = 2025

# Filter to active contracts (not yet expired)
active = contracts[contracts["expiry_season"] >= CURRENT_SEASON].copy()
print(f"Active contracts: {len(active)}")

# ── 3. Calculate NPV for each contract ─────────────────────────────────────

print("\nCalculating NPV...")

results = []

for _, c in active.iterrows():
    player = c["Player"]
    aav = c["cap_hit_num"]
    current_age = int(c["Sign Age"]) + (CURRENT_SEASON - c["first_season"])
    current_war82 = c["WAR_82"] if pd.notna(c["WAR_82"]) else 0
    remaining_years = int(c["expiry_season"] - CURRENT_SEASON + 1)

    if remaining_years <= 0:
        continue

    # ── Cost NPV: discount future cap hits by cap growth ──
    # A $10M hit in year 3 is worth less in "real cap %" terms
    cost_npv = 0
    cost_nominal = 0
    for yr in range(remaining_years):
        discount = (1 + CAP_GROWTH_RATE) ** yr
        cost_npv += aav / discount
        cost_nominal += aav

    # ── Production NPV: project future WAR using age curve ──
    # Start from current WAR_82, decay by age curve ratio year over year
    prod_npv = 0
    prod_nominal = 0
    projected_wars = []

    for yr in range(remaining_years):
        future_age = current_age + yr
        # Project WAR: current WAR_82 * (age_curve[future_age] / age_curve[current_age])
        current_pct = age_curve.get(current_age, 0.5)
        future_pct = age_curve.get(future_age, 0.1)
        if current_pct > 0:
            projected_war = current_war82 * (future_pct / current_pct)
        else:
            projected_war = 0

        war_value = max(0, projected_war) * DOLLARS_PER_WAR
        discount = (1 + CAP_GROWTH_RATE) ** yr
        prod_npv += war_value / discount
        prod_nominal += war_value
        projected_wars.append(round(projected_war, 2))

    contract_npv = prod_npv - cost_npv

    results.append({
        "player": player,
        "position": c["POS"],
        "age": current_age,
        "aav": aav,
        "remaining_yrs": remaining_years,
        "expiry": int(c["expiry_season"]),
        "current_war82": round(current_war82, 2),
        "projected_wars": projected_wars,
        "avg_proj_war": round(np.mean(projected_wars), 2) if projected_wars else 0,
        "cost_nominal": cost_nominal,
        "cost_npv": round(cost_npv),
        "prod_nominal": round(prod_nominal),
        "prod_npv": round(prod_npv),
        "contract_npv": round(contract_npv),
        "npv_per_year": round(contract_npv / max(remaining_years, 1)),
        "contract_type": c["Level"],
        "sign_status": c["Sign Status"],
    })

npv_df = pd.DataFrame(results)
npv_df = npv_df.sort_values("contract_npv", ascending=False)

print(f"NPV computed for {len(npv_df)} active contracts")

# ── 4. Results ──────────────────────────────────────────────────────────────

print(f"\n{'='*90}")
print(f"TOP 30 CONTRACT NPV (best value remaining contracts)")
print(f"{'='*90}\n")

print(f"{'Player':25s} {'Pos':>3s} {'Age':>3s} {'Yrs':>3s} {'AAV':>12s} {'WAR/82':>6s} "
      f"{'Cost NPV':>12s} {'Prod NPV':>12s} {'Contract NPV':>14s}")
print("-" * 100)
for _, r in npv_df.head(30).iterrows():
    print(f"{r['player']:25s} {r['position']:>3s} {r['age']:3d} {r['remaining_yrs']:3d} "
          f"${r['aav']:>11,.0f} {r['current_war82']:>6.2f} "
          f"${r['cost_npv']:>11,.0f} ${r['prod_npv']:>11,.0f} ${r['contract_npv']:>+13,.0f}")

print(f"\n{'='*90}")
print(f"WORST 30 CONTRACT NPV (most underwater contracts)")
print(f"{'='*90}\n")

print(f"{'Player':25s} {'Pos':>3s} {'Age':>3s} {'Yrs':>3s} {'AAV':>12s} {'WAR/82':>6s} "
      f"{'Cost NPV':>12s} {'Prod NPV':>12s} {'Contract NPV':>14s}")
print("-" * 100)
for _, r in npv_df.tail(30).iterrows():
    print(f"{r['player']:25s} {r['position']:>3s} {r['age']:3d} {r['remaining_yrs']:3d} "
          f"${r['aav']:>11,.0f} {r['current_war82']:>6.2f} "
          f"${r['cost_npv']:>11,.0f} ${r['prod_npv']:>11,.0f} ${r['contract_npv']:>+13,.0f}")

# NPV by contract type
print(f"\n{'='*90}")
print("NPV BY CONTRACT TYPE")
print(f"{'='*90}")

for ctype in ["ELC", "STD"]:
    sub = npv_df[npv_df["contract_type"] == ctype]
    if len(sub):
        print(f"\n  {ctype} ({len(sub)} contracts):")
        print(f"    Mean NPV: ${sub['contract_npv'].mean():+,.0f}")
        print(f"    Mean NPV/yr: ${sub['npv_per_year'].mean():+,.0f}")

for status in ["RFA", "UFA"]:
    sub = npv_df[npv_df["sign_status"] == status]
    if len(sub):
        print(f"  {status} ({len(sub)}): mean NPV ${sub['contract_npv'].mean():+,.0f}, "
              f"NPV/yr ${sub['npv_per_year'].mean():+,.0f}")

# Age distribution
print(f"\n{'='*90}")
print("NPV BY AGE GROUP")
print(f"{'='*90}")

for lo, hi, label in [(18, 24, "18-24"), (25, 28, "25-28"), (29, 32, "29-32"), (33, 45, "33+")]:
    sub = npv_df[(npv_df["age"] >= lo) & (npv_df["age"] <= hi)]
    if len(sub):
        print(f"  {label}: n={len(sub):3d}  mean NPV ${sub['contract_npv'].mean():>+12,.0f}  "
              f"mean NPV/yr ${sub['npv_per_year'].mean():>+10,.0f}")

# ── 5. Save ─────────────────────────────────────────────────────────────────

out_cols = ["player", "position", "age", "aav", "remaining_yrs", "expiry",
            "current_war82", "avg_proj_war",
            "cost_nominal", "cost_npv", "prod_nominal", "prod_npv",
            "contract_npv", "npv_per_year",
            "contract_type", "sign_status"]
out = npv_df[out_cols].copy()

out_path = os.path.join(BASE, "contracts", "contract_npv.csv")
out.to_csv(out_path, index=False)
print(f"\nSaved {len(out)} contract NPVs to {out_path}")
