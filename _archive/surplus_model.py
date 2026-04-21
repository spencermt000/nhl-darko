"""
Surplus Value Model — Per-season player value vs cap cost.

For each player-season, calculates:
  - Market value: what their production is worth at the UFA $/WAR rate
  - Cap cost: their actual cap hit that season
  - Surplus: market value - cap cost (positive = team-friendly deal)

The $/WAR rate is calibrated each season from the UFA free-agent market.

Usage: python contracts/surplus_model.py
"""

import os
import numpy as np
import pandas as pd
from unicodedata import normalize as _ucnorm
import warnings
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(BASE, "output")

# NHL salary cap by season (official values)
SALARY_CAP = {
    2015: 71_400_000, 2016: 73_000_000, 2017: 75_000_000, 2018: 79_500_000,
    2019: 81_500_000, 2020: 81_500_000, 2021: 81_500_000, 2022: 82_500_000,
    2023: 83_500_000, 2024: 88_000_000, 2025: 95_500_000,
}

# ── 1. Load data ─────────────────────────────────────────────────────────────

contracts = pd.read_csv(os.path.join(BASE, "contracts", "contracts.csv"))
contracts["cap_hit_num"] = (contracts["Cap Hit"]
                            .str.replace("$", "", regex=False)
                            .str.replace(",", "", regex=False)
                            .astype(float))
contracts["cap_pct"] = (contracts["Cap %"]
                        .str.replace("%", "", regex=False)
                        .astype(float))
contracts["term_yr"] = (contracts["Term"]
                        .str.replace("yr", "", regex=False)
                        .astype(int))
contracts["sign_date"] = pd.to_datetime(contracts["Date"], format="%d-%b-%y")

# Filter to skaters
contracts = contracts[contracts["POS"] != "G"].copy()

daily_war = pd.read_csv(os.path.join(OUTPUT, "v5_daily_war.csv"))
skater_war = pd.read_csv(os.path.join(OUTPUT, "dashboard_skater_war.csv"))

# ── 2. Name matching (reuse from predict_contracts.py) ───────────────────────

_contract_name_fixes = {
    "Joshua Norris": "Josh Norris", "John-Jason Peterka": "JJ Peterka",
    "Matthew Beniers": "Matty Beniers", "Janis Jérôme Moser": "J.J. Moser",
    "Mike Matheson": "Michael Matheson", "Cameron York": "Cam York",
    "Artyom Zub": "Artem Zub", "Nicklaus Perbix": "Nick Perbix",
    "Joseph Veleno": "Joe Veleno", "Alexander Petrovic": "Alex Petrovic",
    "Pat Maroon": "Patrick Maroon", "Yegor Zamula": "Egor Zamula",
    "Nicolai Knyzhov": "Nikolai Knyzhov", "Fyodor Svechkov": "Fedor Svechkov",
    "Callan Foote": "Cal Foote", "Zachary Jones": "Zac Jones",
    "Joshua Dunne": "Josh Dunne", "Samuel Poulin": "Sam Poulin",
    "Matthew Stienburg": "Matt Stienburg", "Ronald Attard": "Ronnie Attard",
    "Danny O'Regan": "Daniel O'Regan", "Nikolay Prokhorkin": "Nikolai Prokhorkin",
    "Yegor Korshkov": "Egor Korshkov", "Matthew Savoie": "Matt Savoie",
    "Benjamin Kindel": "Ben Kindel", "Maxim Shabanov": "Max Shabanov",
    "Cameron Lund": "Cam Lund",
}


def _strip_accents(s):
    return _ucnorm("NFD", s).encode("ascii", "ignore").decode("ascii")


_name_to_id = (skater_war.dropna(subset=["player_name"])
               .drop_duplicates("player_name")
               .set_index("player_name")["player_id"].to_dict())
_name_lower = {n.lower(): n for n in _name_to_id}


def resolve_name(contract_name):
    if contract_name in _contract_name_fixes:
        return _contract_name_fixes[contract_name]
    if contract_name in _name_to_id:
        return contract_name
    stripped = _strip_accents(contract_name)
    if stripped in _name_to_id:
        return stripped
    if stripped.lower() in _name_lower:
        return _name_lower[stripped.lower()]
    no_hyph = _strip_accents(contract_name.replace("-", " "))
    if no_hyph in _name_to_id:
        return no_hyph
    return None


contracts["stats_name"] = contracts["Player"].map(resolve_name)
contracts["player_id"] = contracts["stats_name"].map(_name_to_id)
contracts = contracts.dropna(subset=["player_id"]).copy()
contracts["player_id"] = contracts["player_id"].astype(int)

print(f"Contracts matched to stats: {len(contracts)}")

# ── 3. Expand contracts into per-season rows ─────────────────────────────────
# Each contract covers specific seasons. A contract signed in July 2022 with
# 3yr term covers seasons 2023, 2024, 2025 (the 2022-23, 2023-24, 2024-25 seasons).

rows = []
for _, c in contracts.iterrows():
    sign_date = c["sign_date"]
    # First season of the contract
    if sign_date.month >= 7:
        first_season = sign_date.year + 1  # signed in offseason, starts next season
    else:
        first_season = sign_date.year      # signed mid-season, covers current season

    for yr in range(c["term_yr"]):
        season = first_season + yr
        if season not in SALARY_CAP:
            continue  # outside our data range
        rows.append({
            "player_id": c["player_id"],
            "player_name": c["stats_name"],
            "season": season,
            "cap_hit": c["cap_hit_num"],
            "cap_pct": c["cap_pct"],
            "contract_type": c["Level"],
            "sign_status": c["Sign Status"],
            "sign_age": c["Sign Age"],
            "sign_date": sign_date,
            "position": c["POS"],
            "draft_year": c["Draft Year"],
            "draft_round": c["Draft Round"],
        })

contract_seasons = pd.DataFrame(rows)

# If a player has overlapping contracts for the same season (e.g. extension signed
# mid-season), keep the most recent signing
contract_seasons = (contract_seasons
                    .sort_values("sign_date", ascending=False)
                    .drop_duplicates(["player_id", "season"], keep="first"))

print(f"Contract-season rows: {len(contract_seasons)}")

# ── 4. Join to per-season WAR ───────────────────────────────────────────────

war_cols = ["player_id", "season", "GP", "WAR", "WAR_82", "WAR_O", "WAR_D",
            "EV_O_GAR", "EV_D_GAR", "PP_GAR", "PK_GAR", "PEN_GAR",
            "toi_5v5", "toi_pp", "toi_pk"]
war = daily_war[[c for c in war_cols if c in daily_war.columns]].copy()

surplus = contract_seasons.merge(war, on=["player_id", "season"], how="inner")
print(f"After joining WAR data: {len(surplus)}")

# Add salary cap for the season
surplus["salary_cap"] = surplus["season"].map(SALARY_CAP)
surplus["cap_pct_actual"] = surplus["cap_hit"] / surplus["salary_cap"] * 100

# ── 5. Calibrate $/WAR from UFA market ──────────────────────────────────────
# The idea: UFA contracts are the "free market" price. We estimate the market
# rate of $/WAR each season by looking at what UFAs got paid relative to their
# prior-season production.
#
# Approach: for each season, calculate total UFA cap spend and total UFA WAR
# produced, then $/WAR = total_cap_spend / total_WAR. We use a floor of 0 WAR
# for each player (teams don't pay negative for bad players, they just don't sign them).

# Minimum salary by approximate era
MIN_SALARY = {
    2015: 550_000, 2016: 575_000, 2017: 575_000, 2018: 650_000,
    2019: 700_000, 2020: 700_000, 2021: 700_000, 2022: 750_000,
    2023: 775_000, 2024: 775_000, 2025: 775_000,
}

ufa_contracts = surplus[surplus["sign_status"] == "UFA"].copy()

# Calculate $/WAR per season from UFA market
cost_per_war = {}
for season in sorted(surplus["season"].unique()):
    ufa_szn = ufa_contracts[ufa_contracts["season"] == season]
    if len(ufa_szn) < 10:
        continue
    # Total cap spend above minimum (the "premium" being paid for production)
    min_sal = MIN_SALARY.get(season, 750_000)
    total_premium = (ufa_szn["cap_hit"] - min_sal).clip(lower=0).sum()
    total_war = ufa_szn["WAR"].clip(lower=0).sum()
    if total_war > 0:
        cost_per_war[season] = total_premium / total_war
    else:
        cost_per_war[season] = 0

print("\n$/WAR by season (from UFA market):")
for szn, cpw in sorted(cost_per_war.items()):
    cap = SALARY_CAP[szn]
    print(f"  {szn}-{str(szn+1)[-2:]}: ${cpw:>12,.0f} / WAR  "
          f"({cpw/cap*100:.1f}% of cap per WAR)")

# Smooth: use median across all seasons as a fallback, fill gaps
median_cpw = np.median(list(cost_per_war.values()))
for szn in SALARY_CAP:
    if szn not in cost_per_war:
        cost_per_war[szn] = median_cpw

# ── 6. Calculate surplus value ──────────────────────────────────────────────

surplus["cost_per_war"] = surplus["season"].map(cost_per_war)
surplus["min_salary"] = surplus["season"].map(MIN_SALARY)

# Market value = (WAR * $/WAR) + minimum salary
# This represents what a team would pay on the open market for this production
surplus["market_value"] = (surplus["WAR"].clip(lower=0) * surplus["cost_per_war"]
                           + surplus["min_salary"])

# Surplus = market value - cap hit (positive = bargain, negative = overpaid)
surplus["surplus_value"] = surplus["market_value"] - surplus["cap_hit"]
surplus["surplus_pct"] = surplus["surplus_value"] / surplus["salary_cap"] * 100

# Market value as % of cap for comparability
surplus["market_pct"] = surplus["market_value"] / surplus["salary_cap"] * 100

print(f"\n{'='*70}")
print(f"SURPLUS VALUE SUMMARY")
print(f"{'='*70}")
print(f"Total player-seasons: {len(surplus)}")
print(f"Mean surplus: ${surplus['surplus_value'].mean():,.0f} ({surplus['surplus_pct'].mean():.2f}% of cap)")
print(f"Median surplus: ${surplus['surplus_value'].median():,.0f}")

# ── 7. Results by contract type ─────────────────────────────────────────────

print(f"\n{'='*70}")
print(f"SURPLUS BY CONTRACT TYPE")
print(f"{'='*70}")

for ctype in ["ELC", "STD"]:
    sub = surplus[surplus["contract_type"] == ctype]
    if len(sub) == 0:
        continue
    print(f"\n  {ctype} contracts ({len(sub)} player-seasons):")
    print(f"    Mean surplus: ${sub['surplus_value'].mean():>10,.0f}  ({sub['surplus_pct'].mean():+.2f}% of cap)")
    print(f"    Median WAR:   {sub['WAR'].median():.2f}")
    print(f"    Mean cap hit: ${sub['cap_hit'].mean():>10,.0f}")

print(f"\n  By signing status:")
for status in ["RFA", "UFA"]:
    sub = surplus[surplus["sign_status"] == status]
    if len(sub) == 0:
        continue
    print(f"    {status}: mean surplus ${sub['surplus_value'].mean():>10,.0f}  "
          f"({sub['surplus_pct'].mean():+.2f}% of cap)  n={len(sub)}")

# ── 8. Leaderboards ─────────────────────────────────────────────────────────

print(f"\n{'='*70}")
print(f"TOP 30 SURPLUS SEASONS (best value player-seasons)")
print(f"{'='*70}")

top = surplus.nlargest(30, "surplus_value")
print(f"\n{'Player':25s} {'Szn':>7s} {'WAR':>5s} {'Cap Hit':>12s} {'Mkt Value':>12s} {'Surplus':>12s} {'Type':>4s}")
print("-" * 90)
for _, r in top.iterrows():
    print(f"{r['player_name']:25s} {r['season']}-{str(r['season']+1)[-2:]:>2s} "
          f"{r['WAR']:5.2f} {r['cap_hit']:>12,.0f} {r['market_value']:>12,.0f} "
          f"{r['surplus_value']:>+12,.0f} {r['contract_type']:>4s}")

print(f"\n{'='*70}")
print(f"WORST 30 SURPLUS SEASONS (most overpaid player-seasons)")
print(f"{'='*70}")

bottom = surplus.nsmallest(30, "surplus_value")
print(f"\n{'Player':25s} {'Szn':>7s} {'WAR':>5s} {'Cap Hit':>12s} {'Mkt Value':>12s} {'Surplus':>12s} {'Type':>4s}")
print("-" * 90)
for _, r in bottom.iterrows():
    print(f"{r['player_name']:25s} {r['season']}-{str(r['season']+1)[-2:]:>2s} "
          f"{r['WAR']:5.2f} {r['cap_hit']:>12,.0f} {r['market_value']:>12,.0f} "
          f"{r['surplus_value']:>+12,.0f} {r['contract_type']:>4s}")

# ── 9. Best surplus by career (total across all seasons) ─────────────────────

print(f"\n{'='*70}")
print(f"TOP 30 CAREER SURPLUS (total across all seasons in data)")
print(f"{'='*70}")

career = (surplus.groupby(["player_id", "player_name"])
          .agg(total_surplus=("surplus_value", "sum"),
               total_war=("WAR", "sum"),
               total_cap=("cap_hit", "sum"),
               seasons=("season", "count"),
               avg_surplus_pct=("surplus_pct", "mean"))
          .reset_index()
          .sort_values("total_surplus", ascending=False))

print(f"\n{'Player':25s} {'Seasons':>7s} {'WAR':>6s} {'Total Cap':>12s} {'Total Surplus':>14s} {'Avg S%':>7s}")
print("-" * 80)
for _, r in career.head(30).iterrows():
    print(f"{r['player_name']:25s} {int(r['seasons']):>7d} {r['total_war']:>6.1f} "
          f"{r['total_cap']:>12,.0f} {r['total_surplus']:>+14,.0f} {r['avg_surplus_pct']:>+6.2f}%")

# ── 10. ELC surplus (key for draft pick model downstream) ────────────────────

print(f"\n{'='*70}")
print(f"TOP 30 ELC SURPLUS SEASONS")
print(f"{'='*70}")

elc = surplus[surplus["contract_type"] == "ELC"].nlargest(30, "surplus_value")
print(f"\n{'Player':25s} {'Szn':>7s} {'WAR':>5s} {'Cap Hit':>12s} {'Surplus':>12s} {'Draft':>6s}")
print("-" * 80)
for _, r in elc.iterrows():
    draft = f"R{int(r['draft_round'])}" if pd.notna(r["draft_round"]) else "UDFA"
    print(f"{r['player_name']:25s} {r['season']}-{str(r['season']+1)[-2:]:>2s} "
          f"{r['WAR']:5.2f} {r['cap_hit']:>12,.0f} {r['surplus_value']:>+12,.0f} {draft:>6s}")

# ── 11. Save full surplus data ──────────────────────────────────────────────

out_cols = ["player_id", "player_name", "position", "season", "GP",
            "WAR", "WAR_82", "WAR_O", "WAR_D",
            "EV_O_GAR", "EV_D_GAR", "PP_GAR", "PK_GAR", "PEN_GAR",
            "cap_hit", "cap_pct_actual", "salary_cap",
            "cost_per_war", "market_value", "market_pct",
            "surplus_value", "surplus_pct",
            "contract_type", "sign_status", "sign_age",
            "draft_year", "draft_round"]
out = surplus[[c for c in out_cols if c in surplus.columns]].copy()
out = out.sort_values(["season", "surplus_value"], ascending=[True, False])

out_path = os.path.join(BASE, "contracts", "surplus_values.csv")
out.to_csv(out_path, index=False)
print(f"\nSaved {len(out)} player-season surplus values to {out_path}")

# Also save career totals
career_path = os.path.join(BASE, "contracts", "career_surplus.csv")
career.to_csv(career_path, index=False)
print(f"Saved {len(career)} career surplus totals to {career_path}")
