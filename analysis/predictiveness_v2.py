"""
Predictiveness Test v2 — Compare carry-forward vs raw composite.

Focused comparison: does the carry-forward actually improve YoY prediction
of both itself and basic stats?
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(BASE, 'output')
DATA = os.path.join(BASE, 'data')

MIN_TOI = 400

# ── Load ─────────────────────────────────────────────────────────────────────
print("Loading data...")

# V5 raw composite
comp = pd.read_csv(f'{OUTPUT}/v5_composite_player_seasons.csv')

# V6 carry-forward
cf = pd.read_csv(f'{OUTPUT}/v6_carry_forward.csv')

# Box scores for basic stats
box_cols = ['playerId', 'season', 'situation', 'icetime',
            'I_F_goals', 'I_F_primaryAssists', 'I_F_secondaryAssists',
            'I_F_points', 'I_F_shotsOnGoal', 'I_F_xGoals']
box = pd.read_csv(f'{DATA}/skaters_by_game.csv', usecols=box_cols)
box_all = box[box['situation'] == 'all']
box_season = box_all.groupby(['playerId', 'season']).agg(
    GP=('icetime', 'count'),
    goals=('I_F_goals', 'sum'),
    assists_total=('I_F_primaryAssists', lambda x: x.sum() +
                   box_all.loc[x.index, 'I_F_secondaryAssists'].sum()),
    points=('I_F_points', 'sum'),
    shots=('I_F_shotsOnGoal', 'sum'),
    xGoals=('I_F_xGoals', 'sum'),
).reset_index()
box_season.rename(columns={'playerId': 'player_id'}, inplace=True)
for col in ['goals', 'points', 'shots', 'xGoals']:
    box_season[f'{col}_pg'] = box_season[col] / box_season['GP']

# ── Build master ─────────────────────────────────────────────────────────────
print("Building master dataset...")

master = cf[['player_id', 'player_name', 'position', 'season', 'GP', 'toi_min',
             'curr_O', 'curr_D', 'cf_O', 'cf_D', 'cf_total',
             'prior_O', 'prior_D', 'data_weight_O', 'WAR']].copy()
cf['curr_total'] = cf['curr_O'] + cf['curr_D']
master['curr_total'] = master['curr_O'] + master['curr_D']

# Add original composite components
comp_cols = ['player_id', 'season', 'composite_O', 'composite_D', 'composite',
             'PV_O', 'PV_D', 'IV_O', 'IV_D', 'GV_O', 'GV_D',
             'OOI_O', 'OOI_D', 'RAPM_O', 'RAPM_D']
master = master.merge(comp[comp_cols], on=['player_id', 'season'], how='left')

# Add basic stats
master = master.merge(box_season[['player_id', 'season', 'goals_pg', 'points_pg',
                                   'shots_pg', 'xGoals_pg']],
                      on=['player_id', 'season'], how='left')

master = master[master['toi_min'] >= MIN_TOI].copy()
print(f"Master: {len(master):,} player-seasons")

# ── YoY pairs ────────────────────────────────────────────────────────────────
curr = master.copy()
nxt = master.copy()
nxt['season'] = nxt['season'] - 1
nxt = nxt.add_suffix('_next')
nxt.rename(columns={'player_id_next': 'player_id', 'season_next': 'season'}, inplace=True)
pairs = curr.merge(nxt, on=['player_id', 'season'], how='inner')
print(f"YoY pairs: {len(pairs):,}")

# ── Compute correlations ─────────────────────────────────────────────────────

def r2(pairs, col_a, col_b):
    valid = pairs[[col_a, col_b]].dropna()
    if len(valid) < 30:
        return np.nan, 0
    r, _ = stats.pearsonr(valid[col_a], valid[col_b])
    return r**2, len(valid)


# === TEST 1: Self-prediction (YoY stability) ===
print("\n" + "="*80)
print("TEST 1: YoY STABILITY — Does carry-forward predict itself better?")
print("="*80)

metrics = [
    ('composite',  'composite_next',  'Raw Composite (v5)'),
    ('cf_total',   'cf_total_next',   'Carry-Forward (v6)'),
    ('composite_O','composite_O_next', 'Raw Composite O'),
    ('cf_O',       'cf_O_next',       'Carry-Forward O'),
    ('composite_D','composite_D_next', 'Raw Composite D'),
    ('cf_D',       'cf_D_next',       'Carry-Forward D'),
    ('WAR',        'WAR_next',        'Carry-Forward WAR'),
    ('GV_O',       'GV_O_next',       'Goal Value O'),
    ('OOI_O',      'OOI_O_next',      'On/Off Impact O'),
    ('OOI_D',      'OOI_D_next',      'On/Off Impact D'),
    ('RAPM_O',     'RAPM_O_next',     'RAPM O'),
    ('RAPM_D',     'RAPM_D_next',     'RAPM D'),
    ('goals_pg',   'goals_pg_next',   'Goals/GP'),
    ('points_pg',  'points_pg_next',  'Points/GP'),
    ('xGoals_pg',  'xGoals_pg_next',  'xGoals/GP'),
]

print(f"\n{'Metric':<30} {'r²':>8} {'N':>6}")
print("-"*46)
for col_a, col_b, label in metrics:
    r2_val, n = r2(pairs, col_a, col_b)
    if not np.isnan(r2_val):
        print(f"{label:<30} {r2_val:>8.3f} {n:>6}")


# === TEST 2: Predicting next-season basic stats ===
print("\n" + "="*80)
print("TEST 2: PREDICTING NEXT-SEASON BASIC STATS")
print("="*80)

predictors = [
    ('cf_total',    'Carry-Forward (v6)'),
    ('composite',   'Raw Composite (v5)'),
    ('cf_O',        'CF Offensive'),
    ('composite_O', 'Raw Comp Offensive'),
    ('GV_O',        'Goal Value O'),
    ('OOI_O',       'On/Off Impact O'),
    ('RAPM_O',      'RAPM O'),
    ('goals_pg',    'Goals/GP (baseline)'),
    ('points_pg',   'Points/GP (baseline)'),
]

targets = [
    ('goals_pg_next',   'Next Goals/GP'),
    ('points_pg_next',  'Next Points/GP'),
    ('xGoals_pg_next',  'Next xGoals/GP'),
    ('shots_pg_next',   'Next Shots/GP'),
]

for target_col, target_label in targets:
    print(f"\n--- {target_label} ---")
    print(f"{'Predictor':<30} {'r²':>8}")
    print("-"*40)
    rows = []
    for pred_col, pred_label in predictors:
        r2_val, n = r2(pairs, pred_col, target_col)
        if not np.isnan(r2_val):
            rows.append((pred_label, r2_val))
    for label, r2_val in sorted(rows, key=lambda x: -x[1]):
        print(f"{label:<30} {r2_val:>8.3f}")


# === TEST 3: Predicting next-season COMPOSITE (most important) ===
print("\n" + "="*80)
print("TEST 3: PREDICTING NEXT-SEASON COMPOSITE VALUE")
print("="*80)

target_col = 'composite_next'
print(f"\n--- Next-Season Raw Composite ---")
print(f"{'Predictor':<30} {'r²':>8}")
print("-"*40)

all_preds = [
    ('cf_total',    'Carry-Forward (v6)'),
    ('composite',   'Raw Composite (v5)'),
    ('cf_O',        'CF Offensive'),
    ('cf_D',        'CF Defensive'),
    ('composite_O', 'Raw Comp Offensive'),
    ('composite_D', 'Raw Comp Defensive'),
    ('GV_O',        'Goal Value O'),
    ('GV_D',        'Goal Value D'),
    ('OOI_O',       'On/Off Impact O'),
    ('OOI_D',       'On/Off Impact D'),
    ('RAPM_O',      'RAPM O'),
    ('RAPM_D',      'RAPM D'),
    ('PV_O',        'Production Value O'),
    ('PV_D',        'Production Value D'),
    ('IV_O',        'Impact Value O'),
    ('IV_D',        'Impact Value D'),
    ('goals_pg',    'Goals/GP'),
    ('points_pg',   'Points/GP'),
    ('xGoals_pg',   'xGoals/GP'),
    ('shots_pg',    'Shots/GP'),
]

rows = []
for pred_col, pred_label in all_preds:
    r2_val, n = r2(pairs, pred_col, target_col)
    if not np.isnan(r2_val):
        rows.append((pred_label, r2_val))
for label, r2_val in sorted(rows, key=lambda x: -x[1]):
    print(f"{label:<30} {r2_val:>8.3f}")

# Also test cf_total predicting next cf_total
print(f"\n--- Next-Season Carry-Forward ---")
print(f"{'Predictor':<30} {'r²':>8}")
print("-"*40)
rows = []
for pred_col, pred_label in all_preds:
    r2_val, n = r2(pairs, pred_col, 'cf_total_next')
    if not np.isnan(r2_val):
        rows.append((pred_label, r2_val))
for label, r2_val in sorted(rows, key=lambda x: -x[1]):
    print(f"{label:<30} {r2_val:>8.3f}")


# === TEST 4: By position ===
print("\n" + "="*80)
print("TEST 4: CARRY-FORWARD vs RAW — BY POSITION")
print("="*80)

for pos, pos_label in [('F', 'Forwards'), ('D', 'Defensemen')]:
    if pos == 'F':
        pp = pairs[pairs['position'].isin(['C', 'L', 'R', 'F'])]
    else:
        pp = pairs[pairs['position'] == 'D']

    print(f"\n--- {pos_label} (n={len(pp)}) ---")
    print(f"{'Metric':<30} {'YoY r²':>8} {'→Goals r²':>10} {'→Pts r²':>10}")
    print("-"*60)

    for col, label in [('composite', 'Raw Composite (v5)'),
                        ('cf_total', 'Carry-Forward (v6)')]:
        yoy, _ = r2(pp, col, f'{col}_next')
        goals, _ = r2(pp, col, 'goals_pg_next')
        pts, _ = r2(pp, col, 'points_pg_next')
        print(f"{label:<30} {yoy:>8.3f} {goals:>10.3f} {pts:>10.3f}")


# === TEST 5: Improvement summary ===
print("\n" + "="*80)
print("SUMMARY: CARRY-FORWARD IMPROVEMENT")
print("="*80)

comparisons = [
    ('YoY Self',         'composite', 'composite_next', 'cf_total', 'cf_total_next'),
    ('→ Next Goals/GP',  'composite', 'goals_pg_next',  'cf_total', 'goals_pg_next'),
    ('→ Next Points/GP', 'composite', 'points_pg_next', 'cf_total', 'points_pg_next'),
    ('→ Next xGoals/GP', 'composite', 'xGoals_pg_next', 'cf_total', 'xGoals_pg_next'),
    ('→ Next Composite', 'composite', 'composite_next', 'cf_total', 'composite_next'),
]

print(f"\n{'Test':<25} {'Raw r²':>8} {'CF r²':>8} {'Δ':>8} {'Δ%':>8}")
print("-"*60)
for label, raw_a, raw_b, cf_a, cf_b in comparisons:
    r2_raw, _ = r2(pairs, raw_a, raw_b)
    r2_cf, _ = r2(pairs, cf_a, cf_b)
    delta = r2_cf - r2_raw
    pct = (delta / r2_raw * 100) if r2_raw > 0 else 0
    print(f"{label:<25} {r2_raw:>8.3f} {r2_cf:>8.3f} {delta:>+8.3f} {pct:>+7.1f}%")

print("\nDone!")
