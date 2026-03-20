"""
Predictiveness Test for NHL RAPM Metrics
=========================================
Tests year-over-year stability and predictive validity of all metrics.

Three analyses:
1. YoY Stability: How well does metric_N predict metric_N+1 (same metric)?
2. Cross-Metric: How well does each metric predict other metrics next season?
3. Basic Stats: How well does each metric predict next-season goals, assists, points?
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(BASE, 'output')
DATA = os.path.join(BASE, 'data')

MIN_TOI = 400  # minimum 5v5 TOI (minutes) per season to include

# ─── Load data ───────────────────────────────────────────────────────────────

print("Loading data...")

# V5 composite (main metrics)
comp = pd.read_csv(f'{OUTPUT}/v5_composite_player_seasons.csv')

# V2 final ratings by season (RAPM components)
v2 = pd.read_csv(f'{OUTPUT}/v2_final_ratings_by_season.csv')

# V2 GAR by season
gar = pd.read_csv(f'{OUTPUT}/v2_gar_by_season.csv')

# Box scores - aggregate per player-season for goals/assists/points
print("Aggregating box scores (this may take a moment)...")
box_cols = ['playerId', 'season', 'situation', 'icetime',
            'I_F_goals', 'I_F_primaryAssists', 'I_F_secondaryAssists',
            'I_F_points', 'I_F_shotsOnGoal', 'I_F_xGoals', 'I_F_hits',
            'I_F_takeaways', 'I_F_giveaways', 'I_F_blockedShotAttempts']

box = pd.read_csv(f'{DATA}/skaters_by_game.csv', usecols=box_cols)
# Filter to all situations for counting stats
box_all = box[box['situation'] == 'all'].copy()
box_season = box_all.groupby(['playerId', 'season']).agg(
    GP=('icetime', 'count'),
    goals=('I_F_goals', 'sum'),
    primary_assists=('I_F_primaryAssists', 'sum'),
    secondary_assists=('I_F_secondaryAssists', 'sum'),
    points=('I_F_points', 'sum'),
    shots=('I_F_shotsOnGoal', 'sum'),
    xGoals=('I_F_xGoals', 'sum'),
    hits=('I_F_hits', 'sum'),
    takeaways=('I_F_takeaways', 'sum'),
    blocks=('I_F_blockedShotAttempts', 'sum'),
    toi_total=('icetime', 'sum'),
).reset_index()
box_season.rename(columns={'playerId': 'player_id'}, inplace=True)
box_season['assists'] = box_season['primary_assists'] + box_season['secondary_assists']
# Per-game rates
for col in ['goals', 'assists', 'points', 'shots', 'xGoals', 'hits', 'takeaways', 'blocks']:
    box_season[f'{col}_pg'] = box_season[col] / box_season['GP']

# MoneyPuck uses start-year for season labeling
# V2/V5 also use start-year convention based on exploration
# Just make sure they align

# ─── Build master dataset ────────────────────────────────────────────────────

print("Building master dataset...")

# Merge composite metrics
master = comp[['player_id', 'player_name', 'position', 'season', 'GP', 'toi_min',
               'composite_O', 'composite_D', 'composite',
               'PV_O', 'PV_D', 'IV_O', 'IV_D',
               'GV_O', 'GV_D', 'OOI_O', 'OOI_D',
               'RAPM_O', 'RAPM_D', 'PP_O', 'PK_D']].copy()

# Merge GAR
gar_cols = ['player_id', 'season', 'xEV_O', 'xEV_D', 'FINISH_O', 'FINISH_D',
            'xGAR', 'GAR', 'WAR', 'PP_GAR', 'PK_GAR', 'PEN_GAR']
gar_sub = gar[gar_cols].copy()
master = master.merge(gar_sub, on=['player_id', 'season'], how='left')

# Merge V2 BPR
v2_cols = ['player_id', 'season', 'BPR_O', 'BPR_D', 'BPR', 'xGF_O', 'xGF_D',
           'total_BPR', 'total_BPR_adj']
v2_sub = v2[v2_cols].copy()
master = master.merge(v2_sub, on=['player_id', 'season'], how='left')

# Merge box score stats
master = master.merge(box_season, on=['player_id', 'season'], how='left')

# Filter: minimum TOI
master = master[master['toi_min'] >= MIN_TOI].copy()

print(f"Master dataset: {len(master)} player-seasons, {master['player_id'].nunique()} unique players")
print(f"Seasons: {sorted(master['season'].unique())}")

# ─── Year-over-year pairs ────────────────────────────────────────────────────

# Create N → N+1 pairs
df_curr = master.copy()
df_next = master.copy()
df_next['season'] = df_next['season'] - 1  # shift so season col = previous season
df_next = df_next.add_suffix('_next')
df_next.rename(columns={'player_id_next': 'player_id', 'season_next': 'season'}, inplace=True)

pairs = df_curr.merge(df_next, on=['player_id', 'season'], how='inner')
print(f"Year-over-year pairs: {len(pairs)}")

# ─── Define metric groups ────────────────────────────────────────────────────

advanced_metrics = {
    'composite': 'Composite (total)',
    'composite_O': 'Composite O',
    'composite_D': 'Composite D',
    'BPR': 'BPR (total)',
    'BPR_O': 'BPR O',
    'BPR_D': 'BPR D',
    'WAR': 'WAR',
    'GAR': 'GAR',
    'xGAR': 'xGAR',
    'GV_O': 'Goal Value O',
    'GV_D': 'Goal Value D',
    'OOI_O': 'On/Off Impact O',
    'OOI_D': 'On/Off Impact D',
    'RAPM_O': 'RAPM O',
    'RAPM_D': 'RAPM D',
    'xEV_O': 'xEV O',
    'xEV_D': 'xEV D',
    'PP_O': 'PP Offense',
    'PK_D': 'PK Defense',
    'total_BPR_adj': 'Total BPR (adj)',
}

basic_stats = {
    'goals_pg': 'Goals/GP',
    'assists_pg': 'Assists/GP',
    'points_pg': 'Points/GP',
    'shots_pg': 'Shots/GP',
    'xGoals_pg': 'xGoals/GP',
    'hits_pg': 'Hits/GP',
    'blocks_pg': 'Blocks/GP',
}

all_metrics = {**advanced_metrics, **basic_stats}

# ─── Analysis 1: YoY Stability ──────────────────────────────────────────────

print("\n" + "="*80)
print("ANALYSIS 1: YEAR-OVER-YEAR STABILITY (r² of metric_N → metric_N+1)")
print("="*80)

stability = []
for col, label in all_metrics.items():
    curr_col = col
    next_col = f'{col}_next'
    if curr_col in pairs.columns and next_col in pairs.columns:
        valid = pairs[[curr_col, next_col]].dropna()
        if len(valid) > 30:
            r, p = stats.pearsonr(valid[curr_col], valid[next_col])
            stability.append({'Metric': label, 'col': col, 'r': r, 'r²': r**2,
                            'p_value': p, 'N': len(valid)})

stability_df = pd.DataFrame(stability).sort_values('r²', ascending=False)
print(f"\n{'Metric':<25} {'r':>8} {'r²':>8} {'N':>6}")
print("-" * 50)
for _, row in stability_df.iterrows():
    print(f"{row['Metric']:<25} {row['r']:>8.3f} {row['r²']:>8.3f} {row['N']:>6.0f}")

# ─── Analysis 2: Cross-Metric Prediction ────────────────────────────────────

print("\n" + "="*80)
print("ANALYSIS 2: CROSS-METRIC PREDICTION (r² of metric_A_N → metric_B_N+1)")
print("="*80)

# Focus on key metrics for the cross-prediction matrix
key_metrics = ['composite', 'composite_O', 'composite_D', 'BPR', 'WAR', 'GAR',
               'xGAR', 'GV_O', 'GV_D', 'OOI_O', 'OOI_D', 'RAPM_O', 'RAPM_D',
               'xEV_O', 'xEV_D', 'PP_O', 'PK_D', 'total_BPR_adj']

cross_matrix = pd.DataFrame(index=key_metrics, columns=key_metrics, dtype=float)

for m_curr in key_metrics:
    for m_next in key_metrics:
        next_col = f'{m_next}_next'
        if m_curr in pairs.columns and next_col in pairs.columns:
            valid = pairs[[m_curr, next_col]].dropna()
            if len(valid) > 30:
                r, _ = stats.pearsonr(valid[m_curr], valid[next_col])
                cross_matrix.loc[m_curr, m_next] = round(r**2, 3)

print("\nCross-prediction r² matrix (rows = current season predictor, cols = next season target):")
print(cross_matrix.to_string())

# ─── Analysis 3: Predicting Basic Stats ─────────────────────────────────────

print("\n" + "="*80)
print("ANALYSIS 3: PREDICTING NEXT-SEASON BASIC STATS FROM CURRENT METRICS")
print("="*80)

target_stats = ['goals_pg', 'assists_pg', 'points_pg', 'shots_pg', 'xGoals_pg']

pred_results = []
for adv_col, adv_label in advanced_metrics.items():
    for stat_col in target_stats:
        next_col = f'{stat_col}_next'
        if adv_col in pairs.columns and next_col in pairs.columns:
            valid = pairs[[adv_col, next_col]].dropna()
            if len(valid) > 30:
                r, p = stats.pearsonr(valid[adv_col], valid[next_col])
                pred_results.append({
                    'Predictor': adv_label,
                    'Target': basic_stats.get(stat_col, stat_col),
                    'r': r, 'r²': r**2, 'p_value': p, 'N': len(valid)
                })

pred_df = pd.DataFrame(pred_results)

# Pivot for nice display
for target in ['Goals/GP', 'Assists/GP', 'Points/GP', 'Shots/GP', 'xGoals/GP']:
    subset = pred_df[pred_df['Target'] == target].sort_values('r²', ascending=False)
    print(f"\n--- Predicting Next-Season {target} ---")
    print(f"{'Metric':<25} {'r':>8} {'r²':>8}")
    print("-" * 43)
    for _, row in subset.iterrows():
        print(f"{row['Predictor']:<25} {row['r']:>8.3f} {row['r²']:>8.3f}")

# ─── Analysis 4: Basic stats predicting advanced metrics ────────────────────

print("\n" + "="*80)
print("ANALYSIS 4: BASIC STATS PREDICTING NEXT-SEASON ADVANCED METRICS")
print("="*80)

for adv_col, adv_label in [('composite', 'Composite'), ('WAR', 'WAR'), ('GAR', 'GAR'), ('BPR', 'BPR')]:
    next_col = f'{adv_col}_next'
    if next_col not in pairs.columns:
        continue
    print(f"\n--- Predicting Next-Season {adv_label} ---")
    print(f"{'Predictor':<25} {'r':>8} {'r²':>8}")
    print("-" * 43)
    rows = []
    for stat_col, stat_label in {**basic_stats, **advanced_metrics}.items():
        if stat_col in pairs.columns:
            valid = pairs[[stat_col, next_col]].dropna()
            if len(valid) > 30:
                r, _ = stats.pearsonr(valid[stat_col], valid[next_col])
                rows.append((stat_label, r, r**2))
    for label, r, r2 in sorted(rows, key=lambda x: -x[2]):
        print(f"{label:<25} {r:>8.3f} {r2:>8.3f}")

# ─── Analysis 5: Forward vs Defenseman split ────────────────────────────────

print("\n" + "="*80)
print("ANALYSIS 5: YoY STABILITY BY POSITION (F vs D)")
print("="*80)

for pos, pos_label in [('F', 'Forwards'), ('D', 'Defensemen')]:
    if pos == 'F':
        pos_pairs = pairs[pairs['position'].isin(['C', 'L', 'R', 'F'])]
    else:
        pos_pairs = pairs[pairs['position'] == 'D']

    print(f"\n--- {pos_label} (n={len(pos_pairs)}) ---")
    print(f"{'Metric':<25} {'r':>8} {'r²':>8}")
    print("-" * 43)

    rows = []
    for col, label in advanced_metrics.items():
        next_col = f'{col}_next'
        if col in pos_pairs.columns and next_col in pos_pairs.columns:
            valid = pos_pairs[[col, next_col]].dropna()
            if len(valid) > 20:
                r, _ = stats.pearsonr(valid[col], valid[next_col])
                rows.append((label, r, r**2))
    for label, r, r2 in sorted(rows, key=lambda x: -x[2]):
        print(f"{label:<25} {r:>8.3f} {r2:>8.3f}")

# ─── Save results ────────────────────────────────────────────────────────────

stability_df.to_csv(f'{OUTPUT}/predictiveness_yoy_stability.csv', index=False)
cross_matrix.to_csv(f'{OUTPUT}/predictiveness_cross_matrix.csv')
pred_df.to_csv(f'{OUTPUT}/predictiveness_basic_stats.csv', index=False)
print(f"\nResults saved to {OUTPUT}/predictiveness_*.csv")
print("Done!")
