# NHL Player Ratings — Project Documentation

## Overview

This project builds a comprehensive NHL skater rating system covering 11 seasons (2015–2025) across ~2,000 players and ~9,000 player-seasons. The system is modeled after Evan Miya's **BPR (Box Plus/Minus replacement)** and **DARKO**-style frameworks: multiple independent signal sources (on-ice regression, box score production, on/off impact) are blended via predictive models, then smoothed with a **Bayesian carry-forward** that accumulates information across seasons.

The output is a carry-forward composite — **cf_total** — that captures a player's total value (even strength + special teams) and is designed to be maximally predictive of future performance. The system also produces component-level **GAR/WAR** breakdowns.

---

## Data Sources

| Source | Content | File |
|--------|---------|------|
| MoneyPuck play-by-play | Game events with player IDs, strength, xGoal | `data/raw_pbp.csv` |
| MoneyPuck shots | Shot-level data with xGoal model | `data/shots_2007-2024.csv` |
| MoneyPuck skaters-by-game | Per-player per-game situational stats | `data/skaters_by_game.csv` |
| MoneyPuck player bio | Birth dates, position, metadata | `data/moneypuck_player_bio.csv` |
| RAPM lineup dataset | On-ice lineup data at CHANGE events | `data/rapm_dataset.csv` |

All data is publicly available at [moneypuck.com/data.htm](https://moneypuck.com/data.htm).

---

## Pipeline

```
                                 ┌─────────────────────────────────────────┐
                                 │           DATA PREPARATION              │
                                 └─────────────────────────────────────────┘
  data/raw_pbp.csv ─────► clean_pbp.py ──────► output/v2_clean_pbp.csv
  data/rapm_dataset.csv ─► build_dataset.py ──► (enriched with zone starts, goalies, penalties)
  data/shots_2007-2024.csv ──────────────────► (merged via build_dataset.py)

                                 ┌─────────────────────────────────────────┐
                                 │          CORE RAPM (v2)                 │
                                 └─────────────────────────────────────────┘
  v2_clean_pbp.csv ─────► rapm_bayesian.py ──► output/v2_rapm_raw.csv      (Pass 1: uninformed)
                          (--mode=raw)
                          rapm_bayesian.py ──► output/v2_rapm_results.csv   (Pass 2: prior-informed)
                          (--mode=prior)       output/v2_rapm_by_season.csv

                                 ┌─────────────────────────────────────────┐
                                 │         ROLLING RAPM (v3)               │
                                 └─────────────────────────────────────────┘
  v2_clean_pbp.csv ─────► rolling_rapm.py ──► output/v3_rolling_rapm.csv         (3-season windows)
                                              output/v3_rolling_rapm_latest.csv

                                 ┌─────────────────────────────────────────┐
                                 │       COMPONENT MODELS (v4)             │
                                 └─────────────────────────────────────────┘
  skaters_by_game.csv ──► bpm.py ──────────► output/v4_bpm_player_seasons.csv
  v3_rolling_rapm.csv ──►                    (GV_O/D, OOI_O/D, RAPM_O/D per season)
  pp_rapm.csv ──────────►

  skaters_by_game.csv ──► epm.py ──────────► output/v4_epm_raw_per_game.csv
                                              output/v4_epm_model_O.pkl
                                              output/v4_epm_model_D.pkl

                                 ┌─────────────────────────────────────────┐
                                 │      PREDICTIVE COMPOSITE (v5)          │
                                 └─────────────────────────────────────────┘
  v4_bpm_player_seasons ► composite_v4.py ──► output/v5_composite_player_seasons.csv
  skaters_by_game.csv ──►                     output/v5_season_war.csv

                                 ┌─────────────────────────────────────────┐
                                 │       CARRY-FORWARD (v6)                │
                                 └─────────────────────────────────────────┘
  v5_composite_* ───────► carry_forward.py ─► output/v6_carry_forward.csv
  moneypuck_player_bio ─►
  v2_gar_by_season ─────►

                                 ┌─────────────────────────────────────────┐
                                 │       DAILY RATINGS (v5)                │
                                 └─────────────────────────────────────────┘
  v4_epm_raw_per_game ──► daily.py ─────────► output/v5_daily_ratings.csv
  skaters_by_game.csv ──►                     output/v5_daily_war.csv
  v5_composite_* ───────►

                                 ┌─────────────────────────────────────────┐
                                 │         GAR/WAR (v2)                    │
                                 └─────────────────────────────────────────┘
  v2_rapm_by_season ────► gar.py ───────────► output/v2_gar_by_season.csv
  skaters_by_game.csv ──►                     output/v2_gar_pooled.csv

                                 ┌─────────────────────────────────────────┐
                                 │        TEAM RATINGS (v6)                │
                                 └─────────────────────────────────────────┘
  (roster aggregation) ─► team_ratings.py ──► output/v6_team_game_ratings.csv
                                              output/v6_team_season_ratings.csv

                                 ┌─────────────────────────────────────────┐
                                 │       VISUALIZATION / DASHBOARD         │
                                 └─────────────────────────────────────────┘
  (all outputs) ────────► dashboard/app.py ─► http://127.0.0.1:8050
                          viz_v1.py / viz_v2.py ► PNG tables, HTML scatter
```

Run scripts: `bash run_all.sh` (v1) or `bash run_all_v2.sh` (v2 Bayesian pipeline)

---

## Step 1: Two-Pass Bayesian RAPM (`rapm_bayesian.py`)

### What is RAPM?

**Regularized Adjusted Plus/Minus (RAPM)** isolates each player's individual contribution from a shared outcome by regressing events onto a sparse player design matrix. Every event on ice becomes a row; every player becomes a column.

- **+1** in a player's column when their team is acting (shooting, generating xGoal)
- **−1** in a player's column when their team is defending

The regression coefficient for each player represents their per-event contribution to the outcome, controlling for every other player on the ice simultaneously. Ridge regularization (L2 penalty) handles the multicollinearity that arises when linemates always play together.

The outcome variable is **expected goals (xGoal)** from MoneyPuck's shot model, supplemented with shots on goal, actual goals, turnovers, and giveaways. These are combined into composite **BPR_O** and **BPR_D** components using **learned weights** (LOSO Ridge CV, R²=0.1545 for offense):

```
BPR_O = 1.74 × xGF_O + 0.12 × SOG_O + 0.12 × GF_O + 0.14 × TO_O + 0.08 × GA_O + 1.96 × iFinish
BPR_D = 0.80 × xGF_D + 0.20 × GF_D   (TO/SOG/GA zeroed — low R², zone-usage confounds)
BPR   = BPR_O + BPR_D
```

Values are expressed in **per-60-minute** units (scaled by `EVENTS_PER_60 = 90`), so ratings can be compared across players with different ice time.

### Adaptive Ridge Regularization

Standard ridge applies the same penalty to all players, which over-shrinks stars (who have ample data) and under-shrinks fringe players (who have little). Adaptive ridge scales each player's regularization inversely with their event count:

```
effective_alpha_i = alpha * (n_ref / n_events_i)
```

Implemented via a column-scaling trick: multiply each player's design matrix columns by `sqrt(n_events_i / n_ref)`. High-TOI stars get ~2× less regularization than the median player. Scale range is clamped to [0.30, 3.00] with `n_ref = 2,112` (median skater event count).

This improved offensive LOSO R² from 0.1345 → 0.1545 and EVO correlation with HockeyStats from ~0.75 → 0.84.

### Two-Pass Quality Adjustment

Standard RAPM already controls for teammates and opponents by construction, but collinear linemates (players who always appear together) can have their individual contributions conflated.

**Pass 1** runs standard pooled RAPM (uninformed). **Pass 2** augments the design matrix with two scalar quality covariates computed per event:

- `acting_quality` — mean Pass 1 BPR_O of the 5 acting-team players
- `defending_quality` — mean Pass 1 BPR_D of the 5 defending-team players

This breaks up collinear forward lines (e.g., McDavid's line) where a single player might otherwise absorb credit for the entire unit.

---

## Step 2: Rolling RAPM (`rolling_rapm.py`)

Instead of pooling ALL seasons into one estimate, fits separate Ridge models on overlapping **3-season windows** (e.g., 2015-2017, 2016-2018, ..., 2022-2024). This captures a player's talent at their current level rather than a career average.

Each player-season is matched to the most recent window containing that season. The rolling RAPM values feed into the BPM component model (v4), providing more current RAPM estimates than per-season or career-pooled alternatives.

Uses pre-computed ridge alphas from the full-data fit to skip cross-validation and run fast.

---

## Step 3: Component Models (`bpm.py`)

Three independent signal sources are computed per player-season:

### Goal Value (GV)
Ridge regression from ~25 box-score features → on-ice goals per 60. Captures individual production that translates to on-ice goal generation (offense) and suppression (defense).

**Features:** Individual xGoals, goals, primary assists, shots, finishing talent, defensive actions (blocks, takeaways, giveaways), zone deployment.

### On/Off Impact (OOI)
Ridge regression from the same box-score features → on-ice xGF differential relative to off-ice. Captures a player's effect on team shot quality when they're on vs off the ice.

### RAPM (Rolling)
3-season rolling RAPM values from `rolling_rapm.py`, precision-shrunk before use. Noisy RAPM estimates (high SE) are attenuated toward zero; precise estimates are kept at face value.

All three are per-season, position-aware (with `isD` dummy + zone start controls).

---

## Step 4: Predictive Composite (`composite_v4.py`)

Two predictive Ridge models, each answering a different question:

### Layer 1: Production Value (PV)
Predicts **next-season GV** (individual production). GV dominates this layer — production is stable and self-predictive. This is where Kucherov and Kaprizov get credit.

### Layer 2: Impact Value (IV)
Predicts **next-season on-ice impact** (xGF/xGA differentials). OOI and RAPM dominate this layer — on-ice context predicts future system impact. This is where McAvoy and Tkachuk get credit.

### Final Composite
```
composite_O = λ × PV_O + (1-λ) × IV_O    (λ = 0.5)
composite_D = λ × PV_D + (1-λ) × IV_D
composite   = composite_O + composite_D
```

Both layers are trained via **leave-one-season-out cross-validation** with TOI-weighted samples. Output is position-centered (forwards and defensemen each average zero).

---

## Step 5: Carry-Forward (`carry_forward.py`)

This is the key innovation for predictiveness. Without carry-forward, the composite treats each season independently — a player's 2023 rating has zero influence on their 2024 rating. The carry-forward layer adds DARKO-style season-to-season memory.

### Mechanics

For each player-season:

1. **Raw signal** = current season's composite_O/D from composite_v4.py
2. **Prior** = previous season's carry-forward rating × age regression factor
3. **Blend** = `w × raw + (1-w) × prior`, where w = sigmoid(TOI)

```
w(toi) = 1 / (1 + exp(-0.004 × (toi - 900)))
```

At 900 minutes of 5v5 TOI (~full season), the blend is 50/50 between current data and carried-forward prior. Below that, the prior dominates. This means even full-season players retain meaningful career information, rather than resetting each year.

### Age Curve

The carry-forward retention (`BASE_CARRY = 0.85`) is modified by age:

| Age | Factor | Effective Carry |
|-----|--------|-----------------|
| < 22 | 0.70 | 0.60 — rookies change fast |
| 22-24 | 0.85 | 0.72 — still developing |
| 25-30 | 1.00 | 0.85 — peak stability |
| 31-33 | 0.95 | 0.81 — beginning of decline |
| 34+ | 0.85 | 0.72 — don't anchor to peak ratings |

For a player's first season, the prior is league average (0).

### Per-Season Special Teams

The carry-forward system uses **per-season PP/PK rates** from the GAR module (v2_gar_by_season.csv), replacing the pooled career PP_O/PK_D used in earlier pipeline versions.

---

## Step 6: Daily Bayesian Ratings (`daily.py`)

DARKO-style per-game updates across 5 components with exponential decay (halflife=30 games):

| Component | Source | Description |
|-----------|--------|-------------|
| EV_O | XGBoost EPM | 5v5 offensive impact per 60 |
| EV_D | XGBoost EPM | 5v5 defensive impact per 60 |
| PP | Box score (ixG + 0.7×A1) | Power play production above league avg |
| PK | On-ice xGA | Penalty kill suppression above league avg |
| PEN | Drawn - Taken | Penalty drawing/taking value |

Each game updates the posterior: `posterior = (prior_precision × prior + evidence_weight × observation) / total_precision`. Priors come from the composite (EV) and career PP/PK RAPM. Seasons carry over with 85% decay.

---

## Step 7: GAR/WAR (`gar.py`, `composite_v4.py`)

Goals Above Replacement / Wins Above Replacement, with 8 skater components + goalie WAR:

| Component | % of |GAR| | Description |
|-----------|-----------|-------------|
| xEV_O | 26.0% | Even-strength offensive (expected) |
| xEV_D | 9.9% | Even-strength defensive (expected) |
| FINISH_O | 28.7% | Offensive finishing above expected (includes PP finishing) |
| FINISH_D | 13.8% | Defensive finishing suppression |
| PP | 6.5% | Power play value |
| PK | 0.8% | Penalty kill value |
| PEN | 13.4% | Penalty drawing value |
| FO | 0.8% | Faceoff wins → goal value |

```
xGAR = xEV_O + xEV_D + PP + PK + PEN + FO
GAR  = xGAR + FINISH_O + FINISH_D
WAR  = GAR / 6.0    (goals → wins)
```

Position-specific replacement levels at the 17th percentile.

### Individual Finishing (iFinish)

Individual finishing talent (goals minus expected goals, per 60) is Bayesian-shrunk with `k = 80` (capped from the statistically optimal k=312 to avoid over-shrinking for current-season valuation). The shrunk iFinish is multiplied by learned weight W_iFIN=1.96 and added to the FINISH_O component.

### PP Finishing

PP goals minus PP expected goals is added as a counting stat to FINISH_O_GAR, capturing individual PP finishing talent that 5v5-only iFinish misses.

### Goalie WAR (GSAx)

Goalie WAR uses **Goals Saved Above Expected (GSAx)** computed from event-level shot data:

1. For each shot, identify the facing goalie and the shot's xGoal
2. Aggregate per goalie-season: shots faced, goals against, xGA
3. `GSAx = xGA - goals_against` (positive = saved more than expected)
4. League-normalize per season to remove xG model bias
5. Replacement level at 25th percentile of starters (1,000+ shots faced)
6. `GOALIE_WAR = (GSAx_adj - replacement_level) / 6.0`

---

## Predictiveness Testing

Predictive validity was measured via year-over-year r² (how well season N predicts season N+1) across 4,986 player-season pairs (min 400 5v5 TOI).

### YoY Stability (self-prediction)

| Metric | r² | Notes |
|--------|-----|-------|
| **Carry-Forward (v6)** | **0.563** | Best composite metric |
| CF Offensive (cf_O) | 0.722 | Surpasses goals/GP |
| CF Defensive (cf_D) | 0.864 | Near ceiling |
| CF WAR | 0.514 | ~10x improvement over raw WAR |
| Raw Composite (v5) | 0.266 | Without carry-forward |
| OOI_D | 0.902 | Most stable individual component |
| OOI_O | 0.673 | |
| RAPM_O | 0.637 | |
| RAPM_D | 0.661 | |
| GV_O | 0.204 | |
| Raw BPR (total) | 0.008 | Per-season RAPM is very noisy |
| Raw WAR/GAR | 0.054 | Before carry-forward |

**Baselines (counting stats):**

| Stat | r² |
|------|-----|
| xGoals/GP | 0.804 |
| Points/GP | 0.705 |
| Goals/GP | 0.631 |

### Carry-Forward Improvement Summary

| Test | Raw (v5) r² | Carry-Forward (v6) r² | Improvement |
|------|------------|---------------------|-------------|
| YoY Self | 0.266 | **0.563** | **+112%** |
| → Next Goals/GP | 0.142 | **0.195** | +37% |
| → Next Points/GP | 0.154 | **0.210** | +37% |
| → Next xGoals/GP | 0.153 | **0.212** | +38% |
| → Next Composite | 0.266 | **0.278** | +4% |

### By Position

| Position | Raw YoY r² | CF YoY r² | CF → Goals r² | CF → Points r² |
|----------|-----------|----------|--------------|---------------|
| Forwards | 0.274 | **0.593** | 0.241 | 0.248 |
| Defensemen | 0.204 | **0.415** | 0.045 | 0.055 |

### Key Findings

1. **Carry-forward is the single biggest predictor improvement** — more than doubling YoY stability by accumulating career information instead of resetting each season.
2. **CF_O (offensive) at r²=0.722** surpasses goals/GP self-prediction (0.631), making it a genuinely strong offensive talent metric.
3. **Defensive metrics are harder to predict** across the board — CF_D (0.864) is strong but still trails OOI_D (0.902) as an individual component.
4. **Counting stats still beat composite metrics** at predicting themselves (points/GP at 0.705 vs cf_total at 0.563), but that's expected — the composite blends offense and defense, and defensive prediction is inherently noisier.
5. **Raw per-season BPR/WAR/GAR are too noisy** for single-season use (r² < 0.06). The carry-forward fixes this by smoothing across seasons.
6. **Predicting next-season counting stats:** cf_O is the best advanced metric predictor of goals/GP (0.356) and points/GP (0.488), substantially better than GV_O, OOI_O, or RAPM_O alone.

---

## Outputs

### Core Outputs

| File | Description |
|------|-------------|
| `output/v6_carry_forward.csv` | **Primary output:** carry-forward composite per player-season with WAR |
| `output/v5_composite_player_seasons.csv` | Per-season raw composite + PV/IV/GV/OOI/RAPM components |
| `output/v5_season_war.csv` | Season WAR leaderboard (from raw composite) |
| `output/v5_daily_ratings.csv` | Per-player-game smoothed 5-component ratings |
| `output/v5_daily_war.csv` | Per-player-season aggregated daily WAR |

### RAPM Outputs

| File | Description |
|------|-------------|
| `output/v2_rapm_results.csv` | Pooled prior-informed RAPM (~2,000 players) |
| `output/v2_rapm_by_season.csv` | Per-season prior-informed RAPM (~9,000 player-seasons) |
| `output/v2_rapm_raw.csv` | Pooled uninformed RAPM (training target) |
| `output/v3_rolling_rapm.csv` | 3-season rolling window RAPM |
| `output/v3_rolling_rapm_latest.csv` | Most recent window per player |

### Component Outputs

| File | Description |
|------|-------------|
| `output/v4_bpm_player_seasons.csv` | GV, OOI, RAPM components per player-season |
| `output/v4_epm_raw_per_game.csv` | XGBoost per-game EPM predictions |
| `output/v2_gar_by_season.csv` | Component-level GAR (8 skater components) per season |
| `output/v2_gar_pooled.csv` | Career-pooled GAR/WAR |

### Goalie Outputs

| File | Description |
|------|-------------|
| `output/v2_goalie_war_by_season.csv` | Per-season goalie WAR (1,027 goalie-seasons, GSAx-based) |
| `output/v2_goalie_war.csv` | Career-pooled goalie WAR (238 goalies) |

### Dashboard Outputs

| File | Description |
|------|-------------|
| `output/dashboard_skater_war.csv` | Dashboard-ready skater WAR (9,811 rows, 39 cols: WAR components + box score stats + team + GP) |
| `output/dashboard_goalie_war.csv` | Dashboard-ready goalie WAR (1,027 rows: GSAx, sv%, shots faced) |
| `output/dashboard_combined_war.csv` | Combined skater+goalie leaderboard on one WAR scale (10,838 rows) |

### Other Outputs

| File | Description |
|------|-------------|
| `output/v2_final_ratings.csv` | V2 pooled blended ratings |
| `output/v2_final_ratings_by_season.csv` | V2 per-season blended ratings |
| `output/pp_rapm.csv` | Career PP_O and PK_D ratings |
| `output/v6_team_game_ratings.csv` | Team-level per-game ratings |
| `output/v6_team_season_ratings.csv` | Team-level season ratings |

### Analysis Outputs

| File | Description |
|------|-------------|
| `output/predictiveness_yoy_stability.csv` | YoY r² for all metrics |
| `output/predictiveness_cross_matrix.csv` | Cross-metric prediction matrix |
| `output/predictiveness_basic_stats.csv` | Metrics → basic stats prediction |

### Visualization

| File | Description |
|------|-------------|
| `dashboard/app.py` | Interactive Dash explorer (http://127.0.0.1:8050) |
| `viz_output/top10_YYYY.png` | Top 10 skaters per season by BPR |
| `viz_output/career_trajectories.png` | BPR over time for elite players |
| `viz_output/bpr_scatter.html` | Interactive BPR_O vs BPR_D scatter |

---

## Metric Interpretation Guide

| Metric | What it measures | Units | Stability (YoY r²) |
|--------|-----------------|-------|---------------------|
| `cf_total` | **Carry-forward total value** | per-60 composite | 0.563 |
| `cf_O` | Carry-forward offensive | per-60 composite | 0.722 |
| `cf_D` | Carry-forward defensive | per-60 composite | 0.864 |
| `composite` | Raw single-season total value | per-60 composite | 0.266 |
| `composite_O` | Raw offensive composite | per-60 composite | 0.477 |
| `composite_D` | Raw defensive composite | per-60 composite | 0.746 |
| `GV_O` / `GV_D` | Goal Value (box score → on-ice goals) | per-60 | 0.204 / 0.331 |
| `OOI_O` / `OOI_D` | On/Off Impact (box score → relative xGF) | per-60 | 0.673 / 0.902 |
| `RAPM_O` / `RAPM_D` | Rolling 3-season RAPM | per-60 | 0.637 / 0.661 |
| `BPR_O` / `BPR_D` | Per-season RAPM composite | per-60 | 0.019 / 0.014 |
| `PP_O` | Power play offensive impact | per-60 (PP events) | — |
| `PK_D` | Penalty kill defensive impact | per-60 (PK events) | — |
| `WAR` | Wins Above Replacement | wins | 0.514 (CF) |

**Scale reference:** Composite values above +0.50 represent elite players; values near 0.00 are league average; values below −0.30 are replacement-level or worse.

---

## Validation

### GAR/WAR Validation
- **Team wins prediction:** Skater WAR R²=0.277, adding Goalie WAR R²=0.389
- **YoY player WAR stability:** r=0.585 (F: 0.609, D: 0.519)
- **Component YoY stability:** FO (0.71), PK (0.90), xEV_O (0.56), PP (0.48), FINISH_O (0.36), FINISH_D (0.09)
- **No position bias:** F mean WAR=1.34, D mean WAR=1.28

### Comparison to HockeyStats WAR (2024-25 season)
- **WAR correlation:** 0.778
- **EVO correlation:** 0.840 (near-zero bias — well calibrated)
- **Shoot correlation:** 0.853 (mean delta -0.43, structural xG model difference)
- **Overall mean delta:** +0.63 WAR (ours runs slightly higher)

| Player | Ours | HockeyStats | Δ |
|--------|------|-------------|---|
| MacKinnon | 6.53 | 6.86 | -0.33 |
| Draisaitl | 6.18 | 5.35 | +0.83 |
| McDavid | 6.92 | 5.05 | +1.87 |
| Kucherov | 6.14 | 4.93 | +1.21 |
| Robertson | 4.95 | 4.78 | +0.17 |
| Caufield | 4.62 | 4.51 | +0.11 |

---

## Known Limitations

**Defensive prediction ceiling:** Defensive value is inherently harder to isolate and predict than offensive value. CF_D (r²=0.864) is strong but benefits heavily from carry-forward smoothing — the raw defensive composite (0.746) is more volatile than offense.

**Counting stats baseline:** Simple counting stats like points/GP (r²=0.705) still beat cf_total (0.563) at self-prediction. This is partly structural — the composite tries to capture offense AND defense, and the defensive side adds noise. The offensive component (cf_O at 0.722) does surpass goals/GP (0.631).

**Per-season RAPM is noisy:** Raw per-season BPR has an r² of just 0.008 year-over-year — essentially noise. This is why the carry-forward and rolling windows are critical. Single-season RAPM should not be used as a standalone metric.

**FINISH_D is noisy:** YoY r=0.09 — essentially random. Candidate for removal or rework.

**Shoot WAR gap:** Our Shoot WAR averages ~0.43 lower than HockeyStats, attributed to differences in the underlying xG model (MoneyPuck vs whatever HS uses). An optimal scale factor of ~1.45× suggests HS's xG model is more conservative.

**First-season players:** Players in their first season have a prior of league average (0), so their carry-forward is entirely driven by current-season data. Draft position or junior stats could improve first-year priors.

**Age curve is approximate:** The age regression factors are hand-tuned, not empirically optimized. A data-driven age curve (fit to the actual YoY changes by age) could improve the carry-forward further.
