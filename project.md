# NHL Player Ratings — Project Documentation

## Overview

This project builds a comprehensive NHL skater rating system covering 10 seasons (2015–2025) across 1,999 players and 8,936 player-seasons. The system is modeled after Evan Miya's **BPR (Box Plus/Minus replacement)** and **DARKO**-style frameworks: two independent signal sources (on-ice regression + box score production) are combined via uncertainty-weighting, then extended to cover power play and penalty kill situations.

The output is a single composite metric — **total_BPR_adj** — that captures a player's value at even strength, on the power play, and on the penalty kill, normalized to league-average deployment so players can be compared regardless of how their coach uses them.

---

## Data Sources

| Source | Content | File |
|--------|---------|------|
| MoneyPuck play-by-play | Game events with player IDs, strength, xGoal | `data/events_*.csv` |
| MoneyPuck skaters | Per-player per-season box score stats | `data/skaters_*.csv` |
| MoneyPuck skaters-by-game | Per-player per-game situational TOI | `data/skaters_by_game.csv` |

All data is publicly available at [moneypuck.com/data.htm](https://moneypuck.com/data.htm).

---

## Pipeline

```
data/events_*.csv  ──►  rapm_v2.py   ──►  data/rapm_results.csv
                                      ──►  data/rapm_by_season.csv

data/skaters_*.csv ──►  box_score.py ──►  data/box_score_ratings.csv

data/events_*.csv  ──►  pp_pk_rapm.py ──►  data/pp_rapm.csv

(all of above)     ──►  blend.py     ──►  data/final_ratings.csv
                                      ──►  data/final_ratings_by_season.csv

(final_ratings)    ──►  viz.py       ──►  viz_output/
```

Run with: `bash run_all.sh`

---

## Step 1: Two-Pass RAPM (`rapm_v2.py`)

### What is RAPM?

**Regularized Adjusted Plus/Minus (RAPM)** isolates each player's individual contribution from a shared outcome by regressing events onto a sparse player design matrix. Every event on ice becomes a row; every player becomes a column.

- **+1** in a player's column when their team is acting (shooting, generating xGoal)
- **−1** in a player's column when their team is defending

The regression coefficient for each player represents their per-event contribution to the outcome, controlling for every other player on the ice simultaneously. Ridge regularization (L2 penalty) handles the multicollinearity that arises when linemates always play together.

The outcome variable is **expected goals (xGoal)** from MoneyPuck's shot model, supplemented with shots on goal, actual goals, and turnovers. These are combined into composite **BPR_O** and **BPR_D** components:

```
BPR_O = 0.50 × xGF_O + 0.22 × SOG_O + 0.15 × GF_O + 0.06 × TO_O − 0.04 × GA_O
BPR_D = 0.50 × xGF_D + 0.22 × SOG_D + 0.15 × GF_D + 0.06 × TO_D − 0.04 × GA_D
BPR   = BPR_O + BPR_D
```

Values are expressed in **per-60-minute** units (scaled by `EVENTS_PER_60 = 90`), so ratings can be compared across players with different ice time.

### Two-Pass Quality Adjustment

Standard RAPM already controls for teammates and opponents by construction, but collinear linemates (players who always appear together) can have their individual contributions conflated — their shared ridge estimate absorbs their combined effect.

**Pass 1** runs standard pooled RAPM. **Pass 2** augments the design matrix with two scalar quality covariates computed per event:

- `acting_quality` — mean Pass 1 BPR_O of the 5 acting-team players
- `defending_quality` — mean Pass 1 BPR_D of the 5 defending-team players

Adding these covariates forces the regression to account for the average quality of a player's linemates and opponents as a control variable, leaving the individual player columns to capture residual differentiation. This is particularly useful for breaking up collinear forward lines (e.g., Crosby's line, McDavid's line) where a single player may otherwise absorb credit for the entire unit.

Per-season fits use pooled Pass 1 ratings for quality covariates (avoiding circular dependency within a season).

### Ridge Regression and Alpha Selection

Alpha (regularization strength) is selected by 5-fold cross-validation over candidates `[100, 500, 1000, 5000, 10000]`. For the pooled model (~8,936 player-seasons of events), typically `alpha ≈ 1000`. Higher alpha pushes all coefficients toward zero; lower alpha risks overfitting to small samples.

---

## Step 2: Box Score Model (`box_score.py`)

The box score model provides an **independent signal** based on individual production stats, without relying on who else was on the ice. It acts as a prior that is especially informative for players with limited on-ice event data.

### Features

From MoneyPuck's skaters dataset, per-60 rates are computed (normalized by TOI/60):

| Feature | Description |
|---------|-------------|
| `G60` | Goals per 60 |
| `A1_60` | Primary assists per 60 |
| `A2_60` | Secondary assists per 60 |
| `SOG60` | Shots on goal per 60 |
| `blocks60` | Blocked shots per 60 |
| `hits60` | Hits per 60 |
| `GA60` | Giveaways per 60 |
| `TO60` | Takeaways per 60 |
| `CF60pct` | Corsi for % (shot attempt share) |

Stats are **position-normalized** by subtracting the position-group mean (F vs D) before regression, so forwards and defensemen are evaluated against their own baselines.

### Model

An ElasticNet regression is fit with pooled RAPM BPR as the target, using cross-validated regularization. The learned coefficients produce:

- `box_O` — offensive box score rating (driven by G60, A1_60, SOG60)
- `box_D` — defensive box score rating (driven by blocks60, CF60pct, TO60)
- `box_BPR` = `box_O + box_D`

The model achieves R² ≈ 0.45–0.60 on held-out player-seasons — capturing about half the variance in RAPM with publicly visible stats alone.

---

## Step 3: PP/PK RAPM (`pp_pk_rapm.py`)

Power play and penalty kill situations require a separate model because 5v5 RAPM only reflects even-strength performance. Special teams can represent 15–20% of a player's ice time and are a significant source of offensive and defensive value.

### Design

A single unified Ridge regression runs on all **unequal-strength events**: 5v4, 4v5, 5v3, 3v5.

For each event, the team with more skaters is designated the **PP team** (+1 columns) and the team with fewer skaters is the **PK team** (−1 columns). The outcome variable is signed:

```
y_signed = +xGoal   if PP team is acting (shooting)
y_signed = −xGoal   if PK team is acting (clearing, shooting)
```

This means:
- A positive coefficient for a PP player → they generate more xGoal on the power play (**PP_O**)
- A positive coefficient for a PK player → their team *suppresses* xGoal against on the PK (**PK_D**)

### Alpha Selection

`RidgeCV` runs over candidates `[5000, 10000, 20000, 50000, 100000]`. CV typically selects `alpha = 5000` for the PP dataset, which is smaller and noisier than the 5v5 dataset.

### Minimum Events Filter

Players with fewer than 300 PP/PK events are excluded to reduce noise from fringe players who may have only a handful of appearances, where ridge cannot adequately regularize.

### Caveats

- PP/PK ratings are **pooled career** only (no per-season PP model due to small per-season sample sizes)
- PK_D is noisier than PP_O because penalty killing is heavily system-driven; a player's PK rating partially reflects their team's system quality
- Some fringe retired players appear in the top PK_D list (e.g., P.A. Parenteau, Matt Carle) — a known artifact of team system effects from limited personal data

---

## Step 4: Blend (`blend.py`)

### DARKO-Style Uncertainty Weighting

The 5v5 RAPM and box score ratings are combined using an **uncertainty-weighted blend** inspired by the DARKO framework. The key insight: RAPM is more accurate with more data, but box score ratings provide stable signal even for players with small samples.

The blend weight is a **sigmoid function of TOI (ice time)**:

```python
rapm_weight(toi) = 1 / (1 + exp(−scale × (toi − midpoint)))
```

**Per-season blend:**
- Midpoint: 700 minutes (half a full starter season)
- At 200 min → RAPM weight ≈ 0.14 (lean heavily on box score)
- At 700 min → RAPM weight = 0.50 (equal blend)
- At 1,400 min → RAPM weight ≈ 0.97 (trust RAPM almost entirely)

**Pooled (career) blend:**
- Midpoint: 2,000 minutes (~2 solid seasons)
- At 500 min → RAPM weight ≈ 0.22
- At 2,000 min → RAPM weight = 0.50
- At 7,000 min → RAPM weight ≈ 0.99

Established veterans with thousands of minutes of career ice time get ratings that are almost entirely RAPM-driven. Rookies and call-up players blend significantly toward the box score prior, preventing large-sample noise from dominating their ratings.

```
final_BPR_O = w × rapm_BPR_O + (1 − w) × box_O
final_BPR_D = w × rapm_BPR_D + (1 − w) × box_D
final_BPR   = final_BPR_O + final_BPR_D
```

### PP/PK Situational Blend

Once 5v5 and PP/PK ratings are available, they are combined into a single composite using each player's actual situational TOI:

```
total_BPR = (toi_5v5 × final_BPR + toi_pp × PP_O + toi_pk × PK_D)
            / (toi_5v5 + toi_pp + toi_pk)
```

This weights situations by how much time a player actually spends in each one. A PP specialist who logs 20% of their ice time on the power play gets significant credit for their PP_O.

### Deployment-Neutral Adjustment (`total_BPR_adj`)

A second composite uses **league-average situational weights** computed from total skater ice time across all players and seasons:

```
League-average weights (approx):
  ES  = 0.845   (even strength)
  PP  = 0.086   (power play)
  PK  = 0.069   (penalty kill)

total_BPR_adj = W_ES × final_BPR + W_PP × PP_O + W_PK × PK_D
```

This is the **preferred comparison metric**. Because every player is evaluated at the same deployment weights, rankings are driven by player quality rather than coaching decisions. A player who logs unusually heavy PP time won't be artificially inflated just because they have more of their minutes in a high-value situation.

---

## Outputs

| File | Description |
|------|-------------|
| `data/rapm_results.csv` | Pooled career RAPM for 1,999 players |
| `data/rapm_by_season.csv` | Per-season RAPM for 8,936 player-seasons |
| `data/box_score_ratings.csv` | Box score model ratings per player-season |
| `data/pp_rapm.csv` | PP_O and PK_D ratings for 1,055 players |
| `data/final_ratings.csv` | Pooled blended composite ratings |
| `data/final_ratings_by_season.csv` | Per-season blended composite ratings |
| `viz_output/` | PNGs + interactive HTML charts |

### Visualization Outputs

| File | Description |
|------|-------------|
| `top10_YYYY.png` | Top 10 skaters per season (5v5 BPR table) |
| `career_trajectories.png` | BPR over time for elite players |
| `bpr_scatter.html` | Interactive BPR_O vs BPR_D, all player-seasons |
| `top10_pp_specialists.png` | Top 10 career PP_O |
| `top10_pk_specialists.png` | Top 10 career PK_D |
| `pp_pk_scatter.html` | Interactive PP_O vs PK_D scatter |

---

## Metric Interpretation Guide

| Metric | What it measures | Units |
|--------|-----------------|-------|
| `BPR_O` | Offensive on-ice impact (5v5) | per-60 xGoal equivalent |
| `BPR_D` | Defensive on-ice impact (5v5) | per-60 xGoal equivalent |
| `final_BPR` | Blended 5v5 overall rating | per-60 xGoal equivalent |
| `rapm_weight` | Fraction of blend coming from RAPM | 0–1 |
| `PP_O` | Power play offensive impact | per-60 (PP events) |
| `PK_D` | Penalty kill defensive impact | per-60 (PK events) |
| `total_BPR` | Composite using player's actual deployment | per-60 overall |
| `total_BPR_adj` | **Composite using league-avg deployment** | per-60 overall |

**Scale reference:** BPR values above +0.50 represent elite players; values near 0.00 are league average; values below −0.30 are replacement-level or worse.

---

## Key Results

### Career Leaders (5v5 BPR, pooled 2015–2025)

| Player | Pos | TOI (min) | BPR_O | BPR_D | final_BPR |
|--------|-----|-----------|-------|-------|-----------|
| Charlie McAvoy | D | 11,809 | +0.336 | +0.628 | **+0.964** |
| Auston Matthews | F | 12,492 | +0.339 | +0.544 | **+0.883** |
| Patrice Bergeron | F | 19,542 | +0.515 | +0.344 | **+0.859** |
| Hampus Lindholm | D | 17,020 | +0.289 | +0.535 | **+0.824** |
| Matthew Tkachuk | F | 11,494 | +0.365 | +0.331 | **+0.696** |

### Career Leaders (total_BPR_adj — overall with PP/PK)

| Player | Pos | 5v5 BPR | PP_O | PK_D | total_BPR_adj |
|--------|-----|---------|------|------|---------------|
| Charlie McAvoy | D | +0.964 | +0.025 | +0.131 | **+0.826** |
| Patrice Bergeron | F | +0.859 | +0.678 | +0.071 | **+0.789** |
| Auston Matthews | F | +0.883 | +0.253 | +0.046 | **+0.771** |
| Hampus Lindholm | D | +0.824 | +0.035 | +0.071 | **+0.704** |
| Connor McDavid | F | +0.622 | +0.932 | +0.179 | **+0.618** |

*Note: McAvoy ranks first overall because his 5v5 RAPM is the highest in the dataset and he has solid (though not elite) PP/PK contributions. Bergeron climbs with the overall composite due to his historically elite PP_O (+0.678).*

### Top PP Specialists (career PP_O)

| Player | Pos | PP_O |
|--------|-----|------|
| Connor McDavid | F | +0.932 |
| Ryan Nugent-Hopkins | F | +0.898 |
| Brayden Point | F | +0.800 |
| Mikko Rantanen | F | +0.685 |
| Patrice Bergeron | F | +0.678 |
| Roope Hintz | F | +0.603 |
| Nikita Kucherov | F | +0.595 |
| Leon Draisaitl | F | +0.588 |

### 2022–23 Season Leaders (total_BPR_adj)

| Player | Pos | final_BPR | PP_O | PK_D | total_BPR_adj |
|--------|-----|-----------|------|------|---------------|
| Charlie McAvoy | D | +1.097 | +0.025 | +0.131 | **+0.938** |
| Auston Matthews | F | +0.992 | +0.253 | +0.046 | **+0.863** |
| Patrice Bergeron | F | +0.941 | +0.678 | +0.071 | **+0.858** |
| Connor McDavid | F | +0.795 | +0.932 | +0.179 | **+0.764** |
| Hampus Lindholm | D | +0.823 | +0.035 | +0.071 | **+0.703** |

---

## Known Limitations

**PK_D noise:** Penalty kill ratings are noisier than PP or 5v5 ratings. Because teams have fewer skaters in structured, coach-driven systems, individual PK performance is hard to disentangle from team system effects. Some retired or role players (e.g., P.A. Parenteau, Matt Carle) appear anomalously high in the PK_D list — this is a known artifact.

**PP/PK is career-pooled:** The PP and PK models only produce career ratings, not per-season ratings. For the per-season composite, each season uses the player's career PP_O and PK_D — which is a reasonable approximation but doesn't capture year-to-year special teams changes.

**Small-sample blending:** The box score blend provides stability for low-TOI players, but players with only a season or two of data may still have ratings that are hard to interpret. The `rapm_weight` column shows how much of the rating comes from RAPM (closer to 1.0 = more trustworthy).

**No goalie adjustment:** RAPM implicitly controls for goalies (they appear in every event for their team), but goalie quality isn't separately modeled. Skater defensive ratings partially reflect the quality of their starting goalie.

**Strength of schedule:** Two-pass quality adjustment partially addresses this, but players who spend their entire career on weak teams may still have slightly inflated defensive ratings.
