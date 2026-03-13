# NHL Player Ratings

A comprehensive NHL skater rating system covering 10 seasons (2015–2025) across ~2,000 players. Combines on-ice RAPM (Regularized Adjusted Plus/Minus) with a box score model and special teams ratings into a single composite metric.

See [project.md](project.md) for full methodology, results, and metric interpretation.

---

## Quick Start

```bash
# Install dependencies
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas scikit-learn matplotlib plotly

# Run the full pipeline
bash run_all.sh
```

Outputs land in `data/` (CSVs) and `viz_output/` (PNGs + HTML charts).

---

## Data

Download from [moneypuck.com/data.htm](https://moneypuck.com/data.htm) and place in `data/`:

| File | Description |
|------|-------------|
| `data/events_*.csv` | Play-by-play with on-ice lineups and xGoal |
| `data/skaters_*.csv` | Per-player per-season box score stats |
| `data/skaters_by_game.csv` | Per-player per-game situational TOI |

---

## Scripts

All scripts live in [`scripts/`](scripts/) and are run from the project root (paths like `data/` are relative to root).

| Script | Step | Description |
|--------|------|-------------|
| [`clean_pbp.py`](scripts/clean_pbp.py) | Pre-processing | Filters raw play-by-play to RAPM-relevant events (shots, goals, giveaways, takeaways) and merges on-ice lineups via an as-of join on change events |
| [`build_rapm_dataset.py`](scripts/build_rapm_dataset.py) | Pre-processing | Builds the RAPM design dataset by tracking on-ice player state across each game and producing one row per event with home/away lineup columns |
| [`rapm.py`](scripts/rapm.py) | Modeling (v1) | Original single-pass RAPM: sparse player design matrix + RidgeCV, 5v5 events only, with score-state covariates. Produces BPR_O / BPR_D / BPR per player |
| [`rapm_v2.py`](scripts/rapm_v2.py) | Modeling (v2, **active**) | Two-pass RAPM with quality adjustment. Pass 1 = standard pooled ridge fit. Pass 2 = re-runs with teammate/opponent quality covariates to reduce collinear linemate noise. Outputs pooled and per-season ratings |
| [`box_score.py`](scripts/box_score.py) | Modeling | Box score model: computes per-60 production rates from MoneyPuck skaters data, position-normalizes, and fits an ElasticNet predicting RAPM BPR. Provides an independent signal for low-sample players |
| [`pp_pk_rapm.py`](scripts/pp_pk_rapm.py) | Modeling | Power play and penalty kill RAPM on all unequal-strength events (5v4, 4v5, 5v3, 3v5). Produces career-pooled PP_O and PK_D ratings |
| [`blend.py`](scripts/blend.py) | Blending | Uncertainty-weighted blend of RAPM + box score (sigmoid on TOI). Adds PP/PK via situational TOI-weighted and deployment-neutral (league-average) composites |
| [`viz.py`](scripts/viz.py) | Visualization | Generates top-10 season tables (PNG), career trajectory charts, and interactive BPR + PP/PK scatter plots (HTML) |

### Pipeline order

```
clean_pbp.py  →  build_rapm_dataset.py  →  rapm_v2.py
                                         →  box_score.py
                                         →  pp_pk_rapm.py
                                              └──────────►  blend.py  →  viz.py
```

`run_all.sh` runs steps 3–7 (assumes pre-processing already done).

---

## Outputs

| File | Description |
|------|-------------|
| `data/rapm_results.csv` | Pooled career RAPM for ~2,000 players |
| `data/rapm_by_season.csv` | Per-season RAPM for ~9,000 player-seasons |
| `data/box_score_ratings.csv` | Box score model ratings per player-season |
| `data/pp_rapm.csv` | PP_O and PK_D ratings for ~1,000 players |
| `data/final_ratings.csv` | Pooled blended composite (main output) |
| `data/final_ratings_by_season.csv` | Per-season blended composite |
| `viz_output/top10_YYYY.png` | Top 10 skaters per season |
| `viz_output/career_trajectories.png` | BPR over time for elite players |
| `viz_output/bpr_scatter.html` | Interactive BPR_O vs BPR_D scatter |
| `viz_output/top10_pp_specialists.png` | Top 10 career PP_O |
| `viz_output/top10_pk_specialists.png` | Top 10 career PK_D |
| `viz_output/pp_pk_scatter.html` | Interactive PP_O vs PK_D scatter |

## Key Metrics

| Metric | What it measures |
|--------|-----------------|
| `final_BPR` | Blended 5v5 overall rating (per 60 min) |
| `PP_O` | Power play offensive impact (per 60 PP min) |
| `PK_D` | Penalty kill defensive impact (per 60 PK min) |
| `total_BPR` | Composite weighted by player's actual deployment |
| `total_BPR_adj` | **Composite weighted by league-average deployment** (best for comparisons) |

Values above +0.50 = elite; near 0.00 = league average; below −0.30 = replacement level.
