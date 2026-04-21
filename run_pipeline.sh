#!/bin/bash
# run_pipeline.sh — Full NHL RAPM pipeline (all stages needed for dashboard).
#
# Usage:
#   bash run_pipeline.sh            # full rebuild
#   bash run_pipeline.sh --skip-scrape   # skip NHL API scraper (use existing data)
#   bash run_pipeline.sh --contracts-only # run only the contracts sub-pipeline
#
# Stage map:
#   1  scraper/scrape_nhl.py      → data/raw_pbp_2025.csv
#   2  prep/build_dataset.py      → output/v2_clean_pbp.csv
#   3  rapm/rapm_bayesian.py      → output/v2_rapm_raw.csv  (pass 1: uninformed)
#   4  bpr/box_prior.py           → output/v2_box_prior.csv
#   5  rapm/rapm_bayesian.py      → output/v2_rapm_results.csv  (pass 2: prior-informed)
#   6  rapm/rolling_rapm.py       → output/v3_rolling_rapm.csv
#   7  rapm/pp_pk_rapm.py         → output/pp_rapm.csv
#   8  bpr/learn_weights.py       → output/learned_bpr_weights.json
#   9  bpr/bpm.py                 → output/v4_bpm_player_seasons.csv
#   10 bpr/epm.py                 → output/v4_daily_epm.csv
#   11 bpr/gar.py                 → output/v2_gar_by_season.csv
#   12 bpr/composite_v2.py        → output/v2_final_ratings*.csv
#   13 bpr/composite_v4.py        → output/v5_composite_player_seasons.csv
#   14 bpr/carry_forward.py       → output/v6_carry_forward.csv
#   15 bpr/daily.py               → output/v5_daily_ratings_deploy.csv
#   16 bpr/win_shares.py          → output/win_shares_by_season.csv
#   17 team/team_ratings.py       → output/v6_team_season_ratings.csv
#   18 contracts/*                → contracts/surplus_values_v2.csv, fa_projections_2026.csv, etc.
#   19 viz/viz_v2.py              → viz/output_v2/

set -e

VENV=".venv/bin/python"
LOG="pipeline.log"
SKIP_SCRAPE=false
CONTRACTS_ONLY=false

for arg in "$@"; do
  case $arg in
    --skip-scrape)    SKIP_SCRAPE=true ;;
    --contracts-only) CONTRACTS_ONLY=true ;;
  esac
done

log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG"; }
run() { log "▶ $1"; $VENV $2 2>&1 | tee -a "$LOG"; log "✓ $1 done."; }

echo "" >> "$LOG"
log "========== run_pipeline.sh started =========="

if $CONTRACTS_ONLY; then
  log "--- Contracts sub-pipeline only ---"
  run "surplus_model_v2.py"      contracts/surplus_model_v2.py
  run "player_projections.py"    contracts/player_projections.py
  run "predict_contracts.py"     contracts/predict_contracts.py
  run "fa_projections.py"        contracts/fa_projections.py
  run "draft_pick_value.py"      contracts/draft_pick_value.py
  run "npv_model.py"             contracts/npv_model.py
  run "trade_market_values.py"   contracts/trade_market_values.py
  run "trade_evaluator.py"       contracts/trade_evaluator.py
  log "========== Contracts pipeline complete =========="
  exit 0
fi

# ── Stage 1: Scrape ───────────────────────────────────────────────────────────
if $SKIP_SCRAPE; then
  log "Skipping scraper (--skip-scrape)"
else
  run "scrape_nhl.py (append)"   "scraper/scrape_nhl.py --append --skip-shots"
fi

# ── Stage 2–5: Core RAPM ─────────────────────────────────────────────────────
run "build_dataset.py"           prep/build_dataset.py
run "rapm_bayesian.py (raw)"     "rapm/rapm_bayesian.py --mode=raw"
run "box_prior.py"               bpr/box_prior.py
run "rapm_bayesian.py (prior)"   "rapm/rapm_bayesian.py --mode=prior"

# ── Stage 6–8: Extended RAPM & Weights ───────────────────────────────────────
run "rolling_rapm.py"            rapm/rolling_rapm.py
run "pp_pk_rapm.py"              rapm/pp_pk_rapm.py
run "learn_weights.py"           bpr/learn_weights.py

# ── Stage 9–11: Component Models ─────────────────────────────────────────────
run "bpm.py"                     bpr/bpm.py
run "epm.py"                     bpr/epm.py
run "gar.py"                     bpr/gar.py

# ── Stage 12–16: Composite & Daily Ratings ───────────────────────────────────
run "composite_v2.py"            bpr/composite_v2.py
run "composite_v4.py"            bpr/composite_v4.py
run "carry_forward.py"           bpr/carry_forward.py
run "daily.py"                   bpr/daily.py
run "win_shares.py"              bpr/win_shares.py

# ── Stage 17: Team Ratings ────────────────────────────────────────────────────
run "team_ratings.py"            team/team_ratings.py

# ── Stage 18: Contracts ───────────────────────────────────────────────────────
log "--- Contracts sub-pipeline ---"
run "surplus_model_v2.py"        contracts/surplus_model_v2.py
run "player_projections.py"      contracts/player_projections.py
run "predict_contracts.py"       contracts/predict_contracts.py
run "fa_projections.py"          contracts/fa_projections.py
run "draft_pick_value.py"        contracts/draft_pick_value.py
run "npv_model.py"               contracts/npv_model.py
run "trade_market_values.py"     contracts/trade_market_values.py
run "trade_evaluator.py"         contracts/trade_evaluator.py

# ── Stage 19: Visualizations ─────────────────────────────────────────────────
run "viz_v2.py"                  viz/viz_v2.py

log "========== Pipeline complete. Key outputs: =========="
log "  output/v2_clean_pbp.parquet / .csv"
log "  output/v2_rapm_results.csv"
log "  output/v6_carry_forward.csv"
log "  output/v5_daily_ratings_deploy.csv"
log "  output/v5_daily_war.csv"
log "  output/win_shares_by_season.csv"
log "  contracts/surplus_values_v2.csv"
log "  contracts/fa_projections_2026.csv"
log "  viz/output_v2/"
log ""
log "To upload outputs to R2:  python scripts/sync_r2.py upload"
log "To launch dashboard:      streamlit run dashboard/streamlit_app.py"
