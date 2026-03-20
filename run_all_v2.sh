#!/bin/bash
set -e

LOG="run_all_v2.log"
VENV=".venv/bin/python"

log() {
    echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG"
}

echo "" >> "$LOG"
log "===== run_all_v2.sh started ====="

log "Step 1/7: build_dataset.py (enriched PBP with zone starts, goalies, penalties)..."
$VENV supporting/build_dataset.py 2>&1 | tee -a "$LOG"
log "build_dataset.py done."

log "Step 2/7: rapm_bayesian.py --mode=raw (uninformed RAPM, training target for box_prior)..."
$VENV rapm/rapm_bayesian.py --mode=raw 2>&1 | tee -a "$LOG"
log "rapm_bayesian.py --mode=raw done."

log "Step 3/7: box_prior.py (box score prior + R² calibration)..."
$VENV bpr/box_prior.py 2>&1 | tee -a "$LOG"
log "box_prior.py done."

log "Step 4/7: rapm_bayesian.py --mode=prior (prior-informed Bayesian RAPM)..."
$VENV rapm/rapm_bayesian.py --mode=prior 2>&1 | tee -a "$LOG"
log "rapm_bayesian.py --mode=prior done."

log "Step 5/7: composite_v2.py (final ratings + PP/PK blend)..."
$VENV bpr/composite_v2.py 2>&1 | tee -a "$LOG"
log "composite_v2.py done."

log "Step 6/7: gar.py (component-level xGAR/GAR/WAR)..."
$VENV bpr/gar.py 2>&1 | tee -a "$LOG"
log "gar.py done."

log "Step 7/7: viz_v2.py (season tables, trajectories, scatter plots)..."
$VENV viz/viz_v2.py 2>&1 | tee -a "$LOG"
log "viz_v2.py done."

log "===== All v2 steps complete. Outputs: ====="
log "  output/v2_clean_pbp.csv"
log "  output/v2_rapm_raw.csv"
log "  output/v2_box_prior.csv + v2_prior_calibration.json"
log "  output/v2_rapm_results.csv + v2_rapm_by_season.csv"
log "  output/v2_final_ratings.csv + v2_final_ratings_by_season.csv"
log "  output/v2_gar_pooled.csv + v2_gar_by_season.csv + v2_goalie_war.csv"
log "  viz/output_v2/ (PNGs + HTML)"
