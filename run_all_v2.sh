#!/bin/bash
set -e

LOG="run_all_v2.log"
VENV=".venv/bin/python"

log() {
    echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG"
}

echo "" >> "$LOG"
log "===== run_all_v2.sh started ====="

log "Step 1/6: build_dataset.py (enriched PBP with zone starts, goalies, penalties)..."
$VENV scripts_2/build_dataset.py 2>&1 | tee -a "$LOG"
log "build_dataset.py done."

log "Step 2/6: rapm_v2.py --mode=raw (uninformed RAPM, training target for box_prior)..."
$VENV scripts_2/rapm_v2.py --mode=raw 2>&1 | tee -a "$LOG"
log "rapm_v2.py --mode=raw done."

log "Step 3/6: box_prior.py (box score prior + R² calibration)..."
$VENV scripts_2/box_prior.py 2>&1 | tee -a "$LOG"
log "box_prior.py done."

log "Step 4/6: rapm_v2.py --mode=prior (prior-informed Bayesian RAPM)..."
$VENV scripts_2/rapm_v2.py --mode=prior 2>&1 | tee -a "$LOG"
log "rapm_v2.py --mode=prior done."

log "Step 5/6: composite.py (final ratings + PP/PK + GAR)..."
$VENV scripts_2/composite.py 2>&1 | tee -a "$LOG"
log "composite.py done."

log "Step 6/6: viz_v2.py (season tables, trajectories, scatter plots)..."
$VENV scripts_2/viz_v2.py 2>&1 | tee -a "$LOG"
log "viz_v2.py done."

log "===== All v2 steps complete. Outputs: ====="
log "  data/v2_clean_pbp.csv"
log "  data/v2_rapm_raw.csv"
log "  data/v2_box_prior.csv + v2_prior_calibration.json"
log "  data/v2_rapm_results.csv + v2_rapm_by_season.csv"
log "  data/v2_final_ratings.csv + v2_final_ratings_by_season.csv"
log "  viz_output_v2/ (PNGs + HTML)"
