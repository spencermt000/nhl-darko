#!/bin/bash
set -e

LOG="run_all.log"
VENV=".venv/bin/python"

log() {
    echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG"
}

echo "" >> "$LOG"
log "===== run_all.sh started ====="

log "Step 1/4: rapm_v2.py (pooled + per-season with two-pass quality adjustment)..."
$VENV scripts/rapm_v2.py 2>&1 | tee -a "$LOG"
log "rapm_v2.py done."

log "Step 2/4: box_score.py (MoneyPuck box scores → production ratings)..."
$VENV scripts/box_score.py 2>&1 | tee -a "$LOG"
log "box_score.py done."

log "Step 3/4: pp_pk_rapm.py (PP/PK RAPM — PP_O and PK_D ratings)..."
$VENV scripts/pp_pk_rapm.py 2>&1 | tee -a "$LOG"
log "pp_pk_rapm.py done."

log "Step 4/5: blend.py (RAPM + box score + PP/PK uncertainty-weighted blend)..."
$VENV scripts/blend.py 2>&1 | tee -a "$LOG"
log "blend.py done."

log "Step 5/5: viz.py (season tables, career trajectories, interactive scatter)..."
$VENV scripts/viz.py 2>&1 | tee -a "$LOG"
log "viz.py done."

log "===== All steps complete. Outputs: ====="
log "  data/rapm_results.csv"
log "  data/rapm_by_season.csv"
log "  data/box_score_ratings.csv"
log "  data/pp_rapm.csv"
log "  data/final_ratings.csv"
log "  data/final_ratings_by_season.csv"
log "  viz_output/ (PNGs + bpr_scatter.html)"
