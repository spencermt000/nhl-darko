#!/bin/bash
set -e

LOG="run_all.log"
VENV=".venv/bin/python"

log() {
    echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG"
}

echo "" >> "$LOG"
log "===== run_all.sh started ====="

log "Step 1/6: rapm_v2.py (pooled + per-season with two-pass quality adjustment)..."
$VENV scripts/rapm_v2.py 2>&1 | tee -a "$LOG"
log "rapm_v2.py done."

log "Step 2/6: bootstrap_rapm.py (game-level bootstrap SEs for pooled RAPM)..."
$VENV scripts/bootstrap_rapm.py 2>&1 | tee -a "$LOG"
log "bootstrap_rapm.py done."

log "Step 3/6: box_score.py (MoneyPuck box scores → production ratings)..."
$VENV scripts/box_score.py 2>&1 | tee -a "$LOG"
log "box_score.py done."

log "Step 4/6: pp_pk_rapm.py (PP/PK RAPM — PP_O and PK_D ratings)..."
$VENV scripts/pp_pk_rapm.py 2>&1 | tee -a "$LOG"
log "pp_pk_rapm.py done."

log "Step 5/6: blend.py (precision-weighted RAPM + box score + PP/PK blend)..."
$VENV scripts/blend.py 2>&1 | tee -a "$LOG"
log "blend.py done."

log "Step 6/6: viz.py (season tables, career trajectories, interactive scatter)..."
$VENV scripts/viz.py 2>&1 | tee -a "$LOG"
log "viz.py done."

log "===== All steps complete. Outputs: ====="
log "  data/rapm_results.csv"
log "  data/rapm_by_season.csv"
log "  data/rapm_bootstrap_se.csv"
log "  data/box_score_ratings.csv"
log "  data/pp_rapm.csv"
log "  data/final_ratings.csv"
log "  data/final_ratings_by_season.csv"
log "  viz_output/ (PNGs + bpr_scatter.html)"
