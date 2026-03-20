#!/bin/bash
set -e

LOG="run_all.log"
VENV=".venv/bin/python"

log() {
    echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG"
}

echo "" >> "$LOG"
log "===== run_all.sh started ====="

log "Step 1/6: rapm_enhanced.py (pooled + per-season with two-pass quality adjustment)..."
$VENV rapm/rapm_enhanced.py 2>&1 | tee -a "$LOG"
log "rapm_enhanced.py done."

log "Step 2/6: bootstrap_rapm.py (game-level bootstrap SEs for pooled RAPM)..."
$VENV rapm/bootstrap_rapm.py 2>&1 | tee -a "$LOG"
log "bootstrap_rapm.py done."

log "Step 3/6: box_score.py (MoneyPuck box scores → production ratings)..."
$VENV bpr/box_score.py 2>&1 | tee -a "$LOG"
log "box_score.py done."

log "Step 4/6: pp_pk_rapm.py (PP/PK RAPM — PP_O and PK_D ratings)..."
$VENV rapm/pp_pk_rapm.py 2>&1 | tee -a "$LOG"
log "pp_pk_rapm.py done."

log "Step 5/6: blend.py (precision-weighted RAPM + box score + PP/PK blend)..."
$VENV bpr/blend.py 2>&1 | tee -a "$LOG"
log "blend.py done."

log "Step 6/6: viz_v1.py (season tables, career trajectories, interactive scatter)..."
$VENV viz/viz_v1.py 2>&1 | tee -a "$LOG"
log "viz_v1.py done."

log "===== All steps complete. Outputs: ====="
log "  output/rapm_results.csv"
log "  output/rapm_by_season.csv"
log "  output/rapm_bootstrap_se.csv"
log "  output/box_score_ratings.csv"
log "  output/pp_rapm.csv"
log "  output/final_ratings.csv"
log "  output/final_ratings_by_season.csv"
log "  viz/output_v1/ (PNGs + bpr_scatter.html)"
