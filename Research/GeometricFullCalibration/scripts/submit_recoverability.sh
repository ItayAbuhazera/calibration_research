#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/itayab/PyCharmProjects/GeometricFullCalibration
SBATCH_SCRIPT="$PROJECT_ROOT/scripts/run_recoverability_seed.sbatch"
SCHEDULER_LOG_DIR="$PROJECT_ROOT/logs/recoverability/slurm"
mkdir -p "$SCHEDULER_LOG_DIR"

PREFLIGHT_JOB_ID=$(sbatch --parsable \
    --array=1-5%5 \
    --output="$SCHEDULER_LOG_DIR/preflight_%A_%a.out" \
    --error="$SCHEDULER_LOG_DIR/preflight_%A_%a.err" \
    --export=ALL,RUN_PHASE=preflight \
    "$SBATCH_SCRIPT")
printf 'preflight_job_id=%s\n' "$PREFLIGHT_JOB_ID"

CORRUPTION_JOB_ID=$(sbatch --parsable \
    --array=1-5%5 \
    --dependency="afterok:$PREFLIGHT_JOB_ID" \
    --output="$SCHEDULER_LOG_DIR/corruptions_%A_%a.out" \
    --error="$SCHEDULER_LOG_DIR/corruptions_%A_%a.err" \
    --export=ALL,RUN_PHASE=corruptions \
    "$SBATCH_SCRIPT")
printf 'corruption_job_id=%s dependency=afterok:%s\n' \
    "$CORRUPTION_JOB_ID" "$PREFLIGHT_JOB_ID"

