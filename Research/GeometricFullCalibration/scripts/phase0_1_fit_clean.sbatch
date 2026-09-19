#!/usr/bin/env bash
#SBATCH --job-name=studyAB-p01-fit
#SBATCH --partition=rtx4090
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --array=1-5
#SBATCH --output=/groups/cliron_group/itayab/calibration_research/Research/GeometricFullCalibration/logs/studyAB/p01_fit_%A_%a.out
#SBATCH --error=/groups/cliron_group/itayab/calibration_research/Research/GeometricFullCalibration/logs/studyAB/p01_fit_%A_%a.err

# Phase 0/1 FIT pass: one array task per checkpoint (SLURM_ARRAY_TASK_ID =
# checkpoint seed, 1..5). Clean protocol only -- fits every frozen Phase 0/1
# method on clean train/val, evaluates on clean test, and persists fitted
# state under --fitted_state_dir for the 60 corruption-cell evaluation-only
# jobs to reuse (fit-once/evaluate-many; see
# Experiments/run_unified_benchmark.py::_fit_or_load_state and friends).
# This clean row IS the "clean" condition of "5 checkpoints x (1 clean + 12
# corruption cells)" -- not a duplicate of the earlier timing probe, which
# had no --fitted_state_dir and therefore produced no reusable fitted state.

set -euo pipefail

PROJECT_ROOT=/groups/cliron_group/itayab/calibration_research/Research/GeometricFullCalibration
PYTHON_BIN=/home/itayab/.conda/envs/geo_cuda12/bin/python
RESULTS_DIR=/home/itayab/PyCharmProjects/Research/geometric/GeometricInternalCalibration/GeometricInternalCalibration/aaai_full_experiments/results
SEED="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
FITTED_STATE_DIR="$PROJECT_ROOT/results/studyAB/phase0/fitted_method/checkpoint_seed${SEED}"
OUTPUT_DIR="$PROJECT_ROOT/results/studyAB/phase0/evaluation/checkpoint_seed${SEED}/clean"

mkdir -p "$OUTPUT_DIR" "$FITTED_STATE_DIR"
export PYTHONPATH="$PROJECT_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-$SLURM_JOB_ID"

cd "$PROJECT_ROOT"

{
    printf 'job=%s array_task=%s seed=%s node=%s\n' "$SLURM_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$SEED" "$(hostname)"
    git rev-parse HEAD
    git status --porcelain
    "$PYTHON_BIN" --version
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
} 2>&1 | tee "$OUTPUT_DIR/environment.log"

"$PYTHON_BIN" "$PROJECT_ROOT/Experiments/run_unified_benchmark.py" \
    --dataset cifar100 \
    --model resnet101 \
    --method baseline_cross_entropy \
    --seed "$SEED" \
    --results_dir "$RESULTS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --fitted_state_dir "$FITTED_STATE_DIR" \
    --disable_post_fusion_temperature \
    --disable_rgcl_tail_hybrids \
    --disable_full_vector_geometric_fusion \
    --disable_knn_blend_baseline \
    --disable_kcal_lite_baseline \
    --disable_rgcl_neighbor_correction \
    --disable_kcal_factorial \
    --disable_aar_lightweight \
    --disable_contrastive_beta_sweep \
    --disable_rgcc \
    --disable_gc_tulip \
    --enable_pts_baseline \
    --enable_trust_score_baseline \
    --enable_glad_pi \
    --enable_glad_pi_zero_geometry \
    --enable_mahalanobis_confidence \
    --enable_kcal_baseline \
    2>&1 | tee "$OUTPUT_DIR/run.log"

echo "Fit+clean-eval complete for checkpoint seed${SEED}: $OUTPUT_DIR (fitted state: $FITTED_STATE_DIR)"
