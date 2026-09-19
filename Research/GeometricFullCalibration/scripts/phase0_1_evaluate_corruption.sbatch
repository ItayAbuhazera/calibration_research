#!/usr/bin/env bash
#SBATCH --job-name=studyAB-p01-eval
#SBATCH --partition=rtx4090
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-59
#SBATCH --output=/groups/cliron_group/itayab/calibration_research/Research/GeometricFullCalibration/logs/studyAB/p01_eval_%A_%a.out
#SBATCH --error=/groups/cliron_group/itayab/calibration_research/Research/GeometricFullCalibration/logs/studyAB/p01_eval_%A_%a.err

# Phase 0/1 EVALUATION-ONLY pass: one array task per (checkpoint, corruption
# cell), 5 checkpoints x 12 cells = 60 tasks (SLURM_ARRAY_TASK_ID 0..59).
# --fitted_state_dir MUST already be populated by phase0_1_fit_clean.sbatch's
# corresponding checkpoint task -- every method's fit step is structurally
# skipped (loads frozen state, or the run aborts; see
# Experiments/run_unified_benchmark.py::_fit_or_load_state and friends). No
# fitting/tuning/hyperparameter selection happens in this script; only the
# frozen test evaluation loader is corrupted.

set -euo pipefail

PROJECT_ROOT=/groups/cliron_group/itayab/calibration_research/Research/GeometricFullCalibration
PYTHON_BIN=/home/itayab/.conda/envs/geo_cuda12/bin/python
RESULTS_DIR=/home/itayab/PyCharmProjects/Research/geometric/GeometricInternalCalibration/GeometricInternalCalibration/aaai_full_experiments/results
CIFAR_C_DIR=/home/itayab/PyCharmProjects/Research/geometric/GeometricInternalCalibration/GeometricInternalCalibration/data/cifar100-c

CORRUPTIONS=(gaussian_noise defocus_blur fog jpeg_compression)
SEVERITIES=(1 3 5)

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
CELL_IDX=$(( TASK_ID % 12 ))
CKPT_IDX=$(( TASK_ID / 12 ))
SEED=$(( CKPT_IDX + 1 ))
CORR_IDX=$(( CELL_IDX / 3 ))
SEV_IDX=$(( CELL_IDX % 3 ))
CORRUPTION="${CORRUPTIONS[$CORR_IDX]}"
SEVERITY="${SEVERITIES[$SEV_IDX]}"

FITTED_STATE_DIR="$PROJECT_ROOT/results/studyAB/phase0/fitted_method/checkpoint_seed${SEED}"
OUTPUT_DIR="$PROJECT_ROOT/results/studyAB/phase0/evaluation/checkpoint_seed${SEED}/${CORRUPTION}_s${SEVERITY}"

if [ ! -d "$FITTED_STATE_DIR" ] || [ -z "$(ls -A "$FITTED_STATE_DIR" 2>/dev/null)" ]; then
    echo "FATAL: fitted_state_dir $FITTED_STATE_DIR is missing or empty -- the fit pass for checkpoint seed${SEED} must complete first." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-$SLURM_JOB_ID"

cd "$PROJECT_ROOT"

{
    printf 'job=%s array_task=%s seed=%s corruption=%s severity=%s node=%s\n' \
        "$SLURM_JOB_ID" "$TASK_ID" "$SEED" "$CORRUPTION" "$SEVERITY" "$(hostname)"
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
    --corruption_type "$CORRUPTION" \
    --corruption_severity "$SEVERITY" \
    --cifar_c_dir "$CIFAR_C_DIR" \
    --disable_post_fusion_temperature \
    --disable_rgcl_tail_hybrids \
    --disable_full_vector_geometric_fusion \
    --disable_knn_blend_baseline \
    --disable_kcal_lite_baseline \
    --disable_rgcl_neighbor_correction \
    --disable_kcal_factorial \
    --disable_aar_lightweight \
    --disable_contrastive_beta_sweep \
    --enable_pts_baseline \
    --enable_trust_score_baseline \
    --enable_glad_pi \
    --enable_glad_pi_zero_geometry \
    --enable_mahalanobis_confidence \
    --enable_kcal_baseline \
    2>&1 | tee "$OUTPUT_DIR/run.log"

echo "Evaluation complete: checkpoint seed${SEED}, ${CORRUPTION} severity=${SEVERITY}: $OUTPUT_DIR"
