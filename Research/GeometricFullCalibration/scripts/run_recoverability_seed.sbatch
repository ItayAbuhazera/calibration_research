#!/usr/bin/env bash
#SBATCH --job-name=rgc-recover
#SBATCH --partition=rtx4090
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00

set -euo pipefail

PROJECT_ROOT=/home/itayab/PyCharmProjects/GeometricFullCalibration
DATA_WORKTREE=/home/itayab/PyCharmProjects/geometric/GeometricInternalCalibration/GeometricInternalCalibration
PYTHON_BIN=/home/itayab/.conda/envs/geo_cuda12/bin/python
CIFAR_C_DIR="$DATA_WORKTREE/data/cifar100-c"
SEED="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
PHASE="${RUN_PHASE:?RUN_PHASE must be preflight or corruptions}"
CHECKPOINT="$DATA_WORKTREE/aaai_full_experiments/results/baseline/baseline_cross_entropy/cifar100/resnet101/seed${SEED}/best_model.pth"
LOG_DIR="$PROJECT_ROOT/logs/recoverability/seed${SEED}"
RESULT_DIR="$PROJECT_ROOT/results/seed${SEED}"

mkdir -p "$LOG_DIR" "$RESULT_DIR"

TASK_TMP="${SLURM_TMPDIR:-/tmp}/rgc-shift-${SLURM_JOB_ID}-${SEED}"
mkdir -p "$TASK_TMP"
unzip -q "$PROJECT_ROOT/rgc-shift.zip" -d "$TASK_TMP"
RGC_SHIFT_ROOT="$TASK_TMP/rgc-shift"
export PYTHONPATH="$PROJECT_ROOT:$RGC_SHIFT_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$TASK_TMP/matplotlib"

cd "$DATA_WORKTREE"

{
    printf 'phase=%s seed=%s job=%s\n' "$PHASE" "$SEED" "$SLURM_JOB_ID"
    "$PYTHON_BIN" --version
    "$PYTHON_BIN" -c 'import sklearn, torch; print("torch=" + torch.__version__); print("cuda=" + str(torch.version.cuda)); print("sklearn=" + sklearn.__version__)'
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
} 2>&1 | tee "$LOG_DIR/${PHASE}_environment.log"

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/export_recoverability.py" \
    --phase "$PHASE" \
    --seed "$SEED" \
    --checkpoint "$CHECKPOINT" \
    --cifar-c-dir "$CIFAR_C_DIR" \
    --export-root "$PROJECT_ROOT/exports" \
    --artifact-root "$PROJECT_ROOT/artifacts/recoverability" \
    --harness-script "$RGC_SHIFT_ROOT/scripts/run_recoverability.py" \
    --batch-size 128 \
    --num-workers 4 \
    --device cuda \
    2>&1 | tee "$LOG_DIR/${PHASE}_export.log"

if [[ "$PHASE" == "corruptions" ]]; then
    for ARM in perclass globalknn; do
        mapfile -t CELLS < <(find "$PROJECT_ROOT/exports/seed${SEED}/${ARM}" -maxdepth 1 -type f -name '*.npz' -print | sort)
        if [[ "${#CELLS[@]}" -ne 14 ]]; then
            printf 'expected 14 cells for seed=%s arm=%s, found %s\n' "$SEED" "$ARM" "${#CELLS[@]}" >&2
            exit 1
        fi

        "$PYTHON_BIN" "$RGC_SHIFT_ROOT/scripts/run_recoverability.py" \
            --cells "${CELLS[@]}" \
            --clean-name val \
            --alpha-metric nll \
            --out "$RESULT_DIR/recoverability_${ARM}.json" \
            2>&1 | tee "$LOG_DIR/recoverability_${ARM}_primary_valfit.log"

        "$PYTHON_BIN" "$RGC_SHIFT_ROOT/scripts/run_recoverability.py" \
            --cells "${CELLS[@]}" \
            --clean-name clean \
            --alpha-metric nll \
            --out "$RESULT_DIR/recoverability_${ARM}_mechanism_cleanfit.json" \
            2>&1 | tee "$LOG_DIR/recoverability_${ARM}_mechanism_cleanfit.log"
    done
fi
