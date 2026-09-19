#!/usr/bin/env bash
# Multi-seed parallel benchmark runner for decision_run_v1.
# Runs JOBS seeds at a time (limited by 11 GB VRAM on a single GTX 1080 Ti).
# Skips seeds whose output directory already has all 4 stages completed.
# One failed seed never aborts the remaining batch.
# Usage: bash run_decision_multiseed.sh [--jobs N]
set -uo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
# Seeds with valid model checkpoints (12, 16-19, 30 have no trained model).
# Seed21 is already complete and will be skipped automatically.
SEEDS=(20 22 23 24 25 26 27 28 29 31)
DATASET=cifar100
MODEL=resnet18
RUN_NAME=decision_run_v1
JOBS=2          # parallel seeds; raise to 3 only if VRAM allows
PYTHON="conda run -n geo_cuda12 python"
# ─────────────────────────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case $1 in
        --jobs) JOBS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p logs

_is_complete() {
    local seed=$1
    local progress="results/unified_benchmark/${DATASET}/${MODEL}/seed${seed}/${RUN_NAME}/intermediates/progress.json"
    [[ -f "$progress" ]] || return 1
    python3 - "$progress" <<'PY'
import sys, json
with open(sys.argv[1]) as f:
    p = json.load(f)
required = {"stage1_splits", "stage2_early", "stage3_features_fusion", "stage4_late_outputs"}
done = set(p.get("completed_stages", []))
sys.exit(0 if required.issubset(done) else 1)
PY
}

run_seed() {
    local seed=$1
    local logfile="logs/${DATASET}_${MODEL}_seed${seed}_${RUN_NAME}.log"
    local outdir="results/unified_benchmark/${DATASET}/${MODEL}/seed${seed}/${RUN_NAME}"

    if _is_complete "$seed"; then
        echo "[$(date '+%H:%M:%S')] seed${seed}: already complete, skipping."
        return 0
    fi

    echo "[$(date '+%H:%M:%S')] seed${seed}: starting → ${logfile}"
    if $PYTHON Experiments/run_unified_benchmark.py \
        --dataset ${DATASET} --model ${MODEL} --seed ${seed} \
        --enable_pts_baseline \
        --enable_full_vector_geometric_fusion \
        --fusion_feature_source rgcl \
        --enable_knn_blend_baseline \
        --enable_kcal_lite_baseline --kcal_k_per_class 5 \
        --enable_rgcl_neighbor_correction \
        --enable_trust_score_baseline \
        --enable_aar_lightweight \
        --enable_glad_pi \
        --glad_pi_beta_grid 0,0.01,0.03,0.1,0.3,1,3,10 \
        --glad_pi_nll_tolerance 0.05 \
        --inner_val_fraction 0.5 \
        --inner_val_seed 123 \
        --output_dir "${outdir}" \
        2>&1 | tee "${logfile}"; then
        echo "[$(date '+%H:%M:%S')] seed${seed}: done."
    else
        echo "[$(date '+%H:%M:%S')] seed${seed}: FAILED — check ${logfile}"
    fi
    return 0  # never propagate failure; let other seeds continue
}

export -f run_seed _is_complete
export DATASET MODEL RUN_NAME PYTHON

echo "Running ${#SEEDS[@]} seeds (${SEEDS[*]}) with JOBS=${JOBS}"
echo "Already-complete seeds will be skipped automatically."
echo ""

if command -v parallel &>/dev/null; then
    printf '%s\n' "${SEEDS[@]}" | parallel -j "${JOBS}" --line-buffer run_seed {}
else
    # Manual batching fallback when GNU parallel is unavailable
    for ((i = 0; i < ${#SEEDS[@]}; i += JOBS)); do
        batch=("${SEEDS[@]:i:JOBS}")
        pids=()
        for seed in "${batch[@]}"; do
            run_seed "$seed" &
            pids+=($!)
        done
        for pid in "${pids[@]}"; do
            wait "$pid"
        done
    done
fi

echo ""
echo "All seeds finished."
