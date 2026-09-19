#!/usr/bin/env bash
# Run full KCal baseline on seeds 20, 22, 23 in a separate output dir,
# then merge the 'kcal' method entry into the original decision_run_v1 summary.
set -uo pipefail

SEEDS=(20 22 23)
DATASET=cifar100
MODEL=resnet18
BASE_RUN=decision_run_v1
KCAL_RUN=decision_run_v1_kcal
JOBS=2
PYTHON="conda run -n geo_cuda12 python"

while [[ $# -gt 0 ]]; do
    case $1 in
        --jobs) JOBS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p logs

_kcal_complete() {
    local seed=$1
    local outdir="results/unified_benchmark/${DATASET}/${MODEL}/seed${seed}/${KCAL_RUN}"
    local method_dir="${outdir}/intermediates/method_outputs/kcal"
    [[ -f "${method_dir}/entry.json" && -f "${method_dir}/probs.npy" ]]
}

run_kcal_seed() {
    local seed=$1
    local logfile="logs/${DATASET}_${MODEL}_seed${seed}_${KCAL_RUN}.log"
    local outdir="results/unified_benchmark/${DATASET}/${MODEL}/seed${seed}/${KCAL_RUN}"

    if _kcal_complete "$seed"; then
        echo "[$(date '+%H:%M:%S')] seed${seed}: KCal already complete, merging."
        merge_seed "$seed"
        return 0
    fi

    echo "[$(date '+%H:%M:%S')] seed${seed}: running full KCal → ${logfile}"
    if $PYTHON Experiments/run_unified_benchmark.py \
        --dataset ${DATASET} --model ${MODEL} --seed ${seed} \
        --enable_kcal_baseline \
        --inner_val_fraction 0.5 \
        --inner_val_seed 123 \
        --output_dir "${outdir}" \
        2>&1 | tee "${logfile}"; then
        echo "[$(date '+%H:%M:%S')] seed${seed}: KCal done, merging into ${BASE_RUN}."
        merge_seed "$seed"
    else
        echo "[$(date '+%H:%M:%S')] seed${seed}: FAILED — check ${logfile}"
    fi
    return 0
}

merge_seed() {
    local seed=$1
    local src_dir="results/unified_benchmark/${DATASET}/${MODEL}/seed${seed}/${KCAL_RUN}"
    local dst_dir="results/unified_benchmark/${DATASET}/${MODEL}/seed${seed}/${BASE_RUN}"
    echo "[$(date '+%H:%M:%S')] seed${seed}: merging KCal entry from ${src_dir} → ${dst_dir}"
    $PYTHON Experiments/merge_kcal_into_run.py \
        --src_dir "${src_dir}" \
        --dst_dir "${dst_dir}"
}

export -f run_kcal_seed _kcal_complete merge_seed
export DATASET MODEL BASE_RUN KCAL_RUN PYTHON

echo "Running full KCal for seeds: ${SEEDS[*]} (JOBS=${JOBS})"

if command -v parallel &>/dev/null; then
    printf '%s\n' "${SEEDS[@]}" | parallel -j "${JOBS}" --line-buffer run_kcal_seed {}
else
    for ((i = 0; i < ${#SEEDS[@]}; i += JOBS)); do
        batch=("${SEEDS[@]:i:JOBS}")
        pids=()
        for seed in "${batch[@]}"; do
            run_kcal_seed "$seed" &
            pids+=($!)
        done
        for pid in "${pids[@]}"; do
            wait "$pid"
        done
    done
fi

echo ""
echo "All seeds finished."
