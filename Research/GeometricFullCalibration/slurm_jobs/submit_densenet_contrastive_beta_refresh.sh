#!/usr/bin/env bash
# Submit contrastive-beta refresh jobs for all densenet121 CIFAR-100 seeds.
# Uses the fast-path in run_unified_benchmark.py that skips all pipeline stages
# and only computes the 4 contrastive-beta methods from existing cached files.
# Submits to the CPU partition — no GPU needed.
set -euo pipefail

PROJECT_ROOT="/home/itayab/PyCharmProjects/GeometricFullCalibration"
RESULTS_DIR="/home/itayab/PyCharmProjects/geometric/GeometricInternalCalibration/GeometricInternalCalibration/aaai_full_experiments/results"
LOG_DIR="${PROJECT_ROOT}/slurm_jobs/_unified_benchmark_logs"
CONDA_ENV="geo_cuda12"
PYTHON="/home/itayab/.conda/envs/${CONDA_ENV}/bin/python"

SEEDS=(0 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40)

mkdir -p "${LOG_DIR}"

submitted=0
skipped=0

for seed in "${SEEDS[@]}"; do
    output_dir="${PROJECT_ROOT}/results/unified_benchmark_c100_densenet121_seed${seed}_rankgeom_mix"

    # Only submit if summary_metrics.json exists (run completed) but contrastive beta checkpoint is missing
    if [ ! -f "${output_dir}/summary_metrics.json" ]; then
        echo "SKIP seed ${seed}: no summary_metrics.json"
        ((skipped++)) || true
        continue
    fi
    if [ -d "${output_dir}/intermediates/method_outputs/contrastive_beta_ts_post_temperature" ]; then
        echo "SKIP seed ${seed}: contrastive beta already complete"
        ((skipped++)) || true
        continue
    fi

    job_name="ubm_densenet121_cifar100_s${seed}_cbeta"
    log_file="${LOG_DIR}/${job_name}_%j.out"

    cmd="${PYTHON} ${PROJECT_ROOT}/Experiments/run_unified_benchmark.py \
  --dataset cifar100 --model densenet121 --method baseline_cross_entropy \
  --seed ${seed} \
  --results_dir ${RESULTS_DIR} \
  --output_dir ${output_dir} \
  --batch_size 128 --target_dimension 256 --num_layers 6 --num_coordinates 256 \
  --device cpu \
  --enable_rgcl_tail_hybrids --rgcl_tail_sources base,temperature_scaling,vector_scaling,dirichlet \
  --enable_contrastive_beta_sweep \
  --enable_full_vector_geometric_fusion"

    sbatch \
        --partition=cpu \
        --job-name="${job_name}" \
        --output="${log_file}" \
        --time=00:30:00 \
        --ntasks=1 \
        --cpus-per-task=4 \
        --mem=16G \
        --wrap="
export PYTHONPATH='${PROJECT_ROOT}:\${PYTHONPATH:-}'
cd ${PROJECT_ROOT}
echo '========================================'
echo \"Job ID       : \$SLURM_JOB_ID\"
echo \"Host         : \$(hostname)\"
echo \"Start Time   : \$(date)\"
echo \"Seed         : ${seed}\"
echo '----------------------------------------'
echo 'CMD: ${cmd}'
${cmd}
exit_code=\$?
echo '----------------------------------------'
echo \"End Time     : \$(date)\"
echo \"Exit Code    : \${exit_code}\"
exit \${exit_code}
"
    echo "Submitted seed ${seed} → ${job_name}"
    ((submitted++)) || true
done

echo ""
echo "Done: ${submitted} submitted, ${skipped} skipped."
