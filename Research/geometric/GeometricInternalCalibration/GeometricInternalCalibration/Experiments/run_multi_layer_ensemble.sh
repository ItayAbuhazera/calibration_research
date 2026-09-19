#!/bin/bash
# ==============================================================================
# SLURM Runner — Multi-Layer Geometric Calibration Ensemble
# Runs Experiments/multi_layer_ensemble.py across models/datasets/seeds/methods
# Skips configs whose ensemble_results.json already contains all requested topKs
# ==============================================================================

set -euo pipefail

# --- Paths & env ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONDA_ENV="tamar_n_env"

# --- Defaults (edit as needed) ---
MODELS=("resnet18" "resnet50" "densenet121")
DATASETS=("cifar10" "cifar100")
SEEDS=(16 17)
TRAINING_METHODS=("baseline_cross_entropy")
ALPHAS=(0.5 1.0 1.5)

# Resources
PARTITION="gpu_partition"
GPUS="rtx_4090:1"
CPUS=4
MEM="32G"
TIME="2-00:00:00"

# Inference settings
DEVICE="cuda"
BATCH_SIZE=512
INPUT_H=32
INPUT_W=32

# Ensemble settings
TOPK=(3 5 7 0 1 2 4)                         # 0 = use ALL layers vary weights only; positive = top K deepest layers
SAMPLE_EVERY=1                   # layer sampling stride (1=every layer, 7=every 7th, 0=per-block) - OPTIMIZED
WEIGHTING_METHODS=("learned" "deep" "shallow" "uniform")    # space-separated: uniform learned deep shallow
DEPTH_ALPHA=1.5                  # only used for deep/shallow (legacy, now overridden by ALPHAS array)
LEARNED_LOSS="brier"             # brier|nll
BINS=15
MIN_RAW_ACC=0.70                 # skip ensembles if raw model test accuracy < this
PARALLELIZE_BY_K=false           # OPTIMIZATION: After caching probs, single job is much faster
# Note: EES = exp(-∑ w_i log w_i) is computed in Python and logged as "effective_ensemble_size"

# Additional analysis flags
SCAN_SINGLE_LAYERS=false         # evaluate each layer independently
EVAL_BASELINES=true             # run standard baseline methods (TS, isotonic, etc.)
DOC_METRICS=true                 # document all selector metrics with Top-K picks + holdout ECE
DOC_METRICS_BOTH=true           # document Top-K for both minimize and maximize per metric
METRICS_TOPK=5                   # number of top layers to report per metric
VAL_SPLIT=0.75                    # fraction of val for metric computation (rest for holdout)
EVAL_TOPK_ENSEMBLES=true        # evaluate metric-based ensembles on val-B + test
ENSEMBLE_TEST_KS=(1 3 5 7 2)         # K values for metric-based ensemble eval
EVAL_TOPK_WEIGHTING_METHODS=("learned" "shallow" "deep" "uniform")  # weighting methods
SAVE_PROBS_METRIC_EVAL=false    # save probs for metric-based eval
EVAL_METRIC_PICKS_ON_TEST=true   # evaluate metric picks on TEST with diagnostics
METRIC_PICK_WEIGHTING_METHODS=("learned" "uniform" "deep" "shallow") # weighting methods for TEST eval
METRIC_PICK_KS=(1 3 5 7 2)           # K values for metric-pick TEST eval

# OPTIMIZATION: Faster learned weight learning
LEARNED_MAX_ITER=200             # Reduced from 500 (often converges well before 200)
LEARNED_USE_LBFGS=true           # Use LBFGS for faster convergence
LEARNED_LR=0.1                   # Learning rate for Adam (ignored if using LBFGS)

# Results / outputs
RESULTS_DIR="${PROJECT_ROOT}/aaai_full_experiments/results/baseline"  # pre-trained ckpts root
OUTPUT_DIR="${PROJECT_ROOT}/aaai_full_experiments/results/multi_layer_ensemble_v1"

# Feature cache
FEAT_CACHE_ROOT="${OUTPUT_DIR}/_feat_cache"
# If you want to share feature cache across all runs, pin a seed (like you did with seed42)
FEAT_CACHE_SEED=20

# ------------------------------------------------------------------------------
# CLI overrides
# ------------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case $1 in
    --models)                IFS=' ' read -r -a MODELS <<< "$2"; shift 2;;
    --datasets)              IFS=' ' read -r -a DATASETS <<< "$2"; shift 2;;
    --seeds)                 IFS=' ' read -r -a SEEDS <<< "$2"; shift 2;;
    --training-methods)      IFS=' ' read -r -a TRAINING_METHODS <<< "$2"; shift 2;;
    --alphas)                IFS=' ' read -r -a ALPHAS <<< "$2"; shift 2;;
    --topk)                  IFS=' ' read -r -a TOPK <<< "$2"; shift 2;;
    --sample-every)          SAMPLE_EVERY="$2"; shift 2;;
    --weighting-methods)     IFS=' ' read -r -a WEIGHTING_METHODS <<< "$2"; shift 2;;
    --depth-alpha)           DEPTH_ALPHA="$2"; shift 2;;
    --learned-loss)          LEARNED_LOSS="$2"; shift 2;;
    --bins)                  BINS="$2"; shift 2;;
    --min-raw-acc)           MIN_RAW_ACC="$2"; shift 2;;
    --scan-single-layers)    SCAN_SINGLE_LAYERS=true; shift;;
    --eval-baselines)        EVAL_BASELINES=true; shift;;
    --no-scan-single-layers) SCAN_SINGLE_LAYERS=false; shift;;
    --no-eval-baselines)     EVAL_BASELINES=false; shift;;
    --doc-metrics)           DOC_METRICS=true; shift;;
    --no-doc-metrics)        DOC_METRICS=false; shift;;
    --doc-metrics-both)      DOC_METRICS_BOTH=true; shift;;
    --no-doc-metrics-both)   DOC_METRICS_BOTH=false; shift;;
    --metrics-topk)          METRICS_TOPK="$2"; shift 2;;
    --val-split)             VAL_SPLIT="$2"; shift 2;;
    --eval-topk-ensembles)   EVAL_TOPK_ENSEMBLES=true; shift;;
    --no-eval-topk-ensembles) EVAL_TOPK_ENSEMBLES=false; shift;;
    --ensemble-test-ks)      IFS=' ' read -r -a ENSEMBLE_TEST_KS <<< "$2"; shift 2;;
    --eval-topk-weighting-methods) IFS=' ' read -r -a EVAL_TOPK_WEIGHTING_METHODS <<< "$2"; shift 2;;
    --save-probs-metric-eval) SAVE_PROBS_METRIC_EVAL=true; shift;;
    --no-save-probs-metric-eval) SAVE_PROBS_METRIC_EVAL=false; shift;;
    --eval-metric-picks-on-test) EVAL_METRIC_PICKS_ON_TEST=true; shift;;
    --no-eval-metric-picks-on-test) EVAL_METRIC_PICKS_ON_TEST=false; shift;;
    --metric-pick-weighting-methods) IFS=' ' read -r -a METRIC_PICK_WEIGHTING_METHODS <<< "$2"; shift 2;;
    --metric-pick-ks)        IFS=' ' read -r -a METRIC_PICK_KS <<< "$2"; shift 2;;
    --parallelize-by-k)      PARALLELIZE_BY_K=true; shift;;
    --no-parallelize-by-k)   PARALLELIZE_BY_K=false; shift;;
    --batch-size)            BATCH_SIZE="$2"; shift 2;;
    --device)                DEVICE="$2"; shift 2;;
    --input-h)               INPUT_H="$2"; shift 2;;
    --input-w)               INPUT_W="$2"; shift 2;;
    --conda-env)             CONDA_ENV="$2"; shift 2;;
    --results-dir)           RESULTS_DIR="$2"; shift 2;;
    --output-dir)            OUTPUT_DIR="$2"; shift 2;;
    --feat-cache-root)       FEAT_CACHE_ROOT="$2"; shift 2;;
    --feat-cache-seed)       FEAT_CACHE_SEED="$2"; shift 2;;
    --partition)             PARTITION="$2"; shift 2;;
    --gpus)                  GPUS="$2"; shift 2;;
    --cpus)                  CPUS="$2"; shift 2;;
    --mem)                   MEM="$2"; shift 2;;
    --time)                  TIME="$2"; shift 2;;
    --help|-h)
      echo "Usage: $0 [all|status] [options]"
      echo "Options:"
      echo "  --models 'm1 m2'               Models (default: ${MODELS[*]})"
      echo "  --datasets 'd1 d2'             Datasets (default: ${DATASETS[*]})"
      echo "  --seeds 's1 s2'                Seeds (default: ${SEEDS[*]})"
      echo "  --training-methods 't1 t2'     Training methods"
      echo "  --alphas 'a1 a2'               Depth alpha values (default: ${ALPHAS[*]})"
      echo "  --topk '0 1 3 5'               Ensemble K (0=all layers vary weights, default: ${TOPK[*]})"
      echo "  --sample-every N               Layer sampling stride (default: $SAMPLE_EVERY)"
      echo "  --weighting-methods 'u l d s'  uniform|learned|deep|shallow (default: ${WEIGHTING_METHODS[*]})"
      echo "  --depth-alpha A                Depth bias alpha (default: $DEPTH_ALPHA)"
      echo "  --learned-loss LOSS            brier|nll (default: $LEARNED_LOSS)"
      echo "  --bins B                       ECE bins (default: $BINS)"
      echo "  --min-raw-acc ACC              Skip if raw test accuracy < ACC (default: $MIN_RAW_ACC)"
      echo "  --scan-single-layers           Evaluate each layer independently"
      echo "  --eval-baselines               Run standard baseline methods"
      echo "  --no-scan-single-layers        Skip per-layer evaluation"
      echo "  --no-eval-baselines            Skip baseline evaluation"
      echo "  --doc-metrics                  Document all selector metrics with Top-K picks"
      echo "  --no-doc-metrics               Skip metrics documentation"
      echo "  --doc-metrics-both             Document Top-K for both minimize and maximize per metric"
      echo "  --no-doc-metrics-both          Only document inferred direction (default)"
      echo "  --metrics-topk K               Top layers per metric (default: $METRICS_TOPK)"
      echo "  --val-split P                  Val fraction for metrics (default: $VAL_SPLIT)"
      echo "  --eval-topk-ensembles          Evaluate metric-based ensembles on valB + test"
      echo "  --no-eval-topk-ensembles       Skip metric-based ensemble evaluation"
      echo "  --ensemble-test-ks '1 3 5'     K values for metric ensemble eval (default: ${ENSEMBLE_TEST_KS[*]})"
      echo "  --eval-topk-weighting-methods 'l s d u' Weighting methods (default: ${EVAL_TOPK_WEIGHTING_METHODS[*]})"
      echo "  --save-probs-metric-eval       Save probs for metric-based evaluation"
      echo "  --no-save-probs-metric-eval    Skip saving probs"
      echo "  --eval-metric-picks-on-test    Evaluate metric picks on TEST with diagnostics"
      echo "  --no-eval-metric-picks-on-test Skip metric-pick TEST evaluation"
      echo "  --metric-pick-weighting-methods 'l u d s' Weighting methods for metric-pick eval"
      echo "  --metric-pick-ks '1 3 5'       K values for metric-pick eval (default: ENSEMBLE_TEST_KS)"
      echo "  --parallelize-by-k             Submit separate SLURM job per K (recommended for memory)"
      echo "  --no-parallelize-by-k          Submit one job with all K values (legacy mode)"
      echo "  --batch-size B                 (default: $BATCH_SIZE)"
      echo "  --device DEV                   cuda|cpu (default: $DEVICE)"
      echo "  --input-h H --input-w W        discovery input size (default: $INPUT_H x $INPUT_W)"
      echo "  --results-dir PATH             Phase-1 checkpoints root"
      echo "  --output-dir PATH              Output root for this experiment"
      echo "  --feat-cache-root PATH         Root for feature cache"
      echo "  --feat-cache-seed S            Shared cache seed (default: $FEAT_CACHE_SEED)"
      echo "  --partition P --gpus G --cpus C --mem M --time T  (SLURM resources)"
      exit 0;;
    *) echo "Unknown option: $1"; exit 1;;
  esac
done

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

json_has_all_topk() {
  local json_file="$1"; shift
  # If we explicitly skipped for low accuracy, treat it as done.
  if grep -q "\"skipped_low_acc\": true" "$json_file" 2>/dev/null; then
    return 0
  fi
  # Otherwise, require all requested topKs to be present.
  for k in "$@"; do
    # k==0 means we expect key "K0", otherwise "topK"
    if [[ "$k" -eq 0 ]]; then
      grep -q "\"K0\"" "$json_file" 2>/dev/null || return 1
    else
      grep -q "\"top${k}\"" "$json_file" 2>/dev/null || return 1
    fi
  done
  return 0
}

check_output_exists() {
  local TRAINING_METHOD="$1" DATASET="$2" MODEL="$3" SEED="$4" KVAL="${5:-}"

  local OUT_ROOT="${OUTPUT_DIR}/${TRAINING_METHOD}/${DATASET}/${MODEL}/seed${SEED}/multi_layer_ensemble"
  local SKIP_JSON="${OUT_ROOT}/ensemble_results.json"

  # If explicitly skipped for low accuracy, only require metric docs if requested.
  if grep -q "\"skipped_low_acc\": true" "$SKIP_JSON" 2>/dev/null; then
    [[ "$DOC_METRICS" == "true" && ! -s "${OUT_ROOT}/metric_docs/metric_docs_and_topk.json" ]] && return 1
    return 0
  fi

  # If no K passed: require *all* requested Ks to exist.
  if [[ -z "$KVAL" ]]; then
    for k in "${TOPK[@]}"; do
      if ! check_output_exists "$TRAINING_METHOD" "$DATASET" "$MODEL" "$SEED" "$k"; then
        return 1
      fi
    done
    # metrics docs if requested
    [[ "$DOC_METRICS" == "true" && ! -s "${OUT_ROOT}/metric_docs/metric_docs_and_topk.json" ]] && return 1
    return 0
  fi

  # K-specific check:
  local KDIR="${OUT_ROOT}/k_${KVAL}"
  # learned/uniform share k_{K}/ensemble_results.json
  local KU_JSON="${KDIR}/ensemble_results.json"
  local KKEY=$([[ "$KVAL" -eq 0 ]] && echo "K0" || echo "top${KVAL}")
  # need both learned & uniform keys in KU_JSON
  [[ ! -s "$KU_JSON" ]] && return 1
  grep -q "\"${KKEY}_learned\"" "$KU_JSON" 2>/dev/null || return 1
  grep -q "\"${KKEY}_uniform\"" "$KU_JSON" 2>/dev/null || return 1

  # deep/shallow must exist for each requested alpha under a_{alpha}/ensemble_results.json
  for a in "${ALPHAS[@]}"; do
    for wm in deep shallow; do
      local WM_JSON="${KDIR}/a_${a}/ensemble_results.json"
      [[ ! -s "$WM_JSON" ]] && return 1
      grep -q "\"${KKEY}_${wm}_a${a}\"" "$WM_JSON" 2>/dev/null || return 1
    done
  done

  # metrics docs if requested
  [[ "$DOC_METRICS" == "true" && ! -s "${OUT_ROOT}/metric_docs/metric_docs_and_topk.json" ]] && return 1
  return 0
}

submit_single_job() {
  local TRAINING_METHOD="$1" DATASET="$2" MODEL="$3" SEED="$4"
  local K_VALUE="${5:-}"  # optional: specific K to run (for parallel-by-K mode)
  # Note: ALPHA_VALUE removed - all alphas are now handled in one job

  # 1) Skip if finished (K-aware, all alphas handled in one job)
  if check_output_exists "$TRAINING_METHOD" "$DATASET" "$MODEL" "$SEED" "$K_VALUE" ""; then
    echo "✅ SKIP: ${TRAINING_METHOD} | ${MODEL} | ${DATASET} | seed ${SEED}${K_VALUE:+ | K=$K_VALUE} (done)"
    return 2
  fi

  # 2) Map CIFAR-C -> clean dataset for ckpt & loaders
  local DATASET_CLEAN="$DATASET"
  if [[ "$DATASET" == "cifar10-c" ]]; then DATASET_CLEAN="cifar10"; fi
  if [[ "$DATASET" == "cifar100-c" ]]; then DATASET_CLEAN="cifar100"; fi

  # 3) Verify checkpoint exists (must match load_model_and_data_like_before)
  local CKPT="${RESULTS_DIR}/${TRAINING_METHOD}/${DATASET_CLEAN}/${MODEL}/seed${SEED}/${TRAINING_METHOD}_${DATASET_CLEAN}_${MODEL}_seed${SEED}/best_model.pth"
  if [[ ! -f "$CKPT" ]]; then
    echo "❌ MISSING CKPT: ${CKPT}"
    return 1
  fi

  # 4) Build feat cache path (shared across all K jobs)
  local FEAT_CACHE="${FEAT_CACHE_ROOT}/${TRAINING_METHOD}/${DATASET_CLEAN}/${MODEL}/seed${FEAT_CACHE_SEED}"
  mkdir -p "$FEAT_CACHE"

  # 5) Determine output dir and K settings for this job
  local JOB_OUTPUT_DIR="$OUTPUT_DIR"
  local TOPK_STR ENSEMBLE_TEST_KS_STR METRIC_PICK_KS_STR
  local METRICS_DOC_DIR=""
  
  # Always try to reuse metric docs from the stable location
  METRICS_DOC_DIR="${OUTPUT_DIR}/${TRAINING_METHOD}/${DATASET_CLEAN}/${MODEL}/seed${SEED}/multi_layer_ensemble/metric_docs"
  if [[ ! -d "$METRICS_DOC_DIR" ]]; then
    METRICS_DOC_DIR=""
  fi
  
  if [[ -n "$K_VALUE" ]]; then
    TOPK_STR="$K_VALUE"
    ENSEMBLE_TEST_KS_STR="$K_VALUE"
    METRIC_PICK_KS_STR="$K_VALUE"
  else
    # Legacy mode: all K values in one job
    TOPK_STR="${TOPK[*]}"
    ENSEMBLE_TEST_KS_STR="${ENSEMBLE_TEST_KS[*]}"
    METRIC_PICK_KS_STR="${METRIC_PICK_KS[*]}"
  fi

  # 6) Compose command
  local PY="${PROJECT_ROOT}/Experiments/multi_layer_ensemble.py"
  
  # OPTIMIZATION: Pass all alphas to one job, Python will handle alpha-specific logic
  local WEIGHTING_STR="${WEIGHTING_METHODS[*]}"
  
  local EVAL_TOPK_WEIGHTING_STR="${EVAL_TOPK_WEIGHTING_METHODS[*]}"

  local CMD="python ${PY}"
  CMD+=" --dataset ${DATASET_CLEAN}"
  CMD+=" --model-name ${MODEL}"
  CMD+=" --training-method ${TRAINING_METHOD}"
  CMD+=" --device ${DEVICE}"
  CMD+=" --batch-size ${BATCH_SIZE}"
  CMD+=" --input-h ${INPUT_H} --input-w ${INPUT_W}"
  CMD+=" --topk ${TOPK_STR}"
  CMD+=" --sample-every ${SAMPLE_EVERY}"
  CMD+=" --weighting-methods ${WEIGHTING_STR}"
  CMD+=" --depth-alphas ${ALPHAS[*]}"
  CMD+=" --learned-loss ${LEARNED_LOSS}"
  CMD+=" --learned-max-iter ${LEARNED_MAX_ITER}"
  CMD+=" --learned-lr ${LEARNED_LR}"
  if [[ "$LEARNED_USE_LBFGS" == "true" ]]; then
    CMD+=" --learned-use-lbfgs"
  fi
  CMD+=" --bins ${BINS}"
  CMD+=" --min-raw-acc ${MIN_RAW_ACC}"
  CMD+=" --seed ${SEED}"
  CMD+=" --output-base-dir ${JOB_OUTPUT_DIR}"
  CMD+=" --feat-cache ${FEAT_CACHE}"
  CMD+=" --results-base-dir ${RESULTS_DIR}"
  
  # Add metrics-doc-dir if we're reusing precomputed docs
  if [[ -n "$METRICS_DOC_DIR" ]]; then
    CMD+=" --metrics-doc-dir ${METRICS_DOC_DIR}"
  fi
  
  # Optional flags
  if [[ "$SCAN_SINGLE_LAYERS" == "true" ]]; then
    CMD+=" --scan-single-layers"
  fi
  if [[ "$EVAL_BASELINES" == "true" ]]; then
    CMD+=" --eval-baselines"
  fi
  # Only compute docs if not reusing (avoids redundant computation on secondary K jobs)
  if [[ "$DOC_METRICS" == "true" && -z "$METRICS_DOC_DIR" ]]; then
    CMD+=" --doc-metrics --metrics-topk ${METRICS_TOPK} --val-split ${VAL_SPLIT}"
    if [[ "$DOC_METRICS_BOTH" == "true" ]]; then
      CMD+=" --doc-metrics-both"
    fi
  fi
  if [[ "$EVAL_TOPK_ENSEMBLES" == "true" ]]; then
    CMD+=" --eval-topk-ensembles"
    CMD+=" --ensemble-test-ks ${ENSEMBLE_TEST_KS_STR}"
    CMD+=" --eval-topk-weighting-methods ${EVAL_TOPK_WEIGHTING_STR}"
    if [[ "$SAVE_PROBS_METRIC_EVAL" == "true" ]]; then
      CMD+=" --save-probs-metric-eval"
    fi
  fi
  if [[ "$EVAL_METRIC_PICKS_ON_TEST" == "true" ]]; then
    CMD+=" --eval-metric-picks-on-test"
    if [[ -n "$METRIC_PICK_KS_STR" ]]; then
      CMD+=" --metric-pick-ks ${METRIC_PICK_KS_STR}"
    fi
    if [[ ${#METRIC_PICK_WEIGHTING_METHODS[@]} -gt 0 ]]; then
      local METRIC_PICK_WM_STR="${METRIC_PICK_WEIGHTING_METHODS[*]}"
      CMD+=" --metric-pick-weighting-methods ${METRIC_PICK_WM_STR}"
    fi
  fi

  # 7) SLURM submit
  local JOB_NAME="MLens_${TRAINING_METHOD:0:10}_${MODEL}_${DATASET}_s${SEED}"
  if [[ -n "$K_VALUE" ]]; then
    JOB_NAME="${JOB_NAME}_k${K_VALUE}"
  fi
  # Note: All alphas handled in one job now
  local LOG_DIR="${OUTPUT_DIR}/_logs"
  mkdir -p "$LOG_DIR"
  local LOG_FILE="${LOG_DIR}/${JOB_NAME}_%j.out"

  echo "🚀 SUBMIT: $JOB_NAME"
  sbatch \
    --partition="$PARTITION" \
    --job-name="$JOB_NAME" \
    --output="$LOG_FILE" \
    --time="$TIME" \
    --ntasks=1 \
    --gpus="$GPUS" \
    --cpus-per-task="$CPUS" \
    --mem="$MEM" \
    --wrap="
      echo '========================================'
      echo '🔬 SLURM JOB: Multi-Layer Ensemble'
      echo '========================================'
      echo 'Job ID     : \$SLURM_JOB_ID'
      echo 'Host       : \$(hostname)'
      echo 'Start Time : \$(date)'
      echo '----------------------------------------'
      echo 'Config:'
      echo '  Method   : $TRAINING_METHOD'
      echo '  Model    : $MODEL'
      echo '  Dataset  : $DATASET'
      echo '  Seed     : $SEED'
      echo '----------------------------------------'
      module load anaconda
      source activate $CONDA_ENV
      export PYTHONPATH='$PROJECT_ROOT:\$PYTHONPATH'
      export CUDA_LAUNCH_BLOCKING=1
      cd $PROJECT_ROOT
      echo 'CMD: $CMD'
      $CMD
      echo '----------------------------------------'
      echo 'End Time   : \$(date)'
      echo '========================================'
    "
  return 0
}

run_all() {
  echo "🔥 Submitting Multi-Layer Ensemble grid"
  echo "Models            : ${MODELS[*]}"
  echo "Datasets          : ${DATASETS[*]}"
  echo "Seeds             : ${SEEDS[*]}"
  echo "Methods           : ${TRAINING_METHODS[*]}"
  echo "Alphas            : ${ALPHAS[*]}"
  echo "TopK              : ${TOPK[*]}"
  echo "Parallelize by K  : ${PARALLELIZE_BY_K}"
  echo "Weightings        : ${WEIGHTING_METHODS[*]}"
  echo "Per-layer         : ${SCAN_SINGLE_LAYERS}"
  echo "Baselines         : ${EVAL_BASELINES}"
  echo "Doc-metrics       : ${DOC_METRICS}"
  echo "Eval-topk-ens     : ${EVAL_TOPK_ENSEMBLES}"
  if [[ "$EVAL_TOPK_ENSEMBLES" == "true" ]]; then
    echo "  Test Ks         : ${ENSEMBLE_TEST_KS[*]}"
    echo "  Weightings      : ${EVAL_TOPK_WEIGHTING_METHODS[*]}"
    echo "  Save probs      : ${SAVE_PROBS_METRIC_EVAL}"
  fi
  echo "Eval-metric-test  : ${EVAL_METRIC_PICKS_ON_TEST}"
  if [[ "$EVAL_METRIC_PICKS_ON_TEST" == "true" ]]; then
    if [[ ${#METRIC_PICK_KS[@]} -gt 0 ]]; then
      echo "  Test Ks         : ${METRIC_PICK_KS[*]}"
    else
      echo "  Test Ks         : (default)"
    fi
    if [[ ${#METRIC_PICK_WEIGHTING_METHODS[@]} -gt 0 ]]; then
      echo "  Weightings      : ${METRIC_PICK_WEIGHTING_METHODS[*]}"
    else
      echo "  Weightings      : (default)"
    fi
  fi
  echo "----------------------------------------"

  local total=0 sub=0 skip=0 miss=0
  
  for tm in "${TRAINING_METHODS[@]}"; do
    for m in "${MODELS[@]}"; do
      for d in "${DATASETS[@]}"; do
        for s in "${SEEDS[@]}"; do
          if [[ "$PARALLELIZE_BY_K" == "true" ]]; then
            # Submit separate job per K value (all alphas in each job)
            for k in "${TOPK[@]}"; do
              total=$((total+1))
              if submit_single_job "$tm" "$d" "$m" "$s" "$k"; then
                rc=$?
              else
                rc=$?
              fi
              case $rc in
                0) sub=$((sub+1));;
                1) miss=$((miss+1));;
                2) skip=$((skip+1));;
              esac
              sleep 0.5
            done
          else
            # One job for all K values and all alphas
            total=$((total+1))
            if submit_single_job "$tm" "$d" "$m" "$s" ""; then
              rc=$?
            else
              rc=$?
            fi
            case $rc in
              0) sub=$((sub+1));;
              1) miss=$((miss+1));;
              2) skip=$((skip+1));;
            esac
            sleep 1
          fi
        done
      done
    done
  done

  echo "========================================"
  echo "📊 Submission Summary"
  echo "Total    : $total"
  echo "Submitted: $sub"
  echo "Skipped  : $skip"
  echo "Missing  : $miss"
  echo "========================================"
  echo "Use: squeue -u \$USER --name='MLens_*'"
}

status() {
  echo "📊 Status (completed configs)"
  local total=0 done=0
  for tm in "${TRAINING_METHODS[@]}"; do
    for m in "${MODELS[@]}"; do
      for d in "${DATASETS[@]}"; do
        for s in "${SEEDS[@]}"; do
          if [[ "$PARALLELIZE_BY_K" == "true" ]]; then
            # Count per-K (all alphas in each job)
            for k in "${TOPK[@]}"; do
              total=$((total+1))
              if check_output_exists "$tm" "$d" "$m" "$s" "$k" ""; then
                done=$((done+1))
              fi
            done
          else
            # One job for all K values and all alphas
            total=$((total+1))
            if check_output_exists "$tm" "$d" "$m" "$s" "" ""; then
              done=$((done+1))
            fi
          fi
        done
      done
    done
  done
  local pct=0
  if [[ $total -gt 0 ]]; then pct=$((100*done/total)); fi
  echo "Done: $done / $total (${pct}%)"
  echo "Queued jobs:"
  squeue -u "$USER" --name="MLens_*" --format="%.18j %.8T %.10M" 2>/dev/null || true
}

case "${1:-all}" in
  all) run_all ;;
  status) status ;;
  *) echo "Unknown command '$1' (use all|status)"; exit 1 ;;
esac
