#!/bin/bash

# ==============================================================================
# Slurm Experiment Runner for Layer Selection Analysis
#
# Description:
# This script automates the process of running the layer selection analysis
# experiment (`run_layer_selection_experiment.py`) across multiple models,
# datasets, training methods, and random seeds on a SLURM cluster.
#
# Each job runs a full analysis for a single configuration, determining both
# the selector's predicted best layer and the empirically best layer, and saves
# the results as a JSON file and a PNG plot.
# ==============================================================================

set -e  # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONDA_ENV="tamar_n_env"

# --- Experiment Parameters (Defaults) ---
MODELS=("resnet18" "resnet50" "densenet121" "dinov2_small" "dinov2_base")
DATASETS=("cifar10" "cifar100")
SEEDS=(11 12 13 14 15 16 17 18 19 20)
BATCH_SIZE=500
DEVICE="cuda"
# Optional list of target accuracies (percent) to analyze; if empty, no accuracy suffixing is used.
TARGET_ACCS=()

# Base directory where pre-trained models from Phase 1 are stored.
# The script will look for models in: <RESULTS_DIR>/<training_method>/<dataset>/<model>/seed<seed>/
RESULTS_DIR="${PROJECT_ROOT}/aaai_full_experiments/results/baseline"

# Base directory where the output of this layer analysis will be saved.
OUTPUT_DIR="${PROJECT_ROOT}/aaai_full_experiments/results/layer_selection_analysis_baselines_4"

# --- Training Methods to Evaluate ---
# Define the training methods for which you have pre-trained models.
declare -a TRAINING_METHODS=(
    # "baseline_cross_entropy"
    "augmix"
    "baseline_brier"
    "baseline_focal_adaptive"
    "baseline_mmce_weighted"
    "constellation_original"
)

# ==============================================================================
# Command-Line Argument Parsing
# ==============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --models) MODELS=($2); shift 2;;
        --datasets) DATASETS=($2); shift 2;;
        --seeds) SEEDS=($2); shift 2;;
        --training-methods) TRAINING_METHODS=($2); shift 2;;
        --batch-size) BATCH_SIZE="$2"; shift 2;;
        --device) DEVICE="$2"; shift 2;;
        --conda-env) CONDA_ENV="$2"; shift 2;;
        --results-dir) RESULTS_DIR="$2"; shift 2;;
        --output-dir) OUTPUT_DIR="$2"; shift 2;;
        --target-acc-list) TARGET_ACCS=($2); shift 2;;
        --help|-h)
            echo "Usage: $0 [COMMAND] [OPTIONS]"
            echo ""
            echo "Commands:"
            echo "  all        (Default) Submit all defined experiment jobs to SLURM."
            echo "  status     Show a summary of completed and pending jobs."
            echo "  help       Show this help message."
            echo ""
            echo "Options:"
            echo "  --models 'm1 m2'         Space-separated list of models (default: ${MODELS[*]})."
            echo "  --datasets 'd1 d2'       Space-separated list of datasets (default: ${DATASETS[*]})."
            echo "  --seeds 's1 s2'          Space-separated list of seeds (default: ${SEEDS[*]})."
            echo "  --training-methods 't1'  Space-separated list of training methods (default: ${TRAINING_METHODS[*]})."
            echo "  --batch-size SIZE        Batch size for the Python script (default: $BATCH_SIZE)."
            echo "  --device DEV             Device to use, 'cuda' or 'cpu' (default: $DEVICE)."
            echo "  --conda-env NAME         Name of the Conda environment to activate (default: $CONDA_ENV)."
            echo "  --results-dir PATH       Path to the base directory of pre-trained models."
            echo "  --output-dir PATH        Path to the base directory for saving analysis results."
            echo "  --target-acc-list 'a b'  Space-separated list of target accuracies (e.g., '75.0 85.0 95.0')."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# ==============================================================================
# Helper Functions
# ==============================================================================

# --- Function to check if a valid result file already exists ---
check_job_output_exists() {
    local TRAINING_METHOD="$1"
    local DATASET="$2"
    local MODEL="$3"
    local SEED="$4"
    local TARGET_ACC="${5:-}"

    # Construct the exact path where the final analysis JSON file should be.
    # This must match the path constructed in the Python script.
    # If a target accuracy is specified, results are written under an accXX subfolder.
    local ACC_SUFFIX=""
    if [[ -n "$TARGET_ACC" ]]; then
        # Cast 75.0 -> 75, 85.0 -> 85, etc.
        local ACC_INT
        ACC_INT=$(printf "%.0f" "$TARGET_ACC")
        ACC_SUFFIX="/acc${ACC_INT}"
    fi
    local FINAL_JSON_FILE="${OUTPUT_DIR}${ACC_SUFFIX}/${TRAINING_METHOD}/${DATASET}/${MODEL}/seed${SEED}/multi_composite_analysis.json"

    # Check if the file exists and is not empty.
    if [[ -f "$FINAL_JSON_FILE" ]] && [[ -s "$FINAL_JSON_FILE" ]]; then
        # Check if this is a skipped run (low accuracy) - if so, treat as incomplete
        if grep -q '"skipped".*true' "$FINAL_JSON_FILE" 2>/dev/null; then
            return 1 # Failure (job was skipped, needs re-running if threshold changes)
        fi
        return 0 # Success (file exists and is valid)
    else
        return 1 # Failure (file does not exist or is empty)
    fi
}


# --- Function to submit a single layer analysis job to SLURM ---
submit_layer_analysis_job() {
    local TRAINING_METHOD="$1"
    local DATASET="$2"
    local MODEL="$3"
    local SEED="$4"
    local TARGET_ACC="${5:-}"

    # --- 1. Pre-Submission Checks ---

    # Check if valid results already exist to avoid re-running.
    if check_job_output_exists "$TRAINING_METHOD" "$DATASET" "$MODEL" "$SEED" "$TARGET_ACC"; then
        echo "✅ SKIPPING: Valid results already exist for $TRAINING_METHOD | $MODEL | $DATASET | Seed $SEED"
        return 2 # Special return code for skipping
    fi

    # Normalize dataset name for model path (models are trained on clean data)
    local DATASET_FOR_MODEL="$DATASET"
    if [[ "$DATASET" == "cifar10-c" ]]; then
        DATASET_FOR_MODEL="cifar10"
    elif [[ "$DATASET" == "cifar100-c" ]]; then
        DATASET_FOR_MODEL="cifar100"
    fi

    # Accuracy-aware model root: for early-stopped runs we expect models under
    #   <BASE_RESULTS_DIR>/accXX/baseline_accXX/<TRAINING_METHOD>/...
    # whereas baseline models live under:
    #   <BASE_RESULTS_DIR>/baseline/<TRAINING_METHOD>/...
    local RESULTS_DIR_EFFECTIVE="$RESULTS_DIR"
    if [[ -n "$TARGET_ACC" ]]; then
        local ACC_INT
        ACC_INT=$(printf "%.0f" "$TARGET_ACC")
        local BASE_RESULTS_DIR
        BASE_RESULTS_DIR="$(dirname "$RESULTS_DIR")"  # strip trailing '/baseline'
        RESULTS_DIR_EFFECTIVE="${BASE_RESULTS_DIR}/acc${ACC_INT}/baseline_acc${ACC_INT}"
    fi

    # Check if the required pre-trained model file exists.
    local MODEL_PATH="${RESULTS_DIR_EFFECTIVE}/${TRAINING_METHOD}/${DATASET_FOR_MODEL}/${MODEL}/seed${SEED}/${TRAINING_METHOD}_${DATASET_FOR_MODEL}_${MODEL}_seed${SEED}/best_model.pth"
    if [[ ! -f "$MODEL_PATH" ]]; then
        echo "❌ MISSING MODEL: Cannot find model for $TRAINING_METHOD | $MODEL | $DATASET | Seed $SEED"
        echo "   (Expected at: $MODEL_PATH)"
        return 1 # Return code for missing model
    fi

    # --- 2. Resource Allocation ---
    # The Python script runs a full analysis (selector + empirical loop),
    # so it requires significant resources.
    local TIME="3-00:00:00"  # 3 days
    local MEM="32G"
    local CPUS=4
    local GPUS="1" # Or your specific GPU partition

    # --- 3. Job and Command Definition ---
    local ACC_TAG=""
    if [[ -n "$TARGET_ACC" ]]; then
        local ACC_INT
        ACC_INT=$(printf "%.0f" "$TARGET_ACC")
        ACC_TAG="_acc${ACC_INT}"
    fi
    local JOB_NAME="L-An_${TRAINING_METHOD:9}_${MODEL}_${DATASET}${ACC_TAG}_s${SEED}"
    local JOB_LOG_DIR="${OUTPUT_DIR}/_logs"
    mkdir -p "$JOB_LOG_DIR"
    local LOG_FILE="${JOB_LOG_DIR}/${JOB_NAME}_%j.out"
    
    local PYTHON_SCRIPT_PATH="${PROJECT_ROOT}/Experiments/layer_selection.py"

    # Build the Python command with all required arguments.
    local CMD="python ${PYTHON_SCRIPT_PATH}"
    CMD+=" --model-name ${MODEL}"

    # Normalize CIFAR-C aliases -> clean dataset + flags
    local DATASET_CLEAN="$DATASET"
    local EXTRA_FLAGS=""
    if [[ "$DATASET" == "cifar10-c" ]]; then
      DATASET_CLEAN="cifar10"
      EXTRA_FLAGS=" --eval-corruptions --cifar-c-dir ${PROJECT_ROOT}/data/cifar10-c --mce-only"
    elif [[ "$DATASET" == "cifar100-c" ]]; then
      DATASET_CLEAN="cifar100"
      EXTRA_FLAGS=" --eval-corruptions --cifar-c-dir ${PROJECT_ROOT}/data/cifar100-c --mce-only"
    fi

    CMD+=" --dataset-name ${DATASET_CLEAN}${EXTRA_FLAGS}"
    CMD+=" --training-method ${TRAINING_METHOD}"
    CMD+=" --seed ${SEED}"
    CMD+=" --results-base-dir ${RESULTS_DIR_EFFECTIVE}"
    CMD+=" --output-base-dir ${OUTPUT_DIR}"
    if [[ -n "$TARGET_ACC" ]]; then
        CMD+=" --target-acc ${TARGET_ACC}"
    fi
    CMD+=" --batch-size ${BATCH_SIZE}"
    CMD+=" --device ${DEVICE}"

    # --- 4. SLURM Submission ---
    echo "🚀 SUBMITTING: $JOB_NAME"
    
    sbatch \
        --partition=gpu_partition \
        --job-name="$JOB_NAME" \
        --output="$LOG_FILE" \
        --time="$TIME" \
        --ntasks=1 \
        --gpus="$GPUS" \
        --cpus-per-task="$CPUS" \
        --mem="$MEM" \
        --wrap="
        echo '========================================'
        echo '🔬 SLURM JOB: Layer Selection Analysis'
        echo '========================================'
        echo 'Job ID          : \$SLURM_JOB_ID'
        echo 'Job Name        : $JOB_NAME'
        echo 'Host            : \$(hostname)'
        echo 'Start Time      : \$(date)'
        echo '----------------------------------------'
        echo 'Configuration:'
        echo '  Training Method : $TRAINING_METHOD'
        echo '  Model           : $MODEL'
        echo '  Dataset         : $DATASET'
        echo '  Seed            : $SEED'
        echo '----------------------------------------'

        # Environment Setup
        echo 'Activating Conda environment: $CONDA_ENV'
        module load anaconda
        source activate $CONDA_ENV
        export PYTHONPATH='$PROJECT_ROOT:\$PYTHONPATH'
        export CUDA_LAUNCH_BLOCKING=1 # Useful for debugging CUDA errors

        cd $PROJECT_ROOT

        echo 'Executing Python script:'
        echo '$CMD'
        echo '----------------------------------------'

        # Execute the command
        if $CMD; then
            echo '✅ EXPERIMENT COMPLETED SUCCESSFULLY'
        else
            echo '❌ EXPERIMENT FAILED'
            # The 'set -e' at the top of the main script ensures this will cause
            # the job to fail with a non-zero exit code if the python script fails.
            exit 1
        fi
        
        echo '========================================'
        echo 'Job End Time    : \$(date)'
        echo '========================================'
        "
    return 0 # Return code for successful submission
}


# ==============================================================================
# Main Execution Logic
# ==============================================================================

# --- Function to run all experiment configurations ---
run_all_experiments() {
    echo "🔥 Starting Layer Selection Analysis Pipeline"
    echo "=============================================="
    echo "Models            : ${MODELS[*]}"
    echo "Datasets          : ${DATASETS[*]}"
    echo "Seeds             : ${SEEDS[*]}"
    echo "Training Methods  : ${TRAINING_METHODS[*]}"
    if [[ ${#TARGET_ACCS[@]} -gt 0 ]]; then
        echo "Target Accuracies : ${TARGET_ACCS[*]}"
    else
        echo "Target Accuracies : (none; using baseline models only)"
    fi
    echo "Output Directory  : $OUTPUT_DIR"
    echo "=============================================="

    local total_jobs=0
    local submitted_jobs=0
    local skipped_jobs=0
    local missing_model_jobs=0

    # Determine which accuracy values to iterate over (empty string = no accuracy suffix)
    local ACCS=("${TARGET_ACCS[@]}")
    if [[ ${#ACCS[@]} -eq 0 ]]; then
        ACCS=("")
    fi

    # Iterate through all combinations
    for training_method in "${TRAINING_METHODS[@]}"; do
        for model in "${MODELS[@]}"; do
            for dataset in "${DATASETS[@]}"; do
                for seed in "${SEEDS[@]}"; do
                    for acc in "${ACCS[@]}"; do
                        total_jobs=$((total_jobs + 1))
                        
                        if submit_layer_analysis_job "$training_method" \
                                 "$dataset" \
                                 "$model" \
                                 "$seed" \
                                 "$acc"
                        then
                            submit_status=$?          # will be 0 or 2
                        else
                            submit_status=$?          # will be 1 (missing model) or other
                        fi

                        case $submit_status in
                            0) submitted_jobs=$((submitted_jobs + 1));;
                            1) missing_model_jobs=$((missing_model_jobs + 1));;
                            2) skipped_jobs=$((skipped_jobs + 1));;
                        esac
                        
                        # Optional: Add a small delay between submissions to be nice to the scheduler
                        sleep 2
                    done
                done
            done
        done
    done

    echo "=============================================="
    echo "📊 Pipeline Submission Summary"
    echo "=============================================="
    echo "Total configurations checked : $total_jobs"
    echo "Jobs submitted to SLURM      : $submitted_jobs"
    echo "Jobs skipped (already done)  : $skipped_jobs"
    echo "Jobs skipped (missing model) : $missing_model_jobs"
    echo "=============================================="
    echo "🚀 Use 'squeue -u \$USER' to monitor your jobs."
}

# --- Function to show the status of the experiments ---
show_status() {
    echo "📊 Experiment Status Summary"
    echo "=============================================="
    
    local total_configs=0
    local completed_configs=0
    
    # Determine which accuracy values to iterate over (empty string = no accuracy suffix)
    local ACCS=("${TARGET_ACCS[@]}")
    if [[ ${#ACCS[@]} -eq 0 ]]; then
        ACCS=("")
    fi

    for training_method in "${TRAINING_METHODS[@]}"; do
        for model in "${MODELS[@]}"; do
            for dataset in "${DATASETS[@]}"; do
                for seed in "${SEEDS[@]}"; do
                    for acc in "${ACCS[@]}"; do
                        total_configs=$((total_configs + 1))
                        if check_job_output_exists "$training_method" "$dataset" "$model" "$seed" "$acc"; then
                            completed_configs=$((completed_configs + 1))
                        fi
                    done
                done
            done
        done
    done
    
    local running_jobs=$(squeue -u $USER --name="L-An_*" --noheader 2>/dev/null | wc -l | xargs)
    local percentage=0
    if [ $total_configs -gt 0 ]; then
        percentage=$((completed_configs * 100 / total_configs))
    fi

    echo "Completion        : ${completed_configs} / ${total_configs} (${percentage}%)"
    echo "Jobs in SLURM Queue : ${running_jobs}"
    echo ""
    echo "Running Jobs:"
    squeue -u $USER --name="L-An_*" --format="%.15j %.8T %.10M" 2>/dev/null || echo "None"
    echo "=============================================="
}


# --- Command Router ---
# Use the first command-line argument to decide what to do.
# Defaults to 'all' if no command is given.
case "${1:-all}" in
    "all")
        run_all_experiments
        ;;
    "status")
        show_status
        ;;
    "help")
        # The argument parser handles this, but we can call it explicitly.
        $0 --help
        ;;
    *)
        echo "❌ Unknown command: $1"
        $0 --help
        exit 1
        ;;
esac