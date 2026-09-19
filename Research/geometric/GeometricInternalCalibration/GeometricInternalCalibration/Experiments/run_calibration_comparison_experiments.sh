#!/bin/bash

# Phase 3: Pretrained vs. From-Scratch Calibration Comparison
# This script runs experiments to compare the calibration of:
#   1. Pretrained DINOv2 models (frozen backbone)
#   2. From-scratch DINOv2 models (fully trained)

set -e  # Exit on any error

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
# CONDA_ENV="geo_cuda12" # 🔧 UPDATE with your conda environment name
CONDA_ENV="tamar_n_env"
# --- Experiment Parameters ---
DATASET="cifar10"
SEEDS=(13 14) # Use multiple seeds for robustness
BASE_OUTPUT_DIR="${PROJECT_ROOT}/aaai_full_experiments/results/calibration_comparison"

# Models to compare: pretrained vs. from-scratch for different sizes
declare -a MODELS=(
    # "dinov2_small_pretrained"
    "dinov2_small_scratch"
    # "dinov2_base_pretrained"
    "dinov2_base_scratch"
)

# Training methods to evaluate (can be expanded)
declare -a TRAINING_METHODS=(
    "baseline_cross_entropy"
    # "augmix"
    # "geometric_focal_calibration"
)

# --- SLURM & Job Configuration ---
USE_SLURM=true  # Set to false to run locally
PARTITION="gpu_partition" # 🔧 UPDATE with your SLURM partition
GPU_TYPE="rtx_4090" # 🔧 UPDATE with your GPU type
BASE_MEM="32G"
BASE_CPUS=4
BASE_TIME="1-00:00:00" # 1 day for from-scratch training

# --- Helper Functions ---

# Function to get resource requirements based on model and training method
get_resources() {
    local MODEL="$1"
    local METHOD="$2"
    local MEM="$BASE_MEM"
    local CPUS="$BASE_CPUS"
    local TIME="$BASE_TIME"

    # From-scratch models require more resources
    if [[ "$MODEL" == *"scratch"* ]]; then
        TIME="2-00:00:00" # More time for from-scratch
        CPUS=6
        if [[ "$MODEL" == *"base"* ]]; then
            MEM="64G" # More memory for base model
        else
            MEM="48G"
        fi
    fi
    
    # Geometric methods might need more memory
    if [[ "$METHOD" == *"geometric"* ]]; then
        MEM="64G"
    fi

    echo "$TIME $MEM $CPUS"
}

# Function to submit a single experiment job
submit_job() {
    local MODEL="$1"
    local METHOD="$2"
    local SEED="$3"

    # Construct experiment name and directories
    local OUTPUT_DIR="${BASE_OUTPUT_DIR}/${METHOD}/${DATASET}/${MODEL}/seed${SEED}"
    local JOB_NAME="cal_comp_${DATASET}_${MODEL}_${METHOD}_s${SEED}"
    
    mkdir -p "${BASE_OUTPUT_DIR}/logs"
    cd "$PROJECT_ROOT" || exit
    # Get resource allocation
    local RESOURCES=($(get_resources "$MODEL" "$METHOD"))
    local TIME="${RESOURCES[0]}"
    local MEM="${RESOURCES[1]}"
    local CPUS="${RESOURCES[2]}"
    
    # Build the Python command
    local CMD="python Experiments/run_single_experiment.py"
    CMD+=" --method ${METHOD}"
    CMD+=" --dataset ${DATASET}"
    CMD+=" --model ${MODEL}"
    CMD+=" --seed ${SEED}"
    CMD+=" --output_dir ${BASE_OUTPUT_DIR}"
    CMD+=" --epochs 350" # Standardized epochs
    CMD+=" --batch_size 128"
    CMD+=" --lr 0.01" # Potentially different LR for scratch vs. pretrained
    
    # Add AugMix parameters if needed
    if [[ "$METHOD" == "augmix" ]]; then
        CMD+=" --use_augmix"
    fi
    
    echo "🚀 Submitting Job: $JOB_NAME"
    echo "   Model: $MODEL | Method: $METHOD | Seed: $SEED"
    echo "   Resources: Time=$TIME, Mem=$MEM, CPUs=$CPUS"
    echo "   Output Dir: $OUTPUT_DIR"
    
    # Check if results already exist - MODIFIED TO BE MORE FLEXIBLE
    # Since we add job ID to the name, we check for a directory pattern
    if compgen -G "${OUTPUT_DIR}/${METHOD}_${DATASET}_${MODEL}_seed${SEED}_*/results.json" > /dev/null; then
        echo "✅ Results directory already exists, skipping."
        return
    fi

    if [ "$USE_SLURM" = true ]; then
        # SLURM submission
        # The job ID is now passed to the script to create a unique directory
        CMD_WITH_JOB_ID="$CMD --job_id \$SLURM_JOB_ID"
        
        # Use SLURM's job ID placeholder %j for unique log files
        local SLURM_LOG_FILE="${BASE_OUTPUT_DIR}/logs/${JOB_NAME}_%j.out"

        sbatch \
            --partition="$PARTITION" \
            --job-name="$JOB_NAME" \
            --output="$SLURM_LOG_FILE" \
            --time="$TIME" \
            --gpus="$GPU_TYPE:1" \
            --cpus-per-task="$CPUS" \
            --exclude=cs-4090-08  \
            --mem="$MEM" \
            --wrap="
            echo '🔬 CALIBRATION COMPARISON EXPERIMENT'
            echo '===================================='
            echo 'Job ID: \$SLURM_JOB_ID'
            echo 'Model: $MODEL | Method: $METHOD'
            echo 'Dataset: $DATASET | Seed: $SEED'
            echo 'Host: \$(hostname) | GPU: \$CUDA_VISIBLE_DEVICES'
            echo 'Started: \$(date)'
            echo '===================================='

            module load anaconda
            source activate $CONDA_ENV
            export PYTHONPATH='$PROJECT_ROOT:\$PYTHONPATH'
            
            echo 'Executing command:'
            echo '$CMD_WITH_JOB_ID'
            
            $CMD_WITH_JOB_ID
            
            echo '✅ Experiment finished: \$(date)'
            "
    else
        # Local execution (for testing)
        echo "Running locally..."
        
        # Use PID for local job ID to avoid overwrites
        LOCAL_JOB_ID="local_$$"
        CMD_WITH_JOB_ID="$CMD --job_id $LOCAL_JOB_ID"
        
        export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
        
        # Activate conda env if not in SLURM
        if [[ -z "$SLURM_JOB_ID" ]]; then
            source activate "$CONDA_ENV"
        fi
        
        # Create a unique log file for local execution using the process ID
        local LOCAL_LOG_FILE="${BASE_OUTPUT_DIR}/logs/${JOB_NAME}_pid$$.out"
        $CMD_WITH_JOB_ID > "$LOCAL_LOG_FILE" 2>&1 &
        echo "   PID: $!"
        echo "   Log: $LOCAL_LOG_FILE"
    fi
}

# --- Main Loop ---
echo "🔥 Starting Calibration Comparison Experiments"
echo "============================================"
echo "Dataset: $DATASET"
echo "Models: ${MODELS[*]}"
echo "Methods: ${TRAINING_METHODS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "============================================"

for model in "${MODELS[@]}"; do
    for method in "${TRAINING_METHODS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            submit_job "$model" "$method" "$seed"
            
            if [ "$USE_SLURM" = true ]; then
                echo "⏳ Waiting 30 seconds before next submission..."
                sleep 30
            fi
        done
    done
done

echo "🎉 All jobs submitted."
echo "Monitor with: squeue -u \$USER"
echo "Check logs in: ${BASE_OUTPUT_DIR}/logs"
echo "Check results in: ${BASE_OUTPUT_DIR}" 