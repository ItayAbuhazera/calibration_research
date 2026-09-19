#!/bin/bash
# Modular Runner for AAAI Paper Section 4 experiments
# "Why Physical-Space Approaches Fail" - SLURM Command Line Version

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Determine if we're in Experiments directory or project root
if [[ $(basename "$SCRIPT_DIR") == "Experiments" ]]; then
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
    EXPERIMENTS_DIR="$SCRIPT_DIR"
else
    PROJECT_ROOT="$SCRIPT_DIR"
    EXPERIMENTS_DIR="$PROJECT_ROOT/Experiments"
fi

# ===== EXPERIMENTAL CONFIGURATION ARRAYS =====
# These arrays define the full experimental space

# Models to analyze
MODELS=("resnet18" "resnet50" "densenet121")

# Seeds for robust evaluation
SEEDS=(12)

# Training methods to evaluate
TRAINING_METHODS=("baseline_brier" "baseline_mmce" "baseline_mmce_weighted" "augmix_constellation" "augmix" "ce_fast_separation" "constellation_original" "geometric_focal_calibration")

# Default parameters (can be overridden)
DEFAULT_TRAINING_METHOD="constellation"
DEFAULT_SEED=12
DEFAULT_MODEL="resnet18"
DEFAULT_LAYER="layer3"
PYTHON_ENV="tamar_n_env"
NUM_IMAGES=100
MAX_ANALYZE=200
OUTPUT_DIR="section4_statistical_analysis"
BASE_OUTPUT_DIR="$OUTPUT_DIR"  # Store the base directory
BATCH_SIZE=100

# SLURM Configuration
USE_SLURM=true
PARTITION="gpu_partition"
GPU_TYPE="rtx_4090"

# Available layers per model
declare -A MODEL_LAYERS
MODEL_LAYERS[resnet18]="layer1 layer2 layer3 layer4 layer4.1"
MODEL_LAYERS[resnet50]="layer1 layer2 layer3 layer4 layer4.1"
MODEL_LAYERS[densenet121]="dense1 trans1 dense2 trans2 dense3 trans3 dense4 bn"

# Default layers per model (used for single experiments)
declare -A DEFAULT_LAYERS
DEFAULT_LAYERS[resnet18]="layer3"
DEFAULT_LAYERS[resnet50]="layer3"
DEFAULT_LAYERS[densenet121]="trans3"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            DEFAULT_MODEL="$2"
            shift 2
            ;;
        --models)
            # Parse multiple models: --models "resnet18 resnet50 densenet121"
            IFS=' ' read -ra MODELS <<< "$2"
            shift 2
            ;;
        --layer)
            DEFAULT_LAYER="$2"
            shift 2
            ;;
        --seed)
            DEFAULT_SEED="$2"
            shift 2
            ;;
        --seeds)
            # Parse multiple seeds: --seeds "12 13 14 11"
            IFS=' ' read -ra SEEDS <<< "$2"
            shift 2
            ;;
        --training-method)
            DEFAULT_TRAINING_METHOD="$2"
            shift 2
            ;;
        --training-methods)
            # Parse multiple training methods: --training-methods "constellation contrastive"
            IFS=' ' read -ra TRAINING_METHODS <<< "$2"
            shift 2
            ;;
        --num-images)
            NUM_IMAGES="$2"
            shift 2
            ;;
        --max-analyze)
            MAX_ANALYZE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --python-env)
            PYTHON_ENV="$2"
            shift 2
            ;;
        --no-slurm)
            USE_SLURM=false
            shift
            ;;
        --partition)
            PARTITION="$2"
            shift 2
            ;;
        --gpu-type)
            GPU_TYPE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS] [COMMAND]"
            echo ""
            echo "Modular AAAI Paper Section 4 Experiments - SLURM VERSION"
            echo "========================================================="
            echo ""
            echo "Commands:"
            echo "  single           - Submit single experiment to SLURM queue"
            echo "  all              - Submit ALL experiments across all models/layers/seeds/methods"
            echo "  by-model         - Submit experiments for all layers/seeds for each model"
            echo "  by-method        - Submit experiments for all models/layers/seeds for each method"
            echo "  by-seed          - Submit experiments for all models/layers/methods for each seed"
            echo "  quick            - Submit quick test jobs with reduced parameters"
            echo "  scan             - Scan for existing results"
            echo "  monitor          - Monitor running SLURM jobs"
            echo "  cleanup          - Clean up incomplete results"
            echo "  aggregate        - Aggregate all results with layer comparison"
            echo "  help             - Show this help message"
            echo ""
            echo "Options:"
            echo "  --model MODEL              Single model to analyze (default: resnet18)"
            echo "  --models \"MODEL1 MODEL2\"   Multiple models (default: \"resnet18 resnet50 densenet121\")"
            echo "  --layer LAYER              Single layer to analyze (default: layer3)"
            echo "  --seed SEED                Single seed (default: 12)"
            echo "  --seeds \"SEED1 SEED2\"      Multiple seeds (default: \"12 13 14 11\")"
            echo "  --training-method METHOD   Single training method (default: constellation)"
            echo "  --training-methods \"M1 M2\" Multiple methods (default: \"constellation contrastive baseline_cross_entropy augmix\")"
            echo "  --num-images NUM           Number of images to analyze (default: 100)"
            echo "  --max-analyze NUM          Maximum car examples to analyze (default: 200)"
            echo "  --output-dir DIR           Output directory (default: section4_statistical_analysis)"
            echo "  --batch-size SIZE          Batch size for feature extraction (default: 100)"
            echo "  --python-env ENV           Conda environment name (default: tamar_n_env)"
            echo ""
            echo "SLURM Options:"
            echo "  --no-slurm                 Run directly without SLURM submission"
            echo "  --partition PARTITION      SLURM partition (default: gpu_partition)"
            echo "  --gpu-type TYPE            GPU type (default: rtx_4090)"
            echo ""
            echo "Examples:"
            echo "  # Single experiment"
            echo "  $0 single --model resnet18 --layer layer3 --seed 12"
            echo ""
            echo "  # All experiments with default arrays"
            echo "  $0 all"
            echo ""
            echo "  # All experiments with custom arrays"
            echo "  $0 all --models \"resnet18 resnet50\" --seeds \"12 13\" --training-methods \"constellation contrastive\""
            echo ""
            echo "  # By model (all layers/seeds for each model)"
            echo "  $0 by-model --models \"resnet18 densenet121\""
            echo ""
            echo "  # By training method"
            echo "  $0 by-method --training-methods \"constellation augmix\""
            echo ""
            echo "  # Quick test with reduced parameters"
            echo "  $0 quick --models \"resnet18 resnet50\""
            echo ""
            echo "  # Aggregate all results with layer comparison"
            echo "  $0 aggregate"
            echo ""
            echo "Available Models and Layers:"
            echo "============================="
            echo "ResNet-18: layer1, layer2, layer3, layer4, layer4.1"
            echo "ResNet-50: layer1, layer2, layer3, layer4, layer4.1"
            echo "DenseNet-121: dense1, trans1, dense2, trans2, dense3, trans3, dense4, bn"
            echo ""
            echo "Default Experimental Arrays:"
            echo "==========================="
            echo "Models: ${MODELS[*]}"
            echo "Seeds: ${SEEDS[*]}"
            echo "Training Methods: ${TRAINING_METHODS[*]}"
            echo ""
            echo "SLURM Integration:"
            echo "=================="
            echo "• Jobs are submitted in parallel for maximum efficiency"
            echo "• Automatic resource allocation based on model/layer complexity"
            echo "• GPU memory optimization for different model architectures"
            echo "• Comprehensive logging and error handling"
            echo "• Duplicate job prevention through result checking"
            echo "• Smart loops over experimental arrays"
            echo ""
            exit 0
            ;;
        *)
            # Store command for later
            if [[ -z "$COMMAND" ]]; then
                COMMAND="$1"
            else
                echo "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
            fi
            shift
            ;;
    esac
done

# Set default command if none provided
COMMAND="${COMMAND:-single}"

# Validation functions
validate_model() {
    local model="$1"
    case "$model" in
        resnet18|resnet50|densenet121)
            return 0
            ;;
        *)
            print_error "Invalid model: $model"
            print_info "Available models: resnet18, resnet50, densenet121"
            exit 1
            ;;
    esac
}

validate_layer() {
    local model="$1"
    local layer="$2"
    
    # Check if layer is valid for the model
    local valid_layers="${MODEL_LAYERS[$model]}"
    if [[ ! " $valid_layers " =~ " $layer " ]]; then
        print_error "Invalid layer '$layer' for model '$model'"
        print_info "Available layers for $model: $valid_layers"
        exit 1
    fi
}

validate_training_method() {
    local method="$1"
    local valid_methods=("constellation" "contrastive" "baseline_cross_entropy" "baseline_focal" "baseline_focal_adaptive" "baseline_brier" "baseline_mmce" "baseline_mmce_weighted" "augmix" "augmix_constellation" "ce_fast_separation" "constellation_original" "geometric_focal_calibration")
    
    for valid_method in "${valid_methods[@]}"; do
        if [[ "$method" == "$valid_method" ]]; then
            return 0
        fi
    done
    
    print_error "Invalid training method: $method"
    print_info "Available methods: ${valid_methods[*]}"
    exit 1
}

# Get all layers for a model
get_all_layers_for_model() {
    local model="$1"
    echo "${MODEL_LAYERS[$model]}"
}

# SLURM Resource Allocation
get_section4_resources() {
    local MODEL="$1"
    local TRAINING_METHOD="$2"
    local NUM_IMAGES="$3"
    
    # Base resource allocation
    local TIME="1:00:00"     # 1 hour default
    local MEM="16G"          # 16GB default
    local CPUS=4             # 4 CPUs default
    
    # Scale based on model complexity
    case "$MODEL" in
        "densenet121")
            TIME="2:00:00"   # More time for DenseNet
            MEM="32G"        # More memory for DenseNet
            CPUS=6
            ;;
        "resnet50")
            TIME="1:30:00"   # Moderate time for ResNet-50
            MEM="24G"        # Moderate memory
            CPUS=4
            ;;
        "resnet18")
            TIME="1:00:00"   # Default for ResNet-18
            MEM="16G"
            CPUS=4
            ;;
    esac
    
    # Scale based on training method complexity
    case "$TRAINING_METHOD" in
        "constellation"|"contrastive")
            # These methods may need more resources for feature extraction
            MEM=$(echo "$MEM" | sed 's/G//' | awk '{print int($1 * 1.5) "G"}')
            # Fix time calculation - handle hours and minutes properly
            local hours=$(echo "$TIME" | cut -d: -f1)
            local minutes=$(echo "$TIME" | cut -d: -f2)
            local new_hours=$(echo "$hours * 1.5" | bc | cut -d. -f1)
            TIME="${new_hours}:${minutes}:00"
            ;;
        "augmix")
            # AugMix may need additional memory for augmentations
            MEM=$(echo "$MEM" | sed 's/G//' | awk '{print int($1 * 1.2) "G"}')
            ;;
    esac
    
    # Scale based on number of images
    if [[ $NUM_IMAGES -gt 500 ]]; then
        local hours=$(echo "$TIME" | cut -d: -f1)
        local minutes=$(echo "$TIME" | cut -d: -f2)
        local new_hours=$(echo "$hours * 2" | bc | cut -d. -f1)
        TIME="${new_hours}:${minutes}:00"
        MEM=$(echo "$MEM" | sed 's/G//' | awk '{print int($1 * 1.5) "G"}')
    elif [[ $NUM_IMAGES -gt 200 ]]; then
        local hours=$(echo "$TIME" | cut -d: -f1)
        local minutes=$(echo "$TIME" | cut -d: -f2)
        local new_hours=$(echo "$hours * 1.5" | bc | cut -d. -f1)
        TIME="${new_hours}:${minutes}:00"
        MEM=$(echo "$MEM" | sed 's/G//' | awk '{print int($1 * 1.2) "G"}')
    fi
    
    echo "$TIME $MEM $CPUS"
}

# Check if results already exist
check_existing_results() {
    local MODEL="$1"
    local LAYER="$2"
    local SEED="$3"
    local TRAINING_METHOD="$4"
    local OUTPUT_DIR="$5"
    
    # Construct the actual results directory where refactored script places files
    # The refactored script creates: base_output_dir/results/{training_method}/cifar10/{model}/{layer}_compression/seed{seed}/
    local BASE_OUTPUT_DIR=$(dirname "$(dirname "$OUTPUT_DIR")")  # Go up two levels to get base
    local ACTUAL_RESULTS_DIR="${BASE_OUTPUT_DIR}/results/${TRAINING_METHOD}/cifar10/${MODEL}/${LAYER}_compression/seed${SEED}"
    
    # Construct expected result files in the new location
    local expected_files=(
        "$ACTUAL_RESULTS_DIR/statistical_analysis.png"
        "$ACTUAL_RESULTS_DIR/detailed_results.json"
        "$ACTUAL_RESULTS_DIR/statistical_significance.json"
        "$ACTUAL_RESULTS_DIR/best_examples_grid.png"
        "$ACTUAL_RESULTS_DIR/latex_table.tex"
    )
    
    # Check if all expected files exist
    for file in "${expected_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            return 1  # Results don't exist
        fi
    done
    
    # Validate JSON files if they exist
    for json_file in "$ACTUAL_RESULTS_DIR/detailed_results.json" "$ACTUAL_RESULTS_DIR/statistical_significance.json"; do
        if [[ -f "$json_file" ]]; then
            if ! python -c "import json; json.load(open('$json_file'))" 2>/dev/null; then
                return 1  # Invalid JSON
            fi
        fi
    done
    
    return 0  # Valid results exist
}

# SLURM Job Submission
submit_section4_job() {
    local MODEL="$1"
    local LAYER="$2"
    local SEED="$3"
    local TRAINING_METHOD="$4"
    local NUM_IMAGES="$5"
    local MAX_ANALYZE="$6"
    local OUTPUT_DIR="$7"
    local BATCH_SIZE="$8"
    local BASE_OUTPUT_DIR="${9:-$OUTPUT_DIR}"  # Use provided base or default to OUTPUT_DIR
    
    # Validate parameters
    validate_model "$MODEL"
    validate_layer "$MODEL" "$LAYER"
    validate_training_method "$TRAINING_METHOD"
    
    # Check for existing results
    if check_existing_results "$MODEL" "$LAYER" "$SEED" "$TRAINING_METHOD" "$OUTPUT_DIR"; then
        print_status "Valid results already exist for $MODEL-$LAYER-$SEED-$TRAINING_METHOD"
        print_info "Skipping job submission to avoid duplicate work"
        return 2  # Special code for existing results
    fi
    
    # Get resource requirements
    local RESOURCES=($(get_section4_resources "$MODEL" "$TRAINING_METHOD" "$NUM_IMAGES"))
    local TIME="${RESOURCES[0]}"
    local MEM="${RESOURCES[1]}"
    local CPUS="${RESOURCES[2]}"
    
    # Create job name
    local JOB_NAME="section4_${MODEL}_${LAYER}_${TRAINING_METHOD}_s${SEED}"
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$(dirname "$OUTPUT_DIR")/logs"
    
    # Log file
    local LOG_FILE="$(dirname "$OUTPUT_DIR")/logs/${JOB_NAME}_%j.out"
    
    # Build Python command - use the base directory
    local CMD="python $EXPERIMENTS_DIR/robust_statistical_analysis.py"
    CMD+=" --model $MODEL"
    CMD+=" --layer $LAYER"
    CMD+=" --seed $SEED"
    CMD+=" --training-method $TRAINING_METHOD"
    CMD+=" --num-images $NUM_IMAGES"
    CMD+=" --max-analyze $MAX_ANALYZE"
    CMD+=" --output-dir $BASE_OUTPUT_DIR"
    CMD+=" --results-dir aaai_full_experiments/results"
    CMD+=" --batch-size $BATCH_SIZE"
    
    print_status "Submitting SLURM job: $JOB_NAME"
    print_info "Resources: Time=$TIME, Memory=$MEM, CPUs=$CPUS"
    print_info "Command: $CMD"
    
    # Submit job
    local SUBMIT_OUTPUT
    SUBMIT_OUTPUT=$(sbatch \
        --partition="$PARTITION" \
        --job-name="$JOB_NAME" \
        --output="$LOG_FILE" \
        --time="$TIME" \
        --ntasks=1 \
        --gpus="0" \
        --cpus-per-task="$CPUS" \
        --mem="$MEM" \
        --wrap="
        echo '🔬 AAAI SECTION 4 STATISTICAL ANALYSIS'
        echo '======================================'
        echo 'Job ID: '\$SLURM_JOB_ID
        echo 'Model: $MODEL | Layer: $LAYER'
        echo 'Training Method: $TRAINING_METHOD | Seed: $SEED'
        echo 'Host: '\$(hostname)' | GPU: '\$CUDA_VISIBLE_DEVICES
        echo 'Started: '\$(date)
        echo '======================================'

        # Environment setup
        module load anaconda || { echo 'Failed to load anaconda'; exit 1; }
        
        echo '🔧 Activating conda environment: $PYTHON_ENV'
        source activate $PYTHON_ENV || { echo 'Failed to activate conda environment'; exit 1; }
        
        echo '🔍 Environment verification:'
        which python
        python --version
        
        export CUDA_HOME=/usr/local/cuda
        export CUDA_VISIBLE_DEVICES=\$(echo \$CUDA_VISIBLE_DEVICES | cut -d',' -f1)
        export PYTHONPATH='$PROJECT_ROOT:\$PYTHONPATH'
        
        echo '🔧 GPU memory management setup:'
        export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:False
        export CUDA_LAUNCH_BLOCKING=1
        
        python -c '
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f\"🧹 GPU memory cleared. Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB\")
else:
    print(\"⚠️ CUDA not available\")
'
        
        cd $PROJECT_ROOT || { echo 'Failed to change directory'; exit 1; }

        echo '🚀 Executing Section 4 statistical analysis:'
        echo '$CMD'
        echo '===================='

        if $CMD; then
            echo '✅ SECTION 4 ANALYSIS COMPLETED SUCCESSFULLY'
            echo 'Results saved to: $OUTPUT_DIR'
        else
            echo '❌ SECTION 4 ANALYSIS FAILED'
            exit 1
        fi

        echo 'Finished: '\$(date)
        " 2>&1)
    
    local SUBMIT_EXIT_CODE=$?
    
    # Check submission result
    if [[ $SUBMIT_EXIT_CODE -eq 0 ]] && [[ -n "$SUBMIT_OUTPUT" ]]; then
        # Extract job ID
        local JOB_ID
        JOB_ID=$(echo "$SUBMIT_OUTPUT" | grep -o 'Submitted batch job [0-9]*' | grep -o '[0-9]*')
        
        if [[ -n "$JOB_ID" ]] && [[ "$JOB_ID" =~ ^[0-9]+$ ]]; then
            print_status "✅ Job submitted with ID: $JOB_ID"
            print_info "📝 Log file: $LOG_FILE"
            return 0
        else
            print_error "Could not extract job ID from output: $SUBMIT_OUTPUT"
            return 1
        fi
    else
        print_error "Job submission failed with exit code: $SUBMIT_EXIT_CODE"
        print_error "Error output: $SUBMIT_OUTPUT"
        return 1
    fi
}

# Environment setup
setup_environment() {
    print_status "Setting up environment..."
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$(dirname "$OUTPUT_DIR")/logs"
    
    # Check SLURM if enabled
    if [[ "$USE_SLURM" == true ]]; then
        if ! command -v sbatch &> /dev/null; then
            print_error "SLURM not available! Use --no-slurm for direct execution"
            exit 1
        fi
        print_status "SLURM environment detected"
    else
        print_status "Direct execution mode (no SLURM)"
        
        # Check conda environment for direct execution
        if conda env list | grep -q "$PYTHON_ENV"; then
            print_status "Found conda environment: $PYTHON_ENV"
        else
            print_error "Conda environment '$PYTHON_ENV' not found!"
            echo "Please create it or use --python-env to specify a different environment"
            exit 1
        fi
        
        # Activate conda environment
        eval "$(conda shell.bash hook)"
        conda activate $PYTHON_ENV
        
        print_status "Python version: $(python --version)"
    fi
}

# Single experiment runner
run_single_experiment() {
    local MODEL="${DEFAULT_MODEL}"
    local LAYER="${DEFAULT_LAYER:-${DEFAULT_LAYERS[$MODEL]}}"
    local SEED="${DEFAULT_SEED}"
    local TRAINING_METHOD="${DEFAULT_TRAINING_METHOD}"
    
    print_status "Running SINGLE experiment:"
    print_info "  Model: $MODEL"
    print_info "  Layer: $LAYER"
    print_info "  Seed: $SEED"
    print_info "  Training Method: $TRAINING_METHOD"
    print_info "  Output: $OUTPUT_DIR"
    print_info "  SLURM: $USE_SLURM"
    
    if [[ "$USE_SLURM" == true ]]; then
        # Submit to SLURM - pass base directory separately
        local exp_output_dir="${OUTPUT_DIR}/${MODEL}_${LAYER}_${TRAINING_METHOD}_seed${SEED}"
        submit_section4_job "$MODEL" "$LAYER" "$SEED" "$TRAINING_METHOD" \
                           "$NUM_IMAGES" "$MAX_ANALYZE" "$exp_output_dir" "$BATCH_SIZE" "$BASE_OUTPUT_DIR"
        
        local exit_code=$?
        if [[ $exit_code -eq 0 ]]; then
            print_status "Job submitted successfully!"
            print_info "Monitor with: squeue -u \$USER"
            print_info "Check logs in: $(dirname "$OUTPUT_DIR")/logs/"
        elif [[ $exit_code -eq 2 ]]; then
            print_status "Results already exist - no job submitted"
            show_results_summary "$exp_output_dir"
        else
            print_error "Job submission failed!"
            exit 1
        fi
    else
        # Run directly (original behavior)
        # Validate parameters
        validate_model "$MODEL"
        validate_layer "$MODEL" "$LAYER"
        validate_training_method "$TRAINING_METHOD"
        
        # Build Python command - use the base directory
        local CMD="python $EXPERIMENTS_DIR/robust_statistical_analysis.py"
        CMD+=" --model $MODEL"
        CMD+=" --layer $LAYER"
        CMD+=" --seed $SEED"
        CMD+=" --training-method $TRAINING_METHOD"
        CMD+=" --num-images $NUM_IMAGES"
        CMD+=" --max-analyze $MAX_ANALYZE"
        CMD+=" --output-dir $BASE_OUTPUT_DIR"
        CMD+=" --results-dir aaai_full_experiments/results"
        CMD+=" --batch-size $BATCH_SIZE"
        
        print_status "Executing: $CMD"
        
        # Run the experiment
        if eval "$CMD"; then
            print_status "Single experiment completed successfully!"
            show_results_summary "$OUTPUT_DIR"
        else
            print_error "Single experiment failed!"
            exit 1
        fi
    fi
}

# ===== COMPREHENSIVE EXPERIMENTAL LOOPS =====

# Run ALL experiments across all models/layers/seeds/methods
run_all_experiments() {
    print_status "Running ALL experiments across experimental arrays..."
    print_info "  Models: ${MODELS[*]}"
    print_info "  Seeds: ${SEEDS[*]}"
    print_info "  Training Methods: ${TRAINING_METHODS[*]}"
    print_info "  SLURM: $USE_SLURM"
    
    local total_experiments=0
    local submitted_jobs=0
    local existing_results=0
    local failed_submissions=0
    
    # Count total experiments
    for model in "${MODELS[@]}"; do
        local layers=($(get_all_layers_for_model "$model"))
        for layer in "${layers[@]}"; do
            for seed in "${SEEDS[@]}"; do
                for training_method in "${TRAINING_METHODS[@]}"; do
                    total_experiments=$((total_experiments + 1))
                done
            done
        done
    done
    
    print_status "Planning $total_experiments experiments..."
    
    # Run each experiment
    local experiment_num=0
    for model in "${MODELS[@]}"; do
        local layers=($(get_all_layers_for_model "$model"))
        for layer in "${layers[@]}"; do
            for seed in "${SEEDS[@]}"; do
                for training_method in "${TRAINING_METHODS[@]}"; do
                    experiment_num=$((experiment_num + 1))
                    print_status "Experiment $experiment_num/$total_experiments: $model-$layer-$training_method-seed$seed"
                    
                    # Create experiment-specific output directory
                    local exp_output_dir="${OUTPUT_DIR}/${model}_${layer}_${training_method}_seed${seed}"
                    
                    if [[ "$USE_SLURM" == true ]]; then
                        # Submit to SLURM - pass base directory separately
                        submit_section4_job "$model" "$layer" "$seed" "$training_method" \
                                           "$NUM_IMAGES" "$MAX_ANALYZE" "$exp_output_dir" "$BATCH_SIZE" "$BASE_OUTPUT_DIR"
                        
                        local exit_code=$?
                        if [[ $exit_code -eq 0 ]]; then
                            submitted_jobs=$((submitted_jobs + 1))
                            print_status "✅ $model-$layer-$training_method-seed$seed submitted"
                        elif [[ $exit_code -eq 2 ]]; then
                            existing_results=$((existing_results + 1))
                            print_status "⏭️ $model-$layer-$training_method-seed$seed skipped (results exist)"
                        else
                            failed_submissions=$((failed_submissions + 1))
                            print_error "❌ $model-$layer-$training_method-seed$seed submission failed"
                        fi
                    else
                        # Run directly
                        mkdir -p "$exp_output_dir"
                        
                        # Build Python command - use the base directory
                        local CMD="python $EXPERIMENTS_DIR/robust_statistical_analysis.py"
                        CMD+=" --model $model"
                        CMD+=" --layer $layer"
                        CMD+=" --seed $seed"
                        CMD+=" --training-method $training_method"
                        CMD+=" --num-images $NUM_IMAGES"
                        CMD+=" --max-analyze $MAX_ANALYZE"
                        CMD+=" --output-dir $BASE_OUTPUT_DIR"
                        CMD+=" --results-dir aaai_full_experiments/results"
                        CMD+=" --batch-size $BATCH_SIZE"
                        
                        # Run experiment
                        if eval "$CMD"; then
                            submitted_jobs=$((submitted_jobs + 1))
                            print_status "✅ $model-$layer-$training_method-seed$seed completed"
                        else
                            failed_submissions=$((failed_submissions + 1))
                            print_error "❌ $model-$layer-$training_method-seed$seed failed"
                        fi
                    fi
                done
            done
        done
    done
    
    # Summary
    echo ""
    print_status "ALL EXPERIMENTS SUMMARY:"
    print_info "  Total experiments: $total_experiments"
    if [[ "$USE_SLURM" == true ]]; then
        print_info "  Jobs submitted: $submitted_jobs"
        print_info "  Existing results: $existing_results"
        print_info "  Failed submissions: $failed_submissions"
        
        if [[ $submitted_jobs -gt 0 ]]; then
            print_status "Monitor jobs with: squeue -u \$USER"
            print_info "Check logs in: $(dirname "$OUTPUT_DIR")/logs/"
        fi
    else
        print_info "  Completed: $submitted_jobs"
        print_info "  Failed: $failed_submissions"
        
        if [[ $submitted_jobs -gt 0 ]]; then
            print_status "Creating aggregated results..."
            aggregate_all_results
        fi
    fi
}

# Run experiments by model (all layers/seeds/methods for each model)
run_by_model_experiments() {
    print_status "Running experiments BY MODEL..."
    print_info "  Models: ${MODELS[*]}"
    print_info "  Seeds: ${SEEDS[*]}"
    print_info "  Training Methods: ${TRAINING_METHODS[*]}"
    
    local total_submitted=0
    local total_existing=0
    local total_failed=0
    
    for model in "${MODELS[@]}"; do
        print_status "Processing model: $model"
        local layers=($(get_all_layers_for_model "$model"))
        
        for layer in "${layers[@]}"; do
            for seed in "${SEEDS[@]}"; do
                for training_method in "${TRAINING_METHODS[@]}"; do
                    local exp_output_dir="${OUTPUT_DIR}/${model}_${layer}_${training_method}_seed${seed}"
                    
                    if [[ "$USE_SLURM" == true ]]; then
                        submit_section4_job "$model" "$layer" "$seed" "$training_method" \
                                           "$NUM_IMAGES" "$MAX_ANALYZE" "$exp_output_dir" "$BATCH_SIZE" "$BASE_OUTPUT_DIR"
                        
                        local exit_code=$?
                        if [[ $exit_code -eq 0 ]]; then
                            total_submitted=$((total_submitted + 1))
                        elif [[ $exit_code -eq 2 ]]; then
                            total_existing=$((total_existing + 1))
                        else
                            total_failed=$((total_failed + 1))
                        fi
                    fi
                done
            done
        done
    done
    
    print_status "BY MODEL SUMMARY:"
    print_info "  Jobs submitted: $total_submitted"
    print_info "  Existing results: $total_existing"
    print_info "  Failed submissions: $total_failed"
}

# Run experiments by training method (all models/layers/seeds for each method)
run_by_method_experiments() {
    print_status "Running experiments BY TRAINING METHOD..."
    print_info "  Training Methods: ${TRAINING_METHODS[*]}"
    print_info "  Models: ${MODELS[*]}"
    print_info "  Seeds: ${SEEDS[*]}"
    
    local total_submitted=0
    local total_existing=0
    local total_failed=0
    
    for training_method in "${TRAINING_METHODS[@]}"; do
        print_status "Processing training method: $training_method"
        
        for model in "${MODELS[@]}"; do
            local layers=($(get_all_layers_for_model "$model"))
            for layer in "${layers[@]}"; do
                for seed in "${SEEDS[@]}"; do
                    local exp_output_dir="${OUTPUT_DIR}/${model}_${layer}_${training_method}_seed${seed}"
                    
                    if [[ "$USE_SLURM" == true ]]; then
                        submit_section4_job "$model" "$layer" "$seed" "$training_method" \
                                           "$NUM_IMAGES" "$MAX_ANALYZE" "$exp_output_dir" "$BATCH_SIZE" "$BASE_OUTPUT_DIR"
                        
                        local exit_code=$?
                        if [[ $exit_code -eq 0 ]]; then
                            total_submitted=$((total_submitted + 1))
                        elif [[ $exit_code -eq 2 ]]; then
                            total_existing=$((total_existing + 1))
                        else
                            total_failed=$((total_failed + 1))
                        fi
                    fi
                done
            done
        done
    done
    
    print_status "BY TRAINING METHOD SUMMARY:"
    print_info "  Jobs submitted: $total_submitted"
    print_info "  Existing results: $total_existing"
    print_info "  Failed submissions: $total_failed"
}

# Run experiments by seed (all models/layers/methods for each seed)
run_by_seed_experiments() {
    print_status "Running experiments BY SEED..."
    print_info "  Seeds: ${SEEDS[*]}"
    print_info "  Models: ${MODELS[*]}"
    print_info "  Training Methods: ${TRAINING_METHODS[*]}"
    
    local total_submitted=0
    local total_existing=0
    local total_failed=0
    
    for seed in "${SEEDS[@]}"; do
        print_status "Processing seed: $seed"
        
        for model in "${MODELS[@]}"; do
            local layers=($(get_all_layers_for_model "$model"))
            for layer in "${layers[@]}"; do
                for training_method in "${TRAINING_METHODS[@]}"; do
                    local exp_output_dir="${OUTPUT_DIR}/${model}_${layer}_${training_method}_seed${seed}"
                    
                    if [[ "$USE_SLURM" == true ]]; then
                        submit_section4_job "$model" "$layer" "$seed" "$training_method" \
                                           "$NUM_IMAGES" "$MAX_ANALYZE" "$exp_output_dir" "$BATCH_SIZE" "$BASE_OUTPUT_DIR"
                        
                        local exit_code=$?
                        if [[ $exit_code -eq 0 ]]; then
                            total_submitted=$((total_submitted + 1))
                        elif [[ $exit_code -eq 2 ]]; then
                            total_existing=$((total_existing + 1))
                        else
                            total_failed=$((total_failed + 1))
                        fi
                    fi
                done
            done
        done
    done
    
    print_status "BY SEED SUMMARY:"
    print_info "  Jobs submitted: $total_submitted"
    print_info "  Existing results: $total_existing"
    print_info "  Failed submissions: $total_failed"
}

# Quick test with reduced parameters
run_quick_test() {
    print_status "Running QUICK test with reduced parameters..."
    print_info "  Models: ${MODELS[*]}"
    print_info "  SLURM: $USE_SLURM"
    
    local submitted_jobs=0
    local existing_results=0
    local failed_jobs=0
    
    # Use only first two models and default layer for quick test
    local quick_models=("${MODELS[0]}" "${MODELS[1]}")
    local quick_seed="${SEEDS[0]}"
    local quick_method="${TRAINING_METHODS[0]}"
    
    for model in "${quick_models[@]}"; do
        local layer="${DEFAULT_LAYERS[$model]}"
        
        print_status "Testing $model-$layer..."
        
        # Create test-specific output directory
        local test_output_dir="${OUTPUT_DIR}/quick_test_${model}_${layer}"
        
        if [[ "$USE_SLURM" == true ]]; then
            # Submit to SLURM with reduced parameters for quick test
            submit_section4_job "$model" "$layer" "$quick_seed" "$quick_method" \
                               "50" "100" "$test_output_dir" "$BATCH_SIZE" "$BASE_OUTPUT_DIR"
            
            local exit_code=$?
            if [[ $exit_code -eq 0 ]]; then
                submitted_jobs=$((submitted_jobs + 1))
                print_status "✅ $model-$layer test submitted"
            elif [[ $exit_code -eq 2 ]]; then
                existing_results=$((existing_results + 1))
                print_status "⏭️ $model-$layer test skipped (results exist)"
            else
                failed_jobs=$((failed_jobs + 1))
                print_warning "⚠️ $model-$layer test submission failed"
            fi
        else
            # Run directly
            mkdir -p "$test_output_dir"
            
            # Build Python command - use the base directory
            local CMD="python $EXPERIMENTS_DIR/robust_statistical_analysis.py"
            CMD+=" --model $model"
            CMD+=" --layer $layer"
            CMD+=" --seed $quick_seed"
            CMD+=" --training-method $quick_method"
            CMD+=" --num-images 50"  # Reduced for quick test
            CMD+=" --max-analyze 100"  # Reduced for quick test
            CMD+=" --output-dir $BASE_OUTPUT_DIR"
            CMD+=" --results-dir aaai_full_experiments/results"
            CMD+=" --batch-size $BATCH_SIZE"
            
            if eval "$CMD"; then
                submitted_jobs=$((submitted_jobs + 1))
                print_status "✅ $model-$layer test completed"
            else
                failed_jobs=$((failed_jobs + 1))
                print_warning "⚠️ $model-$layer test failed"
            fi
        fi
    done
    
    print_status "Quick test summary:"
    if [[ "$USE_SLURM" == true ]]; then
        print_info "  Jobs submitted: $submitted_jobs"
        print_info "  Existing results: $existing_results"
        print_info "  Failed submissions: $failed_jobs"
    else
        print_info "  Completed: $submitted_jobs"
        print_info "  Failed: $failed_jobs"
    fi
}

# Monitor SLURM jobs
monitor_jobs() {
    if [[ "$USE_SLURM" != true ]]; then
        print_warning "SLURM monitoring not available in direct execution mode"
        return
    fi
    
    print_status "Monitoring Section 4 SLURM jobs..."
    
    echo ""
    echo "📊 Current Section 4 jobs:"
    squeue -u $USER --name="section4_*" --format="%.15i %.30j %.8T %.10M %.6D %R" 2>/dev/null || echo "No Section 4 jobs in queue"
    
    echo ""
    echo "📈 Recent Section 4 job history:"
    sacct -u $USER --name="section4_*" --starttime=today --format="JobID,JobName,State,ExitCode,Start,End,Elapsed" 2>/dev/null | head -15 || echo "No recent Section 4 jobs"
    
    echo ""
    echo "❌ Recent failed jobs:"
    sacct -u $USER --name="section4_*" --starttime=today --state=FAILED --format="JobID,JobName,ExitCode,Start,End" 2>/dev/null | head -10 || echo "No failed jobs found"
    
    # Show disk usage
    echo ""
    echo "💾 Results disk usage:"
    if [[ -d "$OUTPUT_DIR" ]]; then
        du -sh "$OUTPUT_DIR" 2>/dev/null || echo "Could not calculate disk usage"
    fi
    
    # Show recent log files
    echo ""
    echo "📝 Recent log files:"
    find "$(dirname "$OUTPUT_DIR")/logs" -name "section4_*.out" -mtime -1 2>/dev/null | head -5 || echo "No recent log files"
}

# Scan for existing results
scan_results() {
    print_status "Scanning for existing results..."
    
    local total_found=0
    local valid_found=0
    local invalid_found=0
    
    # Look for result directories
    if [[ -d "$OUTPUT_DIR" ]]; then
        for result_dir in "$OUTPUT_DIR"/*; do
            if [[ -d "$result_dir" ]]; then
                total_found=$((total_found + 1))
                
                # Check for key result files
                local main_fig="$result_dir/statistical_analysis.png"
                local detail_json="$result_dir/detailed_results.json"
                local latex_table="$result_dir/latex_table.tex"
                
                if [[ -f "$main_fig" && -f "$detail_json" && -f "$latex_table" ]]; then
                    valid_found=$((valid_found + 1))
                    print_status "✅ Valid results: $(basename "$result_dir")"
                else
                    invalid_found=$((invalid_found + 1))
                    print_warning "⚠️ Incomplete results: $(basename "$result_dir")"
                fi
            fi
        done
    fi
    
    print_status "SCAN SUMMARY:"
    print_info "  Total result directories: $total_found"
    print_info "  Valid results: $valid_found"
    print_info "  Incomplete results: $invalid_found"
    
    # Show progress by experimental array
    echo ""
    print_status "Progress by experimental dimensions:"
    
    # Progress by model
    echo "📊 By Model:"
    for model in "${MODELS[@]}"; do
        local model_results=$(find "$OUTPUT_DIR" -name "*${model}_*" -type d 2>/dev/null | wc -l)
        echo "   $model: $model_results results"
    done
    
    # Progress by training method
    echo "📊 By Training Method:"
    for method in "${TRAINING_METHODS[@]}"; do
        local method_results=$(find "$OUTPUT_DIR" -name "*${method}_*" -type d 2>/dev/null | wc -l)
        echo "   $method: $method_results results"
    done
    
    # Progress by seed
    echo "📊 By Seed:"
    for seed in "${SEEDS[@]}"; do
        local seed_results=$(find "$OUTPUT_DIR" -name "*seed${seed}*" -type d 2>/dev/null | wc -l)
        echo "   seed$seed: $seed_results results"
    done
}

# Clean up incomplete results
cleanup_results() {
    print_status "Cleaning up incomplete results..."
    
    local cleaned_count=0
    
    if [[ -d "$OUTPUT_DIR" ]]; then
        for result_dir in "$OUTPUT_DIR"/*; do
            if [[ -d "$result_dir" ]]; then
                # Check for key result files
                local main_fig="$result_dir/statistical_analysis.png"
                local detail_json="$result_dir/detailed_results.json"
                local latex_table="$result_dir/latex_table.tex"
                
                if [[ ! -f "$main_fig" || ! -f "$detail_json" || ! -f "$latex_table" ]]; then
                    print_warning "Removing incomplete results: $(basename "$result_dir")"
                    rm -rf "$result_dir"
                    cleaned_count=$((cleaned_count + 1))
                fi
            fi
        done
    fi
    
    print_status "Cleanup completed! Removed $cleaned_count incomplete result directories."
}

# Aggregate results from all experiments
aggregate_all_results() {
    print_status "Aggregating results from all experiments..."
    
    # Create aggregated output directory
    local agg_dir="${OUTPUT_DIR}/aggregated_results"
    mkdir -p "$agg_dir"
    
    # Run aggregation Python script if it exists
    if [[ -f "$EXPERIMENTS_DIR/results_aggregator.py" ]]; then
        local CMD="python $EXPERIMENTS_DIR/results_aggregator.py"
        CMD+=" --input-dir $OUTPUT_DIR"
        CMD+=" --output-dir $agg_dir"
        CMD+=" --training-method $DEFAULT_TRAINING_METHOD"
        CMD+=" --seed $DEFAULT_SEED"
        
        if eval "$CMD"; then
            print_status "Results aggregation completed!"
            print_info "  Aggregated results saved to: $agg_dir"
        else
            print_warning "Results aggregation failed, but individual results are still available"
        fi
    else
        print_warning "Results aggregator script not found, skipping aggregation"
    fi
}

# Show results summary
show_results_summary() {
    local output_dir="${1:-$OUTPUT_DIR}"
    
    print_status "RESULTS SUMMARY:"
    
    # Look for the main result files
    local main_fig="$output_dir/statistical_analysis.png"
    local best_examples="$output_dir/best_examples_grid.png"
    local latex_table="$output_dir/latex_table.tex"
    local detailed_json="$output_dir/detailed_results.json"
    local significance_json="$output_dir/statistical_significance.json"
    
    echo ""
    echo "📁 Generated Files:"
    
    if [[ -f "$main_fig" ]]; then
        echo "   ✅ Main Figure: $main_fig"
    else
        echo "   ❌ Main Figure: Not found"
    fi
    
    if [[ -f "$best_examples" ]]; then
        echo "   ✅ Best Examples: $best_examples"
    else
        echo "   ❌ Best Examples: Not found"
    fi
    
    if [[ -f "$latex_table" ]]; then
        echo "   ✅ LaTeX Table: $latex_table"
    else
        echo "   ❌ LaTeX Table: Not found"
    fi
    
    if [[ -f "$detailed_json" ]]; then
        echo "   ✅ Detailed Results: $detailed_json"
    else
        echo "   ❌ Detailed Results: Not found"
    fi
    
    if [[ -f "$significance_json" ]]; then
        echo "   ✅ Statistical Tests: $significance_json"
    else
        echo "   ❌ Statistical Tests: Not found"
    fi
    
    echo ""
    if [[ "$USE_SLURM" == true ]]; then
        print_status "SLURM job submitted! Monitor progress and check results when complete."
    else
        print_status "Experiment completed! Check the files above for your results."
    fi
}

# Main execution
main() {
    echo "========================================================================"
    echo "MODULAR AAAI PAPER SECTION 4 EXPERIMENTS - SLURM VERSION"
    echo "Command: $COMMAND"
    echo "========================================================================"
    
    # Setup environment for all commands except help
    if [[ "$COMMAND" != "help" ]]; then
        setup_environment
    fi
    
    # Execute command
    case "$COMMAND" in
        "single")
            run_single_experiment
            ;;
        "all")
            run_all_experiments
            ;;
        "by-model")
            run_by_model_experiments
            ;;
        "by-method")
            run_by_method_experiments
            ;;
        "by-seed")
            run_by_seed_experiments
            ;;
        "quick")
            run_quick_test
            ;;
        "scan")
            scan_results
            ;;
        "cleanup")
            cleanup_results
            ;;
        "monitor")
            monitor_jobs
            ;;
        "aggregate")
            print_status "Running results aggregation..."
            
            # Create aggregated output directory
            local agg_dir="${OUTPUT_DIR}/aggregated_results"
            mkdir -p "$agg_dir"
            
            # Activate conda environment
            print_status "Activating conda environment: $PYTHON_ENV"
            eval "$(conda shell.bash hook)"
            conda activate $PYTHON_ENV
            
            # Verify environment
            print_info "Python version: $(python --version)"
            print_info "Python path: $(which python)"
            
            # Build aggregation command
            local CMD="python $EXPERIMENTS_DIR/results_aggregator.py"
            CMD+=" --input-dir $BASE_OUTPUT_DIR"
            CMD+=" --output-dir $agg_dir"
            CMD+=" --training-methods ${TRAINING_METHODS[*]}"
            CMD+=" --models ${MODELS[*]}"
            CMD+=" --seeds ${SEEDS[*]}"
            CMD+=" --dataset cifar10"
            
            print_info "Command: $CMD"
            
            if eval "$CMD"; then
                print_status "Results aggregation completed!"
                print_info "Aggregated results saved to: $agg_dir"
                print_info "Files generated:"
                print_info "  - layer_comparison_analysis.png (layer comparison)"
                print_info "  - comprehensive_analysis.png (main comparison)"
                print_info "  - best_layers_summary.csv (best layer per model-method)"
                print_info "  - all_layers_data.csv (all layer data)"
                print_info "  - comprehensive_latex_table.tex (publication table)"
            else
                print_error "Results aggregation failed!"
                exit 1
            fi
            ;;
        "help")
            # Help is handled in argument parsing
            exit 0
            ;;
        *)
            print_error "Unknown command: $COMMAND"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
}

# Execute main function
main 