#!/bin/bash
# Enhanced AAAI Paper Section 4 Stability Experiments with Aggregation
# Comprehensive Analysis with Built-in Results Aggregation

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

# Training methods to evaluate (comprehensive list)
TRAINING_METHODS=("baseline_brier" "baseline_mmce" "baseline_mmce_weighted" "augmix_constellation" "augmix" "ce_fast_separation" "constellation_original" "geometric_focal_calibration" "baseline_focal" "baseline_focal_adaptive" "baseline_cross_entropy" "constellation")

# Default parameters (can be overridden)
DEFAULT_TRAINING_METHOD="constellation"
DEFAULT_SEED=12
DEFAULT_MODEL="resnet18"
DEFAULT_LAYER="layer3"
DEFAULT_DATASET="cifar100"
PYTHON_ENV="tamar_n_env"
NUM_IMAGES=100
MAX_ANALYZE=200
OUTPUT_DIR="section4_stability_with_aggregation"
BASE_OUTPUT_DIR="$OUTPUT_DIR"  # Store the base directory
BATCH_SIZE=100
CIFAR10C_DIR="data/cifar10-c"  # Default CIFAR-10-C directory
CIFAR100C_DIR="data/cifar100-c"  # Default CIFAR-100-C directory

# Comprehensive analysis parameters
COMPREHENSIVE_ANALYSIS=false
SAMPLES_PER_CLASS_SEVERITY=100

# SLURM Configuration
USE_SLURM=true
PARTITION="gpu_partition"
GPU_TYPE="rtx_4090"

# Available layers per model
declare -A MODEL_LAYERS
MODEL_LAYERS[resnet18]="layer1 layer2 layer3 layer4 layer4.1 pixel"
MODEL_LAYERS[resnet50]="layer1 layer2 layer3 layer4 layer4.1 pixel"
MODEL_LAYERS[densenet121]="dense1 trans1 dense2 trans2 dense3 trans3 dense4 bn pixel"

# Default layers per model (used for single experiments)
declare -A DEFAULT_LAYERS
DEFAULT_LAYERS[resnet18]="layer3"
DEFAULT_LAYERS[resnet50]="layer3"
DEFAULT_LAYERS[densenet121]="trans3"

# All 15 CIFAR-10-C corruption types for stability analysis
ALL_CORRUPTION_TYPES=("gaussian_noise" "shot_noise" "impulse_noise" "defocus_blur" "glass_blur" "motion_blur" "zoom_blur" "snow" "frost" "fog" "brightness" "contrast" "elastic_transform" "pixelate" "jpeg_compression")

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
        --dataset)
            DEFAULT_DATASET="$2"
            shift 2
            ;;
        --model)
            DEFAULT_MODEL="$2"
            shift 2
            ;;
        --models)
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
            IFS=' ' read -ra SEEDS <<< "$2"
            shift 2
            ;;
        --training-method)
            DEFAULT_TRAINING_METHOD="$2"
            shift 2
            ;;
        --training-methods)
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
        --cifar10c-dir)
            CIFAR10C_DIR="$2"
            shift 2
            ;;
        --cifar100c-dir)
            CIFAR100C_DIR="$2"
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
        --comprehensive-analysis)
            COMPREHENSIVE_ANALYSIS=true
            shift
            ;;
        --samples-per-class-severity)
            SAMPLES_PER_CLASS_SEVERITY="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS] [COMMAND]"
            echo ""
            echo "Enhanced AAAI Paper Section 4 Stability Experiments with Aggregation"
            echo "=================================================================="
            echo ""
            echo "NEW FEATURES:"
            echo "• Built-in results aggregation across all experimental conditions"
            echo "• Comprehensive stability analysis with layer selection insights"
            echo "• Publication-ready visualizations and statistical analysis"
            echo "• Support for both individual and batch experiment submission"
            echo ""
            echo "Commands:"
            echo "  single           - Submit single stability experiment to SLURM queue"
            echo "  all              - Submit ALL stability experiments across all conditions"
            echo "  all-comprehensive - Submit ALL experiments with comprehensive multi-dimensional analysis"
            echo "  by-model         - Submit experiments for all layers/seeds for each model"
            echo "  by-method        - Submit experiments for all models/layers/seeds for each method"
            echo "  by-seed          - Submit experiments for all models/layers/methods for each seed"
            echo "  quick            - Submit quick test jobs with reduced parameters"
            echo "  quick-comprehensive - Quick test with comprehensive analysis"
            echo "  scan             - Scan for existing results"
            echo "  monitor          - Monitor running SLURM jobs"
            echo "  cleanup          - Clean up incomplete results"
            echo "  aggregate        - Aggregate all results with comprehensive analysis"
            echo "  full-pipeline    - Run complete pipeline: experiments + aggregation"
            echo "  full-comprehensive - Run complete pipeline with comprehensive analysis"
            echo "  help             - Show this help message"
            echo ""
            echo "Options:"
            echo "  --dataset DATASET          Dataset to use (default: cifar10, choices: cifar10, cifar100)"
            echo "  --model MODEL              Single model to analyze (default: resnet18)"
            echo "  --models \"MODEL1 MODEL2\"   Multiple models (default: \"resnet18 resnet50 densenet121\")"
            echo "  --layer LAYER              Single layer to analyze (default: layer3)"
            echo "  --seed SEED                Single seed (default: 12)"
            echo "  --seeds \"SEED1 SEED2\"      Multiple seeds (default: \"12 13 14 11\")"
            echo "  --training-method METHOD   Single training method (default: constellation)"
            echo "  --training-methods \"M1 M2\" Multiple methods (default: comprehensive list)"
            echo "  --num-images NUM           Number of images to analyze (default: 100)"
            echo "  --max-analyze NUM          Maximum examples to analyze (default: 200)"
            echo "  --output-dir DIR           Output directory (default: section4_stability_with_aggregation)"
            echo "  --cifar10c-dir DIR         CIFAR-10-C data directory (default: data/cifar10-c)"
            echo "  --cifar100c-dir DIR        CIFAR-100-C data directory (default: data/cifar100-c)"
            echo "  --batch-size SIZE          Batch size for processing (default: 100)"
            echo "  --python-env ENV           Conda environment name (default: tamar_n_env)"
            echo ""
            echo "Comprehensive Analysis Options:"
            echo "  --comprehensive-analysis   Enable multi-dimensional analysis across classes and severities"
            echo "  --samples-per-class-severity NUM  Samples per class-severity combination (default: 20)"
            echo ""
            echo "SLURM Options:"
            echo "  --no-slurm                 Run directly without SLURM submission"
            echo "  --partition PARTITION      SLURM partition (default: gpu_partition)"
            echo "  --gpu-type TYPE            GPU type (default: rtx_4090)"
            echo ""
            echo "Examples:"
            echo "  # Complete pipeline (experiments + aggregation)"
            echo "  $0 full-pipeline --dataset cifar10"
            echo ""
            echo "  # Complete pipeline with comprehensive multi-dimensional analysis"
            echo "  $0 full-comprehensive --dataset cifar10 --samples-per-class-severity 30"
            echo ""
            echo "  # Run all experiments with comprehensive analysis"
            echo "  $0 all-comprehensive --dataset cifar100 && $0 aggregate --dataset cifar100"
            echo ""
            echo "  # Run all experiments then aggregate"
            echo "  $0 all --dataset cifar10 && $0 aggregate --dataset cifar10"
            echo ""
            echo "  # Quick test pipeline with comprehensive analysis"
            echo "  $0 quick-comprehensive --dataset cifar100 && $0 aggregate --dataset cifar100"
            echo ""
            echo "  # Quick test pipeline"
            echo "  $0 quick --dataset cifar100 && $0 aggregate --dataset cifar100"
            echo ""
            echo "  # Single experiment with comprehensive analysis"
            echo "  $0 single --dataset cifar10 --model resnet18 --layer layer3 --comprehensive-analysis"
            echo ""
            echo "  # Single experiment on CIFAR-10"
            echo "  $0 single --dataset cifar10 --model resnet18 --layer layer3 --seed 12"
            echo ""
            echo "  # Single experiment on CIFAR-100"
            echo "  $0 single --dataset cifar100 --model densenet121 --layer trans3 --seed 12"
            echo ""
            echo "  # Custom experimental arrays with comprehensive analysis"
            echo "  $0 all --dataset cifar100 --models \"resnet18 resnet50\" --comprehensive-analysis --samples-per-class-severity 25"
            echo ""
            echo "Key Features:"
            echo "============="
            echo "• Comprehensive stability analysis across corruption types"
            echo "• Multi-dimensional analysis across ALL classes and severities (--comprehensive-analysis)"
            echo "• Layer selection optimization and consistency analysis"
            echo "• Semantic vs physical space advantage quantification"
            echo "• Training method robustness assessment"
            echo "• Publication-ready figures and LaTeX tables"
            echo "• Statistical significance testing and confidence intervals"
            echo "• Granular insights: 'For automobile class, layer X is optimal'"
            echo ""
            echo "Output Structure:"
            echo "================"
            echo "section4_stability_with_aggregation/"
            echo "├── results/{training_method}/{dataset}/{model}/{layer}_stability/seed{seed}/"
            echo "│   ├── {corruption}/stability_analysis.png"
            echo "│   ├── {corruption}/stability_detailed_results.json"
            echo "│   └── multi_corruption_summary.png"
            echo "├── aggregated_results/{dataset}/"
            echo "│   ├── comprehensive_stability_analysis_{dataset}.png"
            echo "│   ├── stability_analysis_overall_data.csv"
            echo "│   ├── stability_analysis_aggregated.json"
            echo "│   └── stability_analysis_summary.csv"
            echo "└── logs/"
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
validate_dataset() {
    local dataset="$1"
    case "$dataset" in
        cifar10|cifar100)
            return 0
            ;;
        *)
            print_error "Invalid dataset: $dataset"
            print_info "Available datasets: cifar10, cifar100"
            exit 1
            ;;
    esac
}

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

# Get corruption directory based on dataset
get_corruption_dir() {
    local dataset="$1"
    case "$dataset" in
        cifar10)
            echo "$CIFAR10C_DIR"
            ;;
        cifar100)
            echo "$CIFAR100C_DIR"
            ;;
        *)
            print_error "Unknown dataset: $dataset"
            exit 1
            ;;
    esac
}

# SLURM Resource Allocation for Stability Analysis
get_stability_resources() {
    local MODEL="$1"
    local TRAINING_METHOD="$2"
    local NUM_IMAGES="$3"
    
    # Base resource allocation for stability analysis
    local TIME="4:00:00"     # 4 hours for comprehensive stability analysis
    local MEM="24G"          # 24GB default
    local CPUS=4             # 4 CPUs default
    
    # Scale based on model complexity
    case "$MODEL" in
        "densenet121")
            TIME="6:00:00"   # More time for DenseNet
            MEM="32G"        # More memory for DenseNet
            CPUS=6
            ;;
        "resnet50")
            TIME="5:00:00"   # Moderate time for ResNet-50
            MEM="28G"        # Moderate memory
            CPUS=4
            ;;
        "resnet18")
            TIME="4:00:00"   # Default for ResNet-18
            MEM="24G"
            CPUS=4
            ;;
    esac
    
    # Scale based on training method complexity
    case "$TRAINING_METHOD" in
        "constellation"|"contrastive")
            # These methods may need more resources
            MEM=$(echo "$MEM" | sed 's/G//' | awk '{print int($1 * 1.3) "G"}')
            local hours=$(echo "$TIME" | cut -d: -f1)
            local minutes=$(echo "$TIME" | cut -d: -f2)
            local new_hours=$(echo "$hours * 1.2" | bc | cut -d. -f1)
            TIME="${new_hours}:${minutes}:00"
            ;;
        "augmix"|"augmix_constellation")
            # AugMix methods may need additional resources
            MEM=$(echo "$MEM" | sed 's/G//' | awk '{print int($1 * 1.2) "G"}')
            ;;
    esac
    
    # Scale based on number of images
    if [[ $NUM_IMAGES -gt 300 ]]; then
        local hours=$(echo "$TIME" | cut -d: -f1)
        local minutes=$(echo "$TIME" | cut -d: -f2)
        local new_hours=$(echo "$hours * 1.5" | bc | cut -d. -f1)
        TIME="${new_hours}:${minutes}:00"
        MEM=$(echo "$MEM" | sed 's/G//' | awk '{print int($1 * 1.2) "G"}')
    fi
    
    echo "$TIME $MEM $CPUS"
}

# Check if stability analysis results already exist
check_stability_results() {
    local MODEL="$1"
    local LAYER="$2"
    local SEED="$3"
    local TRAINING_METHOD="$4"
    local OUTPUT_DIR="$5"
    
    # Construct the actual results directory
    local BASE_OUTPUT_DIR=$(dirname "$(dirname "$OUTPUT_DIR")")
    local ACTUAL_RESULTS_DIR="${BASE_OUTPUT_DIR}/results/${TRAINING_METHOD}/${DEFAULT_DATASET}/${MODEL}/${LAYER}_stability/seed${SEED}"
    
    # Check for main analysis files
    local main_files=(
        "$ACTUAL_RESULTS_DIR/stability_analysis.png"
        "$ACTUAL_RESULTS_DIR/detailed_results.json"
        "$ACTUAL_RESULTS_DIR/statistical_significance.json"
    )
    
    # Check if main files exist
    for file in "${main_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            return 1  # Main results don't exist
        fi
    done
    
    # Validate JSON files
    for json_file in "$ACTUAL_RESULTS_DIR/detailed_results.json" "$ACTUAL_RESULTS_DIR/statistical_significance.json"; do
        if [[ -f "$json_file" ]]; then
            if ! python -c "import json; json.load(open('$json_file'))" 2>/dev/null; then
                return 1  # Invalid JSON
            fi
        fi
    done
    
    return 0  # Valid results exist
}

# SLURM Job Submission for Stability Analysis
submit_stability_job() {
    local MODEL="$1"
    local LAYER="$2"
    local SEED="$3"
    local TRAINING_METHOD="$4"
    local NUM_IMAGES="$5"
    local MAX_ANALYZE="$6"
    local OUTPUT_DIR="$7"
    local BATCH_SIZE="$8"
    local BASE_OUTPUT_DIR="${9:-$OUTPUT_DIR}"
    
    # Validate parameters
    validate_dataset "$DEFAULT_DATASET"
    validate_model "$MODEL"
    validate_layer "$MODEL" "$LAYER"
    validate_training_method "$TRAINING_METHOD"
    
    # Check for existing results
    if check_stability_results "$MODEL" "$LAYER" "$SEED" "$TRAINING_METHOD" "$OUTPUT_DIR"; then
        print_status "Valid stability results already exist for $MODEL-$LAYER-$SEED-$TRAINING_METHOD"
        print_info "Skipping job submission to avoid duplicate work"
        return 2  # Special code for existing results
    fi
    
    # Get resource requirements
    local RESOURCES=($(get_stability_resources "$MODEL" "$TRAINING_METHOD" "$NUM_IMAGES"))
    local TIME="${RESOURCES[0]}"
    local MEM="${RESOURCES[1]}"
    local CPUS="${RESOURCES[2]}"
    
    # Create job name
    local JOB_NAME="stab_${MODEL}_${LAYER}_${TRAINING_METHOD}_s${SEED}"
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$(dirname "$OUTPUT_DIR")/logs"
    
    # Log file
    local LOG_FILE="$(dirname "$OUTPUT_DIR")/logs/${JOB_NAME}_%j.out"
    
    # Build Python command for stability analysis
    local CMD="python $EXPERIMENTS_DIR/stability_statistical_analysis.py"
    CMD+=" --model $MODEL"
    CMD+=" --layer $LAYER"
    CMD+=" --seed $SEED"
    CMD+=" --training-method $TRAINING_METHOD"
    CMD+=" --dataset $DEFAULT_DATASET"
    CMD+=" --num-images $NUM_IMAGES"
    CMD+=" --max-analyze $MAX_ANALYZE"
    CMD+=" --output-dir $BASE_OUTPUT_DIR"
    CMD+=" --results-dir aaai_full_experiments/results"
    CMD+=" --batch-size $BATCH_SIZE"
    CMD+=" --all-corruptions"  # CRITICAL: Process all corruption types
    
    # Add comprehensive analysis flags if enabled
    if [[ "$COMPREHENSIVE_ANALYSIS" == true ]]; then
        CMD+=" --comprehensive-analysis"
        CMD+=" --samples-per-class-severity $SAMPLES_PER_CLASS_SEVERITY"
    fi
    
    local CORRUPTION_DIR=$(get_corruption_dir "$DEFAULT_DATASET")
    if [[ "$DEFAULT_DATASET" == "cifar10" ]]; then
        CMD+=" --cifar10c-dir $CORRUPTION_DIR"
    else
        CMD+=" --cifar100c-dir $CORRUPTION_DIR"
    fi
    
    print_status "Submitting Stability Analysis SLURM job: $JOB_NAME"
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
        --gpus=${GPU_TYPE}:1 \
        --cpus-per-task="$CPUS" \
        --mem="$MEM" \
        --wrap="
        echo '🔬 AAAI SECTION 4: STABILITY ANALYSIS'
        echo '====================================='
        echo 'Job ID: '\$SLURM_JOB_ID
        echo 'Model: $MODEL | Layer: $LAYER'
        echo 'Training Method: $TRAINING_METHOD | Seed: $SEED'
        echo 'Host: '\$(hostname)' | GPU: '\$CUDA_VISIBLE_DEVICES
        echo 'Started: '\$(date)
        echo '====================================='

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

        echo '🚀 Executing Stability Analysis:'
        echo '$CMD'
        echo '===================='

        if $CMD; then
            echo '✅ STABILITY ANALYSIS COMPLETED SUCCESSFULLY'
            echo 'Results saved to: $OUTPUT_DIR'
        else
            echo '❌ STABILITY ANALYSIS FAILED'
            exit 1
        fi

        echo 'Finished: '\$(date)
        " 2>&1)
    
    local SUBMIT_EXIT_CODE=$?
    
    # Check submission result
    if [[ $SUBMIT_EXIT_CODE -eq 0 ]] && [[ -n "$SUBMIT_OUTPUT" ]]; then
        local JOB_ID
        JOB_ID=$(echo "$SUBMIT_OUTPUT" | grep -o 'Submitted batch job [0-9]*' | grep -o '[0-9]*')
        
        if [[ -n "$JOB_ID" ]] && [[ "$JOB_ID" =~ ^[0-9]+$ ]]; then
            print_status "✅ Stability job submitted with ID: $JOB_ID"
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
    print_status "Setting up stability analysis environment..."
    
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
    
    print_status "Running SINGLE stability experiment:"
    print_info "  Dataset: $DEFAULT_DATASET"
    print_info "  Model: $MODEL"
    print_info "  Layer: $LAYER"
    print_info "  Seed: $SEED"
    print_info "  Training Method: $TRAINING_METHOD"
    print_info "  Output: $OUTPUT_DIR"
    print_info "  SLURM: $USE_SLURM"
    print_info "  Comprehensive Analysis: $COMPREHENSIVE_ANALYSIS"
    if [[ "$COMPREHENSIVE_ANALYSIS" == true ]]; then
        print_info "  Samples per class-severity: $SAMPLES_PER_CLASS_SEVERITY"
    fi
    
    if [[ "$USE_SLURM" == true ]]; then
        # Submit to SLURM
        local exp_output_dir="${OUTPUT_DIR}/${MODEL}_${LAYER}_${TRAINING_METHOD}_seed${SEED}"
        submit_stability_job "$MODEL" "$LAYER" "$SEED" "$TRAINING_METHOD" \
                           "$NUM_IMAGES" "$MAX_ANALYZE" "$exp_output_dir" "$BATCH_SIZE" "$BASE_OUTPUT_DIR"
        
        local exit_code=$?
        if [[ $exit_code -eq 0 ]]; then
            print_status "Stability job submitted successfully!"
            print_info "Monitor with: squeue -u \$USER"
            print_info "Check logs in: $(dirname "$OUTPUT_DIR")/logs/"
        elif [[ $exit_code -eq 2 ]]; then
            print_status "Results already exist - no job submitted"
        else
            print_error "Job submission failed!"
            exit 1
        fi
    else
        # Run directly
        validate_model "$MODEL"
        validate_layer "$MODEL" "$LAYER"
        validate_training_method "$TRAINING_METHOD"
        
        # Build Python command
        local CMD="python $EXPERIMENTS_DIR/stability_statistical_analysis.py"
        CMD+=" --model $MODEL"
        CMD+=" --layer $LAYER"
        CMD+=" --seed $SEED"
        CMD+=" --training-method $TRAINING_METHOD"
        CMD+=" --dataset $DEFAULT_DATASET"
        CMD+=" --num-images $NUM_IMAGES"
        CMD+=" --max-analyze $MAX_ANALYZE"
        CMD+=" --output-dir $BASE_OUTPUT_DIR"
        CMD+=" --results-dir aaai_full_experiments/results"
        CMD+=" --batch-size $BATCH_SIZE"
        CMD+=" --all-corruptions"  # CRITICAL: Process all corruption types
        
        # Add comprehensive analysis flags if enabled
        if [[ "$COMPREHENSIVE_ANALYSIS" == true ]]; then
            CMD+=" --comprehensive-analysis"
            CMD+=" --samples-per-class-severity $SAMPLES_PER_CLASS_SEVERITY"
        fi
        
        local CORRUPTION_DIR=$(get_corruption_dir "$DEFAULT_DATASET")
        if [[ "$DEFAULT_DATASET" == "cifar10" ]]; then
            CMD+=" --cifar10c-dir $CORRUPTION_DIR"
        else
            CMD+=" --cifar100c-dir $CORRUPTION_DIR"
        fi
        
        print_status "Executing: $CMD"
        
        # Run the experiment
        if eval "$CMD"; then
            print_status "Single stability experiment completed successfully!"
        else
            print_error "Single stability experiment failed!"
            exit 1
        fi
    fi
}

# Run ALL experiments across all models/layers/seeds/methods
run_all_experiments() {
    print_status "Running ALL stability experiments across experimental arrays..."
    print_info "  Dataset: $DEFAULT_DATASET"
    print_info "  Models: ${MODELS[*]}"
    print_info "  Seeds: ${SEEDS[*]}"
    print_info "  Training Methods: ${TRAINING_METHODS[*]}"
    print_info "  SLURM: $USE_SLURM"
    print_info "  Comprehensive Analysis: $COMPREHENSIVE_ANALYSIS"
    if [[ "$COMPREHENSIVE_ANALYSIS" == true ]]; then
        print_info "  Samples per class-severity: $SAMPLES_PER_CLASS_SEVERITY"
        print_info "  🎯 Will analyze ALL classes and ALL severities for multi-dimensional insights!"
    fi
    
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
    
    print_status "Planning $total_experiments stability experiments..."
    
    # Run each experiment
    local experiment_num=0
    for model in "${MODELS[@]}"; do
        local layers=($(get_all_layers_for_model "$model"))
        for layer in "${layers[@]}"; do
            for seed in "${SEEDS[@]}"; do
                for training_method in "${TRAINING_METHODS[@]}"; do
                    experiment_num=$((experiment_num + 1))
                    print_status "Stability experiment $experiment_num/$total_experiments: $model-$layer-$training_method-seed$seed"
                    
                    local exp_output_dir="${OUTPUT_DIR}/${model}_${layer}_${training_method}_seed${seed}"
                    
                    if [[ "$USE_SLURM" == true ]]; then
                        submit_stability_job "$model" "$layer" "$seed" "$training_method" \
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
                        
                        local CMD="python $EXPERIMENTS_DIR/stability_statistical_analysis.py"
                        CMD+=" --model $model"
                        CMD+=" --layer $layer"
                        CMD+=" --seed $seed"
                        CMD+=" --training-method $training_method"
                        CMD+=" --dataset $DEFAULT_DATASET"
                        CMD+=" --num-images $NUM_IMAGES"
                        CMD+=" --max-analyze $MAX_ANALYZE"
                        CMD+=" --output-dir $BASE_OUTPUT_DIR"
                        CMD+=" --results-dir aaai_full_experiments/results"
                        CMD+=" --batch-size $BATCH_SIZE"
                        CMD+=" --all-corruptions"  # CRITICAL: Process all corruption types
                        
                        # Add comprehensive analysis flags if enabled
                        if [[ "$COMPREHENSIVE_ANALYSIS" == true ]]; then
                            CMD+=" --comprehensive-analysis"
                            CMD+=" --samples-per-class-severity $SAMPLES_PER_CLASS_SEVERITY"
                        fi
                        
                        local CORRUPTION_DIR=$(get_corruption_dir "$DEFAULT_DATASET")
                        if [[ "$DEFAULT_DATASET" == "cifar10" ]]; then
                            CMD+=" --cifar10c-dir $CORRUPTION_DIR"
                        else
                            CMD+=" --cifar100c-dir $CORRUPTION_DIR"
                        fi
                        
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
    print_status "ALL STABILITY EXPERIMENTS SUMMARY:"
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
            print_status "Running aggregation after completion..."
            aggregate_all_results
        fi
    fi
}

# Quick test with reduced parameters
run_quick_test() {
    print_status "Running QUICK stability test with reduced parameters..."
    print_info "  Dataset: $DEFAULT_DATASET"
    print_info "  Models: ${MODELS[*]}"
    print_info "  SLURM: $USE_SLURM"
    print_info "  Comprehensive Analysis: $COMPREHENSIVE_ANALYSIS"
    if [[ "$COMPREHENSIVE_ANALYSIS" == true ]]; then
        print_info "  Samples per class-severity: $SAMPLES_PER_CLASS_SEVERITY"
        print_info "  🎯 Quick test with multi-dimensional insights!"
    fi
    
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
        
        local test_output_dir="${OUTPUT_DIR}/quick_test_${model}_${layer}"
        
        if [[ "$USE_SLURM" == true ]]; then
            submit_stability_job "$model" "$layer" "$quick_seed" "$quick_method" \
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
            mkdir -p "$test_output_dir"
            
            local CMD="python $EXPERIMENTS_DIR/stability_statistical_analysis.py"
            CMD+=" --model $model"
            CMD+=" --layer $layer"
            CMD+=" --seed $quick_seed"
            CMD+=" --training-method $quick_method"
            CMD+=" --dataset $DEFAULT_DATASET"
            CMD+=" --num-images 50"
            CMD+=" --max-analyze 100"
            CMD+=" --output-dir $BASE_OUTPUT_DIR"
            CMD+=" --results-dir aaai_full_experiments/results"
            CMD+=" --batch-size $BATCH_SIZE"
            CMD+=" --all-corruptions"  # CRITICAL: Process all corruption types
            
            # Add comprehensive analysis flags if enabled
            if [[ "$COMPREHENSIVE_ANALYSIS" == true ]]; then
                CMD+=" --comprehensive-analysis"
                CMD+=" --samples-per-class-severity $SAMPLES_PER_CLASS_SEVERITY"
            fi
            
            local CORRUPTION_DIR=$(get_corruption_dir "$DEFAULT_DATASET")
            if [[ "$DEFAULT_DATASET" == "cifar10" ]]; then
                CMD+=" --cifar10c-dir $CORRUPTION_DIR"
            else
                CMD+=" --cifar100c-dir $CORRUPTION_DIR"
            fi
            
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
        
        if [[ $submitted_jobs -gt 0 ]]; then
            print_status "Running aggregation after quick test..."
            aggregate_all_results
        fi
    fi
}

# Monitor SLURM jobs
monitor_jobs() {
    if [[ "$USE_SLURM" != true ]]; then
        print_warning "SLURM monitoring not available in direct execution mode"
        return
    fi
    
    print_status "Monitoring Stability Analysis SLURM jobs..."
    
    echo ""
    echo "📊 Current Stability Analysis jobs:"
    squeue -u $USER --name="stab_*" --format="%.18i %.35j %.8T %.10M %.6D %R" 2>/dev/null || echo "No Stability Analysis jobs in queue"
    
    echo ""
    echo "📈 Recent Stability job history:"
    sacct -u $USER --name="stab_*" --starttime=today --format="JobID,JobName,State,ExitCode,Start,End,Elapsed" 2>/dev/null | head -15 || echo "No recent Stability jobs"
    
    echo ""
    echo "❌ Recent failed Stability jobs:"
    sacct -u $USER --name="stab_*" --starttime=today --state=FAILED --format="JobID,JobName,ExitCode,Start,End" 2>/dev/null | head -10 || echo "No failed Stability jobs found"
    
    # Show disk usage
    echo ""
    echo "💾 Results disk usage:"
    if [[ -d "$OUTPUT_DIR" ]]; then
        du -sh "$OUTPUT_DIR" 2>/dev/null || echo "Could not calculate disk usage"
    fi
}

# Aggregate all results
aggregate_all_results() {
    print_status "Running comprehensive stability analysis aggregation..."
    
    # Create aggregated output directory with dataset name
    local agg_dir="${OUTPUT_DIR}/aggregated_results/${DEFAULT_DATASET}"
    mkdir -p "$agg_dir"
    
    # Always activate conda environment for aggregation (runs locally)
    print_status "Activating conda environment: $PYTHON_ENV"
    
    # Load anaconda module if available (for SLURM environments)
    if command -v module &> /dev/null; then
        module load anaconda 2>/dev/null || print_warning "Could not load anaconda module"
    fi
    
    # Activate conda environment
    eval "$(conda shell.bash hook)"
    conda activate $PYTHON_ENV || {
        print_error "Failed to activate conda environment: $PYTHON_ENV"
        print_info "Available environments:"
        conda env list
        exit 1
    }
    
    # Verify environment
    print_info "Python version: $(python --version)"
    print_info "Python path: $(which python)"
    
    # Build aggregation command - use the correct input directory with correlation data
    local CMD="python $EXPERIMENTS_DIR/stability_analysis_aggregator.py"
    CMD+=" --input-dir section4_stability_analysis"
    CMD+=" --output-dir $agg_dir"
    CMD+=" --training-methods ${TRAINING_METHODS[*]}"
    CMD+=" --models ${MODELS[*]}"
    CMD+=" --seeds ${SEEDS[*]}"
    CMD+=" --dataset $DEFAULT_DATASET"
    
    print_info "Aggregation command: $CMD"
    
    if eval "$CMD"; then
        print_status "Stability analysis aggregation completed for $DEFAULT_DATASET!"
        print_info "Aggregated results saved to: $agg_dir"
        print_info "Dataset: $DEFAULT_DATASET"
        print_info ""
        print_info "🎯 Key files generated:"
        print_info "  📊 comprehensive_stability_analysis_${DEFAULT_DATASET}.png - Main publication figure"
        print_info "  📋 stability_analysis_overall_data_${DEFAULT_DATASET}.csv - Raw experimental data"
        print_info "  📈 stability_analysis_aggregated_${DEFAULT_DATASET}.json - Comprehensive statistics"
        print_info "  📝 stability_analysis_summary_${DEFAULT_DATASET}.csv - Key findings summary"
        print_info ""
        print_status "🎉 Section 4 analysis ready for AAAI submission!"
    else
        print_error "Stability analysis aggregation failed!"
        return 1
    fi
}

# Run complete pipeline (experiments + aggregation)
run_full_pipeline() {
    print_status "Running COMPLETE STABILITY ANALYSIS PIPELINE"
    print_info "Phase 1: Running all experiments"
    print_info "Phase 2: Aggregating results"
    print_info "Comprehensive Analysis: $COMPREHENSIVE_ANALYSIS"
    if [[ "$COMPREHENSIVE_ANALYSIS" == true ]]; then
        print_info "🎯 FULL COMPREHENSIVE ANALYSIS: All classes, all severities, dimensional insights!"
        print_info "Samples per class-severity: $SAMPLES_PER_CLASS_SEVERITY"
    fi
    
    # Phase 1: Run all experiments
    run_all_experiments
    
    # Wait for SLURM jobs to complete if using SLURM
    if [[ "$USE_SLURM" == true ]]; then
        print_status "Waiting for SLURM jobs to complete before aggregation..."
        print_info "Monitor job progress with: $0 monitor"
        print_info "Run aggregation manually when jobs complete: $0 aggregate"
        print_warning "Pipeline paused - run '$0 aggregate' after jobs complete"
    else
        # Phase 2: Aggregate results (already called in run_all_experiments for direct mode)
        print_status "Complete pipeline finished successfully!"
    fi
}

# Main execution
main() {
    echo "========================================================================"
    echo "ENHANCED AAAI PAPER SECTION 4 STABILITY EXPERIMENTS WITH AGGREGATION"
    echo "Comprehensive Analysis for Semantic-Space Geometric Calibration"
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
        "all-comprehensive")
            COMPREHENSIVE_ANALYSIS=true
            run_all_experiments
            ;;
        "by-model"|"by-method"|"by-seed")
            print_warning "Individual grouping commands not yet implemented for stability analysis"
            print_info "Use 'all', 'single', or 'quick' commands"
            exit 1
            ;;
        "quick")
            run_quick_test
            ;;
        "quick-comprehensive")
            COMPREHENSIVE_ANALYSIS=true
            run_quick_test
            ;;
        "scan")
            print_status "Scanning for stability results..."
            # Implementation for scanning results
            ;;
        "cleanup")
            print_status "Cleaning up incomplete stability results..."
            # Implementation for cleanup
            ;;
        "monitor")
            monitor_jobs
            ;;
        "aggregate")
            aggregate_all_results
            ;;
        "full-pipeline")
            run_full_pipeline
            ;;
        "full-comprehensive")
            COMPREHENSIVE_ANALYSIS=true
            run_full_pipeline
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