#!/bin/bash
# Enhanced AAAI Paper Section 4 Multi-Corruption Robust Analysis Runner
# Comprehensive Analysis with ALL Classes and ALL Severities Support
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
MODELS=("resnet50" "resnet18" "densenet121" "dinov2_small_scratch" "dinov2_base_scratch")

# Seeds for robust evaluation
SEEDS=(13 14 11 12)

# Training methods to evaluate (comprehensive list)
TRAINING_METHODS=("baseline_cross_entropy")

# Default parameters (can be overridden)
DEFAULT_TRAINING_METHOD="constellation"
DEFAULT_SEED=12
DEFAULT_MODEL="resnet18"
DEFAULT_LAYER="layer3"
DEFAULT_DATASET="cifar10"
PYTHON_ENV="tamar_n_env"
NUM_IMAGES=100
MAX_ANALYZE=200
OUTPUT_DIR="section4_robust_analysis"
BASE_OUTPUT_DIR="$OUTPUT_DIR"  # Store the base directory
RESULTS_DIR="aaai_full_experiments/results" # Base directory containing trained models
BATCH_SIZE=100
K_VALUES="1 3 5" # Default k-NN values
CIFAR10C_DIR="data/cifar10-c"  # Default CIFAR-10-C directory
CIFAR100C_DIR="data/cifar100-c"  # Default CIFAR-100-C directory

# Comprehensive analysis parameters
COMPREHENSIVE_ANALYSIS=false
SAMPLES_PER_CLASS_SEVERITY=30

# SLURM Configuration
USE_SLURM=true
PARTITION="gpu_partition"
GPU_TYPE="rtx_4090"

# Available layers per model
declare -A MODEL_LAYERS
MODEL_LAYERS[resnet18]="layer1 layer2 layer3 layer4 layer4.1"
MODEL_LAYERS[resnet50]="layer1 layer2 layer3 layer4 layer4.1"
MODEL_LAYERS[densenet121]="dense1 trans1 dense2 trans2 dense3 trans3 dense4 bn"
MODEL_LAYERS[dinov2_small_scratch]="block_0 block_2 block_5 block_8 block_11 norm"
MODEL_LAYERS[dinov2_base_scratch]="block_0 block_2 block_5 block_8 block_11 norm"

# Default layers per model (used for single experiments)
declare -A DEFAULT_LAYERS
DEFAULT_LAYERS[resnet18]="layer3"
DEFAULT_LAYERS[resnet50]="layer3"
DEFAULT_LAYERS[densenet121]="trans3"
DEFAULT_LAYERS[dinov2_small_scratch]="block_8"
DEFAULT_LAYERS[dinov2_base_scratch]="block_8"

# All 15 CIFAR-10-C corruption types for robust analysis
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
        --k-values)
            K_VALUES="$2"
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
            echo "🚀 Enhanced AAAI Paper Section 4 Multi-Corruption Robust Analysis Runner"
            echo "========================================================================"
            echo ""
            echo "🆕 NEW ENHANCED FEATURES:"
            echo "• 🎯 Smart Caching: Automatic result caching for faster subsequent runs"
            echo "• 🔍 Comprehensive Analysis: ALL classes and ALL severities analyzed"
            echo "• 📊 Dimensional Analysis: Class-wise, severity-wise, and combo analysis"
            echo "• ⚡ Intelligent Validation: Automatic cache validation and integrity checks"
            echo "• 🧹 Auto-cleanup: Detects and handles incomplete results"
            echo "• 📈 Enhanced Monitoring: Real-time progress tracking and statistics"
            echo "• 🎨 Publication-ready: Enhanced visualizations and LaTeX tables"
            echo ""
            echo "Commands:"
            echo "  single           - Submit single robust analysis experiment to SLURM queue"
            echo "  all              - Submit ALL robust analysis experiments across all conditions"
            echo "  all-comprehensive - Submit ALL experiments with comprehensive multi-dimensional analysis"
            echo "  by-model         - Submit experiments for all layers/seeds for each model"
            echo "  by-method        - Submit experiments for all models/layers/seeds for each method"
            echo "  by-seed          - Submit experiments for all models/layers/methods for each seed"
            echo "  quick            - Submit quick test jobs with reduced parameters"
            echo "  quick-comprehensive - Quick test with comprehensive analysis"
            echo "  scan             - Scan for existing results"
            echo "  monitor          - Monitor running SLURM jobs"
            echo "  cleanup          - Clean up incomplete results"
            echo "  help             - Show this help message"
            echo ""
            echo "Options:"
            echo "  --dataset DATASET          Dataset to use (default: cifar10, choices: cifar10, cifar100)"
            echo "  --model MODEL              Single model to analyze (default: resnet18)"
            echo "  --models \"MODEL1 MODEL2\"   Multiple models (default: \"resnet18 resnet50 densenet121\")"
            echo "  --layer LAYER              Single layer to analyze (default: layer3)"
            echo "  --seed SEED                Single seed (default: 12)"
            echo "  --seeds \"SEED1 SEED2\"      Multiple seeds (default: \"12\")"
            echo "  --training-method METHOD   Single training method (default: constellation)"
            echo "  --training-methods \"M1 M2\" Multiple methods (default: comprehensive list)"
            echo "  --num-images NUM           Number of images to analyze (default: 100)"
            echo "  --max-analyze NUM          Maximum examples to analyze (default: 200)"
            echo "  --output-dir DIR           Output directory (default: section4_robust_analysis)"
            echo "  --cifar10c-dir DIR         CIFAR-10-C data directory (default: data/cifar10-c)"
            echo "  --cifar100c-dir DIR        CIFAR-100-C data directory (default: data/cifar100-c)"
            echo "  --batch-size SIZE          Batch size for processing (default: 100)"
            echo "  --k-values \"K1 K2\"         k-NN values for analysis (default: \"1 3 5\")"
            echo "  --python-env ENV           Conda environment name (default: tamar_n_env)"
            echo ""
            echo "Comprehensive Analysis Options:"
            echo "  --comprehensive-analysis   Enable multi-dimensional analysis across ALL classes and severities"
            echo "  --samples-per-class-severity NUM  Samples per class-severity combination (default: 30)"
            echo ""
            echo "SLURM Options:"
            echo "  --no-slurm                 Run directly without SLURM submission"
            echo "  --partition PARTITION      SLURM partition (default: gpu_partition)"
            echo "  --gpu-type TYPE            GPU type (default: rtx_4090)"
            echo ""
            echo "Examples:"
            echo "  # Complete comprehensive analysis on CIFAR-10 (first run computes, second uses cache)"
            echo "  $0 all-comprehensive --dataset cifar10"
            echo ""
            echo "  # Comprehensive analysis with custom sampling"
            echo "  $0 all-comprehensive --dataset cifar100 --samples-per-class-severity 50"
            echo ""
            echo "  # Single comprehensive experiment with caching"
            echo "  $0 single --dataset cifar10 --model resnet18 --layer layer3 --comprehensive-analysis"
            echo ""
            echo "  # Quick comprehensive test"
            echo "  $0 quick-comprehensive --dataset cifar100"
            echo ""
            echo "  # Scan existing results and show progress"
            echo "  $0 scan --dataset cifar10"
            echo ""
            echo "  # Clean up incomplete results"
            echo "  $0 cleanup --dataset cifar10"
            echo ""
            echo "  # Monitor running jobs with enhanced statistics"
            echo "  $0 monitor"
            echo ""
            echo "Key Features:"
            echo "============="
            echo "• 🎯 Smart caching system for faster subsequent runs"
            echo "• 🔍 Comprehensive robust analysis across ALL 15 corruption types"
            echo "• 📊 Multi-dimensional analysis across ALL classes and ALL severities"
            echo "• ⚡ Semantic vs pixel space advantage quantification"
            echo "• 🧪 Training method robustness assessment with statistical significance"
            echo "• 🎨 Publication-ready figures and LaTeX tables"
            echo "• 🔬 Granular insights: 'For automobile class at severity 3, layer X shows Y advantage'"
            echo "• 📈 Enhanced monitoring and progress tracking"
            echo "• 🧹 Automatic cleanup of incomplete results"
            echo ""
            echo "Output Structure:"
            echo "================"
            echo "section4_robust_analysis/"
            echo "├── results/{training_method}/{dataset}/{model}/{layer}_analysis[_comprehensive]/seed{seed}/"
            echo "│   ├── {corruption}/detailed_results.json"
            echo "│   ├── {corruption}/robust_analysis_data.csv (comprehensive mode)"
            echo "│   ├── aggregated_results.json (🎯 used for caching validation)"
            echo "│   ├── comprehensive_multi_corruption_analysis.png"
            echo "│   └── comprehensive_latex_table.tex"
            echo "└── logs/"
            echo ""
            echo "🎯 Caching Behavior:"
            echo "• Results are automatically cached after successful completion"
            echo "• Cache validation checks experiment configuration and file integrity"
            echo "• Subsequent runs with identical parameters use cached results"
            echo "• Configuration changes trigger automatic recomputation"
            echo "• Use 'scan' command to check cache status"
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
        resnet18|resnet50|densenet121|dinov2_small_scratch|dinov2_base_scratch)
            return 0
            ;;
        *)
            print_error "Invalid model: $model"
            print_info "Available models: resnet18, resnet50, densenet121, dinov2_small_scratch, dinov2_base_scratch"
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

# Check if robust analysis results already exist
check_robust_results() {
    local MODEL="$1"
    local LAYER="$2"
    local SEED="$3"
    local TRAINING_METHOD="$4"
    local OUTPUT_DIR="$5"
    local COMPREHENSIVE="$6"
    
    # Construct the actual results directory
    local mode_suffix=""
    if [[ "$COMPREHENSIVE" == true ]]; then
        mode_suffix="_comprehensive"
    fi
    
    local BASE_OUTPUT_DIR=$(dirname "$(dirname "$OUTPUT_DIR")")
    local ACTUAL_RESULTS_DIR="${BASE_OUTPUT_DIR}/results/${TRAINING_METHOD}/${DEFAULT_DATASET}/${MODEL}/${LAYER}_analysis${mode_suffix}/seed${SEED}"
    
    # Check for main analysis files (these are the key indicators)
    local main_files=(
        "$ACTUAL_RESULTS_DIR/aggregated_results.json"
        "$ACTUAL_RESULTS_DIR/comprehensive_latex_table.tex"
    )
    
    # Check if main files exist
    for file in "${main_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            return 1  # Main results don't exist
        fi
    done
    
    # Validate JSON files and check experiment configuration
    if [[ -f "$ACTUAL_RESULTS_DIR/aggregated_results.json" ]]; then
        # Check if JSON is valid
        if ! python -c "import json; json.load(open('$ACTUAL_RESULTS_DIR/aggregated_results.json'))" 2>/dev/null; then
            print_warning "Invalid JSON file found: $ACTUAL_RESULTS_DIR/aggregated_results.json"
            return 1  # Invalid JSON
        fi
        
        # Check if experiment configuration matches current parameters
        local config_check=$(python -c "
import json
import sys
try:
    with open('$ACTUAL_RESULTS_DIR/aggregated_results.json', 'r') as f:
        data = json.load(f)
    
    config = data.get('experiment_config', {})
    
    # Check key parameters
    # Compare k-values by sorting them to handle different ordering
    current_k_values = sorted([int(k) for k in '$K_VALUES'.split()])
    cached_k_values = sorted(config.get('k_values', [1, 3, 5]))

    if (config.get('dataset') != '$DEFAULT_DATASET' or
        config.get('model') != '$MODEL' or
        config.get('layer') != '$LAYER' or
        config.get('seed') != $SEED or
        config.get('training_method') != '$TRAINING_METHOD' or
        config.get('comprehensive_analysis') != ('$COMPREHENSIVE' == 'true') or
        current_k_values != cached_k_values):
        print('MISMATCH')
        sys.exit(1)
    else:
        print('MATCH')
        sys.exit(0)
except Exception as e:
    print(f'ERROR: {e}')
    sys.exit(1)
" 2>/dev/null)
        
        if [[ "$config_check" == "MISMATCH" ]]; then
            print_warning "Cached results don't match current configuration for $MODEL-$LAYER-$SEED-$TRAINING_METHOD"
            return 1  # Configuration mismatch
        elif [[ "$config_check" == "ERROR:"* ]]; then
            print_warning "Error checking cached configuration: $config_check"
            return 1  # Error in config check
        fi
    fi
    
    # Check for per-corruption results (the analysis script will validate these)
    local corruption_count=0
    local required_corruptions=("gaussian_noise" "shot_noise" "impulse_noise" "defocus_blur" "glass_blur" "motion_blur" "zoom_blur" "snow" "frost" "fog" "brightness" "contrast" "elastic_transform" "pixelate" "jpeg_compression")
    
    for corruption in "${required_corruptions[@]}"; do
        if [[ -f "$ACTUAL_RESULTS_DIR/$corruption/detailed_results.json" ]]; then
            corruption_count=$((corruption_count + 1))
        fi
    done
    
    # Require at least 12 out of 15 corruptions for valid results (allow for some failed corruptions)
    if [[ $corruption_count -lt 12 ]]; then
        print_warning "Insufficient corruption results: $corruption_count/15 found"
        return 1
    fi
    
    # Check if the main visualization file exists (created by the analysis script)
    if [[ ! -f "$ACTUAL_RESULTS_DIR/comprehensive_multi_corruption_analysis.png" ]]; then
        print_warning "Main visualization file missing"
        return 1
    fi
    
    # If we get here, results are valid and complete
    print_info "✅ Valid cached results found: $corruption_count/15 corruptions"
    return 0  # Valid results exist
}

# Enhanced resource allocation based on caching and comprehensive analysis
get_robust_resources() {
    local MODEL="$1"
    local TRAINING_METHOD="$2"
    local NUM_IMAGES="$3"
    local COMPREHENSIVE="$4"
    
    # Base resource allocation for robust analysis
    local TIME="2-12:00:00"     # 2 hours for basic robust analysis (reduced due to caching)
    local MEM="16G"          # 16GB default (reduced due to better memory management)
    local CPUS=4             # 4 CPUs default
    
    # Scale based on comprehensive analysis
    if [[ "$COMPREHENSIVE" == true ]]; then
        TIME="2-16:00:00"       # 6 hours for comprehensive analysis (reduced from 8 due to caching)
        MEM="40G"            # 28GB for comprehensive analysis
        CPUS=6               # More CPUs for comprehensive analysis
    fi
    
    # Scale based on model complexity
    case "$MODEL" in
        "densenet121")
            TIME=$(echo "$TIME" | sed 's/\([0-9]*\):\([0-9]*\):\([0-9]*\)/\1/' | awk '{print int($1 * 1.4) ":" substr("'$TIME'", index("'$TIME'", ":") + 1)}')
            MEM=$(echo "$MEM" | sed 's/G//' | awk '{print int($1 * 1.2) "G"}')
            CPUS=$((CPUS + 1))
            ;;
        "resnet50")
            TIME=$(echo "$TIME" | sed 's/\([0-9]*\):\([0-9]*\):\([0-9]*\)/\1/' | awk '{print int($1 * 1.2) ":" substr("'$TIME'", index("'$TIME'", ":") + 1)}')
            MEM=$(echo "$MEM" | sed 's/G//' | awk '{print int($1 * 1.1) "G"}')
            ;;
    esac
    
    # Scale based on training method complexity
    case "$TRAINING_METHOD" in
        "constellation"|"contrastive"|"augmix_constellation")
            MEM=$(echo "$MEM" | sed 's/G//' | awk '{print int($1 * 1.15) "G"}')
            ;;
        "augmix")
            MEM=$(echo "$MEM" | sed 's/G//' | awk '{print int($1 * 1.1) "G"}')
            ;;
    esac
    
    echo "$TIME $MEM $CPUS"
}

# Enhanced SLURM Job Submission with better error handling
submit_robust_job() {
    local MODEL="$1"
    local LAYER="$2"
    local SEED="$3"
    local TRAINING_METHOD="$4"
    local NUM_IMAGES="$5"
    local MAX_ANALYZE="$6"
    local OUTPUT_DIR="$7"
    local BATCH_SIZE="$8"
    local BASE_OUTPUT_DIR="${9:-$OUTPUT_DIR}"
    local COMPREHENSIVE="${10:-false}"
    
    # Validate parameters
    validate_dataset "$DEFAULT_DATASET"
    validate_model "$MODEL"
    validate_layer "$MODEL" "$LAYER"
    validate_training_method "$TRAINING_METHOD"
    
    # Check if trained model exists before submitting job
    print_info "   🔍 Checking for existing trained model..."
    local MODEL_FOUND=false
    
    # Pattern 1: Nested structure
    local NESTED_MODEL_PATTERN="${RESULTS_DIR}/*/${TRAINING_METHOD}/${DEFAULT_DATASET}/${MODEL}/seed${SEED}/${TRAINING_METHOD}_${DEFAULT_DATASET}_${MODEL}_seed${SEED}/best_model.pth"
    
    # Pattern 2: Flat structure (for DINOv2 and others)
    local FLAT_MODEL_PATTERN="${RESULTS_DIR}/*/${TRAINING_METHOD}_${DEFAULT_DATASET}_${MODEL}_seed${SEED}*/best_model.pth"

    if ls $NESTED_MODEL_PATTERN 1> /dev/null 2>&1; then
        print_info "   ✅ Model found in standard nested path."
        MODEL_FOUND=true
    elif ls $FLAT_MODEL_PATTERN 1> /dev/null 2>&1; then
        print_info "   ✅ Model found in flat path structure."
        MODEL_FOUND=true
    fi

    if [[ "$MODEL_FOUND" != true ]]; then
        print_error "❌ Trained model not found for: $TRAINING_METHOD | $DEFAULT_DATASET | $MODEL | seed$SEED"
        print_info "   Searched patterns:"
        print_info "     - Nested: $NESTED_MODEL_PATTERN"
        print_info "     - Flat:   $FLAT_MODEL_PATTERN"
        print_warning "   Skipping job submission for this configuration."
        return 3 # Special exit code for missing model
    fi
    
    # Check for existing results with enhanced validation
    if check_robust_results "$MODEL" "$LAYER" "$SEED" "$TRAINING_METHOD" "$OUTPUT_DIR" "$COMPREHENSIVE"; then
        print_status "✅ Valid cached results found for $MODEL-$LAYER-$SEED-$TRAINING_METHOD"
        if [[ "$COMPREHENSIVE" == true ]]; then
            print_status "   📊 Comprehensive mode results confirmed"
        else
            print_status "   📊 Legacy mode results confirmed"
        fi
        print_info "   ⚡ Skipping job submission (results will be loaded from cache)"
        return 2  # Special code for existing results
    fi
    
    # Get resource requirements
    local RESOURCES=($(get_robust_resources "$MODEL" "$TRAINING_METHOD" "$NUM_IMAGES" "$COMPREHENSIVE"))
    local TIME="${RESOURCES[0]}"
    local MEM="${RESOURCES[1]}"
    local CPUS="${RESOURCES[2]}"
    
    # Create job name with better naming convention
    local JOB_NAME="robust_${MODEL}_${LAYER}_${TRAINING_METHOD}_s${SEED}"
    if [[ "$COMPREHENSIVE" == true ]]; then
        JOB_NAME="${JOB_NAME}_comp"
    fi
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$(dirname "$OUTPUT_DIR")/logs"
    
    # Log file with timestamp
    local LOG_FILE="$(dirname "$OUTPUT_DIR")/logs/${JOB_NAME}_%j.out"
    
    # Build Python command for robust analysis
    local CMD="python $EXPERIMENTS_DIR/robust_statistical_analysis_multi_corruption.py"
    CMD+=" --model $MODEL"
    CMD+=" --layer $LAYER"
    CMD+=" --seed $SEED"
    CMD+=" --training-method $TRAINING_METHOD"
    CMD+=" --dataset $DEFAULT_DATASET"
    CMD+=" --num-images $NUM_IMAGES"
    CMD+=" --max-analyze $MAX_ANALYZE"
    CMD+=" --output-dir $BASE_OUTPUT_DIR"
    CMD+=" --results-dir $RESULTS_DIR"
    CMD+=" --batch-size $BATCH_SIZE"
    CMD+=" --all-corruptions"  # CRITICAL: Process all corruption types
    CMD+=" --k-values $K_VALUES"
    
    # Add comprehensive analysis flags if enabled
    if [[ "$COMPREHENSIVE" == true ]]; then
        CMD+=" --comprehensive-analysis"
        CMD+=" --samples-per-class-severity $SAMPLES_PER_CLASS_SEVERITY"
    fi
    
    local CORRUPTION_DIR=$(get_corruption_dir "$DEFAULT_DATASET")
    if [[ "$DEFAULT_DATASET" == "cifar10" ]]; then
        CMD+=" --cifar10c-dir $CORRUPTION_DIR"
    else
        CMD+=" --cifar100c-dir $CORRUPTION_DIR"
    fi
    
    print_status "🚀 Submitting Enhanced Robust Analysis SLURM job: $JOB_NAME"
    print_info "   📋 Resources: Time=$TIME, Memory=$MEM, CPUs=$CPUS"
    if [[ "$COMPREHENSIVE" == true ]]; then
        print_info "   🔍 Mode: Comprehensive (ALL classes, ALL severities)"
        print_info "   📊 Samples per class-severity: $SAMPLES_PER_CLASS_SEVERITY"
    else
        print_info "   📊 Mode: Legacy (single target class)"
    fi
    print_info "   ⚡ Enhanced caching enabled for faster subsequent runs"
    print_info "   🎯 Command: $CMD"
    
    # Submit job with enhanced error handling
    local SUBMIT_OUTPUT
    SUBMIT_OUTPUT=$(sbatch \
        --partition="$PARTITION" \
        --job-name="$JOB_NAME" \
        --output="$LOG_FILE" \
        --time="$TIME" \
        --ntasks=1 \
        --gpus=0 \
        --cpus-per-task="$CPUS" \
        --mem="$MEM" \
        --wrap="
        echo '🔬 ENHANCED AAAI SECTION 4: ROBUST ANALYSIS WITH CACHING'
        echo '============================================================'
        echo 'Job ID: '\$SLURM_JOB_ID
        echo 'Model: $MODEL | Layer: $LAYER | Training Method: $TRAINING_METHOD | Seed: $SEED'
        echo 'Dataset: $DEFAULT_DATASET | Comprehensive: $COMPREHENSIVE'
        echo 'Host: '\$(hostname)' | CPU Count: $CPUS | Memory: $MEM'
        echo 'Started: '\$(date)
        echo '============================================================'

        # Enhanced environment setup
        module load anaconda || { echo '❌ Failed to load anaconda'; exit 1; }
        
        echo '🔧 Activating conda environment: $PYTHON_ENV'
        source activate $PYTHON_ENV || { echo '❌ Failed to activate conda environment'; exit 1; }
        
        echo '🔍 Environment verification:'
        which python
        python --version
        echo 'Python path: '\$PYTHONPATH
        
        # GPU and memory management
        export CUDA_HOME=/usr/local/cuda
        export CUDA_VISIBLE_DEVICES=\$(echo \$CUDA_VISIBLE_DEVICES | cut -d',' -f1)
        export PYTHONPATH='$PROJECT_ROOT:\$PYTHONPATH'
        
        echo '🔧 Memory management setup:'
        export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:False
        export CUDA_LAUNCH_BLOCKING=1
        
        # 🔧 FIX for OpenBLAS hang: Prevent nested parallelism
        export OMP_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1
        
        # Clear GPU memory and show available resources
        python -c '
import torch
import psutil
import os

print(f\"💾 System memory: {psutil.virtual_memory().total / 1e9:.1f}GB total, {psutil.virtual_memory().available / 1e9:.1f}GB available\")
print(f\"💻 CPU count: {os.cpu_count()}\")

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gpu_props = torch.cuda.get_device_properties(0)
    print(f\"🚀 GPU: {gpu_props.name}\")
    print(f\"🧹 GPU memory: {gpu_props.total_memory / 1e9:.1f}GB total\")
    print(f\"⚡ GPU memory cleared and ready\")
else:
    print(\"⚠️  CUDA not available - running on CPU only\")
'
        
        cd $PROJECT_ROOT || { echo '❌ Failed to change to project directory'; exit 1; }

        echo '🚀 Executing Enhanced Robust Analysis (with caching):'
        echo '$CMD'
        echo '================================================='

        # Run with enhanced error handling
        if timeout 720000 $CMD; then  # 2 hour timeout as backup
            echo '✅ ROBUST ANALYSIS COMPLETED SUCCESSFULLY'
            echo '📊 Results saved to: $BASE_OUTPUT_DIR'
            echo '⚡ Cached results available for future runs'
            echo '🎯 Check comprehensive_multi_corruption_analysis.png for main results'
        else
            exit_code=\$?
            echo '❌ ROBUST ANALYSIS FAILED OR TIMED OUT'
            echo 'Exit code: '\$exit_code
            echo 'Check logs for detailed error information'
            
            # Show some system info for debugging
            echo '🔍 System status at failure:'
            echo 'Memory usage:' 
            free -h
            echo 'Disk usage:'
            df -h $BASE_OUTPUT_DIR 2>/dev/null || echo 'Could not check disk usage'
            
            exit 1
        fi

        echo '🏁 Job finished: '\$(date)
        echo '============================================================'
        " 2>&1)
    
    local SUBMIT_EXIT_CODE=$?
    
    # Enhanced submission result checking
    if [[ $SUBMIT_EXIT_CODE -eq 0 ]] && [[ -n "$SUBMIT_OUTPUT" ]]; then
        local JOB_ID
        JOB_ID=$(echo "$SUBMIT_OUTPUT" | grep -o 'Submitted batch job [0-9]*' | grep -o '[0-9]*')
        
        if [[ -n "$JOB_ID" ]] && [[ "$JOB_ID" =~ ^[0-9]+$ ]]; then
            print_status "✅ Enhanced robust job submitted successfully!"
            print_info "   🆔 Job ID: $JOB_ID"
            print_info "   📝 Log file: $LOG_FILE"
            print_info "   ⚡ Caching enabled - subsequent runs will be faster"
            return 0
        else
            print_error "❌ Could not extract job ID from output: $SUBMIT_OUTPUT"
            return 1
        fi
    else
        print_error "❌ Job submission failed with exit code: $SUBMIT_EXIT_CODE"
        print_error "Error output: $SUBMIT_OUTPUT"
        return 1
    fi
}

# Environment setup
setup_environment() {
    print_status "Setting up robust analysis environment..."
    
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
    
    print_status "Running SINGLE robust analysis experiment:"
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
        submit_robust_job "$MODEL" "$LAYER" "$SEED" "$TRAINING_METHOD" \
                         "$NUM_IMAGES" "$MAX_ANALYZE" "$exp_output_dir" "$BATCH_SIZE" \
                         "$BASE_OUTPUT_DIR" "$COMPREHENSIVE_ANALYSIS"
        
        local exit_code=$?
        if [[ $exit_code -eq 0 ]]; then
            print_status "Robust analysis job submitted successfully!"
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
        local CMD="python $EXPERIMENTS_DIR/robust_statistical_analysis_multi_corruption.py"
        CMD+=" --model $MODEL"
        CMD+=" --layer $LAYER"
        CMD+=" --seed $SEED"
        CMD+=" --training-method $TRAINING_METHOD"
        CMD+=" --dataset $DEFAULT_DATASET"
        CMD+=" --num-images $NUM_IMAGES"
        CMD+=" --max-analyze $MAX_ANALYZE"
        CMD+=" --output-dir $BASE_OUTPUT_DIR"
        CMD+=" --results-dir $RESULTS_DIR"
        CMD+=" --batch-size $BATCH_SIZE"
        CMD+=" --all-corruptions"  # CRITICAL: Process all corruption types
        CMD+=" --k-values $K_VALUES"
        
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
            print_status "Single robust analysis experiment completed successfully!"
        else
            print_error "Single robust analysis experiment failed!"
            exit 1
        fi
    fi
}

# Run ALL experiments across all models/layers/seeds/methods
run_all_experiments() {
    local COMP_MODE="$COMPREHENSIVE_ANALYSIS"
    
    print_status "Running ALL robust analysis experiments across experimental arrays..."
    print_info "  Dataset: $DEFAULT_DATASET"
    print_info "  Models: ${MODELS[*]}"
    print_info "  Seeds: ${SEEDS[*]}"
    print_info "  Training Methods: ${TRAINING_METHODS[*]}"
    print_info "  SLURM: $USE_SLURM"
    print_info "  Comprehensive Analysis: $COMP_MODE"
    if [[ "$COMP_MODE" == true ]]; then
        print_info "  Samples per class-severity: $SAMPLES_PER_CLASS_SEVERITY"
        print_info "  🎯 Will analyze ALL classes and ALL severities for multi-dimensional insights!"
    fi
    
    local total_experiments=0
    local submitted_jobs=0
    local existing_results=0
    local failed_submissions=0
    local missing_model_count=0
    
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
    
    print_status "Planning $total_experiments robust analysis experiments..."
    
    # Run each experiment
    local experiment_num=0
    for model in "${MODELS[@]}"; do
        local layers=($(get_all_layers_for_model "$model"))
        for layer in "${layers[@]}"; do
            for seed in "${SEEDS[@]}"; do
                for training_method in "${TRAINING_METHODS[@]}"; do
                    experiment_num=$((experiment_num + 1))
                    print_status "Robust experiment $experiment_num/$total_experiments: $model-$layer-$training_method-seed$seed"
                    
                    local exp_output_dir="${OUTPUT_DIR}/${model}_${layer}_${training_method}_seed${seed}"
                    
                    if [[ "$USE_SLURM" == true ]]; then
                        # Temporarily disable exit-on-error to handle return codes gracefully
                        set +e
                        submit_robust_job "$model" "$layer" "$seed" "$training_method" \
                                         "$NUM_IMAGES" "$MAX_ANALYZE" "$exp_output_dir" "$BATCH_SIZE" \
                                         "$BASE_OUTPUT_DIR" "$COMP_MODE"
                        
                        local exit_code=$?
                        set -e # Re-enable exit-on-error
                        
                        if [[ $exit_code -eq 0 ]]; then
                            submitted_jobs=$((submitted_jobs + 1))
                            print_status "✅ $model-$layer-$training_method-seed$seed submitted"
                            # Wait 10 seconds between job submissions
                            sleep 10
                        elif [[ $exit_code -eq 2 ]]; then
                            existing_results=$((existing_results + 1))
                            print_status "⏭️ $model-$layer-$training_method-seed$seed skipped (results exist)"
                        elif [[ $exit_code -eq 3 ]]; then
                            missing_model_count=$((missing_model_count + 1))
                            # The message is already printed inside submit_robust_job
                        else
                            failed_submissions=$((failed_submissions + 1))
                            print_error "❌ $model-$layer-$training_method-seed$seed submission failed with exit code $exit_code"
                        fi
                    else
                        # Run directly
                        mkdir -p "$exp_output_dir"
                        
                        local CMD="python $EXPERIMENTS_DIR/robust_statistical_analysis_multi_corruption.py"
                        CMD+=" --model $model"
                        CMD+=" --layer $layer"
                        CMD+=" --seed $seed"
                        CMD+=" --training-method $training_method"
                        CMD+=" --dataset $DEFAULT_DATASET"
                        CMD+=" --num-images $NUM_IMAGES"
                        CMD+=" --max-analyze $MAX_ANALYZE"
                        CMD+=" --output-dir $BASE_OUTPUT_DIR"
                        CMD+=" --results-dir $RESULTS_DIR"
                        CMD+=" --batch-size $BATCH_SIZE"
                        CMD+=" --all-corruptions"  # CRITICAL: Process all corruption types
                        CMD+=" --k-values $K_VALUES"
                        
                        # Add comprehensive analysis flags if enabled
                        if [[ "$COMP_MODE" == true ]]; then
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
    print_status "ALL ROBUST ANALYSIS EXPERIMENTS SUMMARY:"
    print_info "  Total experiments: $total_experiments"
    if [[ "$USE_SLURM" == true ]]; then
        print_info "  Jobs submitted: $submitted_jobs"
        print_info "  Existing results: $existing_results"
        print_info "  Skipped (model not found): $missing_model_count"
        print_info "  Failed submissions: $failed_submissions"
        
        if [[ $submitted_jobs -gt 0 ]]; then
            print_status "Monitor jobs with: squeue -u \$USER"
            print_info "Check logs in: $(dirname "$OUTPUT_DIR")/logs/"
        fi
    else
        print_info "  Completed: $submitted_jobs"
        print_info "  Failed: $failed_submissions"
        
        if [[ $submitted_jobs -gt 0 ]]; then
            print_status "All robust analysis experiments completed!"
        fi
    fi
}

# Quick test with reduced parameters
run_quick_test() {
    local COMP_MODE="$COMPREHENSIVE_ANALYSIS"
    
    print_status "Running QUICK robust analysis test with reduced parameters..."
    print_info "  Dataset: $DEFAULT_DATASET"
    print_info "  Models: ${MODELS[*]}"
    print_info "  SLURM: $USE_SLURM"
    print_info "  Comprehensive Analysis: $COMP_MODE"
    if [[ "$COMP_MODE" == true ]]; then
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
            submit_robust_job "$model" "$layer" "$quick_seed" "$quick_method" \
                             "50" "100" "$test_output_dir" "$BATCH_SIZE" \
                             "$BASE_OUTPUT_DIR" "$COMP_MODE"
            
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
            
            local CMD="python $EXPERIMENTS_DIR/robust_statistical_analysis_multi_corruption.py"
            CMD+=" --model $model"
            CMD+=" --layer $layer"
            CMD+=" --seed $quick_seed"
            CMD+=" --training-method $quick_method"
            CMD+=" --dataset $DEFAULT_DATASET"
            CMD+=" --num-images 50"
            CMD+=" --max-analyze 100"
            CMD+=" --output-dir $BASE_OUTPUT_DIR"
            CMD+=" --results-dir $RESULTS_DIR"
            CMD+=" --batch-size $BATCH_SIZE"
            CMD+=" --all-corruptions"  # CRITICAL: Process all corruption types
            CMD+=" --k-values $K_VALUES"
            
            # Add comprehensive analysis flags if enabled
            if [[ "$COMP_MODE" == true ]]; then
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
    fi
}

# Enhanced monitoring with better status information
monitor_jobs() {
    if [[ "$USE_SLURM" != true ]]; then
        print_warning "SLURM monitoring not available in direct execution mode"
        return
    fi
    
    print_status "📊 Monitoring Enhanced Robust Analysis SLURM jobs..."
    
    echo ""
    echo "🔄 Current Robust Analysis jobs:"
    squeue -u $USER --name="robust_*" --format="%.18i %.40j %.8T %.10M %.6D %R %C %m" 2>/dev/null | head -20 || echo "No Robust Analysis jobs in queue"
    
    echo ""
    echo "📈 Recent Robust job history (last 20):"
    sacct -u $USER --name="robust_*" --starttime=today --format="JobID,JobName,State,ExitCode,Start,End,Elapsed,MaxRSS,MaxVMSize" 2>/dev/null | head -20 || echo "No recent Robust jobs"
    
    echo ""
    echo "✅ Completed Robust jobs today:"
    sacct -u $USER --name="robust_*" --starttime=today --state=COMPLETED --format="JobID,JobName,End,Elapsed" 2>/dev/null | head -10 || echo "No completed Robust jobs today"
    
    echo ""
    echo "❌ Failed/Timeout Robust jobs:"
    sacct -u $USER --name="robust_*" --starttime=today --state=FAILED,TIMEOUT --format="JobID,JobName,State,ExitCode,Start,End" 2>/dev/null | head -10 || echo "No failed Robust jobs found"
    
    # Show disk usage and results summary
    echo ""
    echo "💾 Results storage summary:"
    if [[ -d "$BASE_OUTPUT_DIR" ]]; then
        echo "📁 Base directory: $BASE_OUTPUT_DIR"
        du -sh "$BASE_OUTPUT_DIR" 2>/dev/null || echo "Could not calculate disk usage"
        
        # Count completed experiments
        local completed_count=0
        local total_experiments=0
        
        for model in "${MODELS[@]}"; do
            local layers=($(get_all_layers_for_model "$model"))
            for layer in "${layers[@]}"; do
                for seed in "${SEEDS[@]}"; do
                    for training_method in "${TRAINING_METHODS[@]}"; do
                        total_experiments=$((total_experiments + 1))
                        
                        if check_robust_results "$model" "$layer" "$seed" "$training_method" "$OUTPUT_DIR" "$COMPREHENSIVE_ANALYSIS"; then
                            completed_count=$((completed_count + 1))
                        fi
                    done
                done
            done
        done
        
        echo "📊 Experiment completion: $completed_count/$total_experiments"
        local completion_pct=$((completed_count * 100 / total_experiments))
        echo "🎯 Progress: $completion_pct%"
    else
        echo "📁 Results directory not found: $BASE_OUTPUT_DIR"
    fi
    
    echo ""
    echo "🔄 To monitor continuously: watch -n 30 '$0 monitor'"
    echo "📝 To check specific job logs: tail -f $(dirname "$OUTPUT_DIR")/logs/robust_*_JOBID.out"
}

# Enhanced cleanup function
cleanup_incomplete_results() {
    print_status "🧹 Cleaning up incomplete robust analysis results..."
    
    local cleaned_count=0
    local total_checked=0
    
    if [[ ! -d "$BASE_OUTPUT_DIR/results" ]]; then
        print_info "No results directory found to clean"
        return
    fi
    
    # Find potentially incomplete results
    find "$BASE_OUTPUT_DIR/results" -type d -name "*_analysis*" | while read -r result_dir; do
        total_checked=$((total_checked + 1))
        
        # Check if this directory looks incomplete
        if [[ -d "$result_dir" ]]; then
            # Look for key files
            local has_aggregated=false
            local has_latex=false
            local has_png=false
            
            if [[ -f "$result_dir/aggregated_results.json" ]]; then
                has_aggregated=true
            fi
            if [[ -f "$result_dir/comprehensive_latex_table.tex" ]]; then
                has_latex=true
            fi
            if [[ -f "$result_dir/comprehensive_multi_corruption_analysis.png" ]]; then
                has_png=true
            fi
            
            # If missing key files, consider it incomplete
            if [[ "$has_aggregated" == false ]] || [[ "$has_latex" == false ]] || [[ "$has_png" == false ]]; then
                print_info "🗑️  Removing incomplete result: $result_dir"
                rm -rf "$result_dir"
                cleaned_count=$((cleaned_count + 1))
            fi
        fi
    done
    
    print_status "🧹 Cleanup complete: $cleaned_count incomplete results removed"
}

# Enhanced results scanning
scan_results() {
    print_status "🔍 Scanning for robust analysis results..."
    
    local total_expected=0
    local total_found=0
    local total_comprehensive=0
    local total_legacy=0
    
    echo ""
    echo "📊 Scanning experimental space:"
    echo "   Models: ${MODELS[*]}"
    echo "   Seeds: ${SEEDS[*]}"
    echo "   Training Methods: ${TRAINING_METHODS[*]}"
    echo ""
    
    # Check all possible experiments
    for model in "${MODELS[@]}"; do
        local layers=($(get_all_layers_for_model "$model"))
        for layer in "${layers[@]}"; do
            for seed in "${SEEDS[@]}"; do
                for training_method in "${TRAINING_METHODS[@]}"; do
                    total_expected=$((total_expected + 1))
                    
                    # Check both comprehensive and legacy modes
                    local found_comp=false
                    local found_legacy=false
                    
                    if check_robust_results "$model" "$layer" "$seed" "$training_method" "$OUTPUT_DIR" "true"; then
                        found_comp=true
                        total_comprehensive=$((total_comprehensive + 1))
                        total_found=$((total_found + 1))
                    fi
                    
                    if check_robust_results "$model" "$layer" "$seed" "$training_method" "$OUTPUT_DIR" "false"; then
                        found_legacy=true
                        if [[ "$found_comp" == false ]]; then
                            total_legacy=$((total_legacy + 1))
                            total_found=$((total_found + 1))
                        fi
                    fi
                    
                    # Report status
                    local status="❌ Missing"
                    if [[ "$found_comp" == true ]]; then
                        status="✅ Comprehensive"
                    elif [[ "$found_legacy" == true ]]; then
                        status="📊 Legacy"
                    fi
                    
                    echo "   $status: $model-$layer-$training_method-seed$seed"
                done
            done
        done
    done
    
    echo ""
    print_status "📊 Results Summary:"
    print_info "   Total expected: $total_expected"
    print_info "   Total found: $total_found"
    print_info "   Comprehensive mode: $total_comprehensive"
    print_info "   Legacy mode: $total_legacy"
    print_info "   Missing: $((total_expected - total_found))"
    
    local completion_pct=$((total_found * 100 / total_expected))
    print_info "   Completion: $completion_pct%"
    
    if [[ $total_found -eq $total_expected ]]; then
        print_status "🎉 All experiments completed!"
    else
        print_info "💡 Run 'all' or 'all-comprehensive' to complete missing experiments"
    fi
}

# Main execution
main() {
    echo "========================================================================"
    echo "🚀 ENHANCED AAAI PAPER SECTION 4 ROBUST ANALYSIS WITH CACHING"
    echo "Multi-Corruption Pixel vs Semantic Space Analysis"
    echo "========================================================================"
    echo "Command: $COMMAND"
    echo "Dataset: $DEFAULT_DATASET"
    echo "Comprehensive Analysis: $COMPREHENSIVE_ANALYSIS"
    if [[ "$COMPREHENSIVE_ANALYSIS" == true ]]; then
        echo "Samples per class-severity: $SAMPLES_PER_CLASS_SEVERITY"
    fi
    echo "SLURM: $USE_SLURM"
    echo "========================================================================"
    
    # Setup environment for all commands except help
    if [[ "$COMMAND" != "help" ]]; then
        setup_environment
    fi
    
    # Execute command
    case "$COMMAND" in
        "single")
            print_status "🎯 Running single robust analysis experiment..."
            run_single_experiment
            ;;
        "all")
            print_status "🔄 Running ALL robust analysis experiments..."
            run_all_experiments
            ;;
        "all-comprehensive")
            print_status "🔍 Running ALL experiments with comprehensive analysis..."
            COMPREHENSIVE_ANALYSIS=true
            run_all_experiments
            ;;
        "by-model")
            print_status "📊 Running experiments grouped by model..."
            run_experiments_by_model
            ;;
        "by-method")
            print_status "🧪 Running experiments grouped by training method..."
            run_experiments_by_method
            ;;
        "by-seed")
            print_status "🌱 Running experiments grouped by seed..."
            run_experiments_by_seed
            ;;
        "quick")
            print_status "⚡ Running quick test..."
            run_quick_test
            ;;
        "quick-comprehensive")
            print_status "⚡ Running quick test with comprehensive analysis..."
            COMPREHENSIVE_ANALYSIS=true
            run_quick_test
            ;;
        "scan")
            print_status "🔍 Scanning for existing results..."
            scan_results
            ;;
        "cleanup")
            print_status "🧹 Cleaning up incomplete results..."
            cleanup_incomplete_results
            ;;
        "monitor")
            print_status "📊 Monitoring SLURM jobs..."
            monitor_jobs
            ;;
        "help")
            # Help is handled in argument parsing
            exit 0
            ;;
        *)
            print_error "❌ Unknown command: $COMMAND"
            echo ""
            echo "Available commands:"
            echo "  single, all, all-comprehensive, by-model, by-method, by-seed"
            echo "  quick, quick-comprehensive, scan, cleanup, monitor, help"
            echo ""
            echo "Use --help for detailed usage information"
            exit 1
            ;;
    esac
}

# New grouped experiment runners
run_experiments_by_model() {
    print_status "Running experiments grouped by model..."
    print_info "  Dataset: $DEFAULT_DATASET"
    print_info "  Models: ${MODELS[*]}"
    print_info "  SLURM: $USE_SLURM"
    print_info "  Comprehensive Analysis: $COMPREHENSIVE_ANALYSIS"
    
    local total_jobs=0
    local submitted_jobs=0
    local existing_results=0
    local failed_jobs=0
    
    for model in "${MODELS[@]}"; do
        print_status "🔄 Processing model: $model"
        
        local layers=($(get_all_layers_for_model "$model"))
        for layer in "${layers[@]}"; do
            for seed in "${SEEDS[@]}"; do
                for training_method in "${TRAINING_METHODS[@]}"; do
                    total_jobs=$((total_jobs + 1))
                    
                    print_info "  Processing: $model-$layer-$training_method-seed$seed"
                    
                    local exp_output_dir="${OUTPUT_DIR}/${model}_${layer}_${training_method}_seed${seed}"
                    
                    if [[ "$USE_SLURM" == true ]]; then
                        submit_robust_job "$model" "$layer" "$seed" "$training_method" \
                                         "$NUM_IMAGES" "$MAX_ANALYZE" "$exp_output_dir" "$BATCH_SIZE" \
                                         "$BASE_OUTPUT_DIR" "$COMPREHENSIVE_ANALYSIS"
                        
                        local exit_code=$?
                        case $exit_code in
                            0) submitted_jobs=$((submitted_jobs + 1)) ;;
                            2) existing_results=$((existing_results + 1)) ;;
                            *) failed_jobs=$((failed_jobs + 1)) ;;
                        esac
                    else
                        # Direct execution logic here
                        print_info "Direct execution not fully implemented for grouped runs"
                    fi
                done
            done
        done
        
        print_info "✅ Completed model: $model"
    done
    
    print_status "📊 By-model execution summary:"
    print_info "  Total jobs: $total_jobs"
    print_info "  Submitted: $submitted_jobs"
    print_info "  Existing: $existing_results"
    print_info "  Failed: $failed_jobs"
}

run_experiments_by_method() {
    print_status "Running experiments grouped by training method..."
    print_info "  Dataset: $DEFAULT_DATASET"
    print_info "  Training Methods: ${TRAINING_METHODS[*]}"
    print_info "  SLURM: $USE_SLURM"
    print_info "  Comprehensive Analysis: $COMPREHENSIVE_ANALYSIS"
    
    local total_jobs=0
    local submitted_jobs=0
    local existing_results=0
    local failed_jobs=0
    
    for training_method in "${TRAINING_METHODS[@]}"; do
        print_status "🔄 Processing training method: $training_method"
        
        for model in "${MODELS[@]}"; do
            local layers=($(get_all_layers_for_model "$model"))
            for layer in "${layers[@]}"; do
                for seed in "${SEEDS[@]}"; do
                    total_jobs=$((total_jobs + 1))
                    
                    print_info "  Processing: $model-$layer-$training_method-seed$seed"
                    
                    local exp_output_dir="${OUTPUT_DIR}/${model}_${layer}_${training_method}_seed${seed}"
                    
                    if [[ "$USE_SLURM" == true ]]; then
                        submit_robust_job "$model" "$layer" "$seed" "$training_method" \
                                         "$NUM_IMAGES" "$MAX_ANALYZE" "$exp_output_dir" "$BATCH_SIZE" \
                                         "$BASE_OUTPUT_DIR" "$COMPREHENSIVE_ANALYSIS"
                        
                        local exit_code=$?
                        case $exit_code in
                            0) submitted_jobs=$((submitted_jobs + 1)) ;;
                            2) existing_results=$((existing_results + 1)) ;;
                            *) failed_jobs=$((failed_jobs + 1)) ;;
                        esac
                    else
                        # Direct execution logic here
                        print_info "Direct execution not fully implemented for grouped runs"
                    fi
                done
            done
        done
        
        print_info "✅ Completed training method: $training_method"
    done
    
    print_status "📊 By-method execution summary:"
    print_info "  Total jobs: $total_jobs"
    print_info "  Submitted: $submitted_jobs"
    print_info "  Existing: $existing_results"
    print_info "  Failed: $failed_jobs"
}

run_experiments_by_seed() {
    print_status "Running experiments grouped by seed..."
    print_info "  Dataset: $DEFAULT_DATASET"
    print_info "  Seeds: ${SEEDS[*]}"
    print_info "  SLURM: $USE_SLURM"
    print_info "  Comprehensive Analysis: $COMPREHENSIVE_ANALYSIS"
    
    local total_jobs=0
    local submitted_jobs=0
    local existing_results=0
    local failed_jobs=0
    
    for seed in "${SEEDS[@]}"; do
        print_status "🔄 Processing seed: $seed"
        
        for model in "${MODELS[@]}"; do
            local layers=($(get_all_layers_for_model "$model"))
            for layer in "${layers[@]}"; do
                for training_method in "${TRAINING_METHODS[@]}"; do
                    total_jobs=$((total_jobs + 1))
                    
                    print_info "  Processing: $model-$layer-$training_method-seed$seed"
                    
                    local exp_output_dir="${OUTPUT_DIR}/${model}_${layer}_${training_method}_seed${seed}"
                    
                    if [[ "$USE_SLURM" == true ]]; then
                        submit_robust_job "$model" "$layer" "$seed" "$training_method" \
                                         "$NUM_IMAGES" "$MAX_ANALYZE" "$exp_output_dir" "$BATCH_SIZE" \
                                         "$BASE_OUTPUT_DIR" "$COMPREHENSIVE_ANALYSIS"
                        
                        local exit_code=$?
                        case $exit_code in
                            0) submitted_jobs=$((submitted_jobs + 1)) ;;
                            2) existing_results=$((existing_results + 1)) ;;
                            *) failed_jobs=$((failed_jobs + 1)) ;;
                        esac
                    else
                        # Direct execution logic here
                        print_info "Direct execution not fully implemented for grouped runs"
                    fi
                done
            done
        done
        
        print_info "✅ Completed seed: $seed"
    done
    
    print_status "📊 By-seed execution summary:"
    print_info "  Total jobs: $total_jobs"
    print_info "  Submitted: $submitted_jobs"
    print_info "  Existing: $existing_results"
    print_info "  Failed: $failed_jobs"
}

# Execute main function
main 