#!/bin/bash
# AAAI Multi-Model Validation: Raw vs Compressed Feature Analysis
# SLURM-compatible script for comprehensive model validation

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

# ===== CONFIGURATION =====
# Models to validate
MODELS=("resnet18" "resnet50" "densenet121" "dinov2_small_scratch")

# Default parameters
DEFAULT_VALIDATION_TYPE="aaai_ready"  # quick, full, aaai_ready
PYTHON_ENV="tamar_n_env"
OUTPUT_DIR="multi_model_validation"
BASE_OUTPUT_DIR="$OUTPUT_DIR"

# SLURM Configuration
USE_SLURM=true
PARTITION="gpu_partition" 
GPU_TYPE="rtx_4090"

# Corruption settings
DEFAULT_CORRUPTIONS=('gaussian_noise' 'shot_noise' 'impulse_noise'
            'defocus_blur' 'glass_blur' 'motion_blur' 'zoom_blur'
            'snow' 'frost' 'fog' 'brightness'
            'contrast' 'elastic_transform' 'pixelate' 'jpeg_compression')
DEFAULT_SEVERITIES=(1 3 5)
DEFAULT_NUM_SAMPLES=150

# Resource allocation based on validation type
declare -A VALIDATION_RESOURCES
VALIDATION_RESOURCES[quick]="4:00:00 32G 4"      # time memory cpus
VALIDATION_RESOURCES[full]="8:00:00 64G 8"
VALIDATION_RESOURCES[aaai_ready]="12:00:00 64G 8"

# ===== UTILITY FUNCTIONS =====
print_status() { echo "🔄 $1"; }
print_info() { echo "   $1"; }
print_error() { echo "❌ ERROR: $1" >&2; }
print_success() { echo "✅ $1"; }

show_usage() {
    echo "🚀 AAAI Multi-Model Validation"
    echo "=============================="
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "📋 COMMANDS:"
    echo "  quick              Run quick validation (ResNet-18 + ResNet-50)"
    echo "  full               Run full validation (all models)"
    echo "  aaai_ready         Run comprehensive AAAI-ready validation"
    echo "  single MODEL       Run validation for specific model"
    echo "  monitor            Monitor running jobs"
    echo "  help               Show this help"
    echo ""
    echo "📋 OPTIONS:"
    echo "  --models MODEL1,MODEL2    Specific models to test"
    echo "  --corruptions CORR1,CORR2 Specific corruptions to test"  
    echo "  --severities SEV1,SEV2    Specific severities to test"
    echo "  --num-samples N           Number of samples per corruption/severity"
    echo "  --output-dir DIR          Output directory"
    echo "  --no-slurm               Run directly without SLURM"
    echo "  --partition PARTITION     SLURM partition to use"
    echo "  --python-env ENV          Conda environment name"
    echo ""
    echo "📋 EXAMPLES:"
    echo "  $0 quick                                    # Quick validation"
    echo "  $0 full --output-dir custom_validation      # Full validation"
    echo "  $0 single resnet18 --no-slurm              # Single model locally"
    echo "  $0 aaai_ready --models resnet18,resnet50   # Custom AAAI validation"
    echo ""
    echo "📋 AVAILABLE MODELS:"
    echo "  resnet18, resnet50, densenet121, dinov2_small_scratch"
    echo ""
}

# Parse command line arguments
parse_arguments() {
    VALIDATION_TYPE="${1:-quick}"
    shift 2>/dev/null || true
    
    SELECTED_MODELS=()
    SELECTED_CORRUPTIONS=()
    SELECTED_SEVERITIES=()
    NUM_SAMPLES="$DEFAULT_NUM_SAMPLES"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --models)
                IFS=',' read -ra SELECTED_MODELS <<< "$2"
                shift 2
                ;;
            --corruptions)
                IFS=',' read -ra SELECTED_CORRUPTIONS <<< "$2"
                shift 2
                ;;
            --severities)
                IFS=',' read -ra SELECTED_SEVERITIES <<< "$2"
                shift 2
                ;;
            --num-samples)
                NUM_SAMPLES="$2"
                shift 2
                ;;
            --output-dir)
                OUTPUT_DIR="$2"
                BASE_OUTPUT_DIR="$OUTPUT_DIR"
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
            --python-env)
                PYTHON_ENV="$2"
                shift 2
                ;;
            *)
                if [[ "$VALIDATION_TYPE" == "single" && ${#SELECTED_MODELS[@]} -eq 0 ]]; then
                    SELECTED_MODELS=("$1")
                else
                    print_error "Unknown option: $1"
                    show_usage
                    exit 1
                fi
                shift
                ;;
        esac
    done
    
    # Set defaults based on validation type
    if [[ ${#SELECTED_MODELS[@]} -eq 0 ]]; then
        case $VALIDATION_TYPE in
            quick)
                SELECTED_MODELS=("resnet18" "resnet50")
                ;;
            full|aaai_ready)
                SELECTED_MODELS=("${MODELS[@]}")
                ;;
            single)
                print_error "Single mode requires a model name"
                show_usage
                exit 1
                ;;
        esac
    fi
    
    if [[ ${#SELECTED_CORRUPTIONS[@]} -eq 0 ]]; then
        SELECTED_CORRUPTIONS=("${DEFAULT_CORRUPTIONS[@]}")
    fi
    
    if [[ ${#SELECTED_SEVERITIES[@]} -eq 0 ]]; then
        SELECTED_SEVERITIES=("${DEFAULT_SEVERITIES[@]}")
    fi
}

# Get resource requirements for validation type
get_validation_resources() {
    local validation_type="$1"
    local resources="${VALIDATION_RESOURCES[$validation_type]}"
    
    if [[ -z "$resources" ]]; then
        resources="${VALIDATION_RESOURCES[quick]}"  # fallback
    fi
    
    echo $resources
}

# Validate model name
validate_model() {
    local model="$1"
    for valid_model in "${MODELS[@]}"; do
        if [[ "$model" == "$valid_model" ]]; then
            return 0
        fi
    done
    return 1
}

# Check if results already exist
check_validation_results() {
    local validation_type="$1"
    local models_str="$2"
    local output_dir="$3"
    
    local results_file="$output_dir/multi_model_comprehensive_results.json"
    local summary_file="$output_dir/aaai_executive_summary_multi_model.md"
    
    if [[ -f "$results_file" && -f "$summary_file" ]]; then
        # Check if results are valid JSON
        if python -c "import json; json.load(open('$results_file'))" 2>/dev/null; then
            return 0  # Valid results exist
        fi
    fi
    
    return 1  # No valid results
}

# Submit multi-model validation job to SLURM
submit_validation_job() {
    local validation_type="$1"
    local selected_models=("${!2}")
    local selected_corruptions=("${!3}")
    local selected_severities=("${!4}")
    local num_samples="$5"
    local output_dir="$6"
    
    # Validate inputs
    for model in "${selected_models[@]}"; do
        if ! validate_model "$model"; then
            print_error "Invalid model: $model"
            print_info "Available models: ${MODELS[*]}"
            return 1
        fi
    done
    
    # Check for existing results
    local models_str=$(IFS=,; echo "${selected_models[*]}")
    if check_validation_results "$validation_type" "$models_str" "$output_dir"; then
        print_status "Valid validation results already exist for $validation_type"
        print_info "Skipping job submission to avoid duplicate work"
        return 2  # Special code for existing results
    fi
    
    # Get resource requirements
    local resources=($(get_validation_resources "$validation_type"))
    local time="${resources[0]}"
    local mem="${resources[1]}"
    local cpus="${resources[2]}"
    
    # Create job name
    local job_name="multimodel_${validation_type}_$(echo "${selected_models[*]}" | tr ' ' '_' | tr '[:upper:]' '[:lower:]')"
    
    # Create output directory and logs
    mkdir -p "$output_dir"
    mkdir -p "$(dirname "$output_dir")/logs"
    local log_file="$(dirname "$output_dir")/logs/${job_name}_%j.out"
    
    # Convert arrays to comma-separated strings for passing to Python
    local models_str=$(IFS=,; echo "${selected_models[*]}")
    local corruptions_str=$(IFS=,; echo "${selected_corruptions[*]}")
    local severities_str=$(IFS=,; echo "${selected_severities[*]}")
    
    # Build Python command
    local cmd="python $EXPERIMENTS_DIR/multi_model_validation.py"
    cmd+=" --validation-type $validation_type"
    cmd+=" --models $models_str"
    cmd+=" --corruptions $corruptions_str"
    cmd+=" --severities $severities_str"
    cmd+=" --num-samples $num_samples"
    cmd+=" --output-dir $output_dir"
    
    print_status "Submitting multi-model validation job..."
    print_info "Validation Type: $validation_type"
    print_info "Models: ${selected_models[*]}"
    print_info "Corruptions: ${selected_corruptions[*]}"
    print_info "Severities: ${selected_severities[*]}"
    print_info "Samples per corruption/severity: $num_samples"
    print_info "Resources: Time=$time, Mem=$mem, CPUs=$cpus"
    print_info "Output: $output_dir"
    
    # Submit to SLURM
    local submit_output
    submit_output=$(sbatch \
        --partition="$PARTITION" \
        --job-name="$job_name" \
        --output="$log_file" \
        --time="$time" \
        --gpus=0 \
        --cpus-per-task="$cpus" \
        --mem="$mem" \
        --exclude=cs-4090-08 \
        --wrap="
        echo '🤖 AAAI MULTI-MODEL VALIDATION'
        echo '================================'
        echo 'Job ID: \$SLURM_JOB_ID'
        echo 'Validation Type: $validation_type'
        echo 'Models: ${selected_models[*]}'
        echo 'Host: \$(hostname) | GPU: \$CUDA_VISIBLE_DEVICES'
        echo 'Started: \$(date)'
        echo '================================'

        module load anaconda
        source activate $PYTHON_ENV
        export PYTHONPATH='$PROJECT_ROOT:\$PYTHONPATH'
        export CUDA_HOME=/usr/local/cuda
        export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:False
        
        echo 'Environment setup complete'
        echo 'Python version:' \$(python --version)
        echo 'PyTorch version:' \$(python -c 'import torch; print(torch.__version__)')
        echo 'CUDA available:' \$(python -c 'import torch; print(torch.cuda.is_available())')
        if python -c 'import torch; print(torch.cuda.is_available())' | grep -q True; then
            echo 'GPU Memory: Available:' \$(python -c 'import torch; print(f\"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB\")')
        else
            echo '⚠️ CUDA not available'
        fi
        
        cd $PROJECT_ROOT || { echo 'Failed to change directory'; exit 1; }

        echo '🚀 Executing Multi-Model Validation:'
        echo '$cmd'
        echo '===================================='

        if $cmd; then
            echo '✅ MULTI-MODEL VALIDATION COMPLETED SUCCESSFULLY'
            echo 'Results saved to: $output_dir'
            
            # Generate summary
            echo '📋 Generating execution summary...'
            if [[ -f \"$output_dir/multi_model_comprehensive_results.json\" ]]; then
                echo '📊 Results file created successfully'
                python -c \"
import json
import sys
try:
    with open('$output_dir/multi_model_comprehensive_results.json', 'r') as f:
        results = json.load(f)
    
    validation_summary = results.get('aggregated_analysis', {}).get('theoretical_validation_summary', {})
    aaai_status = results.get('aggregated_analysis', {}).get('aaai_validation_status', {})
    
    print('📊 VALIDATION SUMMARY:')
    print('=' * 40)
    print(f'Models Tested: {validation_summary.get(\\\"models_tested\\\", 0)}')
    print(f'Max Raw Contraction: {validation_summary.get(\\\"max_raw_contraction_observed\\\", 0):.1f}x')
    print(f'Theory Validated: {validation_summary.get(\\\"theory_strongly_validated\\\", False)}')
    print(f'Cross-Architecture Consistent: {validation_summary.get(\\\"consistent_across_models\\\", False)}')
    print(f'Publication Ready: {aaai_status.get(\\\"publication_ready\\\", False)}')
    print(f'Recommendation: {aaai_status.get(\\\"recommendation\\\", \\\"UNKNOWN\\\")}')
except Exception as e:
    print(f'Could not parse results: {e}')
                \"
            else
                echo '⚠️ Results file not found'
            fi
        else
            echo '❌ MULTI-MODEL VALIDATION FAILED'
            exit 1
        fi

        echo 'Finished: '\$(date)
        " 2>&1)
    
    local submit_exit_code=$?
    
    # Check submission result
    if [[ $submit_exit_code -eq 0 ]] && [[ -n "$submit_output" ]]; then
        local job_id
        job_id=$(echo "$submit_output" | grep -o 'Submitted batch job [0-9]*' | grep -o '[0-9]*')
        
        if [[ -n "$job_id" ]] && [[ "$job_id" =~ ^[0-9]+$ ]]; then
            print_success "Multi-model validation job submitted with ID: $job_id"
            print_info "📝 Log file: $log_file"
            print_info "📊 Monitor progress: squeue -u \$USER"
            print_info "📁 Results will be saved to: $output_dir"
            return 0
        else
            print_error "Could not extract job ID from output: $submit_output"
            return 1
        fi
    else
        print_error "Job submission failed with exit code: $submit_exit_code"
        print_error "Error output: $submit_output"
        return 1
    fi
}

# Direct execution (no SLURM)
run_direct_validation() {
    local validation_type="$1"
    local selected_models=("${!2}")
    local selected_corruptions=("${!3}")
    local selected_severities=("${!4}")
    local num_samples="$5"
    local output_dir="$6"
    
    print_status "Running multi-model validation directly (no SLURM)..."
    
    # Check for existing results
    local models_str=$(IFS=,; echo "${selected_models[*]}")
    if check_validation_results "$validation_type" "$models_str" "$output_dir"; then
        print_status "Valid validation results already exist"
        return 0
    fi
    
    # Setup environment
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
    export CUDA_HOME=/usr/local/cuda
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:False
    
    # Activate conda environment
    if [[ -z "$SLURM_JOB_ID" ]]; then
        eval "$(conda shell.bash hook)"
        conda activate "$PYTHON_ENV"
    fi
    
    # Create output directory
    mkdir -p "$output_dir"
    mkdir -p "$(dirname "$output_dir")/logs"
    
    # Convert arrays to comma-separated strings
    local models_str=$(IFS=,; echo "${selected_models[*]}")
    local corruptions_str=$(IFS=,; echo "${selected_corruptions[*]}")
    local severities_str=$(IFS=,; echo "${selected_severities[*]}")
    
    # Build and execute command
    local cmd="python $EXPERIMENTS_DIR/multi_model_validation.py"
    cmd+=" --validation-type $validation_type"
    cmd+=" --models $models_str"
    cmd+=" --corruptions $corruptions_str"
    cmd+=" --severities $severities_str"
    cmd+=" --num-samples $num_samples"
    cmd+=" --output-dir $output_dir"
    
    local log_file="$(dirname "$output_dir")/logs/direct_validation_$(date +%Y%m%d_%H%M%S).out"
    
    print_info "Command: $cmd"
    print_info "Log file: $log_file"
    
    cd "$PROJECT_ROOT"
    
    # Execute with logging
    $cmd > "$log_file" 2>&1 &
    local pid=$!
    
    print_info "Started with PID: $pid"
    print_info "Monitor log: tail -f $log_file"
    
    return 0
}

# Monitor validation jobs
monitor_validation_jobs() {
    echo "🔍 MULTI-MODEL VALIDATION JOB MONITOR"
    echo "====================================="
    
    # Show running jobs
    echo ""
    echo "🔄 Running validation jobs:"
    if squeue -u $USER --name="multimodel_*" 2>/dev/null | tail -n +2; then
        echo "   Found running validation jobs"
    else
        echo "   No running validation jobs found"
    fi
    
    # Show recent completed jobs
    echo ""
    echo "✅ Recent completed jobs:"
    sacct -u $USER --name="multimodel_*" --starttime=today --state=COMPLETED \
          --format="JobID,JobName,State,ExitCode,Start,End,Elapsed" 2>/dev/null | head -10 || \
          echo "   No completed jobs found"
    
    # Show recent failed jobs  
    echo ""
    echo "❌ Recent failed jobs:"
    sacct -u $USER --name="multimodel_*" --starttime=today --state=FAILED \
          --format="JobID,JobName,State,ExitCode,Start,End" 2>/dev/null | head -10 || \
          echo "   No failed jobs found"
    
    # Show recent results
    echo ""
    echo "📊 Recent results:"
    find "$BASE_OUTPUT_DIR" -name "multi_model_comprehensive_results.json" -mtime -1 2>/dev/null | \
         sort | tail -5 | while read file; do
        echo "   📁 $(dirname "$file")"
    done
    
    # Show disk usage
    echo ""
    echo "💾 Disk usage:"
    if [[ -d "$BASE_OUTPUT_DIR" ]]; then
        du -sh "$BASE_OUTPUT_DIR" 2>/dev/null || echo "   Could not calculate disk usage"
    fi
}

# Environment setup
setup_environment() {
    print_status "Setting up multi-model validation environment..."
    
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
    fi
}

# Main execution function
run_validation() {
    local validation_type="$1"
    local selected_models=("${!2}")
    local selected_corruptions=("${!3}")
    local selected_severities=("${!4}")
    local num_samples="$5"
    local output_dir="$6"
    
    print_status "Starting $validation_type multi-model validation..."
    print_info "Models: ${selected_models[*]}"
    print_info "Corruptions: ${selected_corruptions[*]}"
    print_info "Severities: ${selected_severities[*]}"
    print_info "Samples: $num_samples"
    print_info "Output: $output_dir"
    print_info "SLURM: $USE_SLURM"
    
    if [[ "$USE_SLURM" == true ]]; then
        submit_validation_job "$validation_type" selected_models[@] selected_corruptions[@] selected_severities[@] "$num_samples" "$output_dir"
        local exit_code=$?
        
        case $exit_code in
            0)
                print_success "Multi-model validation job submitted successfully!"
                print_info "Monitor with: $0 monitor"
                ;;
            2)
                print_status "Results already exist - no job submitted"
                ;;
            *)
                print_error "Job submission failed!"
                exit 1
                ;;
        esac
    else
        run_direct_validation "$validation_type" selected_models[@] selected_corruptions[@] selected_severities[@] "$num_samples" "$output_dir"
        print_info "Direct execution started - check logs for progress"
    fi
}

# ===== MAIN EXECUTION =====
# Setup environment
setup_environment

# Parse arguments
parse_arguments "$@"

# Handle different commands
case "$VALIDATION_TYPE" in
    "quick")
        run_validation "quick" SELECTED_MODELS[@] SELECTED_CORRUPTIONS[@] SELECTED_SEVERITIES[@] "$NUM_SAMPLES" "$OUTPUT_DIR/quick_validation"
        ;;
    "full") 
        run_validation "full" SELECTED_MODELS[@] SELECTED_CORRUPTIONS[@] SELECTED_SEVERITIES[@] "$NUM_SAMPLES" "$OUTPUT_DIR/full_validation"
        ;;
    "aaai_ready")
        run_validation "aaai_ready" SELECTED_MODELS[@] SELECTED_CORRUPTIONS[@] SELECTED_SEVERITIES[@] "$NUM_SAMPLES" "$OUTPUT_DIR/aaai_ready_validation"
        ;;
    "single")
        run_validation "single" SELECTED_MODELS[@] SELECTED_CORRUPTIONS[@] SELECTED_SEVERITIES[@] "$NUM_SAMPLES" "$OUTPUT_DIR/single_${SELECTED_MODELS[0]}_validation"
        ;;
    "monitor")
        monitor_validation_jobs
        ;;
    "help")
        show_usage
        ;;
    *)
        print_error "Unknown command: $VALIDATION_TYPE"
        show_usage
        exit 1
        ;;
esac

echo ""
echo "🎯 MULTI-MODEL VALIDATION SCRIPT COMPLETE"
echo "========================================="
echo ""
echo "Next steps:"
echo "• Monitor progress: $0 monitor"
echo "• Check queue: squeue -u \$USER"
echo "• View logs: ls $(dirname "$OUTPUT_DIR")/logs/"
echo "• Check results: ls $OUTPUT_DIR/"