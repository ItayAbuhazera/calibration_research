#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=plot_layer_norms
#SBATCH --output=logs/plot_layer_norms_%j.out
#SBATCH --error=logs/plot_layer_norms_%j.err
#SBATCH --time=0-02:00:00
#SBATCH --ntasks=1
#SBATCH --gpus=rtx_3090:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Activate conda environment
source ~/.bashrc
conda activate geo_cuda12

# Get project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Create logs directory if it doesn't exist
mkdir -p logs
mkdir -p figs

# Print environment info
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "Python: $(which python)"
echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Number of GPUs: $(python -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)')"
nvidia-smi
echo '----------------------------------------'

# Default arguments (can be overridden by command line)
MODEL=${1:-resnet50}
CHECKPOINT=${2:-}
BATCH_SIZE=${3:-256}
OUTPUT=${4:-}

# Build command
CMD="python plot_layer_norms.py --model $MODEL --batch-size $BATCH_SIZE --device cuda"

if [ -n "$CHECKPOINT" ]; then
    CMD="$CMD --checkpoint $CHECKPOINT"
fi

if [ -n "$OUTPUT" ]; then
    CMD="$CMD --output $OUTPUT"
fi

echo "Command: $CMD"
echo '----------------------------------------'

# Run the command
$CMD

EXIT_CODE=$?
echo '----------------------------------------'
echo 'End Time: $(date)'
echo "Exit Code: $EXIT_CODE"
echo '========================================'

exit $EXIT_CODE

