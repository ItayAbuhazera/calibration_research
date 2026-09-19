# Source summaries copied from results/unified_benchmark_tiny_imagenet_resnet50_sbatch/seed{11..15}/summary_metrics.json
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate geo_cuda12
python research_log/scripts/aggregate_batch.py research_log/experiments/2026-04-27_tiny-imagenet-resnet50_path-decision-batch full_vector_distance_fusion
