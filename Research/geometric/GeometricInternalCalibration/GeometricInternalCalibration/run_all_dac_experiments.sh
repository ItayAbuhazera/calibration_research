#!/bin/bash
# Run DAC comparison experiments for all models and seeds across CIFAR10, CIFAR100, and Tiny ImageNet

python Experiments/run_dac_comparison_experiments.py \
    --models densenet121 dinov2_giant dinov2_large resnet101 resnet152 resnet18 resnet50 wide-resnet28-10 wide-resnet28-10_1 \
    --datasets cifar10 cifar100 tiny_imagenet \
    --training-methods baseline_cross_entropy \
    --seeds 0 1 2 3 4 5 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 \
    --results-base-dir aaai_full_experiments/results \
    --output-base-dir calibration_comparison_results
