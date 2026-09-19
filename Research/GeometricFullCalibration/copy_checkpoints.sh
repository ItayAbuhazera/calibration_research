#!/usr/bin/env bash
set -euo pipefail

SRC_ROOT="/home/itayab/PyCharmProjects/geometric/GeometricInternalCalibration/GeometricInternalCalibration/aaai_full_experiments/results/baseline/baseline_cross_entropy/tiny_imagenet/resnet50"
DST_ROOT="results/models/baseline/baseline_cross_entropy/tiny_imagenet/resnet50"

for seed in 11 12 13 14 15 10; do
  src="${SRC_ROOT}/seed${seed}/best_model.pth"
  dst="${DST_ROOT}/seed${seed}/best_model.pth"

  mkdir -p "$(dirname "$dst")"

  if [[ -f "$src" ]]; then
    cp "$src" "$dst"
    echo "Copied: $src -> $dst"
  else
    echo "Missing source file: $src"
  fi
done