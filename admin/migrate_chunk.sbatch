#!/bin/bash
#SBATCH --job-name=research-migrate
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=12:00:00
#SBATCH --output=/groups/cliron_group/itayab/calibration_research/logs/migrate_%A_%a.out
#SBATCH --error=/groups/cliron_group/itayab/calibration_research/logs/migrate_%A_%a.err

set -euo pipefail

ROOT_SRC=/home/itayab/PyCharmProjects/Research
ROOT_DST=/groups/cliron_group/itayab/calibration_research/Research

mkdir -p "$ROOT_DST"

case "$SLURM_ARRAY_TASK_ID" in

0)
    echo "=== GeometricFullCalibration ==="
    rsync -aH \
      --partial \
      --info=progress2 \
      --exclude='__pycache__/' \
      --exclude='.pytest_cache/' \
      --exclude='.mypy_cache/' \
      --exclude='.ruff_cache/' \
      "$ROOT_SRC/GeometricFullCalibration/" \
      "$ROOT_DST/GeometricFullCalibration/"
    ;;

1)
    echo "=== Canonical RGC ==="
    mkdir -p "$ROOT_DST/geometric/GeometricInternalCalibration/GeometricInternalCalibration"

    rsync -aH \
      --partial \
      --info=progress2 \
      --exclude='_backup_unique_from_GeometricInternalCalibration_1_20260627/' \
      --exclude='__pycache__/' \
      --exclude='.pytest_cache/' \
      --exclude='.mypy_cache/' \
      --exclude='.ruff_cache/' \
      "$ROOT_SRC/geometric/GeometricInternalCalibration/GeometricInternalCalibration/" \
      "$ROOT_DST/geometric/GeometricInternalCalibration/GeometricInternalCalibration/"
    ;;

2)
    echo "=== Remaining Research workspace ==="
    rsync -aH \
      --partial \
      --info=progress2 \
      --exclude='/GeometricFullCalibration/' \
      --exclude='/geometric/GeometricInternalCalibration/GeometricInternalCalibration/' \
      --exclude='/geometric/GeometricInternalCalibration/GeometricInternalCalibration_1/' \
      --exclude='/geometric/GeometricInternalCalibration/GeometricInternalCalibration-main/' \
      --exclude='/geometric/GeometricInternalCalibration/GeometricInternalCalibration-main_1/' \
      --exclude='/geometric/GeometricInternalCalibration/GeometricInternalCalibration-main (2).zip' \
      --exclude='__pycache__/' \
      --exclude='.pytest_cache/' \
      --exclude='.mypy_cache/' \
      --exclude='.ruff_cache/' \
      "$ROOT_SRC/" \
      "$ROOT_DST/"
    ;;

*)
    echo "Unknown task ID: $SLURM_ARRAY_TASK_ID"
    exit 2
    ;;
esac

echo "=== DONE task $SLURM_ARRAY_TASK_ID on $(hostname) at $(date) ==="
