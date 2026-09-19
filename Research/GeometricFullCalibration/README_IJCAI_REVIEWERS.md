# Reproducibility Guide for IJCAI Reviewers

This repository is the code release for **"Semantic Geometric Calibration in Randomized Neural Feature Space"**.

This guide is meant to make four things explicit:
- what is included in the repository,
- which scripts reproduce the paper path,
- which pieces are optional,
- how to set up a clean environment without relying on pre-existing local packages.

## What Is Included

- Training code for the backbone models used in the paper.
- The paper calibration pipeline (`RGCL`, `RGCC`, standard post-hoc baselines, and deep ensemble evaluation).
- Dataset loaders for CIFAR-10, CIFAR-100, and Tiny-ImageNet.
- Example checkpoints and result folders that show the expected output layout.
- Auxiliary analysis scripts used to inspect and aggregate outputs.

## What Is Optional

- FAISS: optional acceleration only. The default paper wrapper uses `fast_separation`, so the base reproduction path does not require FAISS.
- PACS / SVHN / CIFAR-C raw data: some broader code paths reference them, but they are not required for the default paper path.
- AugMix helper code: optional calibration variants reference it, but it is not part of the default IJCAI workflow.

## Recommended Setup

### Option 1: Environment file

```bash
conda env create -f environment.yml
conda activate geometric-internal-calibration
```

### Option 2: Manual Conda setup

```bash
conda create -n geometric-internal-calibration python=3.10 pip
conda activate geometric-internal-calibration
pip install -r requirements.txt
```

`requirements.txt` was trimmed to the packages needed for the default IJCAI workflow so that a fresh `pip install -r requirements.txt` succeeds in a clean environment.

## Optional FAISS Installation

FAISS is not required for `run_paper_experiments.py`, but some optional timing or alternate-search scripts use it.

Examples:

```bash
# Linux / Conda examples
conda install -c pytorch faiss-cpu
# or
conda install -c pytorch faiss-gpu
```

On Windows, `pip install faiss-gpu` is usually unavailable, so it is intentionally not part of the base requirements.

## Dataset Preparation

### CIFAR-10 / CIFAR-100

No manual download is needed for the default path. `torchvision` downloads them automatically into `./data/`.

### Tiny-ImageNet

If you want to run Tiny-ImageNet experiments, extract it into:

```text
data/tiny-imagenet-200/
```

### Other Datasets

The broader codebase also references CIFAR-C, SVHN, and PACS, but those datasets are not required for the default paper reproduction path and are not bundled here.

## Main Entry Points

### Quick smoke test

```bash
python run_paper_experiments.py --quick
```

### Specific paper configuration

```bash
python run_paper_experiments.py --model resnet18 --dataset cifar10 --seed 12 --skip-training
```

### Full wrapper run

```bash
python run_paper_experiments.py --full
```

The wrapper script is the easiest place to start because it:
- checks for model checkpoints,
- trains when needed,
- dispatches the calibration script with the expected arguments,
- writes summary output.

### Direct training command

```bash
python Experiments/train_model.py \
  --model-name resnet18 \
  --dataset cifar10 \
  --seed 12 \
  --output-root results/models
```

### Direct calibration command

```bash
python Experiments/run_rgc_experiments.py \
  --model-name resnet18 \
  --dataset cifar10 \
  --training-method baseline \
  --seed 12 \
  --results-base-dir results/models \
  --output-dir calibration_results
```

## Output Layout

Trained checkpoints are expected under:

```text
results/models/baseline/baseline_cross_entropy/{dataset}/{model}/seed{seed}/best_model.pth
```

Paper calibration outputs are expected under:

```text
calibration_results/{model}/{dataset}/baseline/{seed}/paper_results.json
```

Example output folders are already present in the repository so reviewers can inspect the intended structure.
The pre-trained quick-verification checkpoint is included at
`results/models/baseline/baseline_cross_entropy/cifar10/resnet18/seed12/best_model.pth`.

## Portability Notes

- Dataset imports were normalized to the lowercase `data.*` package so the code works reliably on case-sensitive systems.
- FAISS is now treated as optional instead of a hard dependency for the default path.
- The base requirements now include the packages actually imported by the main paper code path, including `Pillow`, `PyYAML`, and `transformers`.

## Lightweight Verification Commands

These are good sanity checks after installation:

```bash
python -c "import torch, torchvision, numpy, scipy, sklearn, pandas, timm, transformers; print('imports-ok')"
python run_paper_experiments.py --help
python Experiments/run_rgc_experiments.py --help
python Experiments/train_model.py --help
```

## Practical Scope

For IJCAI review, the most relevant scripts are:
- `run_paper_experiments.py`
- `Experiments/train_model.py`
- `Experiments/run_rgc_experiments.py`
- the modules under `Calibrators/`, `utils/`, and `data/` that support those scripts

Other files in the repository are exploratory or auxiliary and are not required to follow the default paper reproduction path.
