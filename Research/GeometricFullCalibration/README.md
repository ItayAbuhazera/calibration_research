# Random Geometric Calibration (RGC)

Official code release for **"Semantic Geometric Calibration in Randomized Neural Feature Space"**.

This repository contains:
- the training and calibration code used for the IJCAI submission,
- a one-command wrapper for the paper experiments,
- reviewer-oriented setup instructions in `README_IJCAI_REVIEWERS.md`,
- a separate [research guide for the unified full-vector benchmark](README_FULL_VECTOR_BENCHMARK.md),
- example checkpoints and result folders showing the expected layout.

`README_IJCAI_REVIEWERS.md` remains the stable IJCAI reproduction guide; the
unified full-vector benchmark is a separate research path.

## Quick Start

Create a clean Conda environment and install the standalone requirements:

```bash
conda env create -f environment.yml
conda activate geometric-internal-calibration
pip install -r requirements.txt
```

Run the paper wrapper:

```bash
python run_paper_experiments.py --quick
```

Run a specific configuration:

```bash
python run_paper_experiments.py --model resnet18 --dataset cifar10 --seed 12 --skip-training
```

For quick verification, this repository ships a pre-trained checkpoint at:
`results/models/baseline/baseline_cross_entropy/cifar10/resnet18/seed12/best_model.pth`

## Main Files

- `run_paper_experiments.py`: top-level entry point for quick checks and paper runs.
- `Experiments/train_model.py`: training entry point used by the wrapper.
- `Experiments/run_rgc_experiments.py`: calibration comparison pipeline for the paper.
- `Calibrators/`: calibration methods including DAC and geometric calibration.
- `data/`: dataset loaders used by the paper code.
- `results/` and `calibration_results/`: expected output locations for checkpoints and calibration outputs.

## Installation Notes

`requirements.txt` is the portable base install for the default IJCAI workflow.

FAISS is optional:
- the main `run_paper_experiments.py` path uses `fast_separation` by default and does not require FAISS,
- FAISS-specific experiments remain optional and need a separate local install,
- on Windows, `pip install faiss-gpu` is typically unavailable.

Optional FAISS examples:

```bash
# Linux / Conda examples
conda install -c pytorch faiss-cpu
# or
conda install -c pytorch faiss-gpu
```

## Dataset Notes

- CIFAR-10 and CIFAR-100 download automatically through `torchvision`.
- Tiny-ImageNet should be extracted into `data/tiny-imagenet-200/` if you want to run those experiments.
- CIFAR-C, SVHN, and PACS are referenced by some broader code paths, but their raw data is not bundled in this snapshot.

## Reproducibility

The detailed reviewer guide in `README_IJCAI_REVIEWERS.md` documents:
- the recommended environment setup,
- which scripts correspond to the paper workflow,
- expected output paths,
- optional components that are intentionally outside the default path.

## Citation

```bibtex
@inproceedings{rgc2026,
  title={Semantic Geometric Calibration in Randomized Neural Feature Space},
  author={Anonymous},
  booktitle={Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI)},
  year={2026}
}
```
