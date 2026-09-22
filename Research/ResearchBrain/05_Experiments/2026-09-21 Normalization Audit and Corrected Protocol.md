---
type: experiment
status: completed
date: 2026-09-21
project: Full-Vector Geometric Calibration
benchmark: CIFAR-100 / CIFAR-100-C, ResNet-101, checkpoint seeds 2 and 4
evidence_scope: unpublished_repository_analysis
tags: [preprocessing, normalization, protocol-versioning, integrity]
---

# 2026-09-21 — Normalization audit and corrected protocol

Full write-up (with the provenance table): repo `docs/normalization_audit.md`.
Observation: [[Legacy benchmark evaluated CIFAR-100 with ImageNet statistics on a CIFAR-trained checkpoint]].
Successor experiment: [[2026-09-21 Layer-Selection Pilot]].

## Question
Was the CIFAR-100 evaluation input normalization consistent with the normalization
the checkpoints were trained with, across the whole execution path?

## Result (labels: **verified implementation fact** / **measured result**)
* **Verified implementation fact.** Checkpoints were trained through
  `data/cifar100.py::get_train_valid_loader` (CIFAR statistics). Train, validation,
  the DAC reference bank, all fitting and selection used CIFAR statistics. The clean
  **test** loader (`get_test_loader`) and the unified benchmark's CIFAR-*-C transform used
  **ImageNet** statistics. The caches hold post-transform tensors (value ranges −2.43…2.75
  vs −2.12…2.64); no double normalization. A third set of constants was found in
  `extract_corruption_features.py`.
* **Measured result** (paired diagnostic, same images, same checkpoint): the checkpoint's
  own recorded validation numbers (loss 0.886969 / 0.881487, acc 77.0 / 77.78) are reproduced
  under CIFAR statistics (0.886946 / 0.881432) and not under ImageNet statistics
  (0.896345 / 0.902084). Clean-test accuracy legacy → corrected: 0.7632 → 0.7651 (seed 4),
  0.7620 → 0.7659 (seed 2); top-1 agreement 93 %; neighbour overlap at `layer1` ≈ 21–32 %
  (top-10), at `conv1` ≈ 4 %; test/val kNN-distance ratio at `layer1` 0.95 → 0.995.
* **Repair.** `utils/preprocessing_protocol.py` (one authoritative spec, versioned protocol
  ids `legacy_v1_mixed_norm` / `corrected_v2_train_norm`), guarded stamps, incompatible-cache
  rejection, default of the unified benchmark switched to corrected (for correctness),
  legacy results untouched and labelled. 16 regression tests.
* **Measured.** Fit-side artifacts were numerically unaffected in this case: corrected native
  DAC weights `[0.24462, 0.18134, 0.41245, 0, 0.12380]`, `w_0=0.88537` equal the legacy
  `[0.244625, 0.181287, 0.412497, 0, 0.123806]`, `0.885362` (fit inputs are the CIFAR-normalized
  train/val features). The safe rule (refit everything under a new tree) was applied anyway.

## What this does NOT establish
* How much any legacy calibration or FV-DAC number changed (no end-to-end comparison of all
  methods was run; only the pilot's baselines were recomputed).
* Whether published CIFAR-100 **clean** numbers from the RGC/IJCAI-era scripts are affected:
  those scripts call the same legacy `get_test_loader`. **Unresolved — needs the paper authors'
  check.** Their default behaviour was intentionally left unchanged; `run_post_hoc_calibration.py`
  gained an opt-in flag.
* Anything about SVHN, PACS, DINOv2 paths (not audited; refused by the corrected protocol).

## Artifacts
`Research/GeometricFullCalibration/`: `utils/preprocessing_protocol.py`,
`tests/test_preprocessing_protocol.py`, `Experiments/preprocessing_diagnostic.py`,
`results/preprocessing_audit/preprocessing_diagnostic_seed{2,4}.json`,
`results/studyAB/phase0_corrected_v2/` (corrected baseline tree; legacy tree
`results/studyAB/phase0/` untouched).

## Status note (2026-09-21)
Corrected benchmark clean fit for seeds 2 and 4 (`results/studyAB/phase0_corrected_v2/`) produced Temperature Scaling, Vector Scaling and native DAC and their clean rows match the
layer pilot's independent recomputation to ≤ 1e-4 accuracy / ≤ 3.2e-5 NLL / ≤ 5.6e-4 ECE. The 24 corrected corruption-cell benchmark runs (job array 21533078) were still pending
when this was written; no corrected *benchmark-path* corruption numbers are claimed here.

## Addendum 2026-09-21 (second pass): IJCAI-era artifact trace (unresolved for publications)
Canonical RGC repo `Experiments/run_post_hoc_calibration.py:1120` uses the legacy `Data.cifar100.get_test_loader`; stored `calibration_comparison/ablation_baseline_cross_entropy_cifar100_resnet101_seed{2,4}.json`
(2026-01-11) record uncalibrated accuracy 76.20 % / 76.31 % — the legacy-normalization values from the paired diagnostic (76.20 / 76.32) rather than the corrected ones (76.59 / 76.51). Which artifacts the published tables used is **not established**;
no publication claim was edited and no one was contacted.
