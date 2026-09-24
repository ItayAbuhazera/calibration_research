---
type: experiment
status: completed_descriptive_cached_outputs_only
date: 2026-09-24
project: Full-Vector Geometric Calibration
benchmark: CIFAR-100 / CIFAR-100-C (12 development cells), ResNet-101, checkpoints 2 and 4
preregistered: false
experiment_type: descriptive (reconciliation with a published claim; no fitting)
parent_question: Why do our source-trained intermediate probes not beat the native head when Uselis & Oh (ICLR 2025) report that source-trained intermediate probes beat a retrained last-layer probe under CIFAR-100-C?
exposure_ledger: "[[Representation Correction Exposure Ledger]]"
evidence_scope: unpublished_repository_analysis (descriptive; development cells; target-selected rows are not deployable)
tags: [stage0, reconciliation, uselis-oh, layer-probes, descriptive]
---

# 2026-09-24 — Phase 1B: reconciliation with Uselis & Oh (ICLR 2025) using cached outputs

Parent: [[2026-09-22 Stage 0 Probe-Logit Increment Study]], [[2026-09-24 Stage 0 Evidence Ablation]]. Post-hoc descriptive analysis (script `atlas/stage0_reconcile_uo.py`, output `results/stage0/report/stage0_reconcile_uo.json`); no fitting, no new features, no reserved data. **The published claim below is quoted as stated in the task; the paper's text was not read for this note.** Workflow: uncommitted working-tree version of the canonical workflow and templates.

## Question
Claim scope: our clean-trained GAP layer probes (12 candidate blocks of ResNet-101, CIFAR-100, checkpoints 2 and 4, layer-pilot recipe) versus the native head on clean and the 12 development cells. Reported claim to reconcile: source-trained intermediate probes beat a retrained last-layer probe under CIFAR-100-C when the layer is chosen on OOD validation. Ours do not beat the native head.

## Results (facts; accuracy %, 12-cell macro unless stated)

**Clean-selected layer (best clean accuracy among the 12 probes; deployable rule).**

| | Layer | Clean acc | 12-cell acc | Native head (clean / 12-cell) | Δ vs head (12-cell) |
|---|---|---|---|---|---|
| Checkpoint 2 | layer4.2 | 76.62 | 50.01 | 76.59 / 50.09 | -0.08 pp |
| Checkpoint 4 | layer4.1 | 76.91 | 50.69 | 76.51 / 51.01 | -0.32 pp |

Excluding the final block, the clean-selected intermediate probe is layer4.1 in both checkpoints: 49.88 (-0.21 pp) and 50.69 (-0.32 pp).

**Target-selected layer per corruption family (best mean accuracy over the family's three severities). target-selected (not deployable).**

Checkpoint 2:

| Family | Best layer | Acc | Head | Δ (pp) | Best intermediate (Δ pp) |
|---|---|---|---|---|---|
| gaussian noise | layer4.2 | 27.00 | 27.24 | -0.24 | layer4.1 (-0.49) |
| defocus blur | layer4.2 | 57.95 | 57.85 | +0.10 | layer4.1 (-0.01) |
| fog | layer4.2 | 60.53 | 60.45 | +0.08 | layer4.1 (-0.13) |
| jpeg compression | layer4.1 | 54.60 | 54.82 | -0.22 | layer4.1 (-0.22) |

Checkpoint 4:

| Family | Best layer | Acc | Head | Δ (pp) | Best intermediate (Δ pp) |
|---|---|---|---|---|---|
| gaussian noise | layer4.2 | 28.36 | 28.46 | -0.11 | layer4.1 (-0.61) |
| defocus blur | layer4.2 | 58.07 | 58.07 | +0.00 | layer4.1 (-0.11) |
| fog | layer4.2 | 61.13 | 61.16 | -0.03 | layer4.1 (-0.27) |
| jpeg compression | layer4.2 | 56.29 | 56.35 | -0.07 | layer4.1 (-0.29) |

Macro over families, target-selected: 50.02 (checkpoint 2; head 50.09) and 50.96 (checkpoint 4; head 51.01). Even with target labels choosing the layer per family, the best probe is at or below the native head (−0.24 to +0.10 pp); the best intermediate (layer4.1) is 0.0 to −0.6 pp.

**Retrained last-layer probe analogue (layer4.2 GAP probe).** 50.01 vs head 50.09 (-0.08 pp) and 50.96 vs 51.01 (-0.05 pp); largest absolute difference from the head in any of the 13 conditions 0.47 / 0.40 pp. Best intermediate versus this retrained last-layer probe: −0.1 to −0.3 pp.

## Discrepancy: candidate explanations and the cached numbers that bear on each (no fitting)

1. **Their baseline is a retrained last-layer probe; ours is the native head.** Bears: retraining the last layer on the training features here reproduces the native head to within 0.1 pp (12-cell) and 0.5 pp (any condition), so our "native head" and a "retrained last-layer probe" are the same baseline. This explanation would apply only if their retrained baseline is weaker than a trained head, which our cache cannot test. Against the retrained baseline, our best intermediate probe is still 0.1–0.3 pp lower.
2. **GAP pooling versus their pooling.** No cached probe number bears. The only cached pooling comparison is a distance-readout sensitivity (2×2 pooling +0.3 to +0.44 pp over GAP in the layer pilot), a different readout.
3. **ResNet-18 versus ResNet-101.** No cached number; all cached checkpoints are ResNet-101.
4. **Layer selected on OOD data versus ID data.** Bears directly: target-labelled per-family selection (OOD) does not create a gain (−0.24 to +0.10 pp against the head), so the selection data is not what separates the results here.
5. **Probe recipe.** Bears partly: the layer-pilot recipe selected λ = 0.01, the upper edge of the grid {1e-4, 1e-3, 1e-2}, for all three layer4 probes, with inner-fit NLL still falling at the edge (layer4.2: 1.90 → 1.31 → 1.02, checkpoint 2), and the lower edge (1e-4) for layer1–2 probes. Edge selection means the probes may be under-tuned, especially at layer4; the layer3 probes sit at the interior value 1e-3. This is a recipe difference that could matter but that the cache cannot test (it would need a refit, not authorized here). Probes are also trained on the network's own 45k training features (memorized to a degree).

## What this does not establish
Not a test of the published claim, not a statement about ResNet-18 or their pooling and recipe, and not evidence that intermediate probes cannot beat a last-layer probe elsewhere. The target-selected rows use target labels and are not deployable. Development cells, two checkpoints.

## Post-mortem and allocation
Primary contrast: none (descriptive). More plausible: recipe (grid edge) and pooling/architecture differences; less plausible: OOD-versus-ID selection and baseline strength. Indistinguishable from cache: pooling, architecture, recipe. Allocation: none; Phase 2 (a pretrained ResNet-50) is a separate frozen card. Review status: self-audit, single author.

## Update 2026-09-24 (after attempting to read the paper; appended)
See [[Intermediate Layer Classifiers for OOD generalization]]. The PDF could not be read here; the protocol summary comes from the arXiv HTML render via a summarizing tool and is unverified. Reported protocol: **frozen public pretrained models** (TorchVision ResNets), GAP for ResNets, affine probes trained with Adam for 100 epochs with **ℓ1** regularization, hyperparameters and layer chosen on an **OOD validation set** (best layer among `l ≤ L−2`), baseline last-layer retraining (DFR); the one quoted CIFAR-C number (+2 to +5 pp) is for CIFAR-10-C. Two candidate explanations are added to the list above, neither testable from our cache: **(6) pretrained frozen backbone vs a CIFAR-trained-from-scratch ResNet-101** (the regime-map pilot's state (a) is the matching test), and **(7) probe regularizer and optimizer (ℓ1/Adam vs L2/L-BFGS)**. The task's statement that the claim covers CIFAR-100-C is not confirmed by this reading.

