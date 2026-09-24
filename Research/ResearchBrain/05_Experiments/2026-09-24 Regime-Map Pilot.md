---
type: experiment
status: frozen_v2_launch_authorized
date: 2026-09-24
project: Full-Vector Geometric Calibration
benchmark: CIFAR-100 / CIFAR-100-C (12 development cells), ImageNet-pretrained ResNet-50 (seven model states (a; b1/b3/b10 × two fine-tuning seeds)), Stage 0 protocol
preregistered: true
experiment_type: mechanism discrimination (target-supervised development diagnostic)
parent_question: Is source-recoverability of intermediate evidence controlled by its clean redundancy given Z, rather than by coming from the same network or from mid-depth?
exposure_ledger: "[[Representation Correction Exposure Ledger]]"
evidence_scope: unpublished_repository_analysis (development; not confirmation; not a deployable method)
tags: [regime-map, pretrained, resnet50, recoverability, clean-redundancy, frozen-not-launched]
---

# 2026-09-24 — Regime-map pilot: does clean redundancy control source-recoverability?

**Status: spec v2 frozen (pre-launch amendment of v1, see Amendments); launch authorized by the researcher.** Frozen specification (authoritative): repo `docs/regime_map_pilot_spec_v2.md` (v1 `docs/regime_map_pilot_spec.md` kept unedited). Parents: [[2026-09-22 Stage 0 Probe-Logit Increment Study]], [[2026-09-24 Stage 0 Evidence Ablation]]; hypothesis (proposed): [[H-STAGE0-01 Layer3.22 probe evidence is redundant with the logits on clean data but complementary under corruption, and clean supervision cannot identify the useful combination]]. Workflow: uncommitted working-tree version of `docs/research_workflow.md` and the card template (they had not landed in git).

## Question and confound
Stage 0 (ResNet-101 from scratch): the mid-depth probe evidence is target-recoverable but not source-recoverable; the evidence ablation found that another checkpoint's logits are source-recoverable. **"Same network" and "clean-redundant given `Z`" are fully confounded** in the existing data. This pilot creates a source of evidence from the *same* network whose clean redundancy is manipulated, and tests whether recoverability follows it.
Claim scope: development diagnostic; ImageNet-pretrained ResNet-50, CIFAR-100, the 12 development cells; one training seed per state; not a deployable method, not an information ceiling, not confirmation.

## Design (details in the spec)
* **Backbone:** torchvision ResNet-50 `IMAGENET1K_V1`, file `resnet50-0676ba61.pth`, sha256 `0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a`.
* **Resolution policy:** corrupt at native 32×32, bilinear upsample to 224×224, ImageNet normalization, one policy for every state; strict-fp32 extraction. Chosen because the pretrained stem/BN are matched to ~224 inputs, a modified stem would discard the pretrained `conv1`, and corruptions are defined at 32×32.
* **States:** (a) frozen backbone + linear head trained on CIFAR-100 train (one fit); (b) the same backbone fine-tuned end to end for 10 epochs, saved after epoch 1 (b1), epoch 3 (b3) and the final epoch (b10) as a dose, **in two independent fine-tuning runs (seeds 20260924 and 20260925)**.
* **Evidence `P`:** GAP linear probe on the last block of `layer3`, trained on each state's own 45k train features with the layer-pilot recipe, unchanged across states, plus the **grid-edge rule** (extend the λ grid by one decade toward a selected edge, at most twice per direction; identical for the head of (a) and every probe; every selected λ reported).
* **Protocol:** Stage 0 code, folds, λ grid, bootstrap and cell assignment unchanged; T and S fits at 8k×1 (primary) and 2.5k×1 on the 12 development cells; S fits scored on clean held-out images for the clean increment. Two fine-tuning seeds for the fine-tuned states, one fit for (a); the image-bootstrap intervals do not capture training-seed variance.

## Competing explanations and predictions
* **Clean redundancy controls recoverability:** the state whose `P` is clean-complementary (a, if the manipulation works) has a small gap; fine-tuning (b-final) makes `P` clean-redundant and the gap large; along the doses the gap does not fall as the clean increment falls.
* **Same-network / mid-depth is what matters (redundancy irrelevant):** `P` is clean-complementary in (a) yet still not source-recoverable, so gap(a) ≈ gap(b-final).
* **The Stage 0 pattern is specific to the from-scratch ResNet-101:** gap(b-final) is small or has an interval including 0.

## Primary contrast and frozen rules (v2; per fine-tuning seed the first matching row is that seed's row; thresholds and the dose condition are defined in the spec §5)
gap = Δ_T − Δ_S at 8k×1 (12-cell macro, pp); clean increment = S-8k×1 `q_ZP` − `q_Z` on clean held-out images. **Two-seed rule:** every row must hold for both seeds; if the seeds fire different rows the result is inconclusive.

| # | Outcome | Reading | Decision |
|---|---|---|---|
| 1 | Manipulation check 1 fails: clean increment(a) ≤ +0.3 pp, or its interval includes 0 | Regime (a) did not create a clean-complementary source from the same network | Inconclusive; report only |
| 2 | **Manipulation check 2 fails:** b10's S-8k×1 clean-view increment > +0.3 pp | Fine-tuning did not reach a clean-redundant state; rows 3 and 5 cannot be read | Inconclusive; report the gaps and the dose condition descriptively |
| 3 | gap(b-final) interval includes 0, or gap(b-final) < +1.0 pp | The Stage 0 pattern does not transfer to a fine-tuned pretrained model | Restrict the claim to from-scratch ResNet-101 |
| 4 | gap(a) ≥ 0.8 × gap(b-final) | `P` is clean-complementary yet still not source-recoverable | Kill "clean redundancy controls recoverability" |
| 5 | gap(a) ≤ 0.5 × gap(b-final), **and** the interval of gap(b-final) − gap(a) excludes 0, **and** across doses the gap does not fall as the clean increment falls | Supports clean redundancy as the controlling variable | Propose a confirmation and baselines package; do not start it |
| 6 | otherwise | Inconclusive | Report |

Dose condition (frozen): for every pair of states (i, j) of the same seed with clean increment(i) − clean increment(j) ≥ 0.3 pp, gap(j) ≥ gap(i) − 0.3 pp. Report, per state and per fine-tuning seed: clean increment, Δ_S, Δ_T and gap with intervals, every selected λ, and absolute accuracies (base `Z`, `q_Z`, `q_ZP`, T and S, macro and clean, NLL and Brier); compare gaps **within** a state, since base accuracy and robustness differ between (a) and (b). Practical scale: the +0.3 pp manipulation floor and +1.0 pp final-gap floor are chosen relative to the Stage 0 magnitudes (gap ≈ +2.9 to +3.4 pp, clean increment ≈ 0); interval and power limits: single training seed, image bootstrap only.

## Data and fitting access; exposure-ledger entry
Data: CIFAR-100 train (45k) and validation (5k) benchmark splits for backbone fine-tuning, head and probe fits; the 10,000 test images × 13 conditions for Stage 0 fits and evaluation, as in Stage 0. Target-label access: T regimes as in Stage 0; S regimes use clean labels.
**Proposed entry for the authoritative [[Representation Correction Exposure Ledger]] (not written there because that file carries a separate task's uncommitted edits; append it when the researcher approves):** *"CIFAR-100 test images (same 10,000 IDs), clean plus the 12 development cells; ImageNet-pretrained ResNet-50, seven model states (a; b1/b3/b10 × two fine-tuning seeds); T-regime labels fit target-supervised readouts, S regime clean labels; development reuse of exposed cells on a new model, not confirmation. Backbone fine-tuning, head and probe hyperparameters use only the CIFAR-100 train/validation splits. Not accessed: checkpoints 1/3/5, the 11 unused CIFAR-100-C families, any new test data."*

## Cost (measured, RTX 4090) and execution envelope
Measured in the smoke test: fine-tuning 1,715 img/s, fp32 extraction 1,695 img/s, peak 6.6 GB.

| item | GPU-hours |
|---|---|
| fine-tuning, 2 runs × 10 epochs (898,560 image passes) | 0.15 |
| feature and probe extraction, 7 states × (45k + 5k + 130k) images | 0.21 |
| head and probe fits incl. grid-edge extensions | ≈ 0.15 |
| subtotal | ≈ 0.5 |
| contingency ×2 | **≈ 1.0 GPU-h** |

Plus ≈ 15 CPU-hours of Stage 0 fits (140 runs). Well under the 24 GPU-hour bound. Submit script `atlas/regime_submit.py`. Engineering recovery allowed; scientific changes need an amendment; no reserved data; no push without approval.

## Pre-launch verification (smoke test; not results)
GPU smoke on a few hundred images (`results/regime_map/smoke/smoke.json`): weights hash asserted; fine-tune step and throughput; fp32 extraction; head and probe fits (600 train / 500 val / 300 test images, 3 conditions); artifact schema loads in the Stage 0 loader format. The Stage 0 fit path with `--regime-state` reproduced Stage 0's `q_ZP` exactly when fed a copy of Stage 0's arrays. Unit test for the rule table (`tests/test_stage0.py::test_regime_map_rules_follow_the_frozen_table`). Two GPU nodes (`cs-4090-09`, and `cs-4090-05` flagged unavailable) failed CUDA initialization or were unavailable and are excluded in the submit script. Smoke numbers are not used for any decision.

## Results
*(none yet)*

## Protocol deviations
None.

## Post-mortem and allocation
*(after the run)*

## Amendments / engineering recovery
* 2026-09-24: v1 specification and card frozen before any fit (commit `4557b84`, spec sha256 `b05119fe…defa96`).
* **2026-09-24 pre-launch amendment v1 → v2** (researcher-directed; recorded as an amendment, not a deviation; no fit of the pilot had been run): (1) second manipulation check (row 2: b10's clean increment > +0.3 pp → inconclusive); (2) second fine-tuning seed for b1/b3/b10, every row must hold for both seeds; (3) grid-edge rule for the head of (a) and every probe. Cost re-estimated ≈ 1 GPU-h. Spec: `docs/regime_map_pilot_spec_v2.md`.

## Provenance
* Frozen spec sha256: see `docs/regime_map_pilot_spec.frozen.sha256`.
* Freeze commit: `4557b84` (spec `docs/regime_map_pilot_spec.md`, sha256 `b05119feb28b9ffc835a8d2d0cbd7a3f3408baf01c7e23ce53efcb542cdefa96`, verified equal to the copy inside the snapshot).
* Immutable snapshot: `snapshots/regime_map_v1_d72302a1f54c` (tree hash `d72302a1f54c`, git head `4557b84bdd4374f60a61f770651fb9db6c6e37ef`). Note: the snapshot's tracked-diff hash includes another task's uncommitted non-code files (workflow, templates); no Stage 0 or pilot code file is uncommitted.
* Weights: `results/regime_map/weights/resnet50-0676ba61.pth`, sha256 `0676ba61…fb8a`.
* Smoke artifacts: `results/regime_map/smoke/smoke.json`; nothing else has been run.
