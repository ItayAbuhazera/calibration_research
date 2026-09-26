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

## Results (2026-09-24; artifacts `results/regime_map/report/regime_aggregate.json`, `results/regime_map/<state>/summary.json`; fits from snapshot `regime_map_v2_c222e12d10a2`; the spec file is unchanged, so its sidecar hash still verifies)

**Validation before reading contrasts.** 150 of 150 jobs completed; 140 fold files (7 states × 4 regimes × 5 folds); 280 arm-fits, 0 unconverged, 0 retried, 0 at a Stage 0 λ-grid edge. Fine-tuning reached val accuracy 0.852 / 0.854 at epoch 10 (seeds 1/2; train 0.991 / 0.992). Every probe selected λ = 1e-3 and the head of (a) λ = 1e-2, all interior: **the grid-edge rule triggered no extension** (0 extensions in any state). Stage 0 fitter λ: 1e-3 in ≥ 36 of 40 fits per state, otherwise 1e-2 or 1e-4.

**Frozen v2 verdict: row 4 fires in both fine-tuning seeds → "kill 'clean redundancy controls recoverability'" (P clean-complementary yet not source-recoverable).** Manipulation check 1 passed (clean increment(a) +3.82 pp [3.20, 4.43]); manipulation check 2 passed (b10 clean increment −0.20 and −0.74 pp, not > +0.3); row 3 not met (gap(b10) +1.53 [1.26, 1.78] and +2.04 [1.79, 2.28], both > +1.0 with intervals excluding 0); row 4 met because gap(a) = +2.77 ≥ 0.8 × gap(b10) in both seeds. gap(b10) − gap(a) = −1.24 [−1.63, −0.86] (seed 1) and −0.73 [−1.12, −0.33] (seed 2): the gap is **larger** in (a), not smaller. The dose condition is violated in both seeds (pairs (a, b1), (a, b3), (a, b10) in seed 1; (a, b1), (a, b10), (b3, b10) in seed 2); it would only matter for row 5.

**Per state, 12-cell macro (pp, 95% image-bootstrap intervals; seed 1 = fine-tuning seed 20260924, seed 2 = 20260925; (a) is one fit).**

| state | clean increment (S-8k×1) | Δ_S (8k×1) | Δ_T (8k×1) | gap = Δ_T − Δ_S (8k×1) | gap (2.5k×1) |
|---|---|---|---|---|---|
| a | +3.82 [+3.20, +4.43] | +2.02 [+1.73, +2.31] | +4.79 [+4.47, +5.13] | +2.77 [+2.43, +3.09] | +2.24 [+1.90, +2.63] |
| b1_s1 | +1.65 [+1.06, +2.26] | +1.47 [+1.23, +1.74] | +3.20 [+2.91, +3.49] | +1.73 [+1.42, +2.04] | +0.89 [+0.52, +1.24] |
| b3_s1 | -0.16 [-0.70, +0.43] | +0.17 [-0.06, +0.40] | +1.81 [+1.55, +2.08] | +1.63 [+1.36, +1.92] | +0.91 [+0.57, +1.23] |
| b10_s1 | -0.20 [-0.66, +0.24] | -0.72 [-0.90, -0.54] | +0.81 [+0.58, +1.04] | +1.53 [+1.26, +1.78] | +0.69 [+0.40, +0.98] |
| b1_s2 | +1.05 [+0.48, +1.66] | +0.24 [-0.01, +0.50] | +2.45 [+2.15, +2.73] | +2.21 [+1.88, +2.51] | +1.27 [+0.91, +1.61] |
| b3_s2 | -0.21 [-0.74, +0.32] | -0.51 [-0.74, -0.29] | +2.02 [+1.76, +2.29] | +2.54 [+2.27, +2.82] | +1.12 [+0.79, +1.45] |
| b10_s2 | -0.74 [-1.19, -0.31] | -1.01 [-1.19, -0.84] | +1.02 [+0.80, +1.25] | +2.04 [+1.79, +2.28] | +1.02 [+0.74, +1.31] |

**Absolute accuracies (%; macro 12 cells / clean).**

| state | `Z` | `P` alone | disagreement P vs Z (macro) | T-8k×1 `q_Z` → `q_ZP` (macro) | S-8k×1 `q_Z` → `q_ZP` (macro) | S-8k×1 clean `q_Z` → `q_ZP` |
|---|---|---|---|---|---|---|
| a | 48.8 / 72.4 | 46.3 / 72.4 | 0.491 | 49.4 → 54.2 | 45.4 → 47.4 | 68.1 → 72.0 |
| b1_s1 | 39.6 / 65.4 | 43.9 / 73.0 | 0.551 | 51.4 → 54.6 | 45.0 → 46.4 | 73.5 → 75.2 |
| b3_s1 | 47.0 / 77.6 | 44.5 / 75.7 | 0.482 | 55.3 → 57.1 | 48.3 → 48.5 | 79.2 → 79.0 |
| b10_s1 | 53.7 / 84.2 | 45.8 / 77.0 | 0.436 | 58.8 → 59.6 | 52.5 → 51.8 | 83.0 → 82.8 |
| b1_s2 | 43.5 / 69.0 | 45.0 / 74.2 | 0.525 | 53.1 → 55.5 | 46.3 → 46.5 | 74.0 → 75.0 |
| b3_s2 | 45.0 / 74.9 | 44.5 / 75.5 | 0.490 | 55.2 → 57.3 | 48.2 → 47.7 | 78.6 → 78.4 |
| b10_s2 | 53.3 / 84.6 | 45.5 / 77.7 | 0.445 | 59.1 → 60.2 | 52.1 → 51.1 | 83.6 → 82.9 |

Selected λ (all interior; see `summary.json`): probes 1e-3 in all seven states; head of (a) 1e-2.

## Interpretation (labelled; not established)
* As defined in the frozen rules, clean redundancy did **not** control source-recoverability here: the state whose `P` is most clean-complementary (a: clean increment +3.8 pp, Δ_S +2.0 pp) still has the largest gap (+2.8 pp), and along the doses the clean increment falls (+3.8 → about +1.0/+1.7 → about −0.2 → −0.2/−0.7) while the gap does not rise (about +2.8 → 1.7/2.2 → 1.6/2.5 → 1.5/2.0).
* Fine-tuning shrinks the target-fit gain too (Δ_T +4.8 → +0.8/+1.0) and turns Δ_S negative; the gap is an absolute difference bounded by the headroom, so a smaller Δ_T in the fine-tuned states also pushes the gap down (post-hoc reading; the frozen rule is on the absolute gap).
* The Stage 0 recoverability gap is **not** confined to the from-scratch ResNet-101: it is present in the pretrained ResNet-50 in every state (gap +1.5 to +2.8 pp at 8k×1, intervals excluding 0), including the frozen backbone.
* Still open: which of clean non-identifiability, source-fit regularization, shift of `P` under corruption, or readout mismatch produces the gap; the earlier explanation that "same network" versus "clean-redundant" separates the from-scratch sources is not supported by this manipulation in this design.

## Not established
One backbone family and resolution policy (224 upsampling), one probe layer, two fine-tuning seeds (image-bootstrap intervals only; no training-seed variance beyond the two-seed comparison), development cells reused; nothing about a deployable method, confirmation, other layers or architectures. Row 4 kills only the specific claim "clean redundancy, as manipulated here, controls recoverability of this `P`".

## Protocol deviations
None. The grid-edge rule was never triggered.

## Post-mortem and allocation
* Primary contrast: gap(b10) − gap(a) < 0 with intervals excluding 0 in both seeds; the frozen kill row fires.
* Explanations: "clean redundancy controls recoverability" made less plausible for this evidence and readout; the four remaining candidates for the recoverability gap are not separated.
* Reasoning-chain stage tested: transfer/identifiability of a fixed evidence source under a restricted linear readout, on a pretrained backbone with manipulated clean redundancy.
* Claim supported / not supported: the recoverability gap is not tied to the from-scratch ResNet-101 nor to clean redundancy as manipulated here; not supported: any mechanism.
* Allocation: none started. Vault notes for the hypothesis, Current Evidence and the exposure ledger have **not** been updated with this result (awaiting the researcher's go). Nothing pushed since `fb2e7bd`.
* Review status: single-author agent-assisted; the rule outcome was computed by the pre-frozen, unit-tested code.

## Amendments / engineering recovery
* 2026-09-24: v1 specification and card frozen before any fit (commit `4557b84`, spec sha256 `b05119fe…defa96`).
* **2026-09-24 pre-launch amendment v1 → v2** (researcher-directed; recorded as an amendment, not a deviation; no fit of the pilot had been run): (1) second manipulation check (row 2: b10's clean increment > +0.3 pp → inconclusive); (2) second fine-tuning seed for b1/b3/b10, every row must hold for both seeds; (3) grid-edge rule for the head of (a) and every probe. Cost re-estimated ≈ 1 GPU-h. Spec: `docs/regime_map_pilot_spec_v2.md`.

## Provenance
* Frozen spec sha256: see `docs/regime_map_pilot_spec.frozen.sha256`.
* Freeze commit: `4557b84` (spec `docs/regime_map_pilot_spec.md`, sha256 `b05119feb28b9ffc835a8d2d0cbd7a3f3408baf01c7e23ce53efcb542cdefa96`, verified equal to the copy inside the snapshot).
* Immutable snapshot: `snapshots/regime_map_v1_d72302a1f54c` (tree hash `d72302a1f54c`, git head `4557b84bdd4374f60a61f770651fb9db6c6e37ef`). Note: the snapshot's tracked-diff hash includes another task's uncommitted non-code files (workflow, templates); no Stage 0 or pilot code file is uncommitted.
* Weights: `results/regime_map/weights/resnet50-0676ba61.pth`, sha256 `0676ba61…fb8a`.
* Smoke artifacts: `results/regime_map/smoke/smoke.json`.
* **Spec v2:** `docs/regime_map_pilot_spec_v2.md`, sha256 `30347cae04f54e34c5ff8e4dc72ec09ae54c0fabd47451093c2590fd6741996f`; committed in `27621e7` before any fit; snapshot `snapshots/regime_map_v2_c222e12d10a2` (tree hash `c222e12d10a2`, git head `27621e717be939355bf0f3fa216a0fd068246883`; spec copy verified equal). The v1 snapshot `regime_map_v1_d72302a1f54c` is superseded and was not used for any fit.
* **Launched 2026-09-24** from the v2 snapshot: jobs 21657923–21657960 (`results/regime_map/ledger.json`); GPU: state (a) extraction, two fine-tuning runs, six state extractions; CPU: 28 fit arrays (5 folds each); aggregation 21657960.
* **Ledger:** the Codex workflow changes (including the authoritative exposure ledger) are still uncommitted, so the exposure entries are kept in this card and in the ablation card and were not appended to the ledger.

## Exposure record (appended 2026-09-26; the authoritative ledger is still blocked by another task's uncommitted edits, so this is the entry of record)
* **Models evaluated:** ImageNet-pretrained torchvision ResNet-50 `IMAGENET1K_V1` (weights sha256 `0676ba61…fb8a`): the frozen state (a) and the fine-tuned states b1/b3/b10 of **fine-tuning seeds 20260924 and 20260925**. **No CIFAR-100 ResNet-101 checkpoint (1–5, including 2 and 4) was used by Phase 2**; the earlier Stage 0 arrays were not read.
* **Splits:** training (backbone fine-tuning, linear head of (a), layer3 probes, λ selection): the benchmark 45,000-image train split and 5,000-image validation split (`results/studyAB/phase0_corrected_v2/evaluation/checkpoint_seed2/clean/intermediates/splits`, the seed-2 split). Evaluation and target/source readout fits: the **10,000 CIFAR-100 test images**, read from `results/atlas/shared/test_sets_full.npy`, in 5 outer folds (Stage 0 fold plan, seed 20260922).
* **Conditions read:** exactly the 13 in `spec.CONDITIONS`: clean plus **gaussian_noise, defocus_blur, fog, jpeg_compression × severities 1, 3, 5**. T fits used the labels of the assigned corrupted view (one per image); S fits used clean labels.
* **Corruption families outside {gaussian_noise, defocus_blur, fog, jpeg_compression}: NOT exposed by Phase 2.** Verified from code and the array: the Phase 2 and Stage 0 fit code contain no path to the raw CIFAR-100-C files, and `test_sets_full.npy` has shape (130000, 3, 32, 32) = 13 × 10,000, i.e. only those conditions exist in the array read. The 11 other families remain as recorded in the ledger. Checkpoints 1/3/5 were not accessed.
* Status: development reuse of the exposed cells and images on a new model family; not confirmation.

## Correction 2026-09-26 — increments are relative to the 8k refit, not to the model's own output (appended; nothing above rewritten)
The clean increment, Δ_S, Δ_T and gap reported above are all differences between two **8k-row refits** (Z+H minus Z-only, same protocol). The Z-only refit is itself not the model's output: its accuracy differs from the base head by a state-dependent amount. Per-state (refit − base) and (Z+H − base), in pp, **clean / macro-12**, read from `results/regime_map/report/regime_aggregate.json` (`states.<s>.standalone.acc_Z_clean`, `acc_Z_macro12`; `states.<s>.regimes.<S-8k1|T-8k1>.q_Z_clean_acc`, `q_Z_macro12_acc`, `q_ZP_clean_acc`, `q_ZP_macro12_acc`):

| state | base Z (clean / macro-12, %) | S-8k×1 Z-only refit − base | S-8k×1 Z+H − base | T-8k×1 Z-only refit − base | T-8k×1 Z+H − base |
|---|---|---|---|---|---|
| a | 72.35 / 48.76 | -4.22 / -3.38 | -0.40 / -1.36 | -7.47 / +0.62 | -4.08 / +5.41 |
| b1_s1 | 65.36 / 39.56 | +8.18 / +5.40 | +9.83 / +6.86 | +5.85 / +11.88 | +6.66 / +15.07 |
| b3_s1 | 77.60 / 46.99 | +1.56 / +1.29 | +1.40 / +1.46 | -0.79 / +8.34 | -1.26 / +10.15 |
| b10_s1 | 84.25 / 53.68 | -1.29 / -1.15 | -1.49 / -1.86 | -3.29 / +5.15 | -4.90 / +5.96 |
| b1_s2 | 68.97 / 43.47 | +5.03 / +2.79 | +6.08 / +3.03 | +2.91 / +9.58 | +3.30 / +12.03 |
| b3_s2 | 74.92 / 45.04 | +3.67 / +3.15 | +3.46 / +2.64 | +1.35 / +10.20 | +0.84 / +12.23 |
| b10_s2 | 84.58 / 53.30 | -0.96 / -1.16 | -1.70 / -2.18 | -3.38 / +5.84 | -4.98 / +6.86 |

* **The row-4 verdict is unchanged** (it is a statement about the frozen rule applied to the reported differences).
* **The statement that the frozen backbone has "+3.8 pp clean complementary information" is not supported relative to the base output**: the +3.82 pp is Z+H minus the S-8k×1 Z-only refit, and that refit is below the base head on clean (see the table); relative to the base output the source-fitted Z+H is below it on clean for state (a).
* Nothing else in this note is reinterpreted by this correction.

