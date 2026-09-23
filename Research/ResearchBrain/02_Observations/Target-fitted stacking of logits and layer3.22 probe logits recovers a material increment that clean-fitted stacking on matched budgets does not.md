---
type: observation
status: open
date: 2026-09-23
project: Full-Vector Geometric Calibration
evidence_strength: 3
tags: [stage0, target-supervised, recoverability, layer3.22, probe-logits, development-only]
---

# Target-fitted stacking of logits and layer3.22 probe logits recovers a material increment that clean-fitted stacking on matched budgets does not

Scope: ResNet-101 (CIFAR-100), checkpoints 2 and 4 (two checkpoints; no cross-checkpoint robustness claim), the 12 exposed CIFAR-100-C development cells, the fixed `layer3.22` GAP linear-probe logits, and a linear multinomial readout fitted with target-supervised (corrupted-cell) or clean labels on the same 10,000 test images (grouped, out-of-fold). Development evidence, not confirmation. Source: [[2026-09-22 Stage 0 Probe-Logit Increment Study]].

## Observation

1. **Target-supervised diagnostic.** A target-fitted linear readout of base logits plus probe logits beats a target-fitted linear readout of the base logits alone by +4.23 pp (checkpoint 2, 95% interval [3.99, 4.47]) and +3.98 pp (checkpoint 4, [3.75, 4.22]) in 12-cell macro accuracy (T-8k×12). Shuffled-probe control, convergence and λ-edge audits were clean.
2. **Matched clean-fitted stacking does not.** At 8,000 images × 1 view the target-fitted increment is +2.87 / +2.63 pp while the clean-fitted increment, scored on the same corrupted rows, is −0.48 / −0.29 pp (gap +3.35 / +2.91 pp). At 2,500 images the values are +1.35 / +1.20 against −0.14 / +0.03.
3. **More data helps.** 8k vs 2.5k images: about +2.1 pp (12 views) and +1.4 to +1.5 pp (1 view); 12 vs 1 views: +1.35 pp (8k) and +0.7 pp (2.5k).
4. **The gain is localized.** Largest in Gaussian noise (up to +11 pp) and severe blur/fog; about zero at severity-1 blur and fog.
5. **The probe alone is not better than the logits.** The clean-trained `layer3.22` probe is below base in every condition (−3.0 to −8.7 pp), and no layer met the descriptive 0b flag.
6. **Recalibration explains a large part.** Post-hoc, target recalibration of the logits alone accounts for roughly 42–44 % of the total gain over base (12-cell macro accuracy 50.1 → 53.2 → 57.4 for checkpoint 2; 51.0 → 54.2 → 58.2 for checkpoint 4).
7. **A scalar blend is not enough (post-hoc).** `softmax(α z + β p + b)` recovers 26–29 % of Δ_T with target labels and 8–11 % with clean labels.

## Evidence

* Frozen spec and deviations: repo `docs/stage0_execution_spec.md` (§7); memo Claude Doc <https://claude.ai/code/artifact/ebf21486-1669-4afb-9748-f624a976e2dd>.
* `results/stage0/report/stage0c_aggregate.json` (sha256 `318453572e1d…4a350a359`, byte-identical from `snapshots/stage0_v3_ec359efbe2cc`), `stage0c_requested_tables.json`, `stage0_posthoc_accuracy_ladder.{json,md}`, `stage0_posthoc_alpha_beta.json`, `stage0a_rank_tables.json`, `stage0b_*.json`, `aggregate_provenance.json`; fits from `snapshots/stage0_v1_d0dcfd61aa88`.
* The intervals are 2,000 paired bootstrap resamples over image groups, conditional on the fitted cross-validation predictions; they exclude training-sample uncertainty and historical selection.

## What it does NOT establish

* **Not a deployable or clean-only method.** Target labels of the exposed cells were used to fit; nothing fitted here may enter a clean-only method, layer choice or threshold.
* **Not an information ceiling, not "representations know what logits miss."** It is one restricted linear readout at these budgets. A null or small result would not have bounded information either.
* **Not a cause of the recoverability gap.** Non-identifiability from clean data, clean-fit regularization, a shift in the distribution of the probe logits relative to the logits, and readout mismatch all remain open; selected λ was the same for clean and target fits at a given budget, which argues against only a simple λ story.
* **Not specific to `layer3.22` or to geometry.** No other layer, other second predictor, or nonlinear readout was tested; ordinary target fusion of any comparably informative predictor is not excluded. The shuffled-probe control rules out capacity/leakage artifacts, not this.
* **Not evidence about detection, abstention or kNN candidates** (Stage 0d not started); no relation to the closed fixed-gate result beyond continuity of the layer.
* **Not confirmation.** Same 10,000 images, exposed cells, two checkpoints; checkpoints 1/3/5 and the 11 other families were not accessed by Stage 0 and are not called pristine ([[Representation Correction Exposure Ledger]]).
* One seed-4 fold file was rewritten by an agent-spawned rerun (card deviation 6); its effect is confined to one S-2.5k×1 fold and is unverified.

## Possible mechanisms

Listed, not tested: probe evidence redundant with logits on clean data but informative under corruption (noise, severe blur/fog), so clean data cannot identify how to combine them; shift in the probe-logit distribution; clean-fit regularization; ordinary target recalibration plus classifier fusion. See [[H-STAGE0-01 Layer3.22 probe evidence is redundant with the logits on clean data but complementary under corruption, and clean supervision cannot identify the useful combination]].

## Related failure modes

[[Serial method tests without discriminating outcomes]] (workflow context); [[Clean-fitted geometric confidence mappings can reverse under synthetic corruption]] (clean-fit versus corruption behaviour, different object).

## Related hypotheses

[[H-STAGE0-01 Layer3.22 probe evidence is redundant with the logits on clean data but complementary under corruption, and clean supervision cannot identify the useful combination]]; closed related work [[H-RESID-01 Source-learnable residual decision information at layer3.22]], [[H-LAYER-01 Selected internal layers add decision value beyond logits]], [[H-GATE-01 Candidate selection versus gate utility mismatch]].

## Decisive next test

None is authorized. Any follow-up needs its own frozen card, a stated mechanism, and (for confirmation) explicit authorization to consume a reserved resource. Cheap discriminating candidates are listed in the hypothesis card.
