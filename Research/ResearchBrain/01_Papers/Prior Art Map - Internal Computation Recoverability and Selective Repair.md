---
type: paper_map
status: active
date: 2026-09-29
tags: [prior-art-audit, recoverability, overthinking, selective-repair, causal, graph, early-exit]
---

# Prior Art Map — Internal Computation Recoverability and Selective Repair

Compact index for [[Internal-Evidence Recoverability and Selective Correction]]. Full tables with evidence locations, peer-review status
and collision matrix: `GeometricFullCalibration/docs/internal_computation_recoverability_territory_audit_2026-09-29.md` §4–5, §18
(cutoff 2026-09-29; ≈ 85 primary sources). PR = peer-reviewed; AX = arXiv only.

## Trajectory / multi-layer evidence
- [[Representation Trajectories Matters]] (AX 2026) — per-sample trajectories complement final state; order not privileged.
- [[Improving LLM Final Representations with Inter-Layer Geometry]] (AX/NeurIPS 2026 claimed) — layer assignment irrelevant.
- [[Intermediate Layer Classifiers for OOD generalization]] (PR ICLR 2025) — dataset-level intermediate heads under shift.

## Early-correct → final-wrong (T1)
- Shallow-Deep Networks, Kaya et al. (PR ICML 2019) — destructive overthinking in ≈50 % of errors, https://arxiv.org/abs/1810.07052.
- [[Understanding the Robustness of Multi-Exit Models under Common Corruptions]] (AX 2022) — ≈10-pt oracle vs ≈1-pt realistic under CIFAR-C.
- [[Vertical Fusion - Recoverability in ViT Hierarchies]] (AX 2026) — "recoverability", 18–76 %, incl. CIFAR-100-C.
- CALRD, "MLLMs Get It Right, Then Get It Wrong" (PR IJCAI 2026), https://arxiv.org/abs/2606.17953 — direction signature separates failures from successes; gated restoration.
- Wrong Before Right (AX 2026), https://arxiv.org/abs/2607.04640 — transient wrong dips in correct answers (base-rate warning).
- LogitDynamics (PR workshop CVPR 2026), https://arxiv.org/abs/2604.10643 — layerwise logit trajectories for error detection.

## Recoverability / selective correction (T2, T3)
- [[Self-Checking Deep Neural Networks in Deployment]] (PR ICSE 2021) — internal alternative prediction; reported to hurt on CIFAR-100.
- Orgad et al., LLMs Know More Than They Show (PR ICLR 2025), https://arxiv.org/abs/2410.02707.
- KAPPA (PR ICML 2026), https://arxiv.org/abs/2509.23782 — knowledge–prediction gap + residual-stream intervention.
- [[To Adapt or Not to Adapt - Selective Adaptation for VLMs]] (PR ECCV 2026) — prospective apply/skip with harm, natural shift.
- ALTAS "Route, Don't Fix" (AX 2026), https://arxiv.org/abs/2609.14825 — trajectory-gated correction; gate is the bottleneck.
- Hidden Error Awareness: Diagnostic, Not Causal (AX 2026), https://arxiv.org/abs/2605.09502.
- Repair metrics: PRDNN drawdown (PLDI 2021), Editable NN (ICLR 2020), Arachne repair/break rate (TOSEM 2023), PC-training NFR (CVPR 2021).
- Failure detection is crowded: whitebox meta-models (AISTATS 2019), FD-Shifts (ICLR 2023), Trust Score, ConfidNet, DOCTOR.

## Causal / interventional (T4)
- DeepCorrect (PR IEEE TIP 2019), https://arxiv.org/abs/1705.02406 — clean-activation correction under distortion.
- Surgical Fine-Tuning (PR ICLR 2023), https://arxiv.org/abs/2210.11466 — corruption repair best in early layers.
- BN adaptation / TENT / DUA — label-free internal interventions under CIFAR-C (aggregate).
- Self-repair / Hydra effect — downstream compensation confound.

## Graph-structured computation (T5)
- DeepProv (PR ACSAC 2025), https://arxiv.org/abs/2509.26562; NeuroTrace (AX 2026), https://arxiv.org/abs/2604.14457.
- Topological Uncertainty (PR IJCAI 2021), https://arxiv.org/abs/2105.04404 — per-input edge-contribution graphs.
- CRV (PR ICLR 2026), https://arxiv.org/abs/2510.09312 — topology features matter least.

## Sequential decision / early exit (RL audit)
- PABEE (PR NeurIPS 2020); CALM (PR NeurIPS 2022); Fast yet Safe risk-controlled exits (PR NeurIPS 2024, i.i.d.; shift = future work).
- Learning to Stop While Learning to Predict (PR ICML 2020) — stopping reduced to supervised imitation.
- BlockDrop (PR CVPR 2018) — its RL is "a single-step MDP … contextual bandit"; SkipNet; Runtime Neural Pruning.

## What this map does NOT establish
Paper-level claims are as read (lead-verified abstracts or subagent-read bodies, marked in the report); no claim about our models.

## Addendum 2026-09-29 — red-team (`GeometricFullCalibration/docs/internal_computation_recoverability_redteam_2026-09-29.md`)
Lead full-text verification of all 13 load-bearing sources. Qualifications: Vertical Fusion recovery rates are clean-only and oracle
(E2); Representation Trajectories order evidence is continuity-only (E1); Selective Adaptation pools negligible + harmful and targets
efficiency (E3); DeepCorrect clean replacement is a filter-ranking definition (E4); Fast yet Safe bounds early-exit degradation, not
repair (E5); KAPPA "mitigates" (E6). Corruption Depth (Neural Networks 2024): full text inaccessible; abstract read → SAME PHENOMENON
(T1). New: [[Causality is not Decodability - Counting ViTs]] (moves T4 → KILL); [[When is Test-Time Adaptation Identifiable From Unlabeled Evidence]]
(adjacent; POC-T3 interpretation); LOES (ICML 2026), CODEC (ICLR 2026), Policy Gradient Steering, BEEM (ICLR 2025): adjacent, no verdict change.
