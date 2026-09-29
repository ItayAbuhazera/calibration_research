---
type: paper
status: read_html
year: 2026
venue: "arXiv 2607.26565 v2 (2026-07-30); not peer reviewed (appendix says AAAI format; no acceptance stated)"
url: https://arxiv.org/abs/2607.26565
tags: [trajectories, multi-layer, ood-detection, cifar-100-c, focal-paper, prior-art-audit]
---

# Representation Trajectories Matters: Complementary Evidence for OOD Detection and Image Classification

De la Jara, Rodriguez-Opazo, Damirchi, Gould, Ranasinghe. Read in full 2026-09-29 from the arXiv HTML (all sections, appendices A–P;
tables via text). Full claim map: `GeometricFullCalibration/docs/internal_computation_recoverability_territory_audit_2026-09-29.md` §3.

## Why this matters to me
Focal collision for the [[Internal-Evidence Recoverability and Selective Correction]] line: it owns "per-sample layer trajectory as
complementary evidence", clean and under shift.

## Core contribution
Trajectory τ(x) = (z_1 … z_L), one pooled vector per native block; updates u_l = z_{l+1} − z_l; class routes μ_l^c and residuals
(§3, Eq. 1–4). OOD: ID-only transition-surprise score fused with Mahalanobis++ at fixed weight 0.3 (§4.1). Recognition: linear probes on
updates, non-negative ensemble with a final-state probe and a final-state **margin gate** (§4.2). Backbones frozen.

## Benchmark / task
OpenOOD v1.5 (38 + 4 checkpoints); classification on 12 datasets × 6 backbones; CIFAR-100-C (clean-selected), PACS, Office-Home.

## Strongest result
FPR95 reduced in 131/152 non-saturated comparisons; clean recognition +4.41 pp mean, 71/72 cases; CIFAR-100-C 169/180 conditions,
control-adjusted +0.99 (CLIP) to +3.11 (ResNet-50) pp (Table O1).

## Surprising observation
No privileged order: "Consistent performance under fixed layer permutations further argues against privileged ordering" (§4.3); reverse
transition prediction is easier for five sequence-model families (App. L.4); full-prefix sequence models give no consistent gain (Fig. L2).

## Failure / limitation
Net accuracy only — no wins/harms; "final state" is a linear probe on frozen pretrained features, not a native head; no misclassification
detection experiment ("Confidence scores are better suited to identifying the model's own errors", §6); capacity-matched unordered
control named as required (App. F.2, Table F1) but not reported; causal claims disclaimed; App. I.1 "15 corruption families" vs Table O1
"six corruptions" inconsistency.

## What this changes in my beliefs
"Trajectory as evidence" and "order matters" are not available as our contributions. Any within-model claim must be about per-example
repair vs harm of the native head, or causal restoration.

## Related observations
[[Clean-trained depth probes in ResNet-101 are below the head at every depth in every tested condition]]; [[Improving LLM Final Representations with Inter-Layer Geometry]] (layer assignment irrelevant).

## Novelty relevance
PHENOMENON AND METHOD ESTABLISHED for trajectory-as-complementary-evidence; leaves W/H, native head, causal and matched-order controls open.

## Follow-up
None authorized. Same group: [[Vertical Fusion - Recoverability in ViT Hierarchies]]; Voyager (arXiv 2609.20299, OOD layer routing).

## Correction 2026-09-29 — scope of the order evidence (red-team `GeometricFullCalibration/docs/internal_computation_recoverability_redteam_2026-09-29.md` §3 E1)
"Consistent performance under fixed layer permutations" and "reverse prediction easier" both belong to the §4.3 / App. L.4
transition-continuity validation, not to recognition or OOD utility. App. F.2 says the permutation test "cannot establish that an ordered
predictor beats the same states given to a capacity-matched set encoder". The kill of "ordered trajectory" now rests on the fact that an
indexed set and an ordered sequence carry the same information, not on these results.
