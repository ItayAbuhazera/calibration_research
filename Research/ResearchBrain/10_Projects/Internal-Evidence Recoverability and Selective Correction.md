---
type: project
status: audited
date: 2026-09-29
project: Full-Vector Geometric Calibration
tags: [research-line, recoverability, selective-repair, trajectories, territory-audit, prior-art-audit]
---

# Internal-Evidence Recoverability and Selective Correction (research line; audit stage only)

Opened 2026-09-29 by an adversarial territory audit. **No experiment has been authorized or run in this line.** Durable report (source of
truth for every verdict below): `GeometricFullCalibration/docs/internal_computation_recoverability_territory_audit_2026-09-29.md`.

## Question

When a single end-to-end classifier makes a final error, what happened to evidence for the correct class during its computation, and can
internal evidence support a *safe* correction (repair without harm)? Our substrate: CIFAR-100 ResNet-101 checkpoints 2 and 4, exposed
CIFAR-100-C development cells.

## Program status (vocabulary as elsewhere in the vault)

| Item | Status |
|---|---|
| G1-DP | completed — **INCONCLUSIVE (validity)**, unchanged ([[2026-09-28 G1 Conditional Accessibility Gatekeeper]] lineage; report `docs/g1_decoder_panel_audit_2026-09-28.md`) |
| N1a | completed — INCONCLUSIVE (insufficient precision) ([[2026-09-28 N1a Action Ambiguity Audit]]) |
| **N1a-DP** | **PAUSED** — draft r4, five families, not frozen, not authorized, not submitted (`docs/n1a_dp_spec.md`, `docs/n1a_dp_prefreeze_methodology_audit_2026-09-29.md`); no experiment card exists (none may be created until authorized) |
| N1b | not started |
| This line | draft_for_discussion — audit only |

## Audit verdicts (2026-09-29)

| Candidate | Verdict | Strongest collision |
|---|---|---|
| T1 class-evidence evolution | WATCH | [[Shallow-Deep Networks]]-era overthinking; [[Understanding the Robustness of Multi-Exit Models under Common Corruptions]]; [[Vertical Fusion - Recoverability in ViT Hierarchies]]; CALRD (MLLMs) |
| T2 recoverability (vs detection) | KILL — [[Recoverability of final errors as a standalone new concept]] | Vertical Fusion; [[Self-Checking Deep Neural Networks in Deployment]]; Orgad et al.; KAPPA |
| T3 selective internal repair | WATCH — [[H-SELREP-01 Multi-depth internal evidence discriminates repair from harm beyond output evidence under held-out corruption]] | [[To Adapt or Not to Adapt - Selective Adaptation for VLMs]]; SelfChecker; ALTAS; our [[H-GATE-01 Candidate selection versus gate utility mismatch]] |
| T4 causal recoverability | WATCH (low) | DeepCorrect; surgical fine-tuning; BN-adapt/TENT; KAPPA (LLM) |
| T5 graph-structured computation | KILL — [[Graph-structured computation as a distinct object for a fixed-topology ResNet]] | DeepProv; NeuroTrace; Topological Uncertainty; CRV ablation |
| RL / DRL over depth | CURRENTLY UNJUSTIFIED — [[RL over depth for fixed-network prediction correction]] | Chen et al. ICML 2020; BlockDrop; EENet/CALM oracle-label training |
| **TEST NOW** | **none** | — |

Prior-art map: [[Prior Art Map - Internal Computation Recoverability and Selective Repair]]. Focal paper: [[Representation Trajectories Matters]].

## Evidence constraints from our own program (existing results, not new)

- [[Clean-trained depth probes in ResNet-101 are below the head at every depth in every tested condition]] (Stage-0b artifacts).
- Atlas: deep-layer3 candidates right on ≈13 % of base errors, wrong on ≈40 % of base-correct rows ([[2026-09-21 Representation Atlas Program]]).
- Fixed-gate study: clean-trained gate over a deep candidate, no consistent held-out benefit; H:W ≈ 3.8:1 ([[2026-09-21 Fixed Deep Candidate Gate Study]]).
- Failure mode: [[Oracle any-layer recoverability counts overstate correction headroom]].

## Research tree

Parent question → (T1 descriptive) → (T3 decision question) → (T4 causal). Only the T3 closure test (POC-T3, report §15: read-only,
leave-one-family-out, internal vs output-only selector, W/H) is a candidate next step, and only if the researcher decides it jointly with
N1a-DP. Otherwise this line is recorded as audited and closed. Hypothesis cards: [[H-SELREP-01 Multi-depth internal evidence discriminates repair from harm beyond output evidence under held-out corruption]], [[H-EVO-01 True-class evidence suppression is distinguishable from base-rate depth flips in ResNet-101 under corruption]].

## What this note does NOT establish

No new empirical result; no novelty for any candidate; no authorization for POC-T1/T3/T4, N1a-DP or N1b.

## Correction 2026-09-29 — red-team of the audit (verdicts revised; text above preserved as the original audit record)

Source: `GeometricFullCalibration/docs/internal_computation_recoverability_redteam_2026-09-29.md` (all 13 load-bearing primary sources re-read in full text by the lead; no experiment; no data opened). Status changed
`draft_for_discussion` → `audited`: **territory CLOSED as a next-stage candidate**; T3 kept only as a dormant WATCH item.

| Candidate | Original | Revised |
|---|---|---|
| T1 class-evidence evolution | WATCH | **KILL** (standalone) — [[Class-evidence suppression through depth as a standalone problem in an end-to-end CNN]] |
| T2 recoverability | KILL | KILL (evidence re-based: SelfChecker / Orgad / KAPPA; Vertical Fusion's recovery rates are oracle and clean-only) |
| T3 selective internal repair | WATCH | **WATCH (dormant)** |
| T4 causal recoverability | WATCH (low) | **KILL** — [[Non-oracle causal recoverability via internal edits in a fixed classifier]] |
| T5 graph / ordered trajectory | KILL | KILL (ordered-trajectory reason corrected: indexed set ≡ ordered sequence) |
| RL / DRL | CURRENTLY UNJUSTIFIED | unchanged |
| **POC-T3** | cheap closure test | **DO NOT RUN** |

Why POC-T3 is not run: the fixed-gate study already ran its central contrast. The internal-evidence gate vs output-evidence gate on the
same deep candidate gave Z1 − Z0 = −0.038 [−0.086, 0.012] / −0.037 [−0.127, 0.054] pp (corruption macro-12), and C0 ≈ Z1
([[2026-09-21 Fixed Deep Candidate Gate Study]], 2026-09-22 correction). The candidate band is outcome-informed. The spec lacks
output + candidate-confidence and layer4.2-only controls. No plausible outcome changes allocation. The earlier line "no consistent held-out
benefit" is corrected: the pipeline gain over base is small and positive (+0.075 / +0.213 pp, intervals exclude 0); what is null is the
internal-over-output increment. Reopen triggers are recorded on [[H-SELREP-01 Multi-depth internal evidence discriminates repair from harm beyond output evidence under held-out corruption]].
New prior art: [[Causality is not Decodability - Counting ViTs]], [[When is Test-Time Adaptation Identifiable From Unlabeled Evidence]].
N1a-DP remains **PAUSED** (not edited). G1-DP remains INCONCLUSIVE (validity).
