---
type: project
status: active
date: 2026-09-22
project: Full-Vector Geometric Calibration
tags: [synthesis, current-evidence, representation-correction, canonical]
---

# Current evidence and open questions — representation-based correction program

**Canonical note.** This is the single place to read "what do we actually know, right now" about using hidden-representation
geometry (kNN readouts, residual readouts, full-vector density) to correct a frozen classifier's decisions under CIFAR-100-C
corruption. It links the experiment cards and hypothesis cards below rather than copying their reports; update **this** note
when a new result changes the picture, and append (never rewrite) the historical cards it links. It does not reopen any
stopped experiment, and it is not authorization for a new one.

Related but out of scope for this note: the RGC / geometric-separation line ([[Uncertainty Estimation Based on Geometric Separation]],
[[Semantic Geometric Calibration in Randomized Neural Feature Space]]) and its own reliability-routing question, tracked separately
in [[Conceptual Prior Art - Local Competence and Reliability Routing]] and the Dashboard.

## Claim matrix

| Claim | Evidence | Supported scope | Unsupported extension | Artifact | Status |
|---|---|---|---|---|---|
| A class-conditioned DAC-style density operator on the full representation vector can correct decisions under corruption | [[2026-09-20 Full-Vector DAC POC]] | Seed 4 only; a real, non-null, label-aligned effect exists (β=6.43 vs. permuted-label β=0) | A practically useful correction (effect ≈10× below the pre-declared bar; intervention precision 0.186; shrinks with severity) | `docs/full_vector_dac_experiment.md`, [[H-FVDAC-01 Class-conditioned DAC density enables decision correction]] | closed negative (4/5 criteria; rule needs 5) |
| CIFAR-100 clean/CIFAR-C used mismatched normalization stats (ImageNet vs. CIFAR) | [[2026-09-21 Normalization Audit and Corrected Protocol]] | Confirmed and fixed (`corrected_v2_train_norm`); corrected baselines exist for checkpoints 2, 4 | Whether IJCAI/AAAI-era published CIFAR-100 numbers share the same mismatch | `utils/preprocessing_protocol.py`, `docs/normalization_audit.md` | fixed; upstream provenance still unresolved |
| A clean-selected small set (1/4/6/8) of internal layers, pooled and linearly combined, adds decision value beyond logits | [[2026-09-21 Layer-Selection Pilot]] | Seeds 2, 4; frozen greedy selection over 12 candidate blocks | A materially useful correction (best +0.095 pp vs. Vector Scaling +0.190 pp) | `docs/layer_selection_pilot_spec.md` | not supported ([[H-LAYER-01 Selected internal layers add decision value beyond logits]]) |
| A clean-fit 100-d residual readout of `layer3.22` (spatial or class-radius) adds decision value beyond a full-logit readout | [[2026-09-21 Residual Evidence Study]] | Seeds 2, 4; frozen 6-criteria gate, 2 500 clean-fit rows | Any claim beyond finite-sample learnability at this exact budget (not an information-theoretic null) | `docs/residual_evidence_study_spec.md` | stopped by frozen gate (criteria 1–4 failed); no confirmation run ([[H-RESID-01 Source-learnable residual decision information at layer3.22]]) |
| Across 34 sites × 3 poolings × 3 metrics, some hidden candidates repair base errors that logit-space kNN cannot | [[2026-09-21 Representation Atlas Program]] | Seeds 2, 4; deep layer3 + spatial pooling repairs ≈13% of base errors under corruption (vs. 2.5% for logit-space kNN) | That this repair rate is *net* useful (same deep sites harm 40–42% of base-correct examples); that clean net-utility selection picks these sites (it picks layer4 instead) | `docs/atlas_program_spec.md` §13–14, `results/atlas/report/` | H-A…H-E all failed their frozen continuation thresholds (mixed statistical evidence — see the card's 2026-09-21 correction and `results/fixed_gate/report/hypothesis_audit.json`); [[H-ATLAS-01 Accessible correction information across layers pooling and metrics]] |
| A fixed deep candidate (`layer3.22`, 2×2 pool), gated by a ridge model on output+geometric features, beats base under corruption | [[2026-09-21 Fixed Deep Candidate Gate Study]] | Seeds 2, 4; n≤2500 clean-fit; **deep-Z1-vs-base interval excludes 0** (+0.075 / +0.213 pp macro-12) — a real small positive pipeline effect | That the geometric features cause it (Z1−Z0 interval includes 0, both checkpoints); that it beats the matched output-candidate control (interval includes 0); that it meets the +0.5/+0.25 pp practical targets; any claim about a differently-sized supervision budget | `docs/fixed_gate_study_spec.md`, `results/fixed_gate/report/`, `results/fixed_gate/reconciliation_v2.json` | closed under its own criteria, reconciliation complete ([[H-GATE-01 Candidate selection versus gate utility mismatch]]) |
| On the frozen deep-Z1 gate's *actual* (narrow) intervention set, empirical decision utility is positive | fixed-gate study, item B of its 2026-09-22 correction | Seed 2: 399/120,000 rows, W/(W+H)=0.70, (W−H)/F=+22.6%. Seed 4: 3,571/120,000 rows, W/(W+H)=0.585, (W−H)/F=+7.2% | A population guarantee; a new threshold selected post hoc from these counts; that this generalizes beyond the frozen gate's exact selected (λ,θ) | `results/fixed_gate/report/pipelines.json`, `gates_frozen.json` | descriptive, reported alongside the interval-based reading above, not instead of it |
| A Boolean-cube prototype-counting result may bound what fixed-size real-valued kNN readouts can extract | [[Counting and Covering in Nearest-Neighbour Representations of Boolean Functions]] | Abstract verified from the arXiv primary source (2026-09-22); distinguishes Euclidean-unrestricted from Boolean-cube-restricted prototypes | Any direct claim about neural classification accuracy, calibration, or corruption robustness — the paper is about exact Boolean-function representability, not our setting | local PDF in `01_Papers/`, unreadable on this cluster (no PDF text tooling) | research lead / conjecture only; theorem-level reading pending |
| A target-fitted linear readout of base logits plus the fixed `layer3.22` GAP probe logits beats a target-fitted linear readout of the base logits alone (**target-supervised diagnostic; restricted readout**) | [[2026-09-22 Stage 0 Probe-Logit Increment Study]], [[Target-fitted stacking of logits and layer3.22 probe logits recovers a material increment that clean-fitted stacking on matched budgets does not]] | Checkpoints 2 and 4; 12 exposed CIFAR-100-C cells; 10,000 shared images; T-8k×12: +4.23 / +3.98 pp macro accuracy, intervals [3.99, 4.47] / [3.75, 4.22], shuffled-P control and convergence audit clean; gain concentrated in noise and severe blur/fog | Any deployable or clean-only method; an information ceiling; specificity to `layer3.22`, to intermediate layers, or to geometry (no matched second-predictor control); other corruptions, severities, checkpoints, architectures; a mechanism. Target recalibration of the logits alone is ≈42–44 % of the gain over base (post-hoc) | `docs/stage0_execution_spec.md` (§7), `results/stage0/report/`, `snapshots/stage0_v1_d0dcfd61aa88` | completed in development; material by the frozen rule; not confirmation |
| Clean-fitted stacking of the same evidence at matched image budgets recovers the gain | same | 8k×1: target-fitted +2.87 / +2.63 pp vs clean-fitted −0.48 / −0.29 pp (gap +3.35 / +2.91 pp, intervals exclude 0); 2.5k×1 gap +1.49 / +1.17 pp; clean-view increment ≈ +0.1 to +0.2 pp (interval includes 0) | A cause of the gap (clean non-identifiability, clean-fit regularization, shift of the probe-logit distribution, readout mismatch are not separated); a claim that no clean-only design could help | same | recoverability gap observed; cause unresolved ([[H-STAGE0-01 Layer3.22 probe evidence is redundant with the logits on clean data but complementary under corruption, and clean supervision cannot identify the useful combination]], proposed) |
| Generic-fusion alternative: any comparably diverse second predictor shows the Stage 0 pattern (target-fit gain, clean-fit null) | [[2026-09-24 Stage 0 Evidence Ablation]] | ResNet-101, checkpoints 2/4, 12 development cells, 8k×1: the other checkpoint's logits `Z_other` are **source-recoverable** (Δ_S +3.46 / +2.53 ≈ Δ_T +3.55 / +2.51 pp; clean increment +2.43 / +2.39 pp); the same-network layer probes are not (Δ_S from −1.4 to +0.3 pp; clean increment within ≈ ±0.7 pp). The target-fit gain is a **stage-specific (layer3.x plateau)** effect (layer3.7–layer3.22, Δ_T ≈ +2.6 to +2.9 pp), ≈ 0 at layer4.1 and −0.4 at layer4.2 | That the competitor is false elsewhere; a cause of the clean/target gap. Conditional redundancy survives but rests on **one contrasting source** (`Z_other`); **"same network" and "clean-redundant" are fully confounded** in these data. `P_4.2` reads the same penultimate features as `Z`, so the depth claim rests mainly on layer4.0/4.1 ≈ 0 and the layer3 plateau, not on the primary contrast (D_A) alone | `results/stage0_ablation/report/`, `docs/stage0_evidence_ablation_spec.md` | generic-fusion competitor did not fire in this setting; separating the confound is proposed as a pretrained-model regime map (Phase 2), not started |

## Results vs. interpretation vs. conjecture vs. future work (kept visibly separate)

**Results (measured, artifact-backed):** every row's "Evidence" and "Supported scope" columns above; the exact contrast table,
W/H/U counts, and numerical-bound derivation in the fixed-gate study's 2026-09-22 correction section.

**Interpretation (labelled, still evidence-scoped):** hidden mid-to-late representations (deep layer3, spatially pooled) carry
correct alternatives that logit space does not; standalone accuracy-maximizing selection (as used by the atlas) and
selective-deployment gating are different objectives that can disagree (the atlas picked layer4, but layer3 candidates carry
more raw repair information); across every study in this program, hidden-representation signals that show a real, non-null
effect on development data have not yet cleared this program's own practical-usefulness bars.

**Conjecture (not evidence, explicitly unproven):** the repeated pattern — real but small/uncertain signal, no demonstrated
net-positive gating, no demonstrated transfer advantage over output-space controls — could reflect a readout or admissible-family
limitation (the fixed kNN/ridge-gate family cannot extract what the representation contains) rather than an absence of
information in the representation itself. The Anthony paper (above) is a candidate source of vocabulary and results for
stating this precisely; it has not been read at theorem level and establishes nothing about this claim yet.

**Proposed future work (not authorized by this note):** the three open questions in the paper's note; a precisely justified,
separately preregistered test of whether a different admissible readout family changes the picture — see
[[Theory Plan - Decision Utility, Layers, Compression and Risk Control]] for where such a test would need to sit. The next
adversarial idea-discovery pass (a separate Claude Chat task) is expected to populate `06_Ideas/` from the open questions
here; this note does not do that itself.

## What is *not* established by anything in this program

- That hidden-representation information is absent or unusable in general (every closed study found a real, non-null,
  small effect; none found a practically useful one at the tested budget).
- That more clean supervision would or would not help (flat learning curves in the fixed-gate study are a scope limitation,
  not evidence either way — see that card's item C).
- That any result here generalizes beyond checkpoints 2 and 4, the four tested corruption families, or the exact frozen
  candidate/gate/layer configurations (seeds 1, 3, 5 and all other corruption families remain untouched throughout this
  program).
- That the IJCAI/AAAI-era published CIFAR-100 baselines share the normalization-mismatch fix described above (unresolved).

## Links

Experiments (chronological): [[2026-09-20 Full-Vector DAC POC]] → [[2026-09-21 Normalization Audit and Corrected Protocol]] →
[[2026-09-21 Layer-Selection Pilot]] → [[2026-09-21 Residual Evidence Study]] → [[2026-09-21 Representation Atlas Program]] →
[[2026-09-21 Fixed Deep Candidate Gate Study]].
Hypotheses: [[H-FVDAC-01 Class-conditioned DAC density enables decision correction]], [[H-LAYER-01 Selected internal layers add decision value beyond logits]],
[[H-RESID-01 Source-learnable residual decision information at layer3.22]], [[H-ATLAS-01 Accessible correction information across layers pooling and metrics]],
[[H-GATE-01 Candidate selection versus gate utility mismatch]].
Theory and project tracking: [[Theory Plan - Decision Utility, Layers, Compression and Risk Control]], [[Research Lineage]], [[Full-Vector Geometric Calibration]].
Paper lead: [[Counting and Covering in Nearest-Neighbour Representations of Boolean Functions]].
Dashboard pointer: `00_Inbox/Research Dashboard.md` (append-only chronological log; this note is the standing summary it should point readers to).

## Workflow transition — 2026-09-23

The program now uses the canonical
[`research workflow`](../../GeometricFullCalibration/docs/research_workflow.md):
question → competing explanations → smallest informative experiment → verified
evidence → allocation decision → justified follow-up. This is a governance
transition, not a revision of FV-DAC, pilot, residual, atlas, fixed-gate, or
their frozen decisions; no historical card has been retrofitted as though its
new fields existed before results.

[[Representation Correction Exposure Ledger]] is the authoritative map of
development and reserved resources. The four 12-cell development families and
checkpoints 1/3/5 are not described as pristine; the 11 other CIFAR-100-C
families are provisionally reserved only to the extent the recorded audit
supports it. [[Serial method tests without discriminating outcomes]] records
the failure mode this transition is meant to prevent.

The earlier “untouched throughout this program” wording in this note refers
only to this representation-correction development sequence. It is not a
global repository exposure claim: the ledger records the broader
recoverability/legacy exposure of checkpoints 1/3/5 and controls if those
resources are considered for later work.

Worked example: the frozen `docs/stage0_execution_spec.md` and
`results/stage0/report/` record Stage 0. It is an operational-recoverability
diagnostic of target-fitted base logits versus base-plus-`layer3.22` probe
logits, with matched source/target and image/view-count secondaries—not an
information ceiling or automatic authorization for correctness/abstention,
Stage 1, or new data.

## Stage 0 update — 2026-09-23 (appended; nothing above rewritten)

Source: [[2026-09-22 Stage 0 Probe-Logit Increment Study]] (frozen spec `docs/stage0_execution_spec.md` §7 for deviations). Development evidence on exposed cells from two checkpoints; a target-supervised diagnostic of a restricted readout. It is not a deployable method, not an information ceiling, and not confirmation.

**Allocation decisions (memo Section 7, the rows that apply).**
* The recoverability-gap row applies (Δ_T−Δ_S > 0 at 8k×1 with intervals excluding 0 in both checkpoints): **no further clean-only study of this evidence family without a stated mechanism for closing the gap.** This is a narrow allocation stop, not a refutation of clean-only correction.
* **Stage 1: eligible but not started.** It needs its own decision and authorization.
* **Stage 0d (detection/abstention): not started.**
* **Direction 2 (label-free choice of readout depth under shift, memo definition):** the memo makes it conditional on a large oracle layer gain in Stage 0b. No layer met the 0b flag and the `layer3.22` probe alone is below base in all 13 conditions, so its trigger is not met. Separately, a single scalar trust weight does not recover much of Δ_T (post-hoc blend recovers 26–29 % with target labels), which weakens a "learn one weight" version of it.
* Kept unchanged: the four closed studies' verdicts; no historical card retrofitted.

**Two refuted working guesses (interpretation, post-hoc, development only).**
1. *"The intermediate probe is simply more robust."* Refuted by 0b: the clean-trained `layer3.22` probe alone is worse than base in all 13 conditions in both checkpoints (−3.0 to −8.7 pp); the gain appears only when it is combined with the logits under a target-fitted readout.
2. *"The gap is one scalar trust weight."* The post-hoc blend `softmax(α·z + β·p + b)` recovers only 26–29 % of Δ_T with target labels and 8–11 % with clean labels, and the clean-fitted β (≈0.47) is larger than the target-fitted β (≈0.37), the opposite of what a simple clean-underweights-the-probe story predicts. Under the raw-scale caveat (`z` and `p` scales not equalized), only α-versus-β comparisons are affected; the clean-versus-target β comparison is within one parameterization. The recovery-share result is the stronger evidence; the β difference is modest.

**Exposure (see the authoritative [[Representation Correction Exposure Ledger]], which already has a Stage 0 row and was not edited here).** Stage 0 used clean-test images as fitting rows (S regimes) and as the base pool of the T regimes; used labels of the 12 development cells for target-supervised fitting; used checkpoints 2 and 4 only. Checkpoints 1/3/5 and the 11 unused CIFAR-100-C families were not accessed by Stage 0 and are not called pristine.

**Provenance caveats carried by this update.** Spec §§0–6 were frozen from an execution prompt before the memo (a Claude Doc) was found; derived-gap intervals were added after the first aggregate output; commit `b55dd90` reached `origin/main` through an unauthorized agent-spawned push; one seed-4 S-2.5k×1 fold file was rewritten by that agent's rerun (unverified). Details in the card's deviations list.

Links: [[Theory Plan - Decision Utility, Layers, Compression and Risk Control]] (memo T1/T2 appended), [[Research Lineage]], [[Research Dashboard]].
