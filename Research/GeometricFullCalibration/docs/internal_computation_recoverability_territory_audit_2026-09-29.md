# Internal-computation recoverability and selective repair — adversarial territory audit (2026-09-29)

**Type:** problem-discovery / prior-art audit. **No scientific experiment was run; no new outcome was computed; no data were
accessed beyond reading schemas and already-reported aggregates.** G1-DP remains **INCONCLUSIVE (validity)**; G1 remains Outcome C;
N1a remains INCONCLUSIVE — insufficient precision; **N1a-DP remains DRAFT r4, NOT FROZEN, NOT AUTHORIZED, PAUSED**; N1b not started.
Literature cutoff 2026-09-29. Repository state at start: `origin/main` = HEAD = `56ccff7`, clean tree.

Evidence labels used throughout: **[OURS-EST]** established by a frozen rule in our experiments; **[OURS-DESC]** descriptive in our
experiments; **[LIT]** established in prior literature (primary source cited; peer-review status stated); **[INFERENCE]**;
**[HYPOTHESIS]**. "[V]" = checked by the lead against the primary source in this session; "[S]" = checked by a subagent against the
primary source; "[K]" = standard known result not re-fetched.

# 1. Executive finding

**No candidate earns TEST NOW.** The broad direction "wrong final prediction, but the correct class was represented earlier; recover
it; repair safely" is largely occupied, and what remains open is a narrow intersection of settings rather than a new scientific
problem.

1. **"Trajectory" as the object is taken, and order is not privileged.** Representation Trajectories Matters (arXiv 2607.26565, not
   peer reviewed) already treats per-sample state/update sequences as complementary evidence for OOD detection and classification,
   clean and shifted (CIFAR-100-C 169/180 conditions), and reports that fixed layer permutations do not hurt and reverse prediction is
   easier (§4.3, App. L.4). ILGE (arXiv 2603.22665) reports that layer-to-node assignment barely matters. An "ordered trajectory"
   claim would have to overturn these; nothing in our evidence suggests it would.
2. **"Correct earlier, wrong later" is an established phenomenon**, including under common corruptions: Shallow-Deep Networks (ICML
   2019: destructive overthinking in ≈50 % of misclassifications, clean), Mehra et al. (arXiv 2022: ≈10-point first-correct-exit oracle
   gain under CIFAR-C, realistic exit rules ≈1 point), Vertical Fusion (arXiv 2607.10391: intermediate probes fix 18–76 % of last-layer
   probe errors, incl. CIFAR-100-C). In LLMs/MLLMs the whole T1→T3 story with a base-rate control and gated restoration is published
   (CALRD, IJCAI 2026; Orgad et al., ICLR 2025; KAPPA, ICML 2026).
3. **Recoverability vs failure detection is already an explicit distinction** — named "recoverability" in Vertical Fusion, "advice"
   vs "alarm" in SelfChecker (ICSE 2021), "knowledge vs prediction" in KAPPA, "detecting the correct answer" in Orgad §6. T2 is KILLED as
   a standalone problem; its only residue (a chance-corrected, per-error measurement on a native end-to-end head) is a measurement
   refinement that belongs inside T3's POC.
4. **Selective repair-vs-harm (T3) is claimed as a problem structure** in adjacent settings (Selective Adaptation for VLMs, ECCV 2026:
   per-sample beneficial/harmful/negligible, prospective adapt/skip, natural shift; ALTAS 2026: trajectory-gated correction, "the gate is
   the bottleneck"; SelfChecker: internal-layer alternative on CIFAR-100, reported to *lower* accuracy). The remaining gap — internal
   evidence × one end-to-end CNN × prospective KEEP/APPLY × W/H accounting × held-out natural corruption families — is an intersection of
   settings (DIFFERENT METHOD ≠ DIFFERENT PROBLEM), and **our own program already tested the closest version on this exact substrate and
   found no held-out benefit** (fixed deep-candidate gate study: H:W ≈ 3.8:1 among disagreements; Atlas: deep-layer3 candidates repair
   ≈13 % of errors but break ≈40 % of correct rows). T3 = WATCH.
5. **Causal recoverability (T4)** exists in oracle/aggregate form for vision under distortion (DeepCorrect, TIP 2019: replacing
   distorted activations with clean ones restores accuracy; Surgical Fine-Tuning, ICLR 2023: corruption shifts are best repaired in
   early layers; BN-adaptation/TENT are label-free internal interventions evaluated in aggregate) and in LLMs (KAPPA succeeds; two 2026
   preprints report detection-without-correction). WATCH, low priority.
6. **Graph structure (T5) is not a distinct object for a fixed-topology ResNet** — a GNN over per-layer node features is another
   decoder; per-sample edge observables (edge contributions) are old (Topological Uncertainty, IJCAI 2021); DeepProv (ACSAC 2025) and
   NeuroTrace (2026) already use GNNs over per-input provenance graphs. KILL.
7. **RL/DRL: CURRENTLY UNJUSTIFIED.** With a fixed deterministic network every counterfactual reward of KEEP/TRUST-EARLIER/ABSTAIN is
   observed offline from one forward pass; depth adds no information about y; the problem is supervised selective prediction
   (and, with compute cost, full-information optimal stopping). INTERVENE with known deterministic dynamics is a contextual bandit /
   planning problem.

**Strongest surviving question (WATCH, not TEST NOW):** in one end-to-end CNN under held-out natural corruption families, does
multi-depth internal evidence discriminate *repair from harm* for an internal candidate correction better than output-only evidence
does? It is scientifically meaningful as a decision question, cheaply falsifiable read-only on existing caches (§15), but its novelty is
a settings intersection and our own prior evidence predicts a negative result. It is worth running only as a closure test, jointly
decided with the paused N1a-DP (which uses the same selector machinery), not as the next PhD stage.

# 2. Exact empirical starting point from our repository

All values below are copied from committed artifacts; nothing was recomputed in this audit. Labels: **[OURS-EST]** established by a
frozen rule in our experiments; **[OURS-DESC]** descriptive in our experiments (no verdict weight); **[INFERENCE]**.

Setting (all rows): CIFAR-100 ResNet-101 `baseline_cross_entropy`, checkpoints (training seeds) 2 and 4, corrected_v2 preprocessing,
12 exposed development cells of CIFAR-100-C (gaussian_noise, defocus_blur, fog, jpeg_compression × severity 1/3/5) + clean.

| # | Fact | Source | Label |
|---|---|---|---|
| E1 | G1-DP official outcome: **INCONCLUSIVE (validity)** for B−C, H−C, D−C (4 valid families < 5 required; linear Hs−A = +0.259 pp and MLP Ds−C = +0.202 pp breached the +0.2 pp shuffle rule at checkpoint 2) | `docs/g1_decoder_panel_audit_2026-09-28.md` §8; `results/g1dp/report/g1dp_aggregate.json` (sha256 `0b2c6a48…`) | OURS-EST |
| E2 | B−C (z+P_3.22 vs z+H_L) POS in 5/6 families, +1.12 to +3.12 pp; H−C (raw H_3.22 vs H_L) POS in 5/6, +2.49 to +4.31 pp; kNN ≈ 0 (anchor selected) | same, §6/§8 "Descriptive only" | OURS-DESC |
| E3 | D−C (adding P_3.22 after H_L) +0.13 to +0.36 pp in linear/poly2/MLP/rff (NULL), +0.63/+0.67 pp LightGBM (POS) | same | OURS-DESC |
| E4 | Width/capacity cost: Cs−A (shuffled 2,048-d block) −0.5 to −1.0 pp; I−H −0.7 to −2.3 pp; real H_L beats matched noise by ≈ +1 pp (C−Cs) | same | OURS-DESC |
| E5 | G1 (anchored linear): frozen **Outcome C — capacity/estimation confound**; does not establish absence of information in H_L | `docs/g1_conditional_access_spec.md`; card `2026-09-28 G1 Conditional Accessibility Gatekeeper` | OURS-EST |
| E6 | Clean-trained linear probes at 12 depths (layer1.0 … layer4.2; fit on the network's own 45k training images) are **monotonically worse with decreasing depth and below the native head in every condition**, e.g. seed 2 clean: layer3.22 −3.2 pp, layer4.0 −1.0, layer4.1 −0.1; gaussian_noise_s5: layer3.22 −5.6 pp; fog_s3: layer3.22 −3.8 pp. No aggregate "destructive overthinking" at population level in this model | `results/stage0/report/stage0b_layer_probe_eval.json`, `stage0b_macro_by_layer.json` | OURS-DESC (existing aggregate) |
| E7 | Atlas: kNN candidates at deep layer3 (2×2 pooling) are correct on **12.9 / 13.4 %** of base errors under corruption but wrong on **40 / 42 %** of base-correct rows (net −12 pp); logit-space kNN repairs 2.5 %, harms 1.9 % | card `2026-09-21 Representation Atlas Program` (corrections), `results/fixed_gate/report/` | OURS-DESC |
| E8 | Fixed deep candidate + clean-trained gate: no consistent held-out benefit (Z1−Z0 intervals include 0); on the frozen gate's own narrow intervention set W/(W+H) = 0.70 (seed 2, 399 rows) / 0.585 (seed 4, 3,571 rows); deep candidate H:W ≈ 3.8:1 among fit disagreements | card `2026-09-21 Fixed Deep Candidate Gate Study`; `results/fixed_gate/report/pipelines.json` | OURS-EST (gate verdict) / OURS-DESC (W/H) |
| E9 | Stage 0: target-fitted (z, P_3.22) beats target-fitted z by +4.23 / +3.98 pp (T-8k×12); clean-fitted stacking at matched budgets does not (gap +3.35 / +2.91 pp at 8k×1); P_3.22 alone below base in all 13 conditions | card `2026-09-22 Stage 0 Probe-Logit Increment Study` | OURS-EST (development, target-supervised diagnostic) |
| E10 | Stage-0 ablation: the target-fit gain is a layer3.x plateau effect (layer3.7–3.22 ≈ +2.6 to +2.9 pp), ≈ 0 at layer4.1; "same network" and "clean-redundant" are confounded | card `2026-09-24 Stage 0 Evidence Ablation` | OURS-DESC |
| E11 | N1a (between-model routing): repair ≈ 8–11 %, harm ≈ 7–11 % of rows, nearly balanced; output-only selector captures 10–19 % of oracle headroom; frozen verdict **INCONCLUSIVE — insufficient precision** | `docs/n1a_action_ambiguity_audit_2026-09-28.md` | OURS-EST |
| E12 | N1a-DP: DRAFT r4, five families, **NOT FROZEN / NOT AUTHORIZED / PAUSED**; shuffled-target informativeness joint pass linear 0/4, rff 0/4, LightGBM 1/4, MLP 0/4, kNN 0/4; continuity control reproduces N1a G_Z exactly (+2.0317 / +1.2475 pp) | `docs/n1a_dp_prefreeze_methodology_audit_2026-09-29.md`; `results/n1adp_audit/summary_5family.json` | OURS-EST (audit record) |

Existing per-example artifacts that make a read-only POC possible (schema inspected only; no outcome computed in this audit):
`results/layer_pilot/checkpoint_seed{2,4}/<cond>/per_sample.npz` holds, for 10,000 images × 13 conditions: `labels`, native logits
`z`, `pred__base_model`, and **`raw__probe_logits` (10000 × 12 × 100)** — pre-temperature logits of clean-trained linear probes at the 12
depths listed in E6 (temperatures in `frozen_state.json`), plus `raw__r` / `raw__r_perm` (per-depth class evidence and its
permuted-label control). Atlas per-site kNN outputs live under `results/atlas/seed{2,4}/`. Raw H_3.22 for all conditions:
`results/g1dp/h322/`.

What this starting point does and does not license:

* It licenses the *question* "what happens to true-class evidence through depth for final errors?" because E7/E9 show that
  mid-depth readouts sometimes carry the correct class for base errors (E7) and that target-fitted combinations gain (E9).
* It does **not** show "correct earlier → suppressed later". E6 shows that, *on average*, every earlier clean probe is worse than the
  head in every condition; E7 shows that the mid-depth candidates' correct-on-error rate (≈13 %) is dwarfed by their wrong-on-correct
  rate (≈40 %). A per-example flip rate without a base-rate control is exactly what E7 already measured, and it was net-harmful.
* G1-DP cannot be used as evidence for information loss in H_L (E1, E3, E5).

# 3. Representation Trajectories Matters — claim map

**Primary source:** De la Jara, Rodriguez-Opazo, Damirchi, Gould, Ranasinghe, "Representation Trajectories Matters: Complementary
Evidence for OOD Detection and Image Classification", arXiv:2607.26565 v2 (submitted 2026-07-29, revised 2026-07-30). **arXiv-only;
not peer reviewed** (the appendix says it "remains in the AAAI two-column format"; no acceptance is stated). Read in full from the arXiv
HTML render (converted verbatim to text locally; all sections, appendices A–P; figures/tables only as captions and text).

| # | Question | Answer (with location) |
|---|---|---|
| 1 | What is a trajectory? | τ(x) = (z_1(x), …, z_L(x)), one globally pooled image-level vector per native block/stage (§3 Eq. 1; App. B "one image-level vector at each selected depth"); cross-stage widths unified by a fixed Gaussian projection (App. C). "The complete dynamical evolution of a sample through the network, characterized by its states and the transitions between them" (§3). |
| 2 | States, changes, or both? | Both. Updates u_l = z_{l+1} − z_l; class routes μ_l^c and residuals r_l = z_l − μ_l^c; decomposition "total update = class-coherent transport + sample-specific innovation" (§3.1 Eq. 4). OOD uses residual states and a one-step transition predictor (§4.1); recognition uses probes on updates u_l (§4.2); natural-shift branch concatenates intermediate *states* (App. K). |
| 3 | Does order matter empirically? | **No privileged order is claimed.** "Consistent performance under fixed layer permutations further argues against privileged ordering" (§4.3, p. after Fig. 3); "documenting the absence of a privileged forward arrow" (Contrib. 2); five sequence-model families find **reverse prediction easier** (§4.3, App. L.4, Fig. L1). Full-prefix sequence models give no consistent OOD gain over a one-step MLP (App. L.4, Fig. L2); prefix mean helps by only 0.34 FPR95 points (App. L.2). |
| 4 | Reversed order tested? | Yes, for transition prediction (direction audit, App. L.4): reverse is easier for MLP/GRU/LSTM/attention/TCN; App. F.4 explains why this is not a time arrow. Not tested for recognition. |
| 5 | Permuted order tested? | Only "fixed layer permutations" (one sentence, §4.3). App. F.2 states explicitly that a fixed global permutation is invertible and "cannot establish that an ordered predictor beats the same states given to a capacity-matched set encoder". |
| 6 | Unordered multi-layer comparison? | Table F1 lists "Capacity-matched unordered-state control" as the test required for "information may reside in ordering"; **no result of such a control is reported** (INFERENCE from full-text search). The recognition method itself is an unordered, non-negative weighted ensemble of per-depth probes (§4.2 Eq. 8), and App. I.2 compares state probes vs update probes vs a seeded final-state ensemble. |
| 7 | Supervision | Backbones frozen; no encoder trained (App. E). OOD: ID labels only (class routes, transition MLP, standardization; fixed fusion weight 0.3 chosen a priori, App. L.5). Recognition: linear probes + non-negative weights + temperatures + a final-state margin gate fitted on labelled ID/source splits (§4.2, App. E). |
| 8 | Tasks | OOD detection (OpenOOD v1.5, 38 + 4 checkpoints; FPR95/AUROC) and image classification (12 datasets × 6 backbones clean; CIFAR-100-C; PACS; Office-Home) (§5.1). |
| 9 | Failure (misclassification) detection? | **Not studied as a task.** Only an interpretive statement: "Confidence scores are better suited to identifying the model's own errors. Trajectories instead measure computational typicality" (§6 "Three reliability objects"). |
| 10 | Correct alternative recovered? | Only implicitly through net accuracy gains of fused probes. App. I.3 / Fig. I1 shows hand-picked DeiT3-B DTD final-representation errors that a fixed late-block vote corrects ("block 9 decodes the visible attribute correctly even when block 12 … confuse[s] related textures"); explicitly "a diagnostic of linearly readable class evidence". **No per-example recoverability rate, no base-rate control, no characterization of which errors are recoverable.** |
| 11 | Selective override? | Partly: a **final-state margin gate** "can restrict path corrections to final-state-ambiguous samples" (§4.2), fitted on source validation. It is a fusion gate, not a KEEP/APPLY decision evaluated for repair vs harm. |
| 12 | Repair vs harm modelled? | **No.** Metrics are top-1 accuracy deltas vs the final-state probe and vs a seeded final-state ensemble (Δ_ctl) (App. I.1, Table O1); Fig. 6c reports "disagreement by severity". No wins/harms decomposition, no repair precision. |
| 13 | Modifies the network computation? | **No.** Read-only; frozen backbones. The "final state" baseline in recognition is a *linear probe on the frozen final representation*, not the checkpoint's native head (§4.2; App. C "Native readouts"). |
| 14 | Causal claims? | Explicitly disclaimed: "replicated checkpoint evidence rather than causal architecture interventions" (§6); Shapley attribution "is not a causal effect of deleting a backbone block" (App. I.4). App. N reports a channel-perturbation amplification screen (where perturbations are contracted/amplified), **not** restoration of correct outputs. |
| 15 | Distribution shift | CIFAR-100-C with clean-only selection: fused path improves 169/180 conditions; control-adjusted Δ_ctl +0.99 (CLIP) to +3.11 (ResNet-50) pp, architecture mean +1.70 pp (Table O1: "six corruptions, five severities, and three seeds"; App. I.1 separately mentions resampling "the 15 corruption families" — internal inconsistency, recorded not resolved). PACS/Office-Home leave-one-domain-out: +3.01 / +0.65 pp beyond control, 39/48 positive folds; CLIP negative on Office-Home (§5.4, App. K). |
| 16 | Limitations / future work | Global pooled vectors only; local/patch/spatial trajectories proposed as the "complementary next step" with an equal-capacity local control (App. B); checkpoint (not causal) evidence; classification branch needs labelled source data (§6); "the usefulness, ordering, and recoverability of that evidence remain empirical properties of each architecture, task, and shift" (App. F, end); "a final state can discard evidence useful for reliability" but the path "cannot manufacture distributional information" (§6, App. F.1). |

**What the focal paper takes (INFERENCE from the claim map):** (i) "the per-sample sequence of layer states/updates carries evidence
complementary to the final representation" for OOD detection and for classification, clean and shifted, across 6–42 frozen
checkpoints; (ii) the state/update/class-route decomposition; (iii) the observation that forward order is not privileged.
**What it leaves open:** per-example recoverability of the model's *own* errors (it uses probes, not the native head, and reports net
accuracy only); repair-vs-harm accounting; any intervention through the original downstream network; end-to-end-trained CIFAR models
(its ResNet-50 is an ImageNet checkpoint read by probes); a capacity-matched unordered control for order.

# 4. Historical + 2025–2026 prior-art map

Grouped by neighborhood. Status: PR = peer-reviewed venue; AX = arXiv only / under review. Overlap class relative to our candidates:
SAME PROBLEM / SAME PHENOMENON / SAME METHOD ONLY / ADJACENT.

## 4.1 Layerwise evidence, overthinking, early-correct → final-wrong (T1, T2)

| Work | Status | Link | Overlapping claim and location | Class |
|---|---|---|---|---|
| Kaya, Hong, Dumitraş — Shallow-Deep Networks | PR, ICML 2019 [V abstract] | https://arxiv.org/abs/1810.07052 | "the destructive effect occurs for 50% of misclassifications on natural inputs" (abstract); internal classifiers can be trained on a frozen network (§3); confusion metric (internal disagreement) flags errors (§5.2); clean only | SAME PHENOMENON (T1) |
| Mehra, Seto, Jaitly, Theobald — Robustness of Multi-Exit Models under Common Corruptions | AX 2022, venue not found [V abstract] | https://arxiv.org/abs/2212.01562 | "early-exiting at the first correct exit … significant boost in accuracy (~10%) … realistic early-exit strategies … marginal improvement (1%)"; shift widens the oracle–realistic gap by ≈5 %; defines over/underthinking under CIFAR-C | SAME PHENOMENON under corruption (T1/T2 oracle) |
| Di Salvo, …, Damirchi, Meza De la Jara, … — Vertical Fusion | AX, "Under review", 2026-07 [V abstract] | https://arxiv.org/abs/2607.10391 | "introducing the notion of recoverability: the capacity of intermediate representations to correct last-layer failures … intermediate probes correctly classify 18% to 76% of samples that the last-layer probe misclassifies"; Recovery Rate = (a_oracle − a_L)/(100 − a_L) (§3.2.1 [S]); CIFAR-100-C among shift sets (§5.3 [S]); frozen DINOv2 ViT | SAME PHENOMENON / names T2 |
| Uselis & Oh — Intermediate Layer Classifiers for OOD generalization | PR, ICLR 2025 [vault card; S] | https://arxiv.org/abs/2504.05461 | intermediate-layer linear heads beat penultimate under shift incl. CIFAR-100-C; layer chosen per dataset on OOD validation | ADJACENT (dataset-level) |
| Baldock, Maennel, Neyshabur — Example difficulty / prediction depth | PR, NeurIPS 2021 [S] | https://arxiv.org/abs/2106.09647 | per-example "prediction depth" via kNN probes; early layers predict original label of mislabeled examples, later layers memorize (Fig. 5, subagent reading) | ADJACENT (T1, memorization) |
| Jazbec et al. — Anytime classification / conditional monotonicity | PR, NeurIPS 2023 [S] | https://arxiv.org/abs/2306.02652 | per-sample prediction quality is non-monotone in depth | ADJACENT / SAME PHENOMENON |
| Beigelman & Freiman — LogitDynamics | PR workshop (HOW @ CVPR 2026) [V abstract] | https://arxiv.org/abs/2604.10643 | error detection from layerwise logit trajectories (switch rate, commitment depth, top-K); ablation: dynamics add little in-distribution (subagent reading) | SAME METHOD (detection only) |
| Agarwal, Vatsa, Singh, Ratha — Corruption depth | PR, Neural Networks 2024 (abstract via snippet only) | DOI 10.1016/j.neunet.2023.11.035 | layer at which a corrupted image becomes misclassified, per-layer probes on ResNet/DenseNet | ADJACENT, **not read in full — residual risk for T1** |
| SNAP-UQ; TULIP; Dissector | PR (ICLR 2026; ICML 2024; ICSE 2020) [S; vault] | 2508.12907; PMLR v235 benkert24a; ICSE'20 | depth-wise surprisal / intermediate predictions / depth-weighted per-layer submodels for failure detection or uncertainty | ADJACENT (detection) |
| Halawi, Denain, Steinhardt — Overthinking the Truth | PR, ICLR 2024 [S] | https://arxiv.org/abs/2307.09476 | intermediate decodings correct until a "critical layer"; ablating heads reduces overthinking | SAME PHENOMENON (LLM) |
| Li et al. — CALRD, "MLLMs Get It Right, Then Get It Wrong" | PR, IJCAI 2026 [V abstract] | https://arxiv.org/abs/2606.17953 | "models often get it right initially … before changing their minds"; base-rate/direction control: "85% of failures shift toward text, while 89% of successes shift toward vision"; gated restoration "when we detect a confident visual prediction being suppressed, we restore it" | SAME PROBLEM (T1+T2+T3), other domain |
| Deng — Wrong Before Right | AX 2026-07 [V abstract] | https://arxiv.org/abs/2607.04640 | mid-layer "wrong-dip" in *correct* final answers, verified by activation transplantation — transient flips occur in successes too (base-rate warning) | ADJACENT (T1 control) |
| DoLa; SLED; Datta et al. late suppression | PR (ICLR 2024; NeurIPS 2024); AX 2026 [S] | 2309.03883; 2411.02433; 2604.00778 | contrast/evolve early vs final logits to correct outputs; late "negative circuits" suppress an internally computed answer | SAME METHOD / PHENOMENON (LLM) |

## 4.2 Recoverability, correction and repair-vs-harm (T2, T3)

| Work | Status | Link | Overlapping claim and location | Class |
|---|---|---|---|---|
| Xiao et al. — SelfChecker | PR, ICSE 2021 [V abstract; S table] | https://arxiv.org/abs/2103.02371 | per-layer class-conditional KDE; alarm when layers disagree with output; "also provides advice in the form of an alternative prediction" (abstract); Table III (subagent-read, **not re-verified by lead**): advice lowers CIFAR-100 accuracy (VGG-16 66.79→66.16; ResNet-20 69.52→68.85); clean only | SAME PROBLEM (T2), partial T3 |
| Orgad et al. — LLMs Know More Than They Show | PR, ICLR 2025 [S] | https://arxiv.org/abs/2410.02707 | §6 "Detecting the Correct Answer": models "may encode the correct answer, yet consistently generate an incorrect one"; error-type taxonomy §5.1 | SAME PHENOMENON (T2, LLM) |
| Park, Pyun, Jo — KAPPA | PR, ICML 2026 [V comments] | https://arxiv.org/abs/2509.23782 | "LLMs often fail … even if they encode correct answers in their hidden representations"; knowledge vs prediction subspaces; inference-time residual-stream intervention closes the gap | SAME PROBLEM (T2+T4, LLM) |
| Jiang et al. — To Adapt or Not to Adapt? Selective Adaptation for VLMs | PR, ECCV 2026 [V] | https://arxiv.org/abs/2609.08367 | "a new problem of selective adaptation, which aims to determine whether a given test sample should undergo adaptation or be skipped"; per-sample harmful flips of correct predictions; natural shift (ImageNet variants) | SAME PROBLEM STRUCTURE (T3); correction = TTA, gate = output-only |
| Dong, Wang, Jin — ALTAS "Route, Don't Fix" | AX 2026-09 [V abstract; S body] | https://arxiv.org/abs/2609.14825 | per-question choice between greedy and late-layer trajectory correction from terminal entropy + late-layer linearity; oracle router captures 55 % of gain; "the bottleneck is the gate, not the corrector" (subagent reading) | SAME PROBLEM (T3, LLM) |
| Yuan et al. — Hidden Error Awareness: Diagnostic, Not Causal | AX / ICML 2026 workshop [V] | https://arxiv.org/abs/2605.09502 | probe detects errors (0.95 AUROC) but steering, probe-guided best-of-N, self-correction and patching fail to fix them | ADJACENT (detection ≠ correction) |
| Chen, Navratil, Iyengar, Shanmugam — Whitebox meta-models with linear probes | PR, AISTATS 2019 [S] | PMLR v89 | intermediate linear probes feed a confidence meta-model (CIFAR-10/100, incl. noise) | SAME METHOD ONLY (detection) |
| Jaeger et al. — A Call to Reflect on Failure Detection (FD-Shifts) | PR, ICLR 2023 [S] | https://arxiv.org/abs/2211.15259 | failure detection under corruption/shift; softmax response strongest overall | ADJACENT (detection is crowded) |
| Vilas et al. — ViTs in class-embedding space | PR, NeurIPS 2023 [S] | NeurIPS 2023 | "even misclassified samples retain information … that correspond to the correct class" | SAME PHENOMENON (T2, vision ID) |
| Sotoudeh & Thakur (PRDNN); Sinitsin (Editable NN); Arachne; NeuRecover; PC-training | PR (PLDI 2021; ICLR 2020; TOSEM 2023; AX; CVPR 2021) [S] | 2104.04413; 2004.00345; TOSEM; 2203.00191; CVPR'21 | fixes-vs-breaks metrics: efficacy/drawdown, repair/break rate, negative flip rate — global weight edits, not per-example prospective KEEP/APPLY | ADJACENT (T3 metrics) |

## 4.3 Causal / interventional (T4)

| Work | Status | Link | Overlapping claim | Class |
|---|---|---|---|---|
| Borkar & Karam — DeepCorrect | PR, IEEE TIP 2019 [V] | https://arxiv.org/abs/1705.02406 | ranks distortion-susceptible filters by accuracy gain "upon correction"; replacing distorted activations with clean ones ≈ perfect correction; learned correction units (blur, AWGN) | SAME PHENOMENON (T4 oracle, aggregate) |
| Lee et al. — Surgical Fine-Tuning | PR, ICLR 2023 [V] | https://arxiv.org/abs/2210.11466 | "for image corruptions, fine-tuning only the first few layers works best" — localization of where corruption damage is repairable | ADJACENT (T4 localization, weight-level) |
| Schneider et al. BN adaptation; TENT; DUA | PR (NeurIPS 2020; ICLR 2021; CVPR 2022) [K] | — | label-free changes of internal statistics under CIFAR-C propagated through the original downstream network; aggregate accuracy | ADJACENT (T4, label-free intervention) |
| Meng et al. causal tracing; Heimersheim & Nanda; Zhang & Nanda; Geiger interchange interventions | PR/AX [K; S] | — | clean→corrupted activation restoration as a method | SAME METHOD ONLY |
| Rushing & Nanda self-repair; McGrath Hydra effect | PR ICML 2024; AX 2023 [S] | 2307.15771 | downstream compensation after interventions | ADJACENT (T4 confound) |
| Liu — Decodable but not corrected by linear steering | AX 2026 [S] | https://arxiv.org/abs/2605.05715 | decodable failure mode, steering gives ≈0 correction | ADJACENT |

## 4.4 Graph-structured computation (T5)

| Work | Status | Link | Overlapping claim | Class |
|---|---|---|---|---|
| Ben Hmida et al. — DeepProv | PR, ACSAC 2025 [S] | https://arxiv.org/abs/2509.26562 | per-input inference provenance graph → GNN detects adversarial inputs; graph-guided activation repair (+≈55 % adversarial accuracy, one layer); no non-graph baseline | SAME METHOD (+ repair, adversarial) |
| Ben Hmida et al. — NeuroTrace | AX 2026-04 [S] | https://arxiv.org/abs/2604.14457 | GraphSAGE over ResNet-20 channel graph, adversarial detection; no set/sequence control | SAME METHOD ONLY |
| Lacombe et al. — Topological Uncertainty | PR, IJCAI 2021 [S] | https://arxiv.org/abs/2105.04404 | per-input activation graph with edge weights \|W·x\|, OOD and corruption-shift detection | SAME PHENOMENON (edge observable) |
| Zhao et al. — CRV (verifying CoT via its computational graph) | PR, ICLR 2026 [S] | https://arxiv.org/abs/2510.09312 | per-sample attribution-graph features predict step correctness; ablation: topology features matter least | SAME PROBLEM (LLM); evidence against T5 |
| Frasca et al. — CHARM | AX 2025 [S] | https://arxiv.org/abs/2509.24770 | only found graph-vs-no-graph ablation; sample-specific topology (confounded) | ADJACENT |
| Wang et al. CDRP (CVPR 2018); Qiu et al. effective path (CVPR 2019); NNSlicer (FSE 2020) | PR [S] | — | per-sample routing/effective paths for adversarial detection | SAME METHOD ONLY |
| Graph Metanetworks; Kofinas neural graphs; NFN | PR ICLR 2024 / NeurIPS 2023 [S] | 2403.12143 | graphs of *weights*; permutation symmetry across networks — not applicable to one fixed network | ADJACENT (not a collision) |

## 4.5 Sequential decision / early exit (RL audit)

| Work | Status | Link | Overlapping claim | Class |
|---|---|---|---|---|
| PABEE (Zhou et al.) | PR, NeurIPS 2020 [S] | https://arxiv.org/abs/2006.04152 | patience-based exit motivated by overthinking; beats final layer | SAME METHOD ONLY (trust-earlier heuristic) |
| Jazbec et al. — Fast yet Safe: Early-Exiting with Risk Control | PR, NeurIPS 2024 [V abstract; S body] | https://arxiv.org/abs/2405.20915 | risk-controlled exits incl. performance-gap risk; i.i.d. assumption, shift left as future work (§6, subagent reading) | SAME PROBLEM on "safe" axis, in-distribution |
| CALM (Schuster et al.) | PR, NeurIPS 2022 [S] | https://arxiv.org/abs/2207.07061 | frozen base; exit classifier trained on oracle consistency labels; Learn-then-Test | SAME METHOD ONLY |
| Chen et al. — Learning to Stop While Learning to Predict | PR, ICML 2020 [S] | https://arxiv.org/abs/2006.05082 | closed-form oracle stopping distribution learned by imitation instead of policy gradient | reduction of RL to supervision |
| EENet; Kubaty et al. rethinking EE calibration | AX [S] | 2301.07099; 2508.21495 | exit schedulers trained with BCE on oracle correctness labels | SAME METHOD (supervised reduction) |
| BlockDrop; SkipNet; Runtime Neural Pruning; EPNet | PR (CVPR 2018; ECCV 2018; NeurIPS 2017; CIKM 2020) [S] | — | RL policies for skipping/exiting; BlockDrop explicitly "a single-step MDP … contextual bandit" (§3.2) | ADJACENT (RL, efficiency) |
| UCBEE; CeeBERT; Hari et al.; UAT | PR/AX [S] | — | label-free bandit threshold adaptation under domain shift using confidence as proxy reward | ADJACENT |

# 5. Direct collision matrix (most important papers)

P = problem, Ph = phenomenon, I = information, A = action, Su = supervision, Sh = shift, E = evaluation, M = method. ✓ = same as our
candidate framing (single end-to-end CIFAR-100 ResNet-101; native head; mid/late-depth evidence; natural corruption shift; KEEP vs APPLY
with W/H), ~ = partially, ✗ = different.

| Paper | P | Ph | I | A | Su | Sh | E | M | Net verdict on our candidates |
|---|---|---|---|---|---|---|---|---|---|
| Representation Trajectories Matters (AX 2026) | ~ (complementary evidence for classification) | ~ | ✓ (per-depth pooled states/updates) | ~ (fusion + margin gate) | ✓ (source labels only) | ✓ (CIFAR-100-C, PACS, Office-Home) | ✗ (net accuracy; no W/H) | ~ | takes "trajectory as evidence", "order not privileged"; leaves W/H, native head, causal |
| Vertical Fusion (AX 2026) | ✓ (T2 by name) | ✓ | ✓ | ✗ (fuse all) | ✓/✗ (probes; OOD validation per subagent) | ✓ (CIFAR-100-C) | ~ (oracle recovery rate) | ~ | kills T2 as a named concept; no chance correction, no W/H |
| SelfChecker (ICSE 2021) | ✓ (alarm + alternative) | ✓ | ✓ (per-layer densities) | ✓ (override with advice) | ✓ | ✗ (clean) | ~ (net accuracy; negative on CIFAR-100) | ~ | closest vision T2/T3 precedent; negative in 100-class setting |
| SDN (ICML 2019) | ~ | ✓ | ✓ (internal classifiers) | ~ (exit) | ✓ | ✗ | ~ (oracle) | ~ | takes T1 phenomenon |
| Mehra et al. (AX 2022) | ~ | ✓ | ✓ | ~ (exit policies) | ✓ | ✓ (CIFAR-C) | ~ (oracle vs realistic) | ~ | takes T1 under corruption; realistic policies ≈ 1 pt |
| Selective Adaptation for VLMs (ECCV 2026) | ✓ (skip/apply, harm) | ✓ (harmful flips) | ✗ (output across augmentations) | ✓ | ✓ | ✓ (natural) | ✓ (per-sample beneficial/harmful) | ✗ (TTA) | takes T3 problem structure |
| CALRD (IJCAI 2026) | ✓ (T1+T3) | ✓ | ✓ (layerwise predictions) | ✓ (gated restoration) | ✗ (training-free) | ✗ (conflict benchmarks) | ~ | ~ | full template in MLLMs |
| KAPPA (ICML 2026) | ✓ (T2+T4) | ✓ | ✓ | ✓ (residual intervention) | ~ | ✗ | ~ (net) | ✗ | T4 in LLMs |
| DeepCorrect (TIP 2019) | ~ (T4 oracle) | ✓ | ✓ (filter activations) | ✓ (correct activations) | ✗ (trained units) | ✓ (blur/noise) | ✗ (aggregate) | ✗ | takes oracle causal restoration |
| DeepProv (ACSAC 2025) | ~ | ✗ (adversarial) | ✓ (per-input graph) | ✓ (activation repair) | ~ | ✗ | ✗ | ✓ (GNN) | takes "GNN over provenance graph + repair" |
| Jazbec et al. (NeurIPS 2024) | ~ (safe exit) | ~ | ✓ | ~ | ✓ | ✗ (i.i.d.) | ✓ (risk) | ✗ | shift explicitly open |

# 6. T1 — class-evidence evolution

**Claim audited:** for eventual final errors, true-class evidence becomes strong at an intermediate depth and is later weakened while
another class wins.

**Already taken [LIT]:** existence and magnitude as oracle counts (SDN; Mehra under CIFAR-C; Vertical Fusion incl. CIFAR-100-C); per-
sample non-monotonicity (Jazbec 2023); error detection from layerwise dynamics (SDN confusion, SelfChecker, LogitDynamics, SNAP-UQ);
in MLLMs, the *direction* of the late shift separates failures from successes and gates a restoration (CALRD); transient mid-layer
wrong-dips in correct answers (Wrong Before Right) — the base-rate control is known to matter.

**Possibly open [INFERENCE]:** a controlled characterization in an end-to-end vision CNN under natural corruption of *how* true-class
evidence is lost (rank/margin trajectory, persistence, abrupt vs gradual, where), compared with matched final-correct examples and
against a multiplicity/chance null. Oracle "any layer correct" counts are inflated by multiple attempts; nobody we found corrects for
that in vision. Residual risk: Corruption Depth (Neural Networks 2024) not read in full.

**Our evidence as constraint [OURS-DESC]:** E6 — every clean-trained probe is below the head in every condition and accuracy rises
monotonically with depth; at population level there is no destructive overthinking in this model. E7 — deep-layer3 candidates are right
on ≈13 % of errors but wrong on ≈40 % of correct rows. So a large fraction of any "earlier-correct" count will be chance agreement of
weak readers.

**Stronger claim beyond "an intermediate classifier was correct":** only "suppression is a distinguishable event": among final errors,
true-class evidence that was top-ranked at a stable depth band and then lost exceeds (a) the rate of the analogous event among
final-correct examples (a runner-up that was top-ranked earlier and lost) and (b) a label-permutation / probe-shuffle null. Even if
true, this is a descriptive measurement; its importance comes only from what it enables (T3).

**Verdict: WATCH.** Phenomenon taken; controlled characterization open but not a stage-defining contribution on its own.

# 7. T2 — recoverability

**Distinction audited:** failure detection ("probably wrong") vs recoverability ("internal computation tells us the correct
alternative").

**Already explicit [LIT]:** Vertical Fusion defines "recoverability: the capacity of intermediate representations to correct last-layer
failures" and measures it under CIFAR-100-C; SelfChecker separates alarm from "advice in the form of an alternative prediction"; Orgad
§6 separates error detection from detecting the correct answer; KAPPA separates knowledge from prediction; Hidden Error Awareness (2026)
reports detection without correction. Our own Atlas (E7) and fixed-gate study (E8) already measured correct-alternative rates on base
errors.

**Label-free at inference, label-evaluated offline:** straightforward (candidate class from internal evidence; W/H counted with labels
offline) — and exactly what SelfChecker, Vertical Fusion and our Atlas did. Nothing new here.

**Residue:** a chance-corrected recoverability measure on a native end-to-end head (e.g. true-class-top-1 rate of one pre-specified
internal candidate among final errors, reported with the same candidate's harm rate on final-correct rows). This is a measurement
refinement and is exactly the "candidate-generation quality" half of T3.

**Verdict: KILL** as a standalone problem (substantially the same problem is defined and measured in prior work, including CIFAR-100-C).
The measurement residue is carried into T3's POC.

# 8. T3 — selective internal repair (KEEP vs APPLY, prospective repair-vs-harm)

**Already taken [LIT]:** the problem structure (prospective apply/skip with beneficial/harmful/negligible outcomes, natural shift) for
TTA of VLMs (ECCV 2026); trajectory-gated correction with an explicit do-no-harm aim in LLMs (ALTAS) and MLLMs (CALRD); an
internal-layer alternative prediction on CIFAR-100 (SelfChecker, clean, reported net negative); risk-controlled trust-earlier exits
i.i.d. (Jazbec 2024; shift named as future work); fixes-vs-breaks metrics in repair/editing (drawdown, break rate, NFR); realistic exit
rules recover ≈1 of ≈10 oracle points under CIFAR-C (Mehra).

**Our own prior tests on this substrate [OURS-EST/DESC]:** fixed deep candidate (layer3.22 2×2 kNN) + clean-trained ridge gate on
output and geometric features: no consistent held-out benefit; H:W ≈ 3.8:1 among fit disagreements; on the frozen gate's narrow set
W/(W+H) = 0.70 / 0.585 (E8). Atlas (E7). Stage 0: clean-fitted combination cannot recover the target-fitted gain (E9). N1a: output
features capture 10–19 % of routing headroom between two models (E11).

**What remains open [INFERENCE]:** the intersection (internal multi-depth evidence) × (single end-to-end CNN, native head) ×
(prospective KEEP/APPLY) × (per-example W, H, ΔAcc = (W − H)/N) × (held-out natural corruption *families*, not clean-fitted gates) ×
(matched output-only selector). Our earlier gates were **clean-fitted**; a leave-one-corruption-family-out selector (the N1a design) with
multi-depth probe evidence has not been run. That is a different supervision regime, not a different scientific problem.

**Separation of candidate generation vs selection:** required and feasible: report (i) candidate quality (W_c = candidate right & head
wrong; H_c = candidate wrong & head right) and (ii) selector quality (realized ΔAcc vs oracle-gated headroom, AUC for repair vs harm
among disagreements), each against the output-only selector.

**Verdict: WATCH.** Scientifically meaningful decision question and cheaply falsifiable, but (a) the problem structure is claimed in
adjacent settings, so the gap is a settings intersection; (b) literature priors (SelfChecker CIFAR-100, Mehra ≈1 pt, ALTAS "gate is the
bottleneck") and our own four closed studies predict a small or negative result. It does not meet the TEST NOW bar of "clear gap".

# 9. T4 — causal / interventional recoverability

**Observational vs causal:** an auxiliary probe predicting y_true shows accessibility; causal recoverability requires that changing the
implicated representation and running the *original* downstream network restores the correct output.

**Already taken [LIT]:** oracle clean-activation replacement restores accuracy under blur/noise (DeepCorrect); where corruption damage
is repairable by updating weights (surgical fine-tuning: early layers for corruptions); label-free internal-statistics interventions
under CIFAR-C (BN adaptation, TENT; aggregate); per-sample harm of test-time interventions (ECCV 2026 selective adaptation); LLM
residual-stream interventions closing a knowledge–prediction gap (KAPPA) and failures of steering/patching to correct detected errors
(2026 preprints); downstream self-repair confounds (Hydra, Rushing & Nanda).

**Open [INFERENCE]:** a per-example, depth-resolved restoration profile for an end-to-end CIFAR-100 ResNet under CIFAR-100-C (oracle:
patch the clean image's activation at depth l into the corrupted forward pass; does the native head recover y?), with W/H. The oracle
version is a descriptive localization close to DeepCorrect + surgical fine-tuning; the non-oracle version (an evidence-derived edit)
has an identifiability problem: if the edit is class-directed (push toward the probe's class), restoration through the downstream
network is nearly guaranteed and uninformative; if it is class-agnostic (restore clean statistics), it is TTA.

**Verdict: WATCH (low priority).** Not killed outright because per-example depth-resolved causal restoration under natural shift was not
found, but the problem is close to existing oracle/TTA results and its non-oracle form lacks a clean identifying contrast.

# 10. T5 — graph-structured computation

**Question:** does connectivity of a *fixed-topology* ResNet provide an inductive bias or observable beyond set/sequence models on the
same information?

**Analysis [INFERENCE, supported by LIT]:** with fixed topology and node features, a GNN is a function of the concatenated node features
at fixed positions; any MLP/transformer with depth/channel identity can express it (no permutation symmetry to exploit — unlike weight-
space metanetworks). New information exists only in per-sample edge quantities (w·a contributions, residual-branch vs skip share,
attribution flow), and those were used for detection already (Topological Uncertainty 2021; effective paths 2019; DeepProv 2025) and can
be flattened into per-layer scalars for a set/sequence decoder. Prior ablations point against topology (CRV: topology features matter
least; ILGE: layer-to-node assignment barely matters). The fair control package is known (same information in a DeepSets/transformer
decoder; true vs degree-preserving rewired vs complete graph; within-class edge shuffles) and nothing suggests it would favor the graph.

**Verdict: KILL** as a scientific object (GNN ≠ CONTRIBUTION). A GNN may be used later only as one decoder in a panel with the controls
above.

# 11. RL / DRL / optimal-stopping assessment

Formalization [INFERENCE; reductions documented in LIT]:
* Deterministic frozen net: h_l = g_l(h_{l−1}). Without INTERVENE, actions (CONTINUE, EXIT, KEEP, TRUST-EARLIER, ABSTAIN) do not change
  future states; the only uncertainty is the static label y. Decision-relevant state is the history H_l (the premise is precisely that
  h_{<l} carries evidence lost by h_l), so it is a POMDP with action-independent deterministic observations = Bayesian sequential
  testing / optimal stopping. H_{l+1} is a deterministic function of H_l: going deeper adds computed features, not information about y.
* **Counterfactual rewards are fully observable offline** (one forward pass on a labelled example reveals the payoff of every
  stop/answer/abstain policy). No exploration, no unknown dynamics, no credit assignment → backward induction = supervised regression of
  P(y | H_l) (Chen et al. ICML 2020 do exactly this; EENet, CALM, Kubaty et al. train exit policies on oracle labels).
* With no compute cost (our correction setting), the optimal policy observes H_L and then classifies or abstains: CASE 1, supervised
  selective prediction. Optimal stopping matters only if a compute cost is introduced (not our question).
* INTERVENE makes transitions action-dependent (CASE 2), but dynamics are known and deterministic; each counterfactual costs one forward
  pass → full-feedback contextual bandit (one decision point) or known-model planning (several). BlockDrop's own RL collapses to "a
  single-step MDP … contextual bandit".
* The real difficulty under shift is P_test(y | H) ≠ P_train(y | H) — an estimation/transfer problem RL does not address; label-free
  bandits use confidence as reward, which is what shift corrupts.

**RL verdict: CURRENTLY UNJUSTIFIED.** It would become "possible but method-driven" only with multi-step, non-differentiable,
compute-costly interventions after supervised/planning baselines were shown to fail.

# 12. Claims already taken

1. Per-sample layer-state/update sequences carry evidence complementary to the final representation for OOD detection and for
   classification, clean and shifted (focal paper; ILGE for LLMs; Uselis & Oh at dataset level).
2. Forward layer order is not privileged in these settings (focal paper §4.3, App. L.4; ILGE).
3. Early-correct → final-wrong exists, ≈50 % of errors on clean data (SDN), with ≈10-point oracle gains under CIFAR-C (Mehra) and
   18–76 % "recoverability" of last-layer probe errors incl. CIFAR-100-C (Vertical Fusion).
4. Internal–final disagreement predicts errors (SDN, SelfChecker, Dissector, LogitDynamics, SNAP-UQ, whitebox meta-models).
5. Internal layers can propose an alternative class for flagged errors; on CIFAR-100 this was reported to hurt (SelfChecker).
6. "Recoverability" vs "detection" as named, distinct notions (Vertical Fusion; Orgad; KAPPA).
7. Prospective per-sample apply/skip with beneficial vs harmful outcomes under natural shift (ECCV 2026, for TTA).
8. Trajectory-gated correction with do-no-harm framing; the gate, not the corrector, is the bottleneck (ALTAS; CALRD) — LLM/MLLM.
9. Oracle clean-activation correction restores accuracy under distortions (DeepCorrect); corruption repair localizes to early layers
   (surgical fine-tuning).
10. Per-input computation graphs + GNNs for detection and activation repair (DeepProv, NeuroTrace; edge-contribution topology, TU).
11. Supervised oracle-label training of exit/trust policies; risk-controlled exits i.i.d.; RL/bandit formulations of skipping/exiting.

# 13. Claims that might remain open (none is a new problem; each is a settings intersection or a control)

a. Chance-/multiplicity-corrected recoverability of a *native end-to-end head's* errors under natural corruption, reported with the same
   candidate's harm rate (measurement refinement of Vertical Fusion/Mehra).
b. Whether suppression of true-class evidence is a distinguishable event relative to matched final-correct examples and a null (T1).
c. Whether multi-depth internal evidence discriminates repair from harm *beyond output-only evidence* for a prospective KEEP/APPLY
   decision on held-out corruption families in one CNN (T3 residue).
d. A capacity-matched ordered-vs-unordered test (the focal paper names it as required, App. F.2/Table F1, but reports none).
e. Per-example, depth-resolved oracle restoration profile under CIFAR-100-C (T4 residue).
f. A risk/harm guarantee for switching away from the final prediction under shift (Jazbec 2024 names shift as future work).

# 14. KILL / WATCH / TEST NOW table

| Candidate | Verdict | One-line reason |
|---|---|---|
| T1 class-evidence evolution | **WATCH** | phenomenon taken (SDN, Mehra, Vertical Fusion; CALRD in MLLMs); controlled suppression characterization open but descriptive |
| T2 recoverability | **KILL** | named and measured incl. CIFAR-100-C (Vertical Fusion); alarm vs advice (SelfChecker); knowledge vs prediction (Orgad, KAPPA); residue folded into T3 |
| T3 selective internal repair | **WATCH** | problem structure claimed (ECCV 2026, ALTAS, CALRD, SelfChecker); gap = settings intersection; own prior evidence predicts negative |
| T4 causal recoverability | **WATCH (low)** | oracle/aggregate version taken (DeepCorrect, surgical FT, TTA; KAPPA in LLMs); non-oracle version poorly identified |
| T5 graph-structured computation | **KILL** | fixed topology ⇒ GNN = another decoder; edge observables old; DeepProv/NeuroTrace; topology ablations negative |
| "Ordered trajectory" as object | **KILL** (absent new evidence) | focal paper: permutations do not hurt, reverse easier; ILGE: assignment irrelevant |
| RL / DRL | **CURRENTLY UNJUSTIFIED** | full-information offline counterfactuals; reduces to supervised selective prediction / bandit |
| **Any TEST NOW?** | **No** | — |

# 15. Smallest falsifying POC for every surviving candidate (NOT authorized; specification only)

All three POCs are **read-only retrospective analyses of existing caches**; none needs GPU, new images, new layers, new checkpoints or
reserved families. They are outcome-generating and therefore require explicit researcher authorization and a frozen card first.
Common substrate: `results/layer_pilot/checkpoint_seed{2,4}/<cond>/per_sample.npz` (labels, native logits `z`, `raw__probe_logits`
10000 × 12 × 100 from clean-trained probes at layer1.0 … layer4.2, temperatures in `frozen_state.json`), 12 exposed cells + clean,
Stage-0 image folds. Caveat: probes were fitted on the network's own 45k training images (`fit_data: train(45000)`).

## POC-T3 (primary; the only one that informs an allocation decision)

* **Claim:** in ResNet-101 under held-out CIFAR-100-C families, multi-depth probe evidence predicts whether switching from the native head
  to a fixed internal candidate repairs or harms, better than output-only evidence.
* **Closest prior-art threat:** Selective Adaptation for VLMs (ECCV 2026) for the structure; SelfChecker for the vision internal-advice
  precedent; focal paper margin gate for the fusion baseline.
* **Observable:** per-example Δ = 1[c(x) = y] − 1[ŷ_head(x) = y] ∈ {−1, 0, +1}, where c(x) is ONE pre-specified label-free candidate
  (e.g. argmax of the temperature-scaled mean of the layer3.17–layer4.1 probe log-probabilities; the exact rule frozen before any label is
  read). Candidate-generation quality: W_c, H_c among disagreements. Selector quality: realized ΔAcc = (W − H)/N of "APPLY iff predicted
  P(+1) − P(−1) > 0".
* **Simplest strong baseline:** the N1a output-only feature set F_Z (standardized logits + entropy/margins/max-softmax/max-logit + argmax
  one-hot) with the N1a selector, same candidate, same holdout; plus the constant policies (never/always switch).
* **Critical controls:** (i) probe-evidence block shuffled across images within partition (dimension-matched); (ii) candidate replaced by a
  runner-up-of-head candidate (tests whether "internal" matters vs "any second guess"); (iii) oracle-gated headroom min(W_c, H_c)-style
  bound, reported, not used for selection.
* **Smallest artifact:** the layer-pilot cache above + Stage-0 folds; leave-one-corruption-family-out × image folds exactly as N1a.
* **Success condition:** in both checkpoints, internal-evidence selector − output-only selector ≥ +0.25 pp family-macro with a lower
  interval bound > 0, and ≥ 3/4 held-out families positive, and shuffled control ≤ +0.10 pp.
* **KILL condition:** upper interval bound of (internal − output-only) < +0.25 pp in both checkpoints, or the internal selector does not beat
  "never switch" in held-out families.
* **If positive:** internal evidence carries repair-vs-harm information that output does not, under a supervision regime between the failed
  clean-fit gates and the target-fitted Stage-0 diagnostic; justifies a preregistered confirmation on reserved families and a causal
  follow-up (T4).
* **If negative:** closes the within-model selective-correction line on this substrate, consistent with the fixed-gate study and
  SelfChecker; the program's remaining open question is the between-model N1a-DP ambiguity, not internal evidence.

## POC-T1 (descriptive prerequisite; run only together with POC-T3 if at all)

* **Claim:** loss of top-ranked true-class evidence through depth is more frequent among final errors than the analogous loss of a
  top-ranked non-final class among final-correct examples, beyond a chance null.
* **Threat:** SDN / Mehra / Vertical Fusion (oracle counts); CALRD (direction signature in MLLMs).
* **Observable:** per example, the true-class rank trajectory r_l(y) over the 12 depths; event S = "true class top-1 at ≥ 2 consecutive
  depths among the last 6, and not top-1 at the head".
* **Baseline / control:** matched event on final-correct examples (a non-final class top-1 at ≥ 2 consecutive late depths); confidence-bin
  matching; label-permutation null (y replaced by a random class) and a probe-shuffle null (probe logits permuted across images within a
  condition).
* **Artifact:** same cache; no fitting.
* **Success:** rate(S | error) exceeds the matched correct-example rate and the nulls by ≥ 5 pp in both checkpoints in ≥ 3/4 families.
* **KILL:** rate(S | error) within ±2 pp of the matched control or null.
* **Positive:** suppression is a distinguishable event — a candidate feature for T3, not a result by itself.
* **Negative:** "earlier-correct" is chance agreement of weaker readers in this model (consistent with E6/E7); T1 closes.

## POC-T4 (optional, lowest priority; needs one GPU forward-hook pass, so NOT read-only)

* **Claim:** replacing the corrupted image's activation at depth l with the clean image's activation restores the native prediction for a
  depth-dependent, non-trivial fraction of final errors, with measurable harm on final-correct rows.
* **Threat:** DeepCorrect; surgical fine-tuning; causal tracing.
* **Observable:** per-example restoration indicator by depth l ∈ the 12 cached depths, W/H vs head.
* **Control:** patch with a *different* clean image of the same class and of a different class (distinguishes restoration from
  class-forcing); self-repair check (restoration rate vs number of downstream blocks).
* **Kill:** restoration profile monotone in depth and fully explained by "fraction of network re-run on clean input" (i.e. no localized
  damage) — then nothing beyond surgical fine-tuning.

# 16. Recommended next scientific question

Plainly: **this audit does not find a problem that deserves to be the next stage of the PhD in its broad form.** "Trajectory",
"recoverability", "graph of computation" and "RL over depth" are taken or unjustified.

The single narrowest question worth a decision is POC-T3: *does multi-depth internal evidence of one end-to-end CNN resolve repair-vs-harm
for an internal correction beyond output-only evidence, on held-out natural corruption families?* It is recommended **only as a cheap
closure test**, and it should be decided **jointly with the paused N1a-DP**, because both use the same selector, holdout and Δ-utility
machinery (N1a-DP: between-model routing with output evidence; POC-T3: within-model switching with internal evidence). If the researcher
wants a single next experiment from this territory, it is POC-T3 (read-only, CPU, hours); if not, the territory should be recorded as
audited and closed.

Why POC-T3 is different from the focal paper: the focal paper reports net accuracy of fused probes relative to a *linear probe* on
frozen *pretrained* features, with source-only selection; POC-T3 uses the *native head* of an end-to-end model, decomposes W and H, and
tests internal vs output-only *selectors* for a prospective KEEP/APPLY decision on held-out corruption families. Why it differs from the
strongest other collision (Selective Adaptation for VLMs, ECCV 2026; and SelfChecker): the correction is an internal alternative class,
not TTA; the gate uses internal evidence with a matched output-only gate; shift is held out by family rather than clean-only
(SelfChecker). These are differences of setting and control, not of problem — hence WATCH, not TEST NOW.

N1a-DP: **remain PAUSED and UNFROZEN.** Nothing in this audit resolves its open design items or argues for launching it by itself.

# 17. ResearchBrain changes

See the commit that accompanies this report (vault, `ResearchBrain/`), all new notes linked from the new research-line note:

* `10_Projects/Internal-Evidence Recoverability and Selective Correction.md` — research-line note (status `draft_for_discussion`;
  audit verdicts; N1a-DP recorded as PAUSED).
* `01_Papers/Prior Art Map - Internal Computation Recoverability and Selective Repair.md` — consolidated prior-art map.
* `01_Papers/Representation Trajectories Matters.md`, `01_Papers/Vertical Fusion - Recoverability in ViT Hierarchies.md`,
  `01_Papers/Self-Checking Deep Neural Networks in Deployment.md`, `01_Papers/Understanding the Robustness of Multi-Exit Models under
  Common Corruptions.md`, `01_Papers/To Adapt or Not to Adapt - Selective Adaptation for VLMs.md` — strongest-collision paper cards.
* `02_Observations/Clean-trained depth probes in ResNet-101 are below the head at every depth in every tested condition.md` — from
  existing Stage-0b artifacts.
* `03_Failure_Modes/Oracle any-layer recoverability counts overstate correction headroom.md`.
* `04_Hypotheses/H-SELREP-01 ….md` (T3) and `04_Hypotheses/H-EVO-01 ….md` (T1) — `status: proposed`, with kill/go; no experiment card.
* `07_Killed_Ideas/` — graph-structured computation as a distinct object (T5); RL over depth for fixed-network correction; standalone
  recoverability as a new concept (T2) — each killed by prior-art audit, not by a protocol.
* Appended (not rewritten): `00_Inbox/Research Dashboard.md`, `08_Weekly_Synthesis/2026-W40.md`,
  `10_Projects/Current Evidence - Representation-Based Correction Program.md`.
* Not touched: `06_Ideas/` (zero edits), the exposure ledger (no data accessed), all historical cards and frozen specs.

# 18. Literature/source ledger and cutoff

Cutoff: 2026-09-29. Four scoped subagents (A dynamics, B recoverability/repair/causal, C graph computation, D sequential decision/RL)
plus lead verification. **≈ 85 primary sources seriously examined** (A ≈ 25, B ≈ 30, C ≈ 16, D ≈ 22, overlapping ≈ 10; lead read the
focal paper in full and re-verified 16 abstracts/metadata directly: 2607.26565, 2212.01562, 1810.07052, 2607.10391, 2606.17953,
2607.04640, 2604.10643, 2103.02371, 2609.20299, 2609.08367, 2609.14825, 1705.02406, 2509.23782, 2605.09502, 2210.11466, 2405.20915).

Neighborhoods searched (saturation): intermediate classifiers / overthinking / early exit under corruption (saturated: SDN, PABEE, Mehra,
Jazbec, EEFP cluster recurred); layerwise probes and OOD (saturated); trajectory-based failure detection (saturated); LLM/MLLM
correct-then-overwritten and knowledge–prediction gap (saturated for our purpose); failure detection under shift (crowded, saturated);
DNN repair and editing metrics (saturated); selective TTA / harm of test-time interventions (**not saturated — new preprints monthly;
re-check before any submission**); vision causal interventions under corruption (moderate: DeepCorrect, AR2 only); per-input
computation graphs (saturated for adversarial/TDA; moderate for natural-shift misprediction); RL/bandit exits (saturated).

Searches triggered by discovered papers: focal paper → Vertical Fusion (same group), Voyager 2609.20299 (same group; OOD only),
Damirchi et al. 2026 (LLM trajectories); SelfChecker → Dissector, SelfChecker++ (not examined); CALRD → Wrong Before Right, late
suppression; Jazbec 2024 → shift as stated future work.

Known unverified / residual risks: Corruption Depth (Neural Networks 2024) full text not read; SelfChecker Table III numbers read by a
subagent from the PDF, not re-verified by the lead; SelfChecker++ (TDSC 2022) not examined; Mehra 2022 venue not found; Vertical Fusion
and the focal paper are unrefereed and from one group — revisions may add W/H analyses; focal paper App. I.1 ("15 corruption
families") vs Table O1 ("six corruptions") inconsistency not resolved.

# Appendix — post-audit red-team (2026-09-29; pointer only, no text above changed)

This report was red-teamed the same day: `docs/internal_computation_recoverability_redteam_2026-09-29.md`. Revised verdicts: T1
WATCH → **KILL** (standalone); T4 WATCH (low) → **KILL**; T2, T5, ordered trajectory KILL and RL CURRENTLY UNJUSTIFIED unchanged; T3
WATCH (dormant); **POC-T3: DO NOT RUN**; territory CLOSED as a next-stage candidate. Qualifications to this report (red-team §3): E1 order
evidence is from the transition-continuity analysis, not recognition; E2 Vertical Fusion's 18–76 % recovery rates are clean-only and
oracle (no CIFAR-100-C recovery rate); E3 Selective Adaptation pools negligible + harmful and targets efficiency; E4 DeepCorrect's clean
replacement is a filter-ranking definition; E5/E6 wording (Fast yet Safe, KAPPA); E7 the fixed-gate study's pipeline gain over base is
small and positive, while its internal-over-output increment (Z1 − Z0) is null, which is POC-T3's central contrast.
