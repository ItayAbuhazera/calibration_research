# Red-team of the internal-computation recoverability territory audit (2026-09-29)

**Type:** adversarial review of `docs/internal_computation_recoverability_territory_audit_2026-09-29.md` (commit `c4cb496`; ResearchBrain
commit `83f749f`). **No scientific experiment was run; no new outcome was computed; no data file was opened.** Only primary sources and
already-committed repository cards/reports were read. G1-DP remains **INCONCLUSIVE (validity)**; N1a-DP remains **DRAFT r4, NOT FROZEN,
NOT AUTHORIZED, PAUSED** (not edited); N1b not started; POC-T3 **not authorized** and not run. Literature cutoff 2026-09-29. Repository
state at start: HEAD = `origin/main` = `83f749f`, clean tree.

Labels: **[OURS-EST]** frozen-rule result in our program; **[OURS-DESC]** descriptive in our program; **[LIT]** primary source;
**[INFERENCE]**; **[HYPOTHESIS]**. Verification marks: **[V-full]** read by the lead in full text this session (arXiv HTML, converted
to text and searched); **[V-abs]** abstract/metadata only; **[X]** not accessible.

# 1. Executive verdict

**The original audit is materially correct in direction and slightly too optimistic in two places. It was not too pessimistic anywhere
that matters for allocation.**

* All 13 load-bearing primary sources were re-read in full text (plus the abstract of the residual-risk paper). No citation is
  fabricated or reversed. Five claims need qualification or are overstated (§3). None of the qualifications revives a killed candidate.
* Two verdicts move **down**. **T1 WATCH → KILL** (standalone): the only thing left open is a base-rate-controlled descriptive measurement
  whose only purpose was to feed T3. The MLLM version (CALRD) and the CNN-under-common-corruption localization (Corruption Depth, abstract)
  sit on either side of it, and our own E6 shows no population-level overthinking in this model. **T4 WATCH (low) → KILL**: there is no clean
  non-oracle causal-recoverability question (§8). The oracle version is DeepCorrect's own ranking procedure. Decodability ≠ causality under
  clean→corrupted patching is published for vision (arXiv 2510.09794).
* **T3 stays WATCH, as a dormant item with explicit reopen triggers.** "Settings intersection" is a fair description (§7). No 2025–2026
  paper was found that does the exact experiment.
* **POC-T3: DO NOT RUN** (§11). Our own fixed-gate study already ran its central contrast: internal-evidence gate vs output-only gate on the
  same internal candidate, both checkpoints, clean and 12-cell macro. The increment was null. No plausible POC-T3 outcome changes the
  allocation. A positive would still be a settings intersection. A negative adds little beyond what Stage 0 and the fixed-gate study
  already establish.
* **No stronger question was missed** (§12).
* **Territory: CLOSED** as a candidate next stage of the PhD. T3 is kept only as a dormant WATCH entry with stated triggers.
  **N1a-DP stays PAUSED.** Nothing here argues for or against unpausing it.

# 2. What was independently verified

All full texts were fetched from arXiv HTML on 2026-09-29, converted to text and keyword-searched around every quoted claim. Venues were
checked against arXiv comments / journal-ref.

| # | Source | Status (verified) | What the original audit relies on | Check | Classification |
|---|---|---|---|---|---|
| 1 | Representation Trajectories Matters, arXiv 2607.26565 | arXiv only; appendix "remains in the AAAI two-column format"; no acceptance | trajectory = per-depth pooled states/updates; fused path improves 169/180 CIFAR-100-C conditions; margin gate; no W/H; no causal claim | [V-full] §5.4: "fusing trajectory information improves 169/180 conditions, and every checkpoint has a positive control-adjusted mean"; Table O1 last row "169/168" (route / control-adjusted); §4.2 "The gate can restrict path corrections to final-state-ambiguous samples"; §6 "replicated checkpoint evidence rather than causal architecture interventions" | VERIFIED |
| 1b | same | — | "order is not privileged: fixed permutations do not hurt; reverse prediction is easier" used to KILL ordered trajectory | [V-full] Both statements sit in §4.3 *transition-continuity* validation ("Consistent performance under fixed layer permutations further argues against privileged ordering"), not in recognition. App. F.2: a fixed permutation "cannot establish that an ordered predictor beats the same states given to a capacity-matched set encoder"; App. F.4 "Reverse predictability is not a time arrow" | **NEEDS QUALIFICATION** (§3 E1) |
| 2 | Vertical Fusion, arXiv 2607.10391 | "Under review" | names recoverability; 18–76 %; "measures it under CIFAR-100-C" | [V-full] Abstract/§3.2.1: recoverability = "fraction of final-layer errors corrected by at least one intermediate layer" (oracle, any layer). **Table 1 (the 18–76 % recovery rates) covers 16 clean datasets only.** CIFAR-100-C appears only in Table 3 (fusion accuracy: best layer 73.1, VFusion 74.4, Oracle 81.0); no recovery rate is reported for it | **OVERSTATED** (§3 E2) |
| 3 | Shallow-Deep Networks, ICML 2019, arXiv 1810.07052 | PR | "destructive effect occurs for 50% of misclassifications on natural inputs" | [V-full] abstract verbatim; intro "up to 50% of a CNN's errors" | VERIFIED ("up to") |
| 4 | SelfChecker, ICSE 2021, arXiv 2103.02371 | PR (journal-ref ICSE 2021) | alarm + "advice in the form of an alternative prediction"; Table III advice lowers CIFAR-100 accuracy; clean only | [V-full] Table III "Advice accuracy": CIFAR-100 VGG-16 66.79 → 66.16, ResNet-20 69.52 → 68.85; text "decrease the prediction accuracy" for 100 classes; no corruption/shift evaluation in the text | VERIFIED (previously subagent-only; now lead-verified) |
| 5 | Mehra et al., arXiv 2212.01562 | arXiv only (comments "16 pages, 22 figures"; no venue) | first-correct-exit ≈10 %; realistic ≈1 %; gap widens ≈5 % under shift | [V-full] abstract verbatim; VGG-16 / ResNet-56, CIFAR-10/100-C | VERIFIED |
| 6 | Selective Adaptation for VLMs, ECCV 2026, arXiv 2609.08367 | PR (arXiv comment "ECCV 2026") | "a new problem of selective adaptation"; per-sample beneficial / harmful / negligible; natural shift | [V-full] verbatim. **But:** the binary target pools *negligible + harmful* ("ineffective") against beneficial; the stated goal is efficiency ("skip as many ineffective adaptations as possible while maintaining or even improving overall accuracy"; skips 85 %); scorer = output agreement across augmentations (CAS) | **NEEDS QUALIFICATION** (§3 E3) |
| 7 | CALRD, IJCAI 2026, arXiv 2606.17953 | PR ("Accepted at IJCAI 2026") | right-then-wrong; 85 % / 89 % direction signature; gated restoration | [V-full] verbatim incl. "Such prediction shift occurs in successful cases too … What distinguishes the two outcomes is the direction of the shift"; training-free; conflict VQA benchmarks, not natural shift | VERIFIED |
| 8 | KAPPA, ICML 2026, arXiv 2509.23782 | PR ("Accepted to ICML 2026") | "inference-time residual-stream intervention closes the gap" | [V-full] abstract: "investigate and **mitigate** this knowledge-prediction gap"; knowledge probe is label-trained; App. E.4 reports KAPPA flipping correct → incorrect under cross-dataset transfer | **OVERSTATED (wording)**: "mitigates", not "closes" |
| 9 | DeepCorrect, IEEE TIP 2019, arXiv 1705.02406 | PR | "replacing distorted activations with clean ones ≈ perfect correction; restores accuracy" | [V-full] §IV-A: replacing distorted with clean activations "is akin to perfectly correcting the activations"; the paper uses this as a *definition* to rank filters by "correction priority" (accuracy gain on a validation set per filter), then trains correction units. It does not report whole-network clean replacement as a restoration finding | **NEEDS QUALIFICATION** (§3 E4) |
| 10 | Surgical Fine-Tuning, ICLR 2023, arXiv 2210.11466 | PR | "for image corruptions, fine-tuning only the first few layers works best" | [V-full] abstract verbatim; CIFAR-10→CIFAR-10-C first block beats full FT by ≈3 % | VERIFIED |
| 11 | DeepProv, ACSAC 2025, arXiv 2509.26562 | PR ("To appear in … ACSAC 2025") | per-input provenance graph + GNN; repair +55 % adversarial accuracy; no non-graph baseline | [V-full] abstract "average 55% improvement in adversarial accuracy"; no MLP / set / non-graph ablation found in text | VERIFIED |
| 12 | NeuroTrace, arXiv 2604.14457 | arXiv only | GraphSAGE over ResNet-20 CIFAR-10 provenance; no set/sequence control | [V-full] "3-layer GraphSAGE"; only baseline is CIGA (another graph method) | VERIFIED |
| 13 | Fast yet Safe, NeurIPS 2024, arXiv 2405.20915 | PR (journal-ref NeurIPS 2024) | "SAME PROBLEM on 'safe' axis"; "risk-controlled trust-earlier exits"; shift left as future work | [V-full] §6 "relaxing the i.i.d assumption … could help extend risk-controlling EENNs to scenarios with test-time distribution shifts" — VERIFIED. **But** the performance-gap risk bounds how much an *early exit is worse than the full model* (efficiency with bounded degradation, Eq. 6); it does not aim to repair final errors | **NEEDS QUALIFICATION** (§3 E5) |
| 14 | ALTAS, arXiv 2609.14825 | arXiv only (IEEE copyright notice, no venue) | oracle router 55 %; "gate is the bottleneck" | [V-full] §IV-D: "captures only 55% of that available gain at 3B and 47% at 8B … the bottleneck is deciding which action to take, not the corrector itself" | VERIFIED |
| R | **Corruption Depth**, Agarwal, Vatsa, Singh, Ratha, *Neural Networks* 172:106013 (2024), DOI 10.1016/j.neunet.2023.11.035 | PR | residual risk for T1 | **[X] full text inaccessible.** Elsevier: closed access (OpenAlex `is_oa: false`); SSRN 4386209 and OpenReview `Xj1orI5p6Sv` return bot challenges. **Full abstract read via PubMed efetch (PMID 38354665):** "we introduce a novel concept of corruption depth, which identifies the location of the network layer/depth until the misclassification persists … to understand the processing of examples through the network … a one-dimensional view of sample or query difficulty"; motivation: pruning / explainability | Resolved as far as possible: it is a **SAME PHENOMENON** precedent (per-sample, depth-localized misclassification under common corruptions in CNNs). It pushes T1 toward *more* taken. The abstract states no correction, gate or harm accounting, so T3 is unaffected. Unread body = residual risk only for the details of T1's residue |

Repository facts re-checked (read-only, cards and reports only):

* E8 fixed-gate study (`ResearchBrain/05_Experiments/2026-09-21 Fixed Deep Candidate Gate Study.md`, including its 2026-09-22 correction):
  deep candidate W/H/U at n = 2500 = 123/470/280 (seed 2) and 126/487/295 (seed 4), so H:W ≈ 3.8:1 — VERIFIED. Deep Z1 − base,
  corruption macro-12: **+0.075 pp [0.048, 0.103] / +0.213 pp [0.139, 0.289]** (intervals exclude 0). **Z1 − Z0 (internal geometric
  features over an output-evidence gate on the same deep candidate): −0.038 [−0.086, 0.012] / −0.037 [−0.127, 0.054].** Deep Z1 −
  output-candidate gate: +0.029 [−0.024, 0.086] / −0.022 [−0.128, 0.076]. "C0 (output evidence only) on the deep candidate gives
  +0.18 / +0.19 macro ≈ Z1". [OURS-EST gate verdict; OURS-DESC magnitudes]
* E7 Atlas: repair 12.9 / 13.4 % of base errors and harm 40.2 / 42.0 % of base-correct rows under corruption — VERIFIED from the same card.
* Practical threshold provenance: "+0.25 over output control" is the fixed-gate study's practical target (same card), reused by POC-T3.

# 3. Errors / overstatements in the original audit

No fabricated citation and no factual reversal. Five overstatements or missing qualifications, one internal compression, and one
under-use of our own evidence:

* **E1. Order evidence (report §1 item 1, §12 item 2, §14 row "ordered trajectory").** The report says the focal paper reports "fixed
  layer permutations do not hurt and reverse prediction is easier", and treats this as evidence that order is not privileged. In the paper
  both statements belong to the transition-continuity validation (§4.3, App. L.4), not to recognition or OOD utility. The paper itself
  (App. F.2) says the permutation test cannot compare ordered vs capacity-matched unordered encoders. The report's claim map (§3 rows 3–6)
  is accurate; the executive summary and the table compress it into a stronger claim. **Verdict unaffected:** see §9b for the correct
  reason to kill "ordered trajectory".
* **E2. Vertical Fusion "measures recoverability under CIFAR-100-C" (report §1 item 2, §4.1, §7, killed-idea note).** OVERSTATED.
  The 18–76 % recovery rates are from 16 clean datasets (Table 1). Under CIFAR-100-C the paper reports only fusion accuracy and an oracle
  row (Table 3). Recoverability there is also an *oracle any-layer* count on *MLP probes over a frozen DINOv2*, not a label-free
  candidate on a native head. Consequence for T2 in §6: the KILL survives on SelfChecker (label-free advice, 2021) and on the explicit
  detection-vs-correct-answer distinction (Orgad; KAPPA), not on Vertical Fusion's CIFAR-100-C numbers.
* **E3. Selective Adaptation "takes the T3 problem structure" (report §1 item 4, §5, §8, §12 item 7).** NEEDS QUALIFICATION. The
  per-sample beneficial / harmful / negligible taxonomy and the prospective apply/skip decision are there. But its detection target pools
  negligible and harmful cases, and the stated objective is efficiency at preserved accuracy, not repair-vs-harm discrimination. Its score
  is output-level. T3's *exact* object (a signed repair-vs-harm decision for a correction that changes the label) is therefore **less**
  taken by this paper than the report says. The nearest exact-structure precedents are SelfChecker (vision, clean, net negative), ALTAS
  (LLM) and our own fixed-gate study. This makes the "settings intersection" judgement slightly less pessimistic but does not move it (§7).
* **E4. DeepCorrect "clean replacement ≈ perfect correction; restores accuracy" (report §1 item 5, §4.3, §9).** NEEDS QUALIFICATION. In
  the paper, clean replacement is a *definition* used to rank individual filters by their accuracy gain. It is not a reported
  whole-network restoration finding. The substance (oracle clean-activation substitution under blur/noise, evaluated in aggregate
  accuracy) holds.
* **E5. Fast yet Safe as "risk-controlled trust-earlier exits" (report §8).** OVERSTATED wording. The risk is how much worse the early
  exit is than the full model; the exits are not intended to beat the final layer. The "shift is future work" point is verified.
* **E6. KAPPA "closes the gap" (report §4.2).** The paper says "mitigate". Minor.
* **E7. Under-use of our own evidence (report §1 item 4, §8, research-line note).** The report says our fixed-gate study "found no held-out
  benefit". Its 2026-09-22 correction records a small **positive** pipeline gain over base whose interval excludes 0 in both
  checkpoints. What is null is the **increment of internal (geometric) evidence over an output-evidence gate on the same internal
  candidate**. That null increment is POC-T3's primary contrast, already run under clean-fitting. The original audit cited the right card
  but did not see that it pre-answers the POC's contrast. §11 depends on this.

# 4. Missed prior-art collisions (2025–2026 search, only items that bear on a verdict)

Searches were targeted at each verdict's weakest point: T3's exact intersection; late-layer suppression in vision CNNs; non-oracle causal
restoration in vision; input-dependent effective graphs with fair non-graph controls; RL formulations with action-dependent state;
learned exit policies under domain shift. Abstracts for every item below were read via the arXiv API. No new paper is a SAME-PROBLEM
collision for T3.

| Paper | Status | Exact problem | Overlap | Class | Moves a verdict? |
|---|---|---|---|---|---|
| Huang & Chang — "Causality ≠ Decodability, and Vice Versa: Lessons from Interpreting Counting ViTs", arXiv [2510.09794](https://arxiv.org/abs/2510.09794) | arXiv (2025-10) [V-abs] | activation patching across clean–corrupted image pairs vs linear-probe decodability, ViT counting | "decodability … and causality … reflect complementary dimensions of representation"; mid-layer tokens causal but weakly decodable, final-layer tokens decodable but inert | SAME PHENOMENON for T4's observational-vs-causal separation, in vision | **Yes: T4 → KILL** (together with the §8 identifiability argument) |
| Jhawar & Wang — "When is Test-Time Adaptation Identifiable From Unlabeled Evidence?", arXiv [2609.11235](https://arxiv.org/abs/2609.11235) | arXiv (2026-09-10) [V-full] | whether the evidence given to a TTA action selector (including a KEEP action) can identify the best action at all; CIFAR-100-C, DomainNet-126 | separates "a weak selector versus an information channel that cannot support the desired decision"; deployment-level (batch/stream), not per-example | ADJACENT; takes the *channel-insufficiency vs weak-selector* framing for label-free action selection under shift | Does not move a verdict; **sharpens §11**: a negative POC-T3 cannot by itself separate a weak selector from an insufficient channel |
| Agarwal et al. — Corruption Depth, Neural Networks 2024 | PR [abstract only] | depth until misclassification persists, CNNs, common corruptions | per-sample depth localization of misclassification under corruption | SAME PHENOMENON (T1) | Supports T1 → KILL |
| "Uncovering the Latent Potential of Deep Intermediate Representations" (LOES), arXiv [2605.23033](https://arxiv.org/abs/2605.23033) | ICML 2026 spotlight (comment) [V-abs] | "task-relevant information is distributed non-monotonically across layers and cannot be recovered by naïve aggregation" | layer selection for transfer | ADJACENT (claim 1 of the original §12 now has a peer-reviewed ICML 2026 instance) | No |
| "Causal Interpretation of Neural Network Computations with Contribution Decomposition" (CODEC), arXiv [2603.06557](https://arxiv.org/abs/2603.06557) | ICLR 2026 poster (comment) [V-abs] | per-sample hidden-neuron *contributions* (not activations) as the unit of analysis in image classifiers, with causal manipulations | the only input-dependent edge-like observable with a published payoff, and it needs no graph learner | ADJACENT | Supports T5 KILL |
| "Policy Gradient Steering", arXiv [2607.27574](https://arxiv.org/abs/2607.27574) | arXiv [V-abs] | RL to construct steering vectors when the objective is behavioural over rollouts (gridworld, chess, football) | RL is justified there by non-differentiable rollouts in an external environment | ADJACENT (contrast case) | Supports RL = CURRENTLY UNJUSTIFIED |
| LogitDynamics, arXiv 2604.10643 (HOW @ CVPR 2026) | workshop [V-abs] | ViT error detection from layerwise logit trajectories incl. top-K competitor instability | detection only | SAME METHOD (T1/T2 detection side) | No (already in the original map) |
| BEEM, ICLR 2025, arXiv 2502.00745 | PR [V-abs] | exit criterion aggregating consistent neighbouring exits, aiming to surpass final-layer performance | "trust earlier consensus", i.i.d. only | SAME METHOD ONLY | No |
| "Improving Model Safety by Targeted Error Correction", ICPR 2026, arXiv 2605.02544 | PR [V-abs] | post-hoc GBDT cascade detecting and correcting high-risk errors, output-level | correction precision reported | ADJACENT | No |

Search saturation: T3 exact intersection (5 queries; no hit beyond items already mapped); vision late-layer suppression (hits are LLM or
texture-bias work); non-oracle causal restoration in vision (only 2510.09794 is new); GNN vs same-feature MLP (generic benchmarks
repeatedly show parity; no activation-graph study with a fair control); RL steering (LLM/agents only); learned exits under domain shift
(bandit threshold adaptation only, already mapped).

# 5. T1 verdict challenge — class-evidence evolution

**Strongest case that WATCH is too pessimistic:** vision has no base-rate-controlled "suppression vs refinement" study. CALRD's key
insight, that direction and not occurrence separates failures from successes, is shown only for MLLM modality conflicts. A distinct CNN
mechanism under natural corruption could exist.

**Why it does not hold [LIT + OURS]:**
1. The phenomenon is established in CNNs (SDN, up to 50 % of errors, clean) and under common corruptions (Mehra; Corruption Depth, which
   localizes misclassification depth per sample under common corruptions in CNNs).
2. The control design (compare against successes; base rate) is published (CALRD §3.3; Wrong Before Right).
3. In our substrate there is no population-level overthinking. Every clean-trained probe is below the head at every depth in every
   condition (E6, [OURS-DESC]). Mid-depth candidates are wrong on ≈40 % of correct rows (E7). So any "earlier-correct" event rate here is
   dominated by weak-reader agreement. The original report already conceded this.
4. The original report itself says T1's value is "only what it enables (T3)". With POC-T3 at DO NOT RUN (§11), T1 has no downstream use.

**Revised: KILL (standalone).** The H-EVO-01 matched-event design is kept as a ready-made *control* if the territory ever reopens.

# 6. T2 verdict challenge — recoverability

**Strongest case against KILL:** the load-bearing citation (Vertical Fusion) establishes only *oracle, probe-based* recoverability, and
not under CIFAR-100-C (E2). The user's T2 is *label-free* identification of the correct alternative. Is that really taken?

**Answer:** yes as a concept. Label-free alternative prediction from internal layers with alarm/advice separation exists in vision
(SelfChecker 2021, clean; advice *lowers* CIFAR-100 accuracy, Table III, now lead-verified). Detection vs correct-answer identification is
explicit in LLMs (Orgad §6; KAPPA with label-trained probes). What remains is "label-free recoverability under natural shift on a native
CNN head". That is SelfChecker + shift, and it is exactly the candidate-quality half of T3. **KILL stands**, re-based on SelfChecker /
Orgad / KAPPA rather than on Vertical Fusion's CIFAR-100-C numbers. The killed-idea note is corrected accordingly.

# 7. T3 verdict challenge — selective internal repair

**Strongest case that WATCH is too pessimistic:** after E3, no paper does *signed* repair-vs-harm discrimination for an *internal label
alternative* with a *prospective* gate on a *native end-to-end classifier* under *held-out natural shift*. Selective Adaptation pools
negligible with harmful and optimizes efficiency. SelfChecker is clean-only. ALTAS and CALRD are LLM/MLLM. Fast yet Safe is i.i.d. and
efficiency-oriented.

**Why it stays WATCH, not TEST NOW:**
1. The *decision structure* (KEEP vs APPLY a per-sample correction, judged by wins and harms against an oracle router) exists in ALTAS
   (oracle gap), SelfChecker (advice) and the repair/editing literature (fixes-vs-breaks). What remains is the combination
   {internal alternative × native CNN × natural held-out shift}. The user's own rule classifies that as a settings intersection.
2. The literature prior is negative or small: SelfChecker lowers CIFAR-100 accuracy; Mehra's realistic exit rules recover ≈1 of ≈10 oracle
   points under CIFAR-C; ALTAS captures 15–55 % of oracle gain and "the bottleneck is deciding which action to take".
3. Our own prior is specific and negative (E7, §2): the internal-over-output increment is null for the same decision on the same substrate.

**Revised: WATCH (dormant).** No action. Reopen triggers are in §14.

# 8. T4 verdict challenge — causal / interventional recoverability

**Attempt to find a clean non-oracle question.** A causal recoverability claim needs an edit e(x), computed without the label or the clean
image, applied at depth l, such that the *original* downstream network restores y. The possible edit classes are:

* **Class-directed** (push h_l toward the class an internal probe prefers): restoration through the downstream network is then largely
  a consequence of the edit direction. It tests the probe's candidate (T2/T3), not the network's causal recoverability. Uninformative.
* **Class-agnostic statistics restoration** (renormalize h_l toward clean-source statistics): this is BN adaptation / TENT / DUA [K],
  already per-sample harm-audited in Selective Adaptation. Taken.
* **Oracle clean patching** (substitute the clean image's h_l): this is DeepCorrect's correction-priority procedure (filter level) and
  standard clean→corrupted activation patching. The observational-vs-causal dissociation it would expose is published for vision
  (2510.09794: "middle-layer object tokens exert strong causal influence despite being weakly decodable, whereas final-layer object
  tokens support accurate decoding yet are functionally inert"). A per-example, depth-resolved profile on our ResNet would be a
  descriptive localization, and surgical fine-tuning already predicts its answer (early layers for corruptions).

No edit class yields an identifying contrast that is both non-oracle and non-trivial. **Revised: KILL** (was WATCH (low)). Residue: the
oracle patching profile, a descriptive localization with no problem status.

# 9. T5 verdict challenge — graph structure (and ordered trajectory)

**Strongest case against KILL:** the *effective* graph is input-dependent even with fixed topology: ReLU gating, residual-branch vs skip
share, per-edge contributions w·a. A graph learner might exploit relational structure among these that a flat decoder misses.

**Why KILL stands [INFERENCE + LIT]:** input-dependent edge quantities are per-sample features at fixed, known positions. With node and
edge identity fixed across samples, there is no permutation symmetry for a GNN to exploit, and any such function is expressible by a flat
or set/sequence decoder over the same features. The published payoffs of per-sample edge/contribution observables (Topological
Uncertainty 2021; CODEC, ICLR 2026) do not need a graph learner. DeepProv and NeuroTrace, the only GNN-over-provenance works, report no
non-graph control (verified: NeuroTrace's only baseline is another graph method). CRV's ablation ranks topology features lowest. No
source found gives a fair same-information graph-vs-set comparison favouring the graph. **KILL stands.**

**9b. Ordered trajectory — KILL stands, for a corrected reason.** When states are tagged with their layer index, the ordered sequence
and the indexed set are related by a fixed bijection: the same information. "Order" can matter only as an inductive bias or
sample-efficiency property of a decoder. That is a method property (DIFFERENT METHOD ≠ DIFFERENT PROBLEM), not a scientific object.
The focal paper's permutation and reverse results are not needed for this kill (§3 E1).

# 10. RL verdict challenge

**Search for a formulation with genuinely action-dependent future state:** multi-step INTERVENE (edit h_l, then observe h_{l+1..L}
under the edit) does make future states depend on actions. But the dynamics are the known, deterministic network; every counterfactual
trajectory costs one forward pass; and the reward (correctness) is observable offline with labels and has a differentiable surrogate. That
is known-model planning / search or supervised learning, not RL. At test time under shift the reward is unobservable, so online RL cannot
learn there either. Label-free proxies (confidence) are what shift corrupts (the original §11). RL steering is justified where the
objective is behavioural over rollouts in an external environment (Policy Gradient Steering, 2607.27574). No such environment exists for a
single-pass classifier. **CURRENTLY UNJUSTIFIED — unchanged.**

# 11. POC-T3 red-team

1. **Already published?** Not in this exact form (§4, §7). Closest: SelfChecker (clean, net negative), ALTAS (LLM), and **our own fixed-gate
   study**, which is the same decision on the same substrate with a clean-fitted gate and 4 geometric features instead of 12-depth probe
   logits.
2. **Does our own evidence already answer it?** Largely yes, for the scientifically interesting contrast. The fixed-gate study measured
   internal evidence vs an output-evidence gate on the same internal candidate: Z1 − Z0 = −0.038 [−0.086, 0.012] / −0.037 [−0.127, 0.054]
   (corruption macro-12), and C0 ≈ Z1. Stage 0 bounds the other side: *target-fitted* (z, P_3.22) gains +4.23 / +3.98 pp, but
   *clean-fitted* does not (E9). POC-T3's only new element is the supervision regime between these two (leave-one-family-out), plus a
   different internal feature set. N1a measured that regime for between-model routing with output evidence: 10–19 % of headroom,
   INCONCLUSIVE — insufficient precision. So the open part is narrow: **does target-family supervision transfer across corruption
   families for this internal candidate?** That is a transfer question about our program's supervision regime, not a question about
   internal computation.
3. **Is the candidate outcome-sensitive or arbitrary?** Outcome-informed. The example rule (mean of the layer3.17–layer4.1 probes) is
   chosen from a depth band that the Stage-0 ablation found to hold the target-fit gain on the *exposed* cells (E10, layer3.7–3.22
   plateau). Any band choice made after E7/E9/E10 is a forking path. A defensible candidate would need a clean-data-only rule. But E6
   shows every clean probe is below the head, so a clean-only rule cannot prefer any band on the basis of accuracy. The candidate
   choice cannot be made both principled and non-arbitrary with these artifacts.
4. **Is +0.25 pp meaningful?** No. It is the fixed-gate study's practical target, i.e. 25 images per 10,000-image cell, and it is a
   detectability threshold. Scientifically meaningful quantities are the fraction of oracle-gated headroom captured and the AUROC of
   repair vs harm among disagreements. A +0.25 pp pass could coexist with < 5 % of headroom captured.
5. **Does the selector test anything beyond a stronger output-only decoder?** Not as specified. (a) The internal selector sees the
   candidate's own probe confidence; the output-only F_Z selector does not see the candidate at all. A win could simply mean "knowing how
   confident the second opinion is helps", which is SelfChecker/ensemble territory. The missing control is output features plus the
   candidate's own confidence/margin (2–3 scalars). (b) The 12 × 100 probe-logit block is a much wider input than F_Z. G1-DP showed that
   width/capacity costs and shuffle-rule breaches dominate at these effect sizes (E1, E4). The dimension-matched shuffled block controls
   noise, not capacity-matched informativeness. (c) There is no "final-representation readout only" arm (layer4.2 probe alone), so depth
   is not separated from "any representation-level readout, family-fitted". Without (a)–(c), a positive is not attributable to
   multi-depth internal evidence.
6. **Would a positive still be a settings intersection?** Yes (§7).
7. **Is there a plausible positive that would justify a larger program?** It would need all of: ≥ 25 % of oracle-gated headroom, the full
   internal selector beating output + candidate confidence and beating the layer4.2-only arm, consistency across 4/4 held-out families
   and both checkpoints. Given SelfChecker, Mehra (≈1 of 10 points), ALTAS (15–55 %), N1a (10–19 %) and the null fixed-gate increment, that
   is implausible [INFERENCE]. Even then the result would be a methods contribution colliding with ALTAS / CALRD / Selective Adaptation
   on problem structure, not a new problem.
8. **If negative, how much closure?** Little beyond what exists. By 2609.11235's distinction, a negative cannot separate a weak selector
   from an insufficient channel. In our case Stage 0 already shows the channel is informative under target fitting, so a negative would
   restate "internal evidence does not transfer across corruption families under our supervision regimes". The fixed-gate study, Stage 0
   and N1a already jointly support that statement.

**Recommendation: DO NOT RUN.** No plausible outcome changes the allocation. The contrast of scientific interest was already run
(fixed-gate Z1 − Z0, null). The candidate cannot be fixed non-arbitrarily from these artifacts. As specified, the selector comparison is
confounded by candidate-confidence access and input width. If N1a-DP is ever authorized, a corrected POC-T3 (§14) could be preregistered
as a secondary arm on the same machinery at near-zero marginal cost. That is a decision for then, not a reason to run it now.

# 12. Any stronger missed question

Three candidates that use the same empirical starting point were examined. None survives:

* **Channel sufficiency of internal vs output evidence for the KEEP/APPLY decision** (is the repair/harm label identifiable from the
  internal channel at all?). The framing is published for label-free TTA action selection (2609.11235). Our Stage 0 (target-fit informative)
  and fixed-gate study (clean-fit increment null) already bracket it for this substrate. Not distinct.
* **Readout misalignment vs information loss under shift** (the head is miscalibrated for corrupted features while the information is
  intact). This is our own G1 line (Outcome C; G1-DP INCONCLUSIVE (validity)). In the literature, last-layer retraining / intermediate-layer
  heads under shift is established (Uselis & Oh, ICLR 2025, in the original map; last-layer retraining work [K]). Not new; and G1-DP
  must not be reinterpreted.
* **Decodability ≠ causality under natural corruption in a CNN.** Published for vision ViTs (2510.09794). A CNN replication is a setting
  change.

**None exists.** Using the user's criteria: scientifically distinct, surviving primary prior art, connected to our evidence, small
falsifier, interesting with a simple method. No question derived from this empirical starting point meets all five.

# 13. Revised KILL / WATCH / TEST NOW table

| Candidate | Original | Revised | Reason for (no) change |
|---|---|---|---|
| T1 class-evidence evolution | WATCH | **KILL** (standalone) | phenomenon taken in CNNs incl. common corruptions (SDN, Mehra, Corruption Depth); control design published (CALRD, Wrong Before Right); our E6 shows no population overthinking; only use was feeding T3 |
| T2 recoverability | KILL | **KILL** | unchanged; evidence re-based (Vertical Fusion is oracle and clean-only for its recovery rates; SelfChecker / Orgad / KAPPA carry the kill) |
| T3 selective internal repair | WATCH | **WATCH (dormant)** | settings intersection is fair; Selective Adaptation is less of a collision than stated, but ALTAS, SelfChecker and our fixed-gate null increment cover it |
| T4 causal recoverability | WATCH (low) | **KILL** | no non-oracle, non-trivial identifying edit; oracle version = DeepCorrect procedure; decodability ≠ causality in vision published (2510.09794) |
| T5 graph-structured computation | KILL | **KILL** | unchanged; edge observables flatten; no fair graph-favouring control exists |
| Ordered trajectory | KILL | **KILL** | unchanged verdict; corrected reason (indexed set ≡ ordered sequence; order = inductive bias) |
| RL / DRL | CURRENTLY UNJUSTIFIED | **CURRENTLY UNJUSTIFIED** | unchanged; known deterministic dynamics, offline counterfactuals, no test-time reward |
| POC-T3 | "cheap closure test" | **DO NOT RUN** | §11 |
| Any TEST NOW | No | **No** | — |

# 14. Allocation recommendation

* **Close the internal-computation recoverability territory** as a candidate for the next stage of the PhD. This is a prior-art and
  own-evidence closure, not an experimental negative.
* **Do not run POC-T3.** Keep T3 as a dormant WATCH entry. Reopen only if one of these occurs:
  (i) a primary source shows an internal-evidence gate beating a matched output-plus-candidate-confidence gate for label-changing
  corrections under held-out natural shift in vision;
  (ii) N1a-DP is authorized: then consider a *corrected* POC-T3 as a preregistered secondary arm, with a clean-only candidate rule,
  output + candidate-confidence and layer4.2-only control arms, and headroom-fraction / repair-vs-harm AUROC as primary quantities
  instead of a +0.25 pp threshold;
  (iii) a substrate where clean probes beat the head at some depth (E6 fails), which would make T1/T3 non-trivial there.
* **N1a-DP stays PAUSED.** Nothing in this red-team resolves its four open design questions or argues for launching it.
* The next scientific question should come from outside this territory. This document does not propose one.

# 15. ResearchBrain changes (made after the conclusions above were final)

Appended correction sections with provenance; no historical text rewritten; no experiment card created; `06_Ideas/` untouched:

* `10_Projects/Internal-Evidence Recoverability and Selective Correction.md` — red-team correction section, revised verdict table,
  status `audited` (territory closed; T3 dormant WATCH).
* `04_Hypotheses/H-EVO-01 …` — status `superseded` (T1 killed standalone), correction appended.
* `04_Hypotheses/H-SELREP-01 …` — status kept `proposed` (dormant); POC DO NOT RUN, confound list and reopen triggers appended.
* `07_Killed_Ideas/` — new: T1 standalone and T4 causal recoverability; `Recoverability of final errors …` — correction (E2 re-basing).
* `01_Papers/` — corrections appended to Vertical Fusion (E2), Representation Trajectories Matters (E1), Selective Adaptation (E3); new
  cards for arXiv 2510.09794 and 2609.11235; Prior Art Map addendum.
* Appended one line each: `00_Inbox/Research Dashboard.md`, `08_Weekly_Synthesis/2026-W40.md`,
  `10_Projects/Current Evidence - Representation-Based Correction Program.md`.
* The original territory report receives a one-paragraph pointer appendix (no text changed).

# Source ledger

Full text read by the lead this session [V-full]: 2607.26565, 2607.10391, 1810.07052, 2103.02371, 2212.01562, 2609.08367, 2606.17953,
2509.23782, 1705.02406, 2210.11466, 2509.26562, 2604.14457, 2405.20915, 2609.14825, 2609.11235.
Abstract / metadata only [V-abs]: Corruption Depth (PubMed PMID 38354665; DOI 10.1016/j.neunet.2023.11.035), 2510.09794, 2605.23033,
2607.21973, 2605.02544, 2603.06557, 2604.27529, 2607.27574, 2502.00745, 2604.10643, 2602.06652.
Inaccessible [X]: Corruption Depth full text (Elsevier closed; SSRN 4386209 and OpenReview Xj1orI5p6Sv blocked by bot challenges).
Repository (read-only): `ResearchBrain/05_Experiments/2026-09-21 Fixed Deep Candidate Gate Study.md` (incl. 2026-09-22 correction);
original report `docs/internal_computation_recoverability_territory_audit_2026-09-29.md`.
Not re-verified in this red-team (carried from the original audit as [S]/[K]): Orgad et al. §6, ILGE, CRV ablation, Topological
Uncertainty, Uselis & Oh, BN-adapt/TENT, Chen et al. ICML 2020, BlockDrop. None is the sole support for any revised verdict.
