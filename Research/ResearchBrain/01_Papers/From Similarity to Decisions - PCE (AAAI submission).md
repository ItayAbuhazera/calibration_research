---
type: paper
status: read_partially
venue: AAAI 2027 submission (anonymous)
title: "From Similarity to Decisions: Pairwise Decision Learning for Embedding-Based Systems"
evidence_source: LaTeX sources (main text §3 from AAAI_Paper.zip 2026-07-30; appendix.tex 2026-07-31), NOT the compiled PDF
tags: [pce, pairwise, decision-layer, hadamard, np-threshold, link-to-fv-dac]
---

# From Similarity to Decisions — PCE (AAAI submission)

**Where it lives (verified):** `/groups/gilein_group/itayab/research/`
(`AAAI_Paper.pdf` 2026-07-31, `AAAI_Paper.zip` 2026-07-30 with `oursolution.tex`,
`intro.tex`, `main.tex`; `appendix.tex` 2026-07-31). The PDF could not be
text-extracted on the cluster (no PDF tools), so the main-text quotes below come from
`oursolution.tex` in the zip, which is one day older than the PDF — **check against
the PDF before citing.** Nothing in that directory was modified.

Related: [[Theory Plan - Decision Utility, Layers, Compression and Risk Control]],
[[2026-09-21 Layer-Selection Pilot]], [[Conformal Risk Control]].

## What the paper does (verified from the sources, §3.1–3.3)

* **§3.1 Problem formulation.** A scalar score `s(q,c)`; pairs split into acceptable
  `H1` and unacceptable `H0`. Objective `max TPR s.t. FPR ≤ α`. Threshold
  `τ_α = Q_{1−α}({s : (q,c)∈H0^cal})` — "the split-conformal quantile of the negative
  class"; the paper states it uses the *empirical* quantile and that the
  `⌈(n+1)(1−α)⌉` correction "additionally guarantees marginal coverage".
* **§3.2 Pairwise decision framework.** Retrieval and deployment are separated: an
  explicit pairwise decision layer, any scorer can occupy it.
* **§3.3 PCE.** All experts consume `h(q,c)=e_q ⊙ e_c` on L2-normalized embeddings
  (node2vec-style Hadamard feature); **cosine is the sum of the coordinates**, so the
  pairwise representation exposes structure unavailable to that scalar. Experts: cosine,
  PCA-whitened cosine, LDA, XGBoost, tiny MLP. Standardize on a selection split, convex
  combination `s_PCE=Σ w_j s̃_j`; weights are chosen **not with a classification loss**
  but to maximize low-FPR utility `TPR̂_α(w)` on held-out data; the threshold is
  calibrated separately.

## Two transferable principles (proposed reading, not a claim)

1. **Preserve richer pairwise information before reducing it to a scalar
   similarity.** In FV-DAC/the layer pilot the analogue is: do not reduce
   per-class, per-layer evidence to one scalar temperature `S(x)`; keep the
   class-conditional vector `r_{l,k}(x)` (and per-layer probes) until the decision.
2. **Train the score for the operational decision, then calibrate that decision
   separately.** Analogue: fit the evidence combiner for expected net decision
   utility (W−H) under a harm budget, and set the operating threshold on a
   separate role. The pilot does **not** yet do this: its layer choice uses NLL on
   inner-SELECT (a proxy) and it leaves clean test untouched for a later calibration role.

## Proposed extension (idea only — *not* an established contribution, *not* implemented)

For base class `i`, candidate `j`, layer `l`, with clean-bank class representatives
`μ̄_{l,k}` and pooled query `h̄_l(x)`:

`v_l(x,i,j) = [ h̄_l(x) ⊙ μ̄_{l,i} , h̄_l(x) ⊙ μ̄_{l,j} ]`

feeding a small scorer of the decision target `D(x,y;i,j)=1{j=y}−1{i=y}∈{−1,0,+1}`
(**not** "is `i` wrong?", which merges W with the neutral wrong→wrong U).
Candidate objective: **expected net decision utility `E[gD]` subject to an explicitly
defined harm budget** on `P(H)` (global) or `P(H | g=1)` (conditional) — rather than
accuracy-independent similarity or an ambiguously defined FPR. This is the
Neyman–Pearson form in [[Theory Plan - Decision Utility, Layers, Compression and Risk Control]] §T4 (P4.1).

**Closest confounds (must be ruled out before any novelty claim):** learned
similarity / metric learning (NCA, LMNN `[unverified]`); prototype classifiers
(prototypical networks `[unverified]`); ordinary linear readouts; stacking and
dynamic classifier selection `[unverified]`. In particular a **linear score on
Hadamard products with fixed prototypes is a constrained linear classifier**
(`⟨w, h̄⊙μ̄_k⟩ = ⟨w⊙μ̄_k, h̄⟩`, i.e. a per-class reweighting of the same features), so
the parameterization alone is **not** a source of novelty. The layer pilot's Family B
(per-layer linear probes) is the un-Hadamarded reference point.

**Not in this phase:** the full PCE ensemble is not implemented anywhere in the
FV-DAC / layer-pilot code.

## Limitations of the appendix, recorded accurately (from `appendix.tex`)

* "Where a threshold is calibrated, its target is a **nominal finite-sample operating
  point rather than a guarantee on unseen data**"; "None of these nominal targets
  guarantees the realized evaluation rate." (Shared protocol table caption.) So the
  empirical quantile is not a risk certificate, and PCE's FPR results are not
  finite-sample guarantees. Contrast the NP-umbrella order-statistic construction
  with a violation-rate bound (Tong–Feng–Li, verified) and LTT/CRC in the theory plan.
* **Evaluation-informed anchor construction.** RQ1 fixes one representative per
  semantic region *before every seed-specific query split*, but "evaluation
  embeddings and labels can influence this complete-artifact representative
  construction"; the stated scope is "decision-layer evaluation on known regions and
  fixed representatives, **not** evaluation of anchor construction, unseen regions …
  or a fully representation-inductive pipeline." **Any evaluation-informed anchor
  construction must not be transferred into a new inductive protocol** — relevant if
  clean-bank class representatives `μ̄_{l,k}` are ever built from data that later
  serves as evaluation.
* **Role separation** in the paper: member scorers fit on training groups; PCE
  weights on a separate query-disjoint subset of the outer training split
  ("calibration data are not used for weight selection"); calibration split selects
  the threshold from negatives only; evaluation untouched. Score learning, selection,
  risk calibration and evaluation each need their own role — which the layer pilot's
  split plan (`layer_pilot_split_v1`) respects except that **no role is left for a
  formal risk certificate** (stated in its spec §3).
* An earlier row-level split leaked query identity between train and eval; the
  paper adopted query-grouped splits (appendix "Why Query-Grouped Evaluation Was
  Adopted"). Analogue here: bank/train examples versus validation are disjoint by
  construction, but CIFAR-100 contains 14/45 000 bit-identical duplicates with
  conflicting labels in its own train split (see FV-DAC note §19.2).

## What this note does NOT establish

That PCE's principles improve FV-DAC-style correction; that the Hadamard adaptation is
novel; anything about the PDF beyond the sources listed above.
