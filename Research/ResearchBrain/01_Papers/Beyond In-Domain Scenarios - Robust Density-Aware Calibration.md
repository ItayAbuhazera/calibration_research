---
type: paper
status: audited
year: 2023
venue: ICML
short_name: DAC
tags: [density, hidden-representations, shift, calibration]
---

# Beyond In-Domain Scenarios — Robust Density-Aware Calibration

## Why this matters

DAC uses hidden-representation neighbourhood density to make calibration more
robust beyond the in-domain regime.

## Questions to extract when reading

- Which hidden layers are used?
- How is neighbourhood density defined?
- Is the signal only used to calibrate confidence, or can it change decisions?
- How does its reliability change under shift?
- Which failure modes overlap with [[Validation-fitted neighbourhood reliability features fail under corruption]]?

## Literature extraction (audit 2026-09-15)

- **Signal:** kNN-derived hidden-layer density summaries, aggregated across selected internal layers and combined with a post-hoc calibrator.
- **Target:** calibrated confidence / predictive probabilities under ID, corruption shift, and OOD; it is explicitly accuracy-preserving.
- **Decision:** supports confidence-based trust, abstention, or downstream risk decisions, but does not itself learn a head-versus-geometry router.
- **Can change argmax?** No in its stated accuracy-preserving formulation.
- **Assumptions:** hidden-layer density remains informative about calibration under the evaluated shifts; a validation-fitted layer weighting transfers; the selected kNN reference bank represents relevant ID structure.
- **Benchmark:** CIFAR-10/100 and ImageNet-1k, with CIFAR-C and ImageNet-C severities 1--5; ObjectNet-OOD; multiple CNN/transformer architectures.
- **Distribution shift?** Yes: common corruptions and an ImageNet-based OOD set.
- **Does it ask when geometry itself is reliable?** Partly: it tests whether density improves calibration after shift, but not whether density predicts when a geometric classifier should override the head.
- **Strongest vault overlap:** it is the closest published counterexample to the claim that clean-fitted neighbourhood information necessarily fails under corruption.
- **Remaining gap:** calibration robustness is not oracle-recoverability or matched-risk routing quality; its argmax-preservation means it cannot recover complementary accuracy through class changes.
- **Primary source:** https://proceedings.mlr.press/v202/tomani23a.html

## Extension branch in this workspace (2026-09-20)

[[Full-Vector Density-Aware Calibration]] takes DAC's representations,
pooling, normalization, distance operator, layer set and reference bank
unchanged, and class-conditions only the *search domain* so the operator
returns a per-class distance vector instead of a scalar. See
[[H-FVDAC-01 Class-conditioned DAC density enables decision correction]] and
[[2026-09-20 Full-Vector DAC POC]].

Two implementation facts about **this repository's** DAC, verified
2026-09-20 from the frozen Phase 0/1 state (not claims about the paper):

- The unified benchmark's `native_dac` uses **5** layers (`conv1`,
  `layer1`..`layer4`); the repository's own docstrings describe the paper's
  set as those 5 **plus LOGITS = 6**. The 5-layer version is the frozen
  canonical baseline here. Recorded as a code/paper discrepancy, not fixed.
- The fitted state retains the bank features but **not** the bank labels,
  and discards neighbour IDs — which is exactly why the class-conditional
  extension has to rebuild a labelled bank rather than read one off.

## Addendum 2026-09-21 (layer study)
The layer-selection pilot ([[2026-09-21 Layer-Selection Pilot]]) treats DAC's five sources as aliases of block outputs (`layer1`→`layer1.2`, `layer2`→`layer2.3`,
`layer3`→`layer3.22`, `layer4`→`layer4.2`; `conv1` is the pre-BN stem used only for the temperature) and re-fits native DAC under the corrected preprocessing
protocol: fitted weights are unchanged to 4e-5 (fit inputs were already consistently normalized). The paper-vs-repo layer-set discrepancy (paper includes LOGITS) is
**not** resolved by this study; the five-layer benchmark version is used unchanged.
