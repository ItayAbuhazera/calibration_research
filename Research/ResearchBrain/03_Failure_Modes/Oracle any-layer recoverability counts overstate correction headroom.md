---
type: failure_mode
status: open
project: Full-Vector Geometric Calibration
tags: [recoverability, oracle, multiplicity, repair-vs-harm, overthinking]
---

# Oracle any-layer recoverability counts overstate correction headroom

## Failure
"Fraction of final errors for which some intermediate readout is correct" (first-correct-exit oracle; Recovery Rate with a_oracle over
all layers) is treated as available correction headroom. With many weak readers over 100 classes it mixes genuine retained evidence
with chance agreement, and it ignores the same readers' harm on correct examples.

## Where observed
- Our Atlas: deep-layer3 2×2 kNN candidates are right on ≈12.9/13.4 % of base errors but wrong on ≈40/42 % of base-correct rows under
  corruption (net −12 pp) ([[2026-09-21 Representation Atlas Program]] corrections; [[2026-09-21 Fixed Deep Candidate Gate Study]]).
- Literature: ≈10-pt oracle vs ≈1-pt realistic early-exit gains under CIFAR-C ([[Understanding the Robustness of Multi-Exit Models under Common Corruptions]]);
  18–76 % oracle recoverability without chance correction ([[Vertical Fusion - Recoverability in ViT Hierarchies]]); transient wrong
  dips occur in correct answers too (Wrong Before Right, arXiv 2607.04640).

## Why existing signal/rule fails
The oracle uses the label to choose the layer; a deployable rule must choose before seeing the label and pays the harm on correct rows.

## Competing explanations
Genuine suppressed evidence (the oracle is partly real) vs multiplicity (k weak readers × 100 classes) vs readout weakness.

## What would falsify this failure mode
A chance-corrected recoverability (matched final-correct control + label-permutation null) close to the raw oracle count, and a
prospective selector capturing most of it (POC-T1/POC-T3 in `docs/internal_computation_recoverability_territory_audit_2026-09-29.md`).

## Related observations
[[Clean-trained depth probes in ResNet-101 are below the head at every depth in every tested condition]].

## Research opportunity
Report candidate-generation (W_c, H_c) and selection (realized (W − H)/N vs oracle-gated) separately in any recoverability claim.
