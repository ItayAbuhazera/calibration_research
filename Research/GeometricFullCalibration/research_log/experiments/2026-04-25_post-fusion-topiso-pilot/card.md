# Experiment Card -- 2026-04-25_post-fusion-topiso-pilot

Status: draft | Date: 25/04/2026 | Commit: 3bfb11f997f30028e179b4ba61aa279f98361970

## Pre-registration (filled BEFORE seeds/ is populated)
- Hypothesis: applying isotonic regression to the top coordinate of full_vector_distance_fusion's output (fit on validation: input = top probability of fused validation output, target = base correctness indicator), then renormalizing non-top probabilities to sum to 1 - calibrated_top, produces a method (post_fusion_topiso) that achieves lower top_label_ece than full_vector_distance_fusion while remaining non-inferior on NLL, Brier, and accuracy. Argmax cannot change under this transform up to numerical noise.
- Primary metric: paired delta (post_fusion_topiso.top_label_ece minus full_vector_distance_fusion.top_label_ece) on CIFAR-100 test set, 5 seeds (21-25). Direction: lower is better.
- Primary baseline: full_vector_distance_fusion.
- Secondary frontier baseline (per ADR-004): rgcl_tail_vector_scaling. The pilot must dominate this method on the joint criterion (top_label_ece non-inferior, NLL/Brier/accuracy strictly better with paired CI excluding zero favorably) for the constructive Path B contribution to claim a free lunch.
- Secondary metrics and acceptable regression bounds, all paired vs full_vector_distance_fusion on CIFAR-100:
  - nll: paired 95% CI upper bound <= +0.01 nat per sample
  - brier: paired 95% CI upper bound <= +0.005
  - accuracy: paired 95% CI lower bound >= -0.001 (essentially zero, since argmax is structurally preserved)
  - adaptive_ece: paired 95% CI upper bound <= +0.005 vs full_vector_distance_fusion
- Secondary metric vs gc_dac (the top-ECE specialist baseline from ADR-002):
  - top_label_ece: paired 95% CI upper bound <= gc_dac.top_label_ece + 0.01 (we accept up to 1pp top-ECE loss vs the specialist in exchange for fusion's accuracy and NLL gains)
- Decision rule:
  - Continue: primary paired CI upper bound < 0 AND all secondary bounds vs fusion respected on CIFAR-100 AND post_fusion_topiso dominates rgcl_tail_vector_scaling on the joint criterion. Promote to CIFAR-10 and Tiny-ImageNet.
  - Modify: primary favorable but exactly one secondary violates its bound; try post_fusion_toptemp (single-temperature scaling on the top coordinate, monotonically equivalent but lower-variance) and re-evaluate once.
  - Pause: primary CI crosses 0.
  - Kill: primary CI lower bound > 0, OR paused result fails to clarify after one modify round, OR post_fusion_topiso fails to dominate rgcl_tail_vector_scaling on the joint criterion. In these cases, fall back to ADR-003's Path A and pre-register the third-dataset confirmation run instead.
- What would change my mind: if post_fusion_topiso achieves the primary criterion on CIFAR-100 but not on CIFAR-10, this is consistent with fusion being a CIFAR-100-and-harder method (CIFAR-10 fusion selected beta = 0 in prior batches). Promote to Tiny-ImageNet anyway and treat CIFAR-10 separately. Conversely, if argmax_change_rate exceeds 0.001 in the implementation, reject results and debug -- the rgcl_tail family has measured argmax_change_rate of 0.00064 from purely numerical effects (per CIFAR-100 aggregate), so 0.001 is the practical noise ceiling for an argmax-preserving transform.
- Seeds: 21, 22, 23, 24, 25.
- Dataset/model: CIFAR-100 ResNet18 primary; CIFAR-10 ResNet18 secondary context. Reuses the per-seed fused outputs from experiments/2026-04-25_cifar100-resnet18_path-decision-batch and experiments/2026-04-25_cifar10-resnet18_path-decision-batch (no new fusion runs required).

Implementation note:
post_fusion_topiso requires per-sample fused validation probabilities to fit the isotonic regressor (input = top probability of fused validation output; target = 1 if argmax(fused_val) == y_val else 0). The implementation prompt for this method must verify that fused validation probabilities are persisted in the existing batches before any code is written. If they are not persisted, the implementation prompt's first step is to add their persistence to the fusion pipeline and re-run the fusion stage; only then proceed to fit the isotonic regressor. Do not skip this check.

Linked ADRs:
- ADR-002 (primary baselines)
- ADR-003 (Path A adopted)
- ADR-004 (rgcl_tail_vector_scaling as secondary frontier baseline; status proposed at time of pre-registration -- accept jointly with this card)

Pre-registered on 2026-04-25. The pilot is implemented and run AFTER this card is committed; no Cursor coding work begins until the pre-registration block is reviewed by the human and ADR-004 is accepted alongside this card.

## Commands
See commands.sh. Environment: see env.txt.

## Results (from aggregate.json -- do not retype numbers here)
- Primary: {{method}} on {{metric}} at {{aggregate.json key path}} = {{value}} [{{ci_lo}}, {{ci_hi}}]
- Secondaries: see aggregate.md.

## Observations
Factual statements only, each grounded in an aggregate.json key path.

## Decision (filled after reviewing results)
- Classification: {{continue | modify | pause | kill | inconclusive}}
- Reason (must cite the pre-registered decision rule above):
- Linked ADR: ADR-{{NNN}} (draft until accepted by human)

## Recommended next experiment
One sentence. Justification must cite this card's metrics.

## Addendum -- 2026-04-26 (pre-registration error noted post-smoke, pre-pilot)

The original Pre-registration block stated that post_fusion_topiso is
argmax-preserving up to numerical noise, and set 0.001 as the invariant
ceiling for argmax_change_rate_vs_fusion. This claim was mathematically
incorrect for the formula as specified.

The smoke run on CIFAR-100 / ResNet18 / seed 21 produced
argmax_change_rate_vs_fusion = 0.1712. Diagnosis: proportional tail
renormalization after top-coordinate isotonic recalibration is
structurally decision-changing whenever isotonic lowers the top
probability enough for some non-top class to exceed it after the
rescale. With mean_top_prob_before = 0.712 and mean_top_prob_after
= 0.601, the non-top scaling factor (1-u')/(1-u) = 1.39, and any
non-top class previously at >= 0.43 of the old top probability becomes
the new argmax. This is the formula behaving as designed, not an
implementation bug.

Corrections, narrow:
- method_family is reclassified from decision_preserving_full_vector
  to decision_changing_full_vector
- can_change_argmax is reclassified from false to true
- the 0.001 argmax-change invariant is withdrawn; argmax_change_rate
  is now informational only
- no implementation change to the calibration formula

Corrections explicitly NOT made:
- the primary metric (paired Δ top_label_ece vs full_vector_distance_fusion)
  is unchanged
- all secondary bounds vs fusion (NLL +0.01, Brier +0.005, accuracy
  -0.001, adaptive_ECE +0.005) are unchanged
- the secondary tolerance vs gc_dac (top_label_ece +0.01) is unchanged
- the secondary frontier-baseline comparison vs rgcl_tail_vector_scaling
  named in ADR-004 is unchanged
- the kill/continue/modify decision rule is unchanged

Rationale: only the argmax-preservation claim was structurally wrong;
no other success threshold has been disproved by the smoke result, so
none should be moved before the 5-seed paired CIs land. If the 5-seed
results show that any unchanged threshold no longer reflects a sensible
test of the (now correctly classified) method, that revision happens
in a successor ADR after results are in, not in this addendum.

Pilot proceeds on CIFAR-100 (primary) and CIFAR-10 (secondary context)
under the corrected method classification and the unchanged thresholds.