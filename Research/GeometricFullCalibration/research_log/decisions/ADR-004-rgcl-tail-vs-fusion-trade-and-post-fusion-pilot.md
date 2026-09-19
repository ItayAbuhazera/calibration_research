# ADR-004: rgcl_tail_vector_scaling is the existing-method top-label-ECE specialist; post_fusion_topiso is pre-registered as the constructive Path B pilot to dominate it

Status: accepted
Date: 2026-04-25
Deciders: Itay

## Context
The CIFAR-100 seed21-25 aggregate reveals that rgcl_tail_vector_scaling already beats full_vector_distance_fusion on top-label ECE with paired 95% CI excluding zero favorably (paired delta -0.056364, CI [-0.085665, -0.027063]), but pays the cost in NLL, Brier, and accuracy because anchoring locks fusion's argmax flips out (argmax_change_rate ~ 0.0006 vs fusion's 0.082, accuracy 0.6317 vs fusion 0.6576, NLL 1.465 vs fusion 1.304, Brier 0.559 vs fusion 0.466). This sits exactly on the Pareto frontier ADR-003 describes; it is not a dominating method. The constructive question becomes: can a less restrictive recalibration of fusion's top probability -- specifically, isotonic regression fit on validation against base-correctness applied only to fusion's top coordinate, with non-top renormalization to 1 - calibrated_top -- recover top-label ECE without paying the full NLL/accuracy cost that locking imposes?

## Decision
Pre-register post_fusion_topiso as the constructive Path B pilot. rgcl_tail_vector_scaling is named the existing-method baseline that post_fusion_topiso must beat on the joint criterion (top-label ECE non-inferior, NLL/Brier/accuracy strictly better) for the constructive contribution to claim a free lunch over the existing frontier. If post_fusion_topiso fails to beat rgcl_tail_vector_scaling on the joint criterion, the paper proceeds with Path A only.

## Evidence
- per_method.rgcl_tail_vector_scaling.top_label_ece.mean = 0.032071 vs per_method.full_vector_distance_fusion.top_label_ece.mean = 0.088435 at experiments/2026-04-25_cifar100-resnet18_path-decision-batch/aggregate.json
- paired_vs_baseline.rgcl_tail_vector_scaling.top_label_ece.mean = -0.056364, CI [-0.085665, -0.027063] at same aggregate.json (favorable, CI excludes 0)
- per_method.rgcl_tail_vector_scaling.nll.mean = 1.464573 vs per_method.full_vector_distance_fusion.nll.mean = 1.303600
- per_method.rgcl_tail_vector_scaling.accuracy.mean = 0.631660 vs per_method.full_vector_distance_fusion.accuracy.mean = 0.657580
- per_method.rgcl_tail_vector_scaling.argmax_change_rate.mean = 0.000640 vs per_method.full_vector_distance_fusion.argmax_change_rate.mean = 0.082400

## Consequences
- Stop doing: treating rgcl_tail_vector_scaling as a placeholder; it is now the named existing-method baseline for the post_fusion_topiso pilot.
- Start doing: pre-register the post_fusion_topiso pilot card with rgcl_tail_vector_scaling and full_vector_distance_fusion as the two named baselines. The pilot reuses the per-seed fused outputs from the already-ported batches; no new fusion runs are required.
- New primary baseline (if changed): n/a (full_vector_distance_fusion remains primary; rgcl_tail_vector_scaling is the new secondary "frontier baseline" specifically for ADR-004's pilot).

## Revisit conditions
after the post_fusion_topiso pilot results land. If post_fusion_topiso fails to dominate rgcl_tail_vector_scaling on the joint criterion, this ADR is superseded by an ADR that confirms Path A as the sole contribution.
