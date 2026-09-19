# ADR-003: Path A adopted -- gated-anchor Path B is refuted by CIFAR-100 seed21-25 evidence

Status: accepted
Date: 2026-04-25
Deciders: Itay

## Context
Prior planning identified two paths after the first multi-seed batch. Path A reframes the contribution as a Pareto-frontier characterization between top-label ECE and full-vector proper scoring. Path B (gated-anchor) used anchored methods where gc_dac confidence c_t is high and fell back to vector_scaling where c_t is low. Path B's viability hinged on whether anchored methods dominate or match vector_scaling on NLL in any c_t stratum on CIFAR-100, and whether the rank-geom mixture grid selection ever picks lambda > 0 on CIFAR-100.

## Decision
Adopt Path A. The gated-anchor Path B is refuted on CIFAR-100 and not pursued further. The contribution is reframed as a characterization of the structural tension between top-label calibration and full-vector proper scoring across the geometric calibration family.

## Evidence
- per_method.anchored_rankgeom_tail_mixture.selected_lambda at experiments/2026-04-25_cifar100-resnet18_path-decision-batch/aggregate.json: mean 0.0, CI [0.0, 0.0], n = 5. The grid selected lambda = 0 on every CIFAR-100 seed; the rank-geom mixture is structurally dead and reduces to anchored_model_tail.
- per_method.anchored_rankgeom_tail_mixture.selected_alpha at same aggregate.json: mean 0.0, CI [0.0, 0.0], n = 5. Confirms the dead-mixture finding.
- per_method.rgcl_tail_dirichlet.delta_nll_on_base_correct.mean = +0.158484, CI [+0.090331, +0.226637] at same aggregate.json. The anchored family hurts NLL on the base-correct subset with paired CI excluding zero unfavorably -- the failure is not localized to confidently-wrong predictions.
- per_method.rgcl_tail_vector_scaling.nll.mean = 1.464573 vs per_method.full_vector_distance_fusion.nll.mean = 1.303600 at same aggregate.json: the best rgcl_tail variant is +0.16 nat per sample worse than fusion on overall NLL on CIFAR-100.
- Supporting (separate pipeline, not in aggregate.json): the c_t-quartile analysis from c100_method_quantile_metrics_mean_ci95.csv showed rgcl_tail_dirichlet uniformly worse than vector_scaling across q0_q25 through q75_q100 with paired gaps of +0.215 to +0.241 nat. This is consistent with the seed-aggregate finding above.
- Reproducibility note: metrics_by_gc_dac_anchor_ct_bin in the seed-level summaries uses fixed thresholds (<0.1, 0.1-0.5, 0.5-0.9, >=0.9) and on CIFAR-100 places all 10000 samples in the 0.5-0.9 bin for every seed, so the seed-level stratification cannot distinguish a high-c_t subregion. Any future stratification must use distribution-adaptive bins (quartiles) computed at analysis time, not the seed-embedded fixed thresholds.

## Consequences
- Stop doing: pursuing methods that gate anchoring by c_t with anchoring as the high-confidence branch; pursuing rank-geom mixture variants on CIFAR-100.
- Start doing: framing the paper as a Pareto-frontier characterization. Adding a third dataset/architecture (Tiny-ImageNet ResNet50, or ResNet50 on CIFAR-100) as the next experiment to confirm the frontier generalizes.
- New primary baseline (if changed): n/a (full_vector_distance_fusion remains primary).

## Revisit conditions
only if a new dataset shows an anchored method beating vector_scaling on NLL in any distribution-adaptive c_t stratum, or if rank-geom mixture grid selects lambda > 0 on a new dataset.
