# N1a — Action ambiguity audit (frozen specification v1)

Frozen 2026-09-28 before any N1a code or output existed. Parent commit `dfd137b` (G1 consolidated and pushed).
Authorization: researcher instruction of 2026-09-28 (design, implement, execute N1a on exposed development data; N1b design only).
**Development diagnostic on exposed cells. Not confirmation. Not an internal-evidence experiment: no internal representation is used anywhere in N1a.**

## 0. Question

Is there a per-example **action-selection problem** — examples for which the available pre-action output evidence is insufficient to
determine whether routing to another independently trained checkpoint helps or harms?

Base b and other o: ResNet-101 CIFAR-100 checkpoints (independent training seeds) 2 and 4, run symmetrically (2 -> 4, 4 -> 2).
For row r = (image i, corrupted cell c):

    U_keep(r)  = 1[argmax Z_b(r) = y]      U_route(r) = 1[argmax Z_o(r) = y]      Delta_route(r) = U_route - U_keep in {-1, 0, +1}

Repair = +1, harm = -1. Model disagreement is not the object; the object is heterogeneity in Delta_route.
Secondary (descriptive only): ensemble, Delta_ens = 1[argmax(softmax Z_b + softmax Z_o) = y] - U_keep. Abstain: not analysed (its
advantage at a fixed cost is a function of current-error only; it cannot show action identifiability).

## 1. Data, exposure, evidence boundary

- Z_b, Z_o: canonical strict-FP32 atlas logits `results/atlas/seed{2,4}/u0/<cell>.npz["logits"]`; labels via `atlas.stage0_data.load_cell`.
- Conditions: the 12 exposed development cells (gaussian_noise, defocus_blur, fog, jpeg_compression x severities 1/3/5). Clean rows not used.
  **Not accessed:** the 11 reserved families, checkpoints 1/3/5, new images, any internal representation.
- **Pre-action output evidence (primary channel) F_Z(b)**, computed from Z_b only: standardized z (100), softmax entropy, p1-p2, p1-p3,
  p1-p5 (sorted softmax), max softmax, max logit, one-hot argmax (100) = 206 features. **Z_o is never in F_Z** (asserted in code).
- Reference channel (NOT pre-action evidence; labelled "post-action reference"): F_Z(b) concatenated with F_Z(o). It measures how much
  of the residual ambiguity is resolvable by any observable output at all (here, the action's own output), not deployable evidence.
- **Prior exposure disclosed:** before freezing, aggregate per-cell accuracies, Z-vs-Z_other disagreement (~0.42 macro) and both-wrong
  rates (~0.41-0.44) were known from `results/stage0_ablation/report/ablation_secondary.json`. No per-example Delta, repair/harm table,
  or selector output had been computed. Thresholds below are derived from the estimand, not from those rates.

## 2. Holdout (family AND image identity)

For held-out family f (all 3 severities) and outer fold k of the Stage-0 plan (`results/stage0/shared/fold_plan.json`, 5 folds, duplicate
groups never split): **train rows** = cells of the 3 other families x images in fold k's `train_idx`; **evaluation rows** = cells of
family f x images in fold k's `test_idx`. No training image identity appears in evaluation (asserted). Every (base, family) evaluation
covers all 10,000 images x 3 severities exactly once across folds. lambda is selected on the Stage-0 inner split (`inner_fit_mask`) of
the training images (inner-fit vs inner-val images, all 9 training cells each).

## 3. Selector (primary target and model)

Primary target: 3-way Delta_route in {-1, 0, +1}. Model: unanchored penalized multinomial logistic regression (`atlas.stage0_fit.fit_arm`,
unchanged: mean CE + lambda ||W||^2, per-coordinate standardization on fit rows, float64 L-BFGS, lambda grid {1e-1..1e-5}, min inner-val NLL).
Predicted advantage e(r) = P(+1 | r) - P(-1 | r). **Policy: route iff e(r) > 0.** No architecture search.
Arms: `Z` (F_Z(b), primary) and `ZZo` (F_Z(b) + F_Z(o), post-action reference).

## 4. Quantities (per base, per held-out family = macro over its 3 severities; family-macro = mean over 4 families)

- Oracle gain G_or = P(Delta = +1). Best constant gain G_const = max(0, mean Delta) (evaluation rows; reference).
- **Heterogeneity headroom h = G_or - G_const = min(P(+1), P(-1)) when mean Delta >= 0 else P(+1)** — the value available only
  through per-example selection (always equals min(P(+1), P(-1)) exactly).
- Realized policy gains over keep: G_Z, G_ZZo (mean Delta over routed rows / all rows). Train-chosen constant policy gain G_const_train.
- **Output-policy ambiguity M_A**: bin evaluation rows (per base x family, pooled over folds; cross-fitted) into deciles of e_Z;
  M_A = sum_bins w_b * min(p+_b, p-_b), the regret that any policy constant within e_Z-bins must pay even with the per-bin rates known.
- **Local-Z ambiguity M_B**: within each (family, evaluation fold), standardize z_b with the training rows' statistics; for each row the
  k = 20 nearest evaluation rows in standardized full-logit space **excluding rows of the same image**; M_B = mean over rows of
  min(p+, p-) in the neighbourhood (self included). Uses no labels across image folds.
- Resolvable gap Q = G_ZZo - G_Z (post-action reference minus pre-action output policy).
- Support: counts of repair and harm evaluation rows per (base, family).
- Intervals: 95% image-group bootstrap (groups from `stage0_aggregate.group_id_for_bootstrap`), B = 2000, one shared index array per
  base (seed 20261010 + 50000 + base). M_B: interval on the row mean only (neighbourhoods fixed).

## 5. Frozen thresholds and their justification

- **h_min = 1.0 pp** (family-macro, per base). A deployable selector captures only part of the headroom; to leave >= 0.5 pp of realized
  value (the repository's material scale for a utility difference) at <= 50 % capture, headroom must be >= 1.0 pp.
- **Support: >= 300 repair and >= 300 harm evaluation rows per (base, family).** M_A is estimated from 10 bins; 300 events of each
  kind give ~30 per bin on average, i.e. a relative standard error of roughly 18 % on a per-bin rate — the minimum at which a bin-level
  min(p+, p-) is not dominated by counting noise.
- **Residual ambiguity: M_A >= 1.0 pp (point) with lower 95 % bound >= 0.5 pp** (family-macro, per base), same logic as h_min; and
  M_A >= 1.0 pp in >= 3 of 4 held-out families per base (stability); and **M_B >= 1.0 pp** (point, family-macro, per base).
- **Resolvability: Q >= 0.5 pp with lower 95 % bound > 0** (family-macro, per base): some observable channel must actually reduce the
  residual, otherwise it is irreducible noise at this granularity and no richer pre-action channel is expected to help.

## 6. Decision (first matching row; both bases evaluated separately and must agree for GO/STOP)

0. **INCONCLUSIVE (validity)**: any unit test fails, an image-overlap or Z_o-in-F_Z assertion fires, a selector fit is unconverged after
   the frozen retry, or support fails in >= 2 families of either base.
1. **STOP — insufficient action heterogeneity**: h < 1.0 pp in either base.
2. **STOP — full Z already resolves the useful ambiguity**: in both bases, M_A upper 95 % bound < 1.0 pp, or Q upper 95 % bound < 0.5 pp.
3. **GO — meaningful residual action ambiguity remains after full Z**: in both bases, every §5 residual-ambiguity and resolvability
   criterion holds.
4. **INCONCLUSIVE — insufficient precision**: otherwise (including disagreement between bases).

GO means only: residual action-relevant ambiguity exists after strong output-only evidence on this exposed substrate, and it is not pure
noise. It is **not** evidence that internal representations resolve it. STOP ends N1 on this substrate. N1b is designed only after GO and
is not run without new authorization.

## 7. Execution envelope

CPU only; Slurm `cpu` partition from an immutable snapshot (`python -m atlas.snapshot n1a`); 2 bases x 4 families x 5 folds x 2 arms
= 80 fits. Engineering recovery allowed; any implementation change after outcomes are visible only for a demonstrable bug, recorded as an
amendment.
