---
type: exposure_ledger
status: active
project: Full-Vector Geometric Calibration
date: 2026-09-23
canonical_workflow: ../../GeometricFullCalibration/docs/research_workflow.md
tags: [exposure, development, confirmation, representation-correction]
---

# Representation-correction exposure ledger

**Authoritative ledger for this program.** Entries distinguish label use for
fitting, selection, evaluation, and design decisions. It is a provenance map,
not permission to consume a reserved resource. Update it from manifests and
frozen cards before a new study; do not inspect protected outcomes merely to
make this table look complete.

| Dataset/split and grouped unit | Checkpoint(s) | Corruption family/severity | Purpose and label access | Exposure / reservation status | Evidence |
|---|---|---|---|---|---|
| CIFAR-100 clean validation; validation-image rows | 2, 4 | clean | clean labels used for fit, selection, calibration, and design in layer-pilot, residual, atlas, and fixed-gate programs | development-exposed; not confirmation | `docs/layer_selection_pilot_spec.md`, `docs/residual_evidence_study_spec.md` |
| CIFAR-100 test images (10,000 original image IDs); clean plus aligned copies | 2, 4 | gaussian noise, defocus blur, fog, JPEG compression; severities 1/3/5 (12 cells) | labels used for evaluation and repeated design decisions; Stage 0 also uses declared target labels in T-regime fits | development-exposed; no untouched-confirmation claim | `docs/layer_selection_pilot_spec.md` §2, `docs/stage0_execution_spec.md` §0 |
| Same test IDs / 12 cells | 1, 3, 5 | same four families/severities where recoverability was recorded | evaluation labels used in the RGC shift/recoverability program; checkpoint branch did not design later layer studies | historically exposed in the broader program; **not pristine** | `results/recoverability_aggregate.json`, `docs/residual_evidence_study_spec.md` §3 |
| CIFAR-100 test images (same underlying 10,000 IDs) | 1, 3, 5 | any later new corruption family | no new label access recorded here | new-family evaluation would test a new corruption axis, not new images; do not call it independent-image confirmation | `docs/residual_evidence_study_spec.md` §3 (shared-image caveat) |
| CIFAR-100-C families not named above: shot noise, impulse noise, glass blur, motion blur, zoom blur, snow, frost, brightness, contrast, elastic transform, pixelate; all severities | any | 11 families × severity 1/3/5 | no completed result artifact or vault card naming their outcomes was found in the 2026-09-23 repository/vault audit; a prior **plan** names them as a separate confirmation summary, not as completed access | provisionally reserved for confirmation, subject to a pre-consumption manifest check and explicit authorization | `docs/residual_evidence_study_spec.md` §9; audit recorded in the workflow transition |
| Stage 0 target-fitted diagnostic: same test IDs, clean and 12 development cells | 2, 4 | four development families/severities | T-regime labels fit inner/outer target-supervised classifiers; S-regime uses clean-source labels; evaluation remains out-of-fold within the declared pool | development diagnostic; not a deployable clean-only fit and not confirmation | `docs/stage0_execution_spec.md` §§0–6; `results/stage0/report/` |
| G1 conditional-accessibility gatekeeper (added 2026-09-28): same 10,000 test IDs, clean + 12 development cells; new H_L (layer4 GAP) extraction | 2, 4 (Z_other = the other of 2/4) | four development families × severities 1/3/5 | T-8k×1 and T-2.5k×1 labels of the 12 cells fit anchored target-supervised readouts (Stage-0 folds/cell assignment); clean labels used for descriptive evaluation only; decision on 12-cell macro out-of-fold accuracy | further development reuse; target-supervised diagnostic; not confirmation. **Not accessed:** the 11 reserved families, checkpoints 1/3/5, new images | `docs/g1_conditional_access_spec.md`; `results/g1/`; [[2026-09-28 G1 Conditional Accessibility Gatekeeper]] |

## Rules carried by the ledger

- `fit`, `selection`, `evaluation`, and `design decision` are separate roles;
  record each when adding an entry. Target-supervised diagnostic fitting never
  becomes clean-only fitting by abbreviation.
- The 11-family status is supported only by the current repository/vault
  history audit. Before consuming any family, reconcile the relevant manifest
  and add any discovered prior exposure; reserve only what remains unexposed.
- Checkpoints 1/3/5 are held out from some later design choices, but their
  historical use means they are not pristine. New corruptions of the same
  images are a different generalization axis from new images/architectures.
- A confirmation requires a frozen justified candidate and explicit approval
  to consume its reserved row. This ledger never supplies that approval.
