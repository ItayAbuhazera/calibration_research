# Decoder Panel v1 — durable master report / execution journal (2026-09-28)

This file is the persistent source of truth for the Decoder Panel v1 program (panel spec, G1-DP, N1a-DP, conditional N1b design).
Together with SLURM accounting (`sacct -j <ids>`) and the result manifests listed below, a new session must be able to reconstruct
the full state. Watchers are convenience only.

## 0. Status header (updated at every stage)

| Field | Value |
|---|---|
| Current stage | A — repository / dependency / cluster inspection |
| Git HEAD at start | `ec4bf50` (N1a results) |
| Push status | 4 N1a commits (3b2ab7b, 3d5ac7b, 5cefe65, ec4bf50) **pushed** to origin/main (fast-forward dfd137b..ec4bf50) |
| Frozen specs | none yet |
| Implementation commit | — |
| Snapshot | — |
| Jobs running | none |
| Next action | finish inspection of G1/N1a code, artifacts, cluster; draft Decoder Panel v1 spec |

## 1. Git verification log (Stage 0)

- 2026-09-28: `git status` clean; `git fetch origin`; origin/main = dfd137b; HEAD = ec4bf50; `origin/main..HEAD` = exactly the
  four expected N1a commits; `HEAD..origin/main` empty (no divergence); diff touches only N1a files plus the small submit-script
  log-directory fix in `atlas/g1_submit.py` (part of 5cefe65, expected). Pushed: `dfd137b..ec4bf50 main -> main`.

## 2. Discrepancies between the task prompt and the repository (§61 log)

| # | Assumed by prompt | Repository shows | Class |
|---|---|---|---|
| 1 | G1 fits a separate target-supervised readout per corruption family x severity cell | `atlas/g1_fit.py` + spec §3: ONE fit per (checkpoint, regime, arm, outer fold) pooling all 12 cells (T-8k x 1: every outer-train image appears once at a preassigned cell, `cell_assignment_local`); evaluation on all 13 conditions; 12-cell macro accuracy | Engineering-level for G1-DP (the prompt explicitly says: preserve the original unit). G1-DP HPO unit = checkpoint x outer fold x evidence arm x decoder family (pooled over cells). Label: TARGET-POOLED HPO — ACCESSIBILITY DIAGNOSTIC ONLY. |

## 3. Journal

- Stage A started 2026-09-28.
