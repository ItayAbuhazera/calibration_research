---
type: hypothesis
status: not-supported-under-this-protocol
project: Full-Vector Geometric Calibration
benchmark: CIFAR-100 / CIFAR-100-C, ResNet-101, checkpoints 2 and 4 (development)
novelty: unknown
tags: [H-RESID, residual-readout, spatial-pooling, decision-utility]
---

# H-RESID-01 — A clean-fit residual readout of `layer3.22` evidence adds decision value beyond full-logit readouts under corruption

Decisive (and stopping) experiment: [[2026-09-21 Residual Evidence Study]]; frozen spec in repo `docs/residual_evidence_study_spec.md` §§0–13.

**Statement (tested combination).** With `q=softmax(B(z)+Wφ_F(x))`, `B` a clean-selected output-only anchor and `φ_F` a 100-d spatial (2×2) or class-radius evidence map of `layer3.22`, fit on 2 500 clean rows and selected by the frozen Decision policy, the
hidden-evidence procedure gains ≥ +0.50 pp over base under the 12 development corruption cells, ≥ +0.25 pp over VS, the anchor and the output-evidence Decision control, positive in each checkpoint and ≥ 3/4 families, with NLL/ECE ≤ 0.02 worse than native DAC and no clean regression (> 0.20 pp / 0.01 nats).

**Go:** all six criteria on development → held-out-checkpoint (1,3,5) confirmation. **Kill:** any failure → stop (no new seeds/layers/pooling/k/classifiers/gates).

**Status: kill criterion met (criteria 1–4 failed), 2026-09-21.** Measured +0.102 pp (control +0.118, VS +0.128, anchor +0.108). Scope: two development checkpoints, mean/2×2 pooling, K_c=5, 100-d compressed evidence, ridge-regularized linear residuals, 2 500 fit rows.
Not an information-theoretic null; the observed limit is finite-sample learnability under the clean fit budget. Related: [[H-LAYER-01 Selected internal layers add decision value beyond logits]].
