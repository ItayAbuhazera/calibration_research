---
type: killed_idea
date: 2026-09-29
project: Full-Vector Geometric Calibration
reason: CURRENTLY UNJUSTIFIED — reduces to supervised selective prediction / contextual bandit; killed by formal analysis and prior art
tags: [rl, optimal-stopping, early-exit, T-RL, prior-art-audit]
---

# RL over depth for fixed-network prediction correction

## Original idea
Treat depth as time; at each layer choose CONTINUE / EXIT / KEEP / TRUST-EARLIER / INTERVENE / ABSTAIN with an RL/DRL policy.

## Why it died
For a frozen deterministic network, KEEP/TRUST-EARLIER/ABSTAIN do not change future states; every counterfactual reward is observed
offline from one labelled forward pass; depth adds computed features but no information about y. The problem is a POMDP with
action-independent observations = full-information optimal stopping, and with no compute cost it collapses to supervised selective
prediction on the full history. INTERVENE has known deterministic dynamics → contextual bandit / planning. The real difficulty under
shift (P_test(y|H) ≠ P_train(y|H)) is not something RL addresses.

## Evidence that killed it
Learning to Stop While Learning to Predict (ICML 2020: stopping learned by imitation of a closed-form oracle); EENet, CALM, Kubaty et al.
(exit policies from oracle labels); BlockDrop (its RL is "a single-step MDP … contextual bandit"); label-free bandit exits under shift
use confidence as reward. Territory audit report §11.

## What remains useful
Supervised action-value / selective-prediction formulations (H-SELREP-01); risk control for switching under shift (open per Fast yet
Safe, NeurIPS 2024).

## Conditions under which to revisit
Multi-step, non-differentiable, compute-costly interventions where supervised and planning baselines are shown to fail — then "possible but method-driven".
