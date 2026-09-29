---
type: hypothesis
status: proposed
project: Full-Vector Geometric Calibration
benchmark: CIFAR-100 / CIFAR-100-C (12 exposed development cells), ResNet-101, checkpoints 2 and 4
novelty: unknown (territory audit 2026-09-29: WATCH — phenomenon taken; controlled characterization possibly open)
tags: [H-EVO, class-evidence-evolution, overthinking, suppression, T1]
---

# H-EVO-01 — True-class evidence suppression is distinguishable from base-rate depth flips

## Formal statement
Claim scope: end-to-end ResNet-101; 12 cached depth probes; event S = true class top-1 at ≥ 2 consecutive depths among the last 6 and not
top-1 at the head. rate(S | final error) exceeds the matched event among final-correct examples (a non-final class top-1 at ≥ 2
consecutive late depths, confidence-bin matched) and label-permutation / probe-shuffle nulls.

## Why it follows from evidence
Overthinking is documented in vision (SDN; under CIFAR-C, Mehra et al.); Atlas shows deep-layer candidates are right on some errors.

## Competing explanation
Chance agreement of weaker readers: all earlier probes are below the head everywhere ([[Clean-trained depth probes in ResNet-101 are below the head at every depth in every tested condition]]); transient flips also occur in correct predictions (Wrong Before Right).

## Benchmark
Existing layer-pilot cache; no fitting.

## Baselines
Raw oracle count (any depth correct); matched correct-example event rate; nulls.

## Decisive experiment
Primary type: descriptive. Contrast: rate(S | error) − matched control, per family and checkpoint. Uncertainty tested: representation
information retained by a statistic. Exposure: exposed development cells only.

## Outcome-to-decision matrix

| Possible outcome | Explanation supported/weakened | Still unresolved | Next decision |
|---|---|---|---|
| ≥ 5 pp above control and nulls, both checkpoints, ≥ 3/4 families | suppression is a distinguishable event | whether it is selectable (H-SELREP-01) | feed as feature into H-SELREP-01 only |
| within ±2 pp of control/null | chance agreement | — | close T1 |
| in between | — | precision | inconclusive; no further variants |

## Kill criterion
rate(S | error) within ±2 pp of the matched control or null.

## Go criterion
As row 1 (≥ 5 pp). Descriptive only — a positive result is not a contribution by itself.

## Prior-art threats
Shallow-Deep Networks; [[Understanding the Robustness of Multi-Exit Models under Common Corruptions]]; [[Vertical Fusion - Recoverability in ViT Hierarchies]];
CALRD (direction signature in MLLMs); Corruption Depth (Neural Networks 2024, not read in full). Not authorized.
