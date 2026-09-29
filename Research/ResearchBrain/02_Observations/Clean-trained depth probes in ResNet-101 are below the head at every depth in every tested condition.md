---
type: observation
status: open
date: 2026-09-29
project: Full-Vector Geometric Calibration
evidence_strength: 2
tags: [layer-probes, overthinking, depth, cifar-100-c, stage0]
---

# Clean-trained depth probes in ResNet-101 are below the head at every depth in every tested condition

## Observation
For the end-to-end CIFAR-100 ResNet-101 (checkpoints 2 and 4), linear probes at 12 depths (layer1.0 … layer4.2; fit on the network's
own 45k training images, temperature-scaled) have accuracy that rises with depth and stays below the native head at every depth in
every condition reported, e.g. checkpoint 2: clean layer3.22 −3.2 pp, layer4.0 −1.0, layer4.1 −0.1; gaussian_noise_s5 layer3.22 −5.6;
fog_s3 layer3.22 −3.8 (checkpoint 4 similar: clean layer3.22 −3.4; fog_s3 −5.3). Layer4.1/4.2 occasionally reach ±0.4 pp of the head.
At population level there is no "destructive overthinking" gain from any single earlier probe in this model.

## Evidence
`GeometricFullCalibration/results/stage0/report/stage0b_layer_probe_eval.json` and `stage0b_macro_by_layer.json` (Stage 0b; already
reported in [[2026-09-22 Stage 0 Probe-Logit Increment Study]]). Probe definitions: `results/layer_pilot/checkpoint_seed{2,4}/frozen_state.json`.
Two checkpoints; 12 exposed development cells + clean. Read (not recomputed) during the 2026-09-29 territory audit.

## What it does NOT establish
That no example is correct earlier and wrong later (per-example flips are not measured here); anything about other probe recipes,
pooling (see Atlas 2×2 results), models, or reserved families; that earlier layers lack correct-class information.

## Possible mechanisms
End-to-end training aligns late features with the head; single-layer linear readouts of earlier blocks are weaker readers; contrast with
frozen-pretrained settings where intermediate heads win ([[Intermediate Layer Classifiers for OOD generalization]]).

## Related failure modes
[[Oracle any-layer recoverability counts overstate correction headroom]].

## Related hypotheses
[[H-EVO-01 True-class evidence suppression is distinguishable from base-rate depth flips in ResNet-101 under corruption]]; [[H-SELREP-01 Multi-depth internal evidence discriminates repair from harm beyond output evidence under held-out corruption]].

## Decisive next test
POC-T1 in the territory audit report §15 (not authorized).
