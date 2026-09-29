---
type: paper
status: abstract_verified_table_via_subagent
year: 2021
venue: ICSE 2021 (peer-reviewed)
url: https://arxiv.org/abs/2103.02371
tags: [selfchecker, internal-layers, alternative-prediction, cifar-100, prior-art-audit]
---

# Self-Checking Deep Neural Networks in Deployment (SelfChecker)

Xiao, Beschastnikh, Rosenblum, Sun, Elbaum, Lin, Dong. Abstract verified by the lead 2026-09-29; Table III numbers read from the PDF by
a subagent (lead's PDF text extraction failed to confirm them — treat as subagent-reported).

## Why this matters to me
Closest vision precedent for T2/T3: internal layers both **flag** an error and **propose an alternative class**.

## Core contribution
Per-layer class-conditional KDE infers a class at each layer; alarm if selected layers disagree with the output; "SelfChecker also
provides advice in the form of an alternative prediction" (abstract). Layer subsets chosen by search.

## Benchmark / task
MNIST, FMNIST, CIFAR-10, CIFAR-100; ConvNet, VGG-16, ResNet-20; self-driving scenarios. Clean test data (no corruption shift).

## Strongest result
Correct alarms on 60.56 % of wrong predictions, false alarms on 2.04 % of correct ones (abstract).

## Surprising observation
Subagent-reported (Table III, RQ2): advice **lowers** CIFAR-100 accuracy (VGG-16 66.79→66.16; ResNet-20 69.52→68.85), attributed by the
authors to 100 classes and few validation samples per class.

## Failure / limitation
No distribution shift; net accuracy only (no W/H split); no output-only matched gate.

## What this changes in my beliefs
The within-model "alternative from internal layers" idea has a documented negative precedent in exactly the 100-class regime; consistent
with our [[2026-09-21 Fixed Deep Candidate Gate Study]] (H:W ≈ 3.8:1).

## Novelty relevance
SAME PROBLEM for T2 (in-distribution), partial T3.

## Follow-up
SelfChecker++ (TDSC 2022) not examined — check before any claim.
