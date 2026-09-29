---
type: killed_idea
date: 2026-09-29
project: Full-Vector Geometric Calibration
reason: killed by prior-art audit and a representational argument, not by an experiment
tags: [gnn, graph, provenance, T5, prior-art-audit]
---

# Graph-structured computation as a distinct object for a fixed-topology ResNet

## Original idea
Represent a sample's realized computation as a graph (architecture connectivity + sample-specific activations/contributions) and use a
GNN / graph transformer to diagnose, recover or repair final errors.

## Why it died
A ResNet's topology is fixed across samples, so a GNN over node features is a function of concatenated features at fixed positions —
expressible by a set/sequence model with depth/channel identity; there is no cross-network permutation symmetry to exploit (unlike
weight-space metanetworks). GNN ≠ CONTRIBUTION.

## Evidence that killed it
DeepProv (ACSAC 2025) and NeuroTrace (arXiv 2026) already use GNNs over per-input provenance graphs (adversarial detection; DeepProv adds
activation repair) without set/sequence controls; per-sample edge-contribution graphs are old (Topological Uncertainty, IJCAI 2021;
effective paths CVPR 2019); CRV (ICLR 2026) finds topology features matter least; ILGE finds layer-to-node assignment barely matters;
[[Representation Trajectories Matters]] finds no privileged order. Details: territory audit report §10, [[Prior Art Map - Internal Computation Recoverability and Selective Repair]].

## What remains useful
Per-sample edge quantities (residual-branch vs skip share, w·a contributions) as *features* for any decoder; the control package
(same information in DeepSets/transformer; true vs degree-preserving rewired vs complete graph; within-class edge shuffles).

## Conditions under which to revisit
A result where per-sample edge information beats a matched set/sequence decoder AND rewiring degrades the GNN, on our substrate.
