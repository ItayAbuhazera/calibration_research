---
type: paper
status: abstract_verified_protocol_from_html_render_unverified_against_pdf
year: 2025
venue: ICLR 2025
arxiv: 2504.05461
access_date: 2026-09-24
tags: [intermediate-layers, linear-probe, ood-generalization, last-layer-retraining, cifar-c]
---

# Intermediate Layer Classifiers for OOD generalization (Uselis & Oh, ICLR 2025)

Authors: Arnas Uselis and Seong Joon Oh. arXiv:2504.05461; ICLR 2025 proceedings (`proceedings.iclr.cc`, paper hash 71c3451f6cd6a4f82bb822db25cea4fd). Referenced in [[2026-09-24 Stage 0 Reconciliation with Uselis and Oh]] and used as motivation for the frozen [[2026-09-24 Regime-Map Pilot]].

## Reading status (honest record)
* **Abstract:** confirmed by a web search result (2026-09-24): the paper questions last-layer representations for OOD generalization and introduces Intermediate Layer Classifiers (ILCs), linear probes on intermediate-layer representations of frozen pretrained models; it reports that intermediate layers often generalize substantially better than the penultimate layer, and that zero-shot OOD generalization with earlier layers approaches the few-shot performance of retraining on penultimate representations.
* **The PDF could not be read here.** The arXiv PDF was downloaded but its text is not extractable on this cluster (the extractor returned only figure labels). The protocol below was obtained from the **arXiv HTML render (`arxiv.org/html/2504.05461`) through a summarizing fetch tool**, which returns paraphrase with short quotes, not the verbatim text; treat every item as *reported by that summary, not verified against the paper*. Items the summary said the paper does not state are listed as not stated. Nothing here is a theorem or a number I checked.

## Probe protocol, as reported by the fetch summary (unverified against the PDF)
* **Pooling.** ResNets: the model's adaptive average pooling after the residual blocks (GAP); ViTs: the [CLS] token followed by layer normalization. Any additional spatial/token aggregation: not stated.
* **Probe form and training.** An affine map from the layer representation to logits, `ILC_l(x) = W_l r_l(x) + b_l`; cross-entropy loss; the summary states probes for all layers are trained simultaneously; Adam optimizer; 100 epochs. Batch size and stopping criteria beyond the epoch count: not stated.
* **Regularization and selection.** ℓ1 regularization. Hyperparameter grids: zero-shot (η, ℓ1) ∈ {1e-4, 1e-3, 1e-2} × {0, 1e-3, 1e-2}; few-shot (η, ℓ1) ∈ {1e-4, 1e-3, 1e-2} × {0, 1e-4, 1e-3, 1e-2}. Selection is on an **OOD validation set** (stated for both settings). The layer is chosen as the best validation-accuracy layer among `l ≤ L−2` (the last two layers are excluded).
* **Base models.** Publicly available pretrained weights, kept **frozen** (no fine-tuning); ResNets from TorchVision; ViTs from public repositories. The base models' training recipes and augmentation: not stated.
* **Data and baseline.** CIFAR-10-C and CIFAR-100-C are among the datasets (the summary mentions a noise-corruption setting); few-shot probes are trained on half of the OOD test set and evaluated on the other half. Baseline: last-layer retraining (DFR). The one explicit CIFAR-C claim the summary quotes is for **CIFAR-10-C**: "+2%p to +5%p improvements in mean accuracies when the best layer is used, instead of the last layer". Input resolution for CIFAR, the exact corruptions, and any CIFAR-100-C numbers: not stated by the summary (so the task's CIFAR-100-C description is **not confirmed** by this reading).

## What this changes for our reconciliation (interpretation, labelled)
* Their base models are **frozen public pretrained networks**, not CIFAR-trained-from-scratch ResNet-101; the regime-map pilot's state (a) is the closest match. This explanation was not in the Phase 1B list because the task text did not state it.
* Their probes are L1-regularized with Adam and select layer and hyperparameters on OOD validation labels; ours are L2/L-BFGS with the layer-pilot recipe and λ edge selections. The recipe difference is therefore larger than a hyperparameter grid.
* Their few-shot variant trains probes on target data; the zero-shot variant selects on OOD validation. Our cached target-selected rows do not beat the native head, which argues against selection data alone.

## What it does NOT establish
Not verified against the paper text; no claim about CIFAR-100-C, our ResNet-101 checkpoints, or our recipe follows from it.
