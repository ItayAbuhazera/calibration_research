---
type: paper
status: read_html_v3_and_code_readme
year: 2026
venue: "NeurIPS 2026 (author-announced in the eyali123/ILGE README; not verified against proceedings); earlier ICLR 2026 GRaM workshop version seen only in a search snippet (UNVERIFIED)"
url: https://arxiv.org/abs/2603.22665
tags: [multi-layer, llm, gnn, cayley-graph, prediction-information]
---

# Improving LLM Final Representations with Inter-Layer Geometry (Ulanovski, Blyachman, Bechler-Speicher)

arXiv 2603.22665 (v1 2026-03-24, v3 2026-05-13). Code: https://github.com/tomulanovski/ILSE ; package https://github.com/eyali123/ILGE. Read 2026-09-28 from the arXiv HTML (v3) and READMEs; the PDF was not parsed.

## Why this matters to me

Strong evidence that multiple internal layers carry complementary **task-predictive** information in frozen LLMs; a candidate evidence channel, not a reliability or action result.

## Core contribution

Mean-pool each layer's hidden states (z_ℓ = mean over tokens; layer counts 25/27/33 for Pythia-410M/Gemma2-2B/Llama3-8B suggest the embedding output is included — inference). Treat layers as graph nodes; a GNN over a fully connected graph (FC-Encoder) or a 4-regular, logarithmic-diameter Cayley graph of SL(2, ℤ_n) (Cayley-Encoder) aggregates them (≈ 0.1 % extra parameters).

## Benchmark / task

13 MTEB tasks (5 classification: Banking77, Emotion, MTOP Domain/Intent, Poem Sentiment; 8 STS), 9 LLMs incl. the Pythia suite. Shift-like test only STSBenchmark → STS12–16 zero-shot.

## Strongest result

Up to +40 pp over baselines; beats Last-Layer, Best-Layer, MLP-on-last-layer, ELMo-style Weighted, DWAtt, DeepSet and LoRA; few-shot gains from ≈ 8 samples per label.

## Surprising observation

Layer-to-node assignment barely matters (< 2 pp SD over 10 permutations) and learned layer-position encodings **hurt**; GNNExplainer attribution: all layers contribute roughly equally.

## Failure / limitation

No generative experiments; no calibration/reliability analysis; token×layer variant (L×T nodes) only for classification; single runs.

## What this changes in my beliefs

Multi-layer prediction information is established for frozen LLM classification; it does **not** establish reliability information (is the output wrong?) or action information (which intervention helps?). Compact aggregators matter — consistent with G1's finding that full-rank readouts are estimation-limited.

## Related observations

[[A full-rank penultimate target readout does not reproduce the compact layer3 probe increment in ResNet-101]].

## Novelty relevance

PHENOMENON ALREADY ESTABLISHED for prediction information; COMPONENT for any future reliability/action study.

## Follow-up

None scheduled (candidate "N2" in the G1 next-step synthesis, not authorized).
