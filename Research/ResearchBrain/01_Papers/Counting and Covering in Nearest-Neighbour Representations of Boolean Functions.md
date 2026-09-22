---
type: paper
status: abstract_verified_theorem_pending
year: 2026
arxiv: 2609.23094
access_date: 2026-09-22
tags: [nearest-neighbour, prototypes, VC-dimension, boolean-functions, representation-limits, research-lead]
---

# Counting and Covering in Nearest-Neighbour Representations of Boolean Functions

**Martin Anthony.** arXiv:2609.23094 [cs.DM] (also cs.LG, math.CO), submitted 2026-09-19 (v1). Retrieved from the arXiv abstract page (primary source) on 2026-09-22.

## Reading status

**Abstract verified against the primary source; theorem-level reading is pending.** The abstract below was fetched directly from `https://arxiv.org/abs/2609.23094` on 2026-09-22 and is quoted verbatim. An attempt to fetch and read the PDF (`https://arxiv.org/pdf/2609.23094`, saved locally at `01_Papers/Counting and Covering in Nearest-Neighbour Representations of Boolean Functions.pdf`, 543,779 bytes) did not yield extractable text: this cluster has no `pdftotext`/`pdftoppm`/`poppler-utils`, and no `PyPDF2`/`fitz`/`pdfplumber` Python packages (consistent with the existing PDF-extraction limitation recorded for the AAAI-PCE source in the assistant's session memory, `project_normalization_and_layer_pilot.md`). **No theorem number, exact formula, exponential-separation claim, or VC-dimension rate below has been verified against the full text — do not cite one as established from this note.**
A second candidate source, a prior ChatGPT conversation link (`https://chatgpt.com/c/6ab26fcf-2e58-83ed-8bac-c2dfeaae0151`), returned HTTP 403 (private/authenticated) and was not read. Per the task's own instruction, an earlier assistant discussion is not a verified theorem source in any case, so nothing from that link is recorded here even indirectly.

## Abstract (verbatim, from the arXiv abstract page)

> We study the number of prototypes needed to represent Boolean functions by nearest-neighbour classification. There are two distinct settings: the prototypes may be arbitrary points of Euclidean space, or they may themselves be required to lie in the Boolean cube. For unrestricted prototypes, we strengthen a known lower bound for almost all Boolean functions. The bound applies simultaneously to nearest-neighbour voting rules with any number of voting neighbours, and substantially narrows the gap with the known general upper bound. We obtain a VC-dimension bound for classes with a bounded number of prototypes, and show that it is sharp in order in dimensions at least four. We then study Boolean prototypes, beginning with symmetric threshold functions. A connection with covering designs expresses the minimum number of prototypes at every threshold level exactly in terms of a covering number, and leads to further exact results for related monotone functions, including disjunctive extensions and a characterisation of when a representation with a single negative prototype is possible. For a uniformly random Boolean function, the Boolean nearest-neighbour complexity, as a proportion of the cube, is asymptotically close either to one half or to one, with explicit limiting probabilities. In particular, almost every Boolean function requires at least approximately half as many prototypes as there are points in the cube, and one half is the largest proportion for which such a lower bound holds. Finally, we consider arbitrary symmetric Boolean functions. Their Boolean nearest-neighbour complexity is closely approximated by a weighted vertex-cover problem on paths. As a consequence, a uniformly random symmetric function typically requires prototypes amounting to 11/20 of the cube. This is much larger than the upper bounds known when the prototypes are allowed to lie anywhere in Euclidean space.

## Why this is a research lead, not an established solution

The abstract's headline distinction — prototypes unrestricted in Euclidean space vs. prototypes restricted to the Boolean cube — is exactly the axis our nearest-neighbour readouts sit on: [[2026-09-21 Representation Atlas Program]] and [[2026-09-21 Fixed Deep Candidate Gate Study]] both use real-valued (unrestricted) hidden-layer prototypes with exact kNN/kNN-voting, never a Boolean-cube-restricted representation. That match is suggestive, not diagnostic: the paper's object is exact Boolean-function representability on the cube, not neural classification accuracy, calibration, or corruption robustness on continuous image features. **It does not directly establish, weaken, or explain any measured result in this vault.** Any bridge from "how many prototypes are needed to represent a Boolean function exactly" to "how much decision-relevant information a fixed-size real-valued kNN readout can extract from a trained network's hidden layer" is an analogy to be built and tested, not a transferred result.

## Proposed questions only (not evaluated, not authorized as new experiments)

- Could the admissible prototype family (fixed bank size, fixed k, a single distance metric) be the limiting factor in a geometric readout, in the sense this paper's prototype-count bounds formalize for Boolean functions?
- Could a scalar/low-dimensional distance summary (the kNN vote / `p_geo`) discard usable task information that remains present in the hidden coordinates themselves, analogous to the gap the paper draws between Euclidean-unrestricted and cube-restricted prototype counts?
- What observational or theoretical evidence would distinguish a **representation** limitation (the hidden layer lacks the information), a **statistic** limitation (kNN counting discards it), a **readout** limitation (this gate family cannot use it), an **estimation** limitation (not enough clean-fit rows), and a **transfer** limitation (shift breaks a clean-fit relationship) for the negative results already on record?

These are open questions to motivate future theory work, cross-linked from [[Theory Plan - Decision Utility, Layers, Compression and Risk Control]]. **Boolean-cube representability results do not directly establish neural classification accuracy, calibration quality, or corruption robustness**, and this paper does not reopen [[2026-09-21 Fixed Deep Candidate Gate Study]], [[H-GATE-01 Candidate selection versus gate utility mismatch]], or [[H-ATLAS-01 Accessible correction information across layers pooling and metrics]].

## Next step (not performed here)

Obtain the full text through a channel where PDF text extraction works (a non-cluster machine or an OCR-capable tool), read the theorem statements listed as open in the reading status above, and only then update this note's `status` to reflect a verified theorem-level reading.
