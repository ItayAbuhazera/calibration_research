---
type: observation
status: supported_exploratory
date: 2026-04-25
project: full-vector-calibration
evidence_strength: 3
tags: [anchoring, argmax, tradeoff]
---

# High-confidence anchoring can block useful full-vector decision changes

## Observation

On CIFAR-100, `rgcl_tail_vector_scaling` achieved substantially better
top-label ECE than full-vector distance fusion, but had lower accuracy and worse
NLL.

Its argmax-change rate was approximately 0.00064 versus ~0.0824 for fusion.

## Method naming

Method naming resolved 2026-09-15 by author decision: the top-label specialist is called `rgcl_tail_vector_scaling` everywhere; `gc_dac` was an earlier informal name for the same method. This is a naming choice, not a re-check of the run outputs.

## Interpretation

Locking or strongly anchoring the predicted class can preserve top-label
calibration while preventing beneficial class changes that a full-vector method
would make.

## Broader lesson

"Preserve the head unless very certain" is not automatically a free lunch.
There is a real opportunity cost to preventing decision changes.

## Partly true by construction

An anchored method is *designed* not to move the argmax, so the near-zero
argmax-change rate (0.00064 vs 0.0824) is definitional, not a finding. The
empirical content is narrower: the changes it forbids were, on this benchmark,
net beneficial. That content rests on one exploratory CIFAR-100 batch.
