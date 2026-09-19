---
type: observation
status: open
date: 2026-09-15
project: rgc-shift
evidence_strength: 3
tags: [blend, clean-fit, negative-result]
---

# Clean-selected probability blending collapses to head-only

## Observation

Every clean/validation blend sweep selected alpha=0, i.e. the head alone.

## Interpretation

The kNN probability distribution is not useful as a globally blended
probability source under the fitted criterion. Any useful geometric signal
appears to be conditional / sparse rather than globally blendable.

## Related failure mode

[[Static global blending cannot exploit sparse geometric complementarity]]
