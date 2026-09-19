---
type: killed_idea
date: 2026-09-15
project: rgc-shift
reason: pre-registered label-free gate captured no usable oracle headroom
status: killed_protocol_contingent
integrity_flags: [protocol-contingent-k_vote]
tags: [geometry, gate, negative-result]
---

# Simple neighbourhood-statistics gate for geometric correction

## Original idea

Use head-neighbour agreement, radius, neighbour margin, and neighbourhood
concentration to decide when a geometric predictor should override the head.

## Why it died

On CIFAR-100-C, substantial oracle headroom existed but the fitted gate captured
approximately zero or negative headroom and did not beat the head or published
beta=30 arm.

## Evidence that killed it

[[2026-09-15 RGC Shift Recoverability]]

## Protocol-contingent (k_vote) - the kill is not yet final

The killing run recorded `k_vote=200`; the pre-registration pinned `k_vote=50`.
Head-neighbour agreement and neighbourhood concentration - two of the four gate
features - are averaged over `k_vote` neighbours, so at 200 the features may be
too global to carry the signal the gate needed. **[inference]**

Until the gate arm is re-run at `k_vote=50`, this note records a
**protocol-contingent kill**: the idea as *executed* failed; the idea as
*specified* has not been tested. Do not cite it as a clean negative result.

## What remains useful

The negative result exposed a sharper question: why is complementary geometric
information present but not observable through the current label-free features?

## Conditions under which to revisit

1. **Mandatory first**: re-run the specified gate at `k_vote=50` on the same
   seeds. If it still captures nothing, the kill becomes unconditional.
2. Otherwise, only with genuinely new observables or a new invariance argument.
   Do not revisit by tuning thresholds on corruptions.
