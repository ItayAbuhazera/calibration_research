# Research OS

This vault is organized around a scientific chain:

**Evidence → Observation → Failure mode → Hypothesis → Experiment → Idea / Kill**

The goal is not to archive everything you read. Create notes only when something
changes what you believe, exposes a failure, creates a testable hypothesis, or
alters an experiment.

## Core rules

1. No idea without a benchmark.
2. No hypothesis without a falsifiable test.
3. No claim of novelty without a prior-art audit.
4. Store negative results and killed ideas explicitly.
5. Link every idea back to concrete observations and experiments.
6. Prefer matched operating-point comparisons (e.g. same risk) over raw-score comparisons.
7. Separate evidence from interpretation.

## Recommended workflow

- Capture raw material in `00_Inbox`.
- Convert meaningful papers into concise notes in `01_Papers`.
- Extract reusable empirical facts into `02_Observations`.
- Promote recurring failures into `03_Failure_Modes`.
- State falsifiable mechanisms in `04_Hypotheses`.
- Pre-register / log runs in `05_Experiments`.
- Create `06_Ideas` notes only when an idea has:
  - a phenomenon,
  - a benchmark,
  - a decisive experiment,
  - a novelty threat.
- Move dead ideas to `07_Killed_Ideas` with the reason preserved.
- Every week, synthesize across projects in `08_Weekly_Synthesis`.

## Current high-value question

A useful current pattern is:

> Geometry can contain accuracy-relevant information under corruption, yet a
> validation-fitted label-free reliability rule may fail to identify when that
> information should override the head.

That is an observation, not yet a paper claim. Two open caveats: no null baseline
separates "geometric information" from "any second disagreeing predictor", and
the gate run deviated from its pre-registered `k_vote`.

## Literature rule

The vault now includes a curated reading map in
`01_Papers/00 Paper Map - Geometric Uncertainty to Actionable Confidence.md`.

Paper notes are deliberately selective. Add a paper when it is:
- an ancestor of the research line,
- a direct conceptual competitor,
- a benchmark source,
- or a bridge from uncertainty estimation to operational decisions.

Do not add papers merely because they appeared in a related-work section.

