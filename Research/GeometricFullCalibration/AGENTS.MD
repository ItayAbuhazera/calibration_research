# AGENTS

- Prefer additive changes over invasive refactors.
- Preserve backward compatibility for the legacy paper workflow.
- Do new implementation work in the unified benchmark path unless explicitly asked otherwise.
- Keep legacy scripts runnable and their baseline behavior stable.
- Document calibration formulas in comments/docstrings near where they are applied.
- Do not change defaults of existing methods unless necessary for correctness.
- If defaults must change, state the reason and expected impact clearly.
- Surface assumptions explicitly in code comments and benchmark summaries.
- Record benchmark metadata (split, seed, checkpoint, method settings) clearly.
- Keep outputs comparable across methods by using aligned evaluation settings.
- Avoid output schema churn that breaks historical result comparisons.