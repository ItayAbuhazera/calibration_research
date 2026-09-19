# Working in this workspace (for any AI coding/research agent)

Full reference: `RESEARCH_WORKSPACE.md` (read it before any non-trivial task
here — this file is only the fast-start summary).

## Canonical repos — do not confuse with their siblings

- `geometric/GeometricCalibration/` — Geometric Separation paper (JMLR 2023).
- `geometric/GeometricInternalCalibration/GeometricInternalCalibration/` — RGC
  paper (IJCAI-ECAI 2026 preprint). Ignore
  `GeometricInternalCalibration_1`, `-main`, `-main_1`, `*.zip`, and any
  `_backup_*` directory unless a finding genuinely exists nowhere else — if
  so, cite it but flag it as backup-sourced in whatever you write.
- `GeometricFullCalibration/` — active full-vector/decision/shift research.
  Keep its legacy paper-reproduction path and its unified-benchmark research
  path distinct; see its own `AGENTS.MD` for code-style rules.
- `ResearchBrain/` — the Obsidian vault holding scientific memory (renamed
  from `Research_Obsidian_Vault`). It is interpretation and pointers, not raw
  data: never copy result files into it, only cite exact artifact paths.

## Evidence hierarchy (always, no exceptions)

Paper PDF > repository artifacts > any prior summary/archaeology note (an
index, never a source). Every claim added must trace to one of the first
two, or be explicitly labeled interpretation/hypothesis. Do not silently
reconcile conflicting artifacts — record the conflict instead.

## After running or reading about an experiment, updates allowed

- A dated `ResearchBrain/05_Experiments/` card citing exact artifact paths
  and seeds.
- Extensions to `02_Observations/` or `03_Failure_Modes/` notes, but only
  with a concrete evidence source and a "what this does NOT establish"
  section.
- Updates to the relevant `10_Projects/` artifact ledger.
- Appending to (never rewriting) the current `08_Weekly_Synthesis/` note.

Not allowed, without the user explicitly asking: editing historical result
artifacts/checkpoints/logs in a project repo, overwriting a preregistration,
or rewriting past synthesis history instead of appending.

## Never auto-create or edit `06_Ideas/`

Ideas are the researcher's deliberate call, never a byproduct of running an
experiment or writing documentation. A recurring pattern across observations
is not grounds for a new Idea note — cross-link the existing notes instead
and mark the pattern as an open question, not a claim. If told not to touch
`06_Ideas/`, that means zero edits, including "just a link."

## Raw result / observation / hypothesis / killed direction

- **Raw result**: one run's numeric output, one exact artifact path — stays
  in the repo or a `05_Experiments/` card, never copied into prose.
- **Observation**: a distilled, evidence-scoped fact with a named source and
  explicit non-conclusions. Check the seed count before treating it as
  settled.
- **Hypothesis**: falsifiable, not yet resolved, has a kill and a go
  criterion — created deliberately, not auto-derived from an observation.
- **Killed direction**: a specific approach that failed under a *stated*
  protocol — never generalize the title beyond what was actually tested.
- **Idea** (`06_Ideas/`): a candidate future project, `novelty: unknown` until
  vetted — never created or edited as a side effect of other work.

See `RESEARCH_WORKSPACE.md` for the full folder taxonomy and integrity-label
conventions (`verified implementation issue` /
`suspicious artifact / likely mechanism` / `under-replicated` /
`insufficient evidence`).
