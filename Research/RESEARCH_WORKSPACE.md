# Research Workspace

This file is the canonical map of this workspace: which repo is which, what
`ResearchBrain` is and is not, and the rules that keep it trustworthy across
sessions and agents. `CLAUDE.md` and `AGENTS.md` are short operating-rule
summaries for AI agents; this file is the full reference they point back to.

## Canonical projects

### Confidence Estimation using Geometric Separation
Canonical code: `geometric/GeometricCalibration/`
Role: Original representation-space geometric uncertainty research (raw
pixel/feature distance to training examples, calibrated post-hoc).
Paper: "Uncertainty Estimation Based on Geometric Separation" (JMLR 2023).

### Semantic Geometric Calibration / RGC
Canonical code:
`geometric/GeometricInternalCalibration/GeometricInternalCalibration/`
Role: Randomized multi-layer/coordinate geometric calibration research
(RGCL/RGCC), replacing architecture-specific layer selection with randomized
sampling.
Paper: "Semantic Geometric Calibration in Randomized Neural Feature Space"
(IJCAI-ECAI 2026 preprint).

**Do NOT use these sibling directories as a source of canonical claims unless
explicitly needed and clearly flagged as non-canonical when you do:**
- `GeometricInternalCalibration_1`, `GeometricInternalCalibration-main`,
  `GeometricInternalCalibration-main_1`, any `*.zip`
- `_backup_unique_from_GeometricInternalCalibration_1_20260627/` and any other
  `_backup_*` directory

They are historical copies or backups and may silently diverge from the
canonical tree. If a finding exists *only* in a backup (this has happened —
see the composite-layer-selector failure in the RGC project's vault page),
cite it, but flag it prominently as backup-sourced in the note itself.

### Full-Vector / Decision / Shift Research
Canonical code: `GeometricFullCalibration/`
Role: Active research extensions after RGC — full-vector calibration,
decision-changing calibration, KCal comparisons, RGC shift/recoverability,
reliability-routing diagnostics. This is the **same repo** that ships the
official RGC paper code (`run_paper_experiments.py`,
`README_IJCAI_REVIEWERS.md`) *and* a separately-labeled research extension
(`Experiments/run_unified_benchmark.py`, `README_FULL_VECTOR_BENCHMARK.md`).
Keep these two paths distinct when writing anything about this repo — the
paper reproduction path and the unified-benchmark research path make
different claims and have different evidentiary status.
See also `GeometricFullCalibration/AGENTS.MD` for code-style rules specific to
that repo (additive changes, backward compatibility for the legacy paper
workflow, output-schema stability).

### Persistent Research Memory — ResearchBrain
Canonical location: `ResearchBrain/` (an Obsidian vault; renamed from
`Research_Obsidian_Vault` on 2026-09-15 for shorter day-to-day typing — if you
have that vault open in the Obsidian app, you will need to relink/reopen it
at the new path).

This is the cross-project **scientific source of truth for interpretation,
not for raw computation**. Raw computational artifacts (checkpoints, CSVs,
logs, notebooks) always remain in their own project repositories above and
are never copied wholesale into ResearchBrain. ResearchBrain notes cite exact
artifact paths; they do not duplicate the artifacts.

---

## What ResearchBrain is

An Obsidian vault with a fixed folder taxonomy, each folder holding one
distinct *kind* of scientific-memory object:

| Folder | Contains |
|---|---|
| `00_Inbox/` | The live dashboard: current priorities, open integrity items |
| `01_Papers/` | One note per paper: research question, method, published evidence, limitations, vault descendants |
| `02_Observations/` | Distilled, evidence-scoped measured facts |
| `03_Failure_Modes/` | Things that didn't work, with competing explanations |
| `04_Hypotheses/` | Explicit, falsifiable, not-yet-settled claims with kill/go criteria |
| `05_Experiments/` | Dated experiment cards — one per run/batch |
| `06_Ideas/` | Candidate future research directions — see restrictions below |
| `07_Killed_Ideas/` | Narrowly-scoped approaches that were tried and failed |
| `08_Weekly_Synthesis/` | Weekly rollups — append addenda, never rewrite prior weeks |
| `09_Benchmarks/` | Notes about shared benchmarks (e.g. CIFAR-C) and their caveats |
| `10_Projects/` | One page per project: research question, method, published contribution, repository-only findings, artifact ledger |

### Evidence hierarchy (always in this order)

1. **Paper PDF** — authoritative for what a paper claims, published scope,
   published limitations, headline results.
2. **Repository artifacts** — authoritative for unpublished ablations, failed
   directions, implementation details, integrity problems in historical
   outputs.
3. **Any prior archaeology/summary note** — an index into evidence, never a
   source by itself. If you didn't read the paper or the artifact yourself,
   don't cite it as if you did.

Every scientific statement in ResearchBrain must be traceable to (1) or (2),
or be explicitly labeled an interpretation/hypothesis. Do not silently
reconcile conflicting artifacts — record the conflict and move on.

### Integrity labels

When citing a repository-only finding, label it as one of:
`verified implementation issue` · `suspicious artifact / likely mechanism` ·
`under-replicated` (state n/seeds) · `insufficient evidence`.
Add `evidence_scope: unpublished_repository_analysis` to any observation/
failure-mode note whose evidence is repo-only, not paper-published.

---

## What is allowed to update after an experiment

After a run completes, an agent (or you) may:

- Add a dated experiment card to `05_Experiments/` (status:
  `completed` / `completed_exploratory` / `results_missing_from_repo`, etc.),
  citing exact artifact paths and seeds.
- Add or extend an `02_Observations/` note **only** when there is a concrete
  measured fact with an exact evidence source — every observation note must
  include a "what this does NOT establish" section.
- Add or extend a `03_Failure_Modes/` note when something did not work, with
  competing explanations listed — do not assert a single causal mechanism
  unless the evidence actually proves it.
- Update the relevant `10_Projects/` page's artifact ledger and
  repository-only-findings section.
- Append to (never rewrite) the current `08_Weekly_Synthesis/` note and the
  `00_Inbox/Research Dashboard.md` integrity queue.
- Update a `04_Hypotheses/` note's status if its kill/go criterion was just
  resolved by the new evidence.

What is **not** allowed, ever, without the user explicitly asking:
- Editing historical result artifacts, checkpoints, or logs inside a project
  repo.
- Overwriting a preregistration.
- Rewriting a past week's synthesis instead of appending an addendum.
- Deleting or silently editing an existing note's measured facts because a
  new run disagrees — record the conflict as a new note or a clearly-marked
  addendum instead.

## What must NOT automatically become an Idea

`06_Ideas/` is reserved for candidate future research directions, and is the
one folder documentation/archaeology work must never populate on its own
initiative:

- Do not create or promote a note into `06_Ideas/` as a byproduct of running
  an experiment, writing up results, or an archaeology/documentation pass.
  Ideas are created deliberately, typically by the researcher, after a
  dedicated ideation review — not automatically inferred from an interesting
  result.
- A recurring pattern across multiple observations/failure modes (e.g. "the
  signal looks useful but we don't know when to trust it") is not by itself
  grounds for a new Idea note. Capture it instead as cross-links between the
  existing Observation/Failure-Mode/Project notes (see `10_Projects/Research
  Lineage.md` for the pattern: record it as a recurring *question*, explicitly
  not yet a claim or a project).
- Never edit an existing `06_Ideas/` note's `novelty`, `top_tier_potential`,
  or verdict fields from a documentation-only or archaeology pass. Those are
  the researcher's calls.
- If a task instructs you not to touch `06_Ideas/`, that means the whole
  folder — no edits of any kind, including "just adding a related-work link."

## How to tell raw result / observation / hypothesis / killed direction apart

- **Raw result**: a specific run's numeric output tied to one exact artifact
  path (a CSV/JSON/log with seeds and config). Lives in the project repo, or
  is pointed to from a `05_Experiments/` card — never copied wholesale into
  ResearchBrain as prose.
- **Observation** (`02_Observations/`): a distilled, evidence-scoped factual
  claim — "X was measured to be Y, per evidence Z" — with an explicit "what
  this does NOT establish" section, `evidence_strength`/seed-count context,
  and links to its project and related notes. One raw result rarely earns a
  strong observation by itself; check the seed count before treating a number
  as settled.
- **Hypothesis** (`04_Hypotheses/`): an explicit, falsifiable, not-yet-tested
  claim with a stated kill criterion and go criterion. Created deliberately
  when an observation raises an open question worth testing — not
  auto-generated from an observation note.
- **Killed direction** (`07_Killed_Ideas/`): a narrowly-scoped approach that
  was tried and failed under a *specific, stated* protocol. State exactly
  what was killed (representation, dataset, fitting protocol) and under what
  conditions it could be revisited — never generalize to "X doesn't work" in
  the title or body.
- **Idea** (`06_Ideas/`): a candidate future research direction, marked
  `novelty`/`top_tier_potential: unknown` until vetted. See restrictions
  above — this is the one type this workspace's day-to-day/documentation work
  must never auto-create.

## Quick pointers

- Reading map / paper triage: `ResearchBrain/01_Papers/00 Paper Map - Geometric Uncertainty to Actionable Confidence.md`
- Current priorities: `ResearchBrain/00_Inbox/Research Dashboard.md`
- Cross-project lineage and the recurring open questions: `ResearchBrain/10_Projects/Research Lineage.md`
