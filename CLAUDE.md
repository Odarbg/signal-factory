# signal-factory: working notes for Claude

## What this is

A research project: can an LLM propose derived fraud signals that beat routine automated
aggregations, measured fairly on unseen later time periods? Read, in order:

1. `docs/superpowers/specs/2026-11-02-signal-factory-design.md` (the approved design,
   pre-registered evaluation).
2. `docs/superpowers/plans/2026-11-02-signal-factory-plan-1-foundations.md` (the current
   plan; tasks have checkboxes).
3. Background and the wider programme: `~/Projects/personal/ideas/ai-os/` and
   `~/Projects/personal/ideas/2026-11-02-high-level-plan.md`.

## Current state

- Plan 1 (foundations) closed 2026-11-03 on the real data; results in `docs/results/`.
- Plan 2 written 2026-11-03: `docs/superpowers/plans/2026-11-03-signal-factory-plan-2-proposer-and-final.md`
  (proposer, selection, final evaluation, report; 14 tasks).
- Next: execute plan 2 from Task 1 with `superpowers:executing-plans` (native execution
  chosen by the owner on 2026-11-03) unless the plan file shows checked boxes further
  along. Collaboration mode for all of plan 2 was also chosen on 2026-11-03: Claude writes
  tests and implements, commits on main; do not re-ask per task. Owner gates stay:
  confirming `domain_notes.md` before arm D runs, and the real `propose`, `select`,
  `run-final` commands.
- Local data: Kaggle CSVs in `~/.kaggle/`, copied to `data/raw/`; `data/split/`,
  `data/parquet/all.parquet` and `data/features/row_b.parquet` exist on the owner's laptop.

## How the owner works

- TDD, always. Before each task, ask which collaboration mode: the owner writes tests
  and Claude implements; Claude writes tests and the owner implements; or review-only.
- Commit per task, small commits, messages in the plan's style.
- GitHub account: `Odarbg` (personal). Use `GH_TOKEN=$(gh auth token --user Odarbg) gh …`;
  never `gh auth switch`.
- `make check` (pytest, ruff, mypy strict) must pass before every commit.

## Hard rules

- `data/` and `results/` are gitignored. Never commit dataset files or any row-level
  derived data. Aggregate numbers go in `docs/results/`.
- `isFraud` is never an input to any signal, filter or routine aggregation.
- Nothing in plan 1 reads `data/split/final.parquet`. The final period is touched once,
  at the end of plan 2, by the ablation runner.
- The pre-registered values in the design (seeds, FPR, split days, acceptance lift,
  bootstrap resamples, row-B grid) are not tuned. Changing one is a new experiment and
  must be recorded as such in the design.
- No Osprey data, numbers, identifiers or internal hypotheses anywhere in this repo or
  in prompts. Generic, public-source fraud knowledge only.
- The proposer (plan 2) may read only the dev period, the data dictionary and the
  domain notes. Enforced in code, not by prompting.

## Dataset

IEEE-CIS Fraud Detection, Kaggle, train split only. The owner accepts the competition
rules on Kaggle and runs the download script; Claude does not need Kaggle credentials.
