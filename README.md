# signal-factory

A supervised research loop that asks an LLM to propose derived fraud signals from a data
dictionary, computes them without leaking the future, and measures, against a fair
baseline, whether they help a fixed learner on unseen later time periods.

Research question (milestone 1):

> Can an LLM use domain descriptions and relational data to construct derived signals
> that improve a fixed learner's fraud detection on unseen time periods, beyond the
> original features and routine automated aggregations?

A negative answer, reliably shown, counts as a result.

## Status

Design stage. See `docs/superpowers/specs/2026-11-02-signal-factory-design.md`.

## Data

IEEE-CIS Fraud Detection (Kaggle). The competition rules allow non-commercial,
academic use and forbid redistribution, so the dataset is never committed. Accept the
rules on Kaggle under your own account, then use the download script (to come) which
wraps the Kaggle CLI. Everything under `data/` is gitignored.

## Principles

- Pre-registered evaluation: thresholds and acceptance criteria are fixed before any
  candidate is evaluated.
- The proposer's data access is enforced in code, not by prompting.
- Public data only. No company data anywhere.
- Test-first.
