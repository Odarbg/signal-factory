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

Plan 1 (foundations) closed 2026-11-03: data pipeline, leakage-tested signal engine,
fixed LightGBM learner, pre-registered metrics, paired bootstrap, row A runner and the
row B grid, all run on the real data. Measured on validation (days 106–138) at FPR ≤ 1%:
row A (431 raw columns) recall 0.533, PR-AUC 0.625; row B (plus 1,626 routine
aggregations) recall 0.538, PR-AUC 0.622. uid collision rate 2.18%. Details in
`docs/results/plan-1-row-a.md`, `docs/results/plan-1-rows-a-b.md` and
`docs/decisions/row-b-backend.md`. Next: plan 2 (proposer, ablation runner, final
evaluation). Design: `docs/superpowers/specs/2026-11-02-signal-factory-design.md`.

## Reproducing rows A and B on validation

Nothing below reads `final.parquet`; the final period is touched once, in plan 2.

```sh
uv sync --all-groups
uv run python -c "from pathlib import Path; from signal_factory.data.download import download; download(Path('data/raw'))"
uv run signal-factory convert data/raw data/parquet/all.parquet
uv run signal-factory split data/parquet/all.parquet data/split
uv run signal-factory stats data/parquet/all.parquet            # uid collision rate
uv run python scripts/freeze_params.py data/split/dev.parquet   # once; commit params.json
uv run signal-factory row-a data/split results/row_a           # row A on validation
uv run python -c "from pathlib import Path; from signal_factory.routine.compute import compute_row_b; compute_row_b(Path('data/split/history.parquet'), Path('data/features/row_b.parquet'))"
uv run python -c "from pathlib import Path; from signal_factory.ablation.row_a import run_row; print(run_row('B', Path('data/split'), Path('results/row_b'), [Path('data/features/row_b.parquet')]))"
```

Optional featuretools comparison (time-boxed, spec §7): `uv sync --group routine-ft`
then `uv run python scripts/try_featuretools.py data/split/history.parquet`.

## Data

IEEE-CIS Fraud Detection (Kaggle). The competition rules allow non-commercial,
academic use and forbid redistribution, so the dataset is never committed. Accept the
rules on Kaggle under your own account, then use the download script
(`signal_factory.data.download`), which wraps the Kaggle CLI. Everything under `data/` is gitignored.

## Principles

- Pre-registered evaluation: thresholds and acceptance criteria are fixed before any
  candidate is evaluated.
- The proposer's data access is enforced in code, not by prompting.
- Public data only. No company data anywhere.
- Test-first.

## Development

```sh
uv sync --all-groups   # Python 3.12 via uv, all dependencies incl. dev tools
make check             # pytest, ruff, mypy --strict
```

`make test`, `make lint` and `make typecheck` run the three parts separately.
