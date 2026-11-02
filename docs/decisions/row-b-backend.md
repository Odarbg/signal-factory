# Decision: which backend computes row B

Date opened: 2026-11-02. Status: **pending the owner's time-boxed run on real data.**

Spec §7 pre-registers the row-B grid and allows either backend: featuretools DFS with
cutoff times, or the project's own signal engine over the same grid. Fallback to the
engine is permitted on either trigger: featuretools fails the leakage test, or it does
not finish the full grid on the owner's laptop within 8 hours of compute.

## What is in place

- `signal_factory.routine.grid.routine_grid()`: the grid as 1,626 signal specs
  (3 windows × (4 keys × (count + time_since_last + 4 aggregations × 31 numeric columns)
  + 38 count_distinct cells)). Tested for size, membership and that no cell uses the
  label or its own grouping key.
- `signal_factory.routine.compute.compute_row_b()`: computes the grid with the signal
  engine in batches of 150 window expressions. Inherits the engine's leakage property
  test.
- `scripts/try_featuretools.py`: the time-boxed featuretools attempt, installed through
  the optional dependency group `routine-ft` (`uv sync --group routine-ft`). It compares
  a 2,000-row sample of count cells with the engine and exits non-zero on mismatch or
  timeout.

## Outcome (owner fills in)

| Item | Value |
|---|---|
| featuretools version | |
| Finished the probe within the time box? | |
| Sampled cells matched the engine? | |
| featuretools wall time | |
| Engine wall time for the full grid (`compute_row_b` on `history.parquet`) | |
| Backend used for row B | |

Rationale for the backend chosen:

(to be written after the run; per spec §7 either outcome is acceptable and the report
states which backend produced row B)
