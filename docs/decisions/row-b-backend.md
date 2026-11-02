# Decision: which backend computes row B

Date opened: 2026-11-02. Status: **engine chosen; featuretools cross-check blocked by the
environment** (see finding below). The owner can revisit by running the probe in a
separate pandas-2 environment.

Spec §7 pre-registers the row-B grid and allows either backend: featuretools DFS with
cutoff times, or the project's own signal engine over the same grid. Fallback to the
engine is permitted on either trigger: featuretools fails the leakage test, or it does
not finish the full grid on the owner's laptop within 8 hours of compute. The spec also
says "plan the engine path from the start; the grid is the same either way".

## What is in place

- `signal_factory.routine.grid.routine_grid()`: the grid as 1,626 signal specs
  (3 windows × (4 keys × (count + time_since_last + 4 aggregations × 31 numeric columns)
  + 38 count_distinct cells)). Tested for size, membership and that no cell uses the
  label or its own grouping key.
- `signal_factory.routine.compute.compute_row_b()`: computes the grid with the signal
  engine in float32 batches of 150 window expressions, joined in DuckDB. Inherits the
  engine's leakage property test (Hypothesis, 200 examples, `entity_key="uid"`).
- `scripts/try_featuretools.py`: the featuretools attempt, installed through the
  dependency group `routine-ft`. **It is a probe, not the full grid**: 2 keys × 3 windows
  × 4 primitives on a sampled cutoff set, every probe cell compared with the engine. A
  "finished within the time box" verdict from it is an extrapolation from the probe's
  wall time, and must be recorded as such.

## Finding 2026-11-02 (builder, synthetic data, this environment)

| Item | Value |
|---|---|
| featuretools / woodwork / pandas | 1.31.0 / 0.31.0 / 3.1.0 |
| Can featuretools build an EntitySet here? | **No.** `DataFrame.ww.init()` returns with `ww.schema is None` under pandas 3.1.0, on a two-column frame; `EntitySet.add_dataframe` then raises `WoodworkNotInitError`. woodwork declares `pandas >= 2.0.0` with no upper bound. |
| Probe finished within the time box? | Not applicable; it cannot start. The script exits 3 on this condition. |
| Sampled cells matched the engine? | Not measurable here. |
| featuretools wall time | – |
| Engine wall time for the full grid on real `history.parquet` | *(owner fills in after `compute_row_b`)* |
| **Backend used for row B** | **Signal engine** (`compute_row_b`). |

Rationale: the engine path was pre-registered as an acceptable backend for the identical
grid, it carries the leakage property test the spec names as the acceptance gate, and
featuretools is not runnable against the project's pinned data stack. This is a stronger
condition than the two spec triggers (it cannot finish because it cannot start) and is
reported as such. The report will say row B was produced by the signal engine and that
the featuretools cross-check was not performed because of the pandas 3 incompatibility.

If the owner wants the cross-check anyway: create a separate environment with
`pandas<3` and featuretools, run `scripts/try_featuretools.py` there on
`data/split/history.parquet`, and record its exit code, wall time and any mismatches
here. Either result leaves the engine as the backend unless a mismatch reveals an engine
bug, which would be a finding against the engine, not a reason to switch.
