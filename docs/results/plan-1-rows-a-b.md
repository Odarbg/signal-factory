# Plan 1, rows A and B on validation

Measured 2026-11-03, frozen parameters from commit 5287438 (127 leaves, learning
rate 0.02, 1,000 trees, feature_fraction 0.5, bagging_fraction 0.8,
min_child_samples 100), seeds 0–4 averaged, threshold chosen once per row on
the whole validation set at FPR ≤ 1%. Aggregate numbers only; row-level files
live under the gitignored `results/`. Validation has 97,976 rows and 3,942
fraud rows; dev has 347,363 rows and 11,766 fraud rows.

- **Row A**: the 431 raw columns (`signal-factory row-a`). 600 s wall.
- **Row B**: row A plus the 1,626 pre-registered routine aggregations (spec §7
  grid, computed by the signal engine, see `docs/decisions/row-b-backend.md`).
  2,057 features. 2,326 s wall for training; the grid itself took 1,585 s.

## Whole validation set

| Row | Threshold | Recall at FPR ≤ 1% | Actual FPR | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| A | 0.15858 | 0.5330 | 0.00999 | 0.6250 | 0.9278 |
| B | 0.11633 | 0.5375 | 0.00999 | 0.6218 | 0.9278 |

Row B catches 18 more of the 3,942 fraud rows than row A at the same false
positive budget (+0.45 points of recall) and loses 0.3 points of PR-AUC. The
routine grid is, on this split, close to a no-op for the frozen learner. No
paired bootstrap interval is given here: `run_row` writes metrics, not per-row
scores, so the interval is left to the plan 2 ablation runner, which must keep
the scores of every row it compares.

## Seen and unseen uids

"Seen" means the row's `uid` occurs in dev; a NULL `uid` counts as unseen. The
threshold is the whole-set one above, so the slices do not each sit at 1% FPR.

| Row | Slice | n | Recall | Actual FPR | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|---|
| A | seen | 38,103 | 0.3839 | 0.00043 | 0.6531 | 0.9620 |
| B | seen | 38,103 | 0.4185 | 0.00021 | 0.6570 | 0.9629 |
| A | unseen | 59,873 | 0.5601 | 0.01634 | 0.6354 | 0.9071 |
| B | unseen | 59,873 | 0.5592 | 0.01648 | 0.6293 | 0.9057 |

The grid's whole gain is on returning customers, where per-entity history
exists; on unseen uids it is flat to slightly negative. Six of the 1,626 grid
cells are constant by construction (`count_distinct(addr1)` under `uid` and
`card1_addr1`, since `addr1` is a component of both keys); they were
pre-registered and kept.

## Checks run on the real data

- Append-later-rows: 40 grid signals sampled with seed 0, history truncated at
  day 60 (223,738 rows) recomputed and compared with the full-history
  `row_b.parquet`: 0 mismatches. Values of past rows do not depend on later
  rows.
- `scripts/try_featuretools.py` on the real history exits 3 (featuretools
  cannot build an EntitySet under pandas 3.1); see the decision record.
- `final.parquet` was not opened by any plan 1 command.
