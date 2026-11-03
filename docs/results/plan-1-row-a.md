# Plan 1, row A: baseline on validation

Measured 2026-11-03 on the IEEE-CIS train split, after the parameter freeze
(commit 5287438). Aggregate numbers only; the row-level files live under the
gitignored `results/row_a/`.

## Data

| Item | Value |
|---|---|
| Rows after convert | 590,540 (437 columns: 434 raw + `day`, `card1_addr1`, `uid`) |
| Days covered | 1–182 |
| dev (days 1–98) | 347,363 |
| gap (99–105) | 39,508 |
| validation (106–138) | 97,976 |
| final (146–182) | 105,693 (not read in plan 1) |
| Rows with NULL `uid` | 66,794 (11.3%) |

## uid collision rate (spec §10)

`signal-factory stats data/parquet/all.parquet`

| Item | Value |
|---|---|
| Distinct uids | 199,070 |
| uids whose rows carry more than one `card2..card6` tuple | 4,333 |
| Collision rate | 2.18% |

Low; the uid guard stays as defined in spec §4.

## Row A: raw columns, frozen LightGBM, five seeds

`signal-factory row-a data/split results/row_a` (431 features, seeds 0–4,
params from `src/signal_factory/baseline/params.json`; 600 s wall on the
builder's laptop).

Threshold chosen once on the whole validation set at FPR ≤ 1%: 0.15858.

| Slice | n | Recall at that threshold | Actual FPR | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| validation (all) | 97,976 | 0.5330 | 0.00999 | 0.6250 | 0.9278 |
| seen uids | 38,103 | 0.3839 | 0.00043 | 0.6531 | 0.9620 |
| unseen uids (incl. NULL) | 59,873 | 0.5601 | 0.01634 | 0.6354 | 0.9071 |

"Seen" means the row's `uid` occurs in dev; a NULL `uid` counts as unseen.
The threshold is set on the whole validation set, so the two slices do not
each sit at 1% FPR: the model is much more conservative on returning
customers (0.04% FPR) and spends the budget on unseen ones (1.63% FPR). This
is the baseline every later row is compared against at the same frozen
threshold rule.
