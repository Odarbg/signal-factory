"""Choose the frozen LightGBM parameters once, on dev data only (spec §6, §10).

Train on days 1–84, hold out days 85–98, score PR-AUC for a fixed grid of six
candidates, write the winner to ``src/signal_factory/baseline/params.json`` and print the
table. Run once by the owner before any derived signal exists; the result is committed.

    uv run python scripts/freeze_params.py data/split/dev.parquet
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import average_precision_score

from signal_factory.ablation.features import base_features, labels
from signal_factory.baseline.learner import PARAMS_PATH, SEEDS, train_predict_single

TRAIN_END_DAY = 84
FIXED = {"feature_fraction": 0.5, "bagging_fraction": 0.8, "min_child_samples": 100}
GRID: list[dict[str, Any]] = [
    {"num_leaves": leaves, "learning_rate": lr, "n_estimators": n, **FIXED}
    for leaves in (31, 127)
    for lr, n in ((0.05, 500), (0.02, 1000))
]
# 2 × 2 = 4 above; the plan names six candidates, so add the two mixed-rate variants.
GRID += [
    {"num_leaves": 31, "learning_rate": 0.02, "n_estimators": 500, **FIXED},
    {"num_leaves": 127, "learning_rate": 0.05, "n_estimators": 1000, **FIXED},
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dev_parquet", type=Path)
    parser.add_argument("--out", type=Path, default=PARAMS_PATH)
    args = parser.parse_args()

    dev = pd.read_parquet(args.dev_parquet)
    train = dev[dev["day"] <= TRAIN_END_DAY]
    holdout = dev[dev["day"] > TRAIN_END_DAY]
    x_train, y_train = base_features(train), labels(train)
    x_hold, y_hold = base_features(holdout), labels(holdout)
    print(f"train rows {len(train)} (days 1–{TRAIN_END_DAY}), holdout rows {len(holdout)}")

    rows: list[tuple[float, float, dict[str, Any]]] = []
    for params in GRID:
        start = time.perf_counter()
        scores = train_predict_single(x_train, y_train, x_hold, params, seed=SEEDS[0])
        pr_auc = float(average_precision_score(y_hold, scores))
        elapsed = time.perf_counter() - start
        rows.append((pr_auc, elapsed, params))
        print(
            f"num_leaves={params['num_leaves']:>4} learning_rate={params['learning_rate']:<5} "
            f"n_estimators={params['n_estimators']:>5}  holdout PR-AUC={pr_auc:.4f}  "
            f"({elapsed:.0f}s)"
        )

    best = max(rows, key=lambda r: r[0])
    args.out.write_text(json.dumps(best[2], indent=2) + "\n")
    print(f"winner written to {args.out}: {best[2]} (PR-AUC {best[0]:.4f})")


if __name__ == "__main__":
    main()
