"""Row A (and any row with extra feature files) trained on dev, measured on validation.

Reads ``dev.parquet`` and ``validation.parquet`` only. ``final.parquet`` is never opened
here; plan 2's ablation runner touches it exactly once.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from signal_factory.ablation.features import base_features, labels
from signal_factory.baseline.learner import SEEDS, load_params, train_predict
from signal_factory.baseline.metrics import (
    FPR,
    pr_auc,
    recall_at_fpr,
    recall_at_threshold,
    roc_auc,
    seen_mask,
)

ID_COLUMN = "TransactionID"


@dataclass(frozen=True)
class Metrics:
    recall: float
    actual_fpr: float
    pr_auc: float
    roc_auc: float
    n: int


@dataclass(frozen=True)
class RowResult:
    name: str
    validation: Metrics
    validation_seen: Metrics
    validation_unseen: Metrics
    threshold: float
    params: dict[str, Any]
    seeds: tuple[int, ...]
    feature_columns: tuple[str, ...]


def _metrics(scores: np.ndarray, y: np.ndarray, threshold: float) -> Metrics:
    r = recall_at_threshold(scores, y, threshold)
    both_classes = 0 < int(y.sum()) < len(y)
    return Metrics(
        recall=r.recall,
        actual_fpr=r.actual_fpr,
        pr_auc=pr_auc(scores, y) if y.sum() else math.nan,
        roc_auc=roc_auc(scores, y) if both_classes else math.nan,
        n=len(y),
    )


def _features(df: pd.DataFrame, feature_parquets: Sequence[Path]) -> pd.DataFrame:
    x = base_features(df)
    for path in feature_parquets:
        extra = pd.read_parquet(path)
        extra = extra.set_index(ID_COLUMN).reindex(df[ID_COLUMN].to_numpy())
        x = pd.concat([x.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)
    return x


def run_row(
    name: str,
    split_dir: Path,
    out_dir: Path,
    feature_parquets: Sequence[Path] = (),
    seeds: Sequence[int] = SEEDS,
    params: dict[str, Any] | None = None,
) -> RowResult:
    """Train on dev, freeze the threshold on all validation rows, write ``row_<name>.json``."""
    params = dict(params or load_params())
    dev = pd.read_parquet(split_dir / "dev.parquet")
    validation = pd.read_parquet(split_dir / "validation.parquet")

    x_dev, y_dev = _features(dev, feature_parquets), labels(dev)
    x_val, y_val = _features(validation, feature_parquets), labels(validation)
    scores = train_predict(x_dev, y_dev, x_val, params, seeds=seeds)

    frozen = recall_at_fpr(scores, y_val, FPR)
    seen = seen_mask(validation["uid"], dev["uid"])
    result = RowResult(
        name=name,
        validation=_metrics(scores, y_val, frozen.threshold),
        validation_seen=_metrics(scores[seen], y_val[seen], frozen.threshold),
        validation_unseen=_metrics(scores[~seen], y_val[~seen], frozen.threshold),
        threshold=frozen.threshold,
        params=params,
        seeds=tuple(seeds),
        feature_columns=tuple(str(c) for c in x_dev.columns),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"row_{name.lower()}.json").write_text(json.dumps(asdict(result), indent=2) + "\n")
    return result


def run_row_a(
    split_dir: Path,
    out_dir: Path,
    seeds: Sequence[int] = SEEDS,
    params: dict[str, Any] | None = None,
) -> RowResult:
    return run_row("A", split_dir, out_dir, [], seeds=seeds, params=params)
