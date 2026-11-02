"""LightGBM with one frozen parameter set and five seeds, predictions averaged (spec §6).

``params.json`` is written once by ``scripts/freeze_params.py`` on dev data before any
derived signal exists, then committed and never tuned again.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
PARAMS_PATH = Path(__file__).with_name("params.json")

FIXED_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "bagging_freq": 1,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}
"""Infrastructure settings that are not tuned and apply to every fit."""


def load_params() -> dict[str, Any]:
    params: dict[str, Any] = json.loads(PARAMS_PATH.read_text())
    return params


def train_predict_single(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_eval: pd.DataFrame,
    params: dict[str, Any] | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Probabilities for ``x_eval`` from one model trained with ``seed``."""
    model = lgb.LGBMClassifier(**FIXED_PARAMS, **(params or load_params()), random_state=seed)
    model.fit(x_train, y_train)
    proba = np.asarray(model.predict_proba(x_eval))[:, 1]
    return proba.astype(np.float64)


def train_predict(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_eval: pd.DataFrame,
    params: dict[str, Any] | None = None,
    seeds: Sequence[int] = SEEDS,
) -> np.ndarray:
    """Seed-averaged probabilities for ``x_eval``."""
    params = params or load_params()
    stacked = np.stack(
        [train_predict_single(x_train, y_train, x_eval, params, seed=s) for s in seeds]
    )
    return stacked.mean(axis=0)
