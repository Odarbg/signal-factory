import json
from pathlib import Path

import numpy as np
import pandas as pd

from signal_factory.ablation.features import base_features
from signal_factory.baseline.learner import (
    PARAMS_PATH,
    SEEDS,
    load_params,
    train_predict,
    train_predict_single,
)

XY = tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]
FAST = {"num_leaves": 7, "learning_rate": 0.1, "n_estimators": 20, "min_child_samples": 5}


def test_seeds_are_pre_registered() -> None:
    assert SEEDS == (0, 1, 2, 3, 4)


def test_load_params_has_required_keys() -> None:
    params = load_params()
    assert {"num_leaves", "learning_rate", "n_estimators"} <= params.keys()
    assert {"feature_fraction", "bagging_fraction", "min_child_samples"} <= params.keys()


def test_params_file_is_the_committed_json() -> None:
    assert PARAMS_PATH == Path(__file__).parents[2] / "src/signal_factory/baseline/params.json"
    assert json.loads(PARAMS_PATH.read_text()) == load_params()


def test_train_predict_returns_probabilities_in_unit_interval(synthetic_xy: XY) -> None:
    x_train, y_train, x_eval, _ = synthetic_xy
    p = train_predict(base_features(x_train), y_train, base_features(x_eval), FAST, seeds=(0, 1))
    assert p.shape == (len(x_eval),)
    assert p.dtype == np.float64
    assert np.all((p >= 0) & (p <= 1))


def test_train_predict_is_better_than_chance(synthetic_xy: XY) -> None:
    from sklearn.metrics import roc_auc_score

    x_train, y_train, x_eval, y_eval = synthetic_xy
    p = train_predict(base_features(x_train), y_train, base_features(x_eval), FAST, seeds=(0,))
    assert roc_auc_score(y_eval, p) > 0.6


def test_train_predict_is_seed_deterministic(synthetic_xy: XY) -> None:
    x_train, y_train, x_eval, _ = synthetic_xy
    a = train_predict_single(base_features(x_train), y_train, base_features(x_eval), FAST, seed=0)
    b = train_predict_single(base_features(x_train), y_train, base_features(x_eval), FAST, seed=0)
    assert np.allclose(a, b)


def test_different_seeds_differ(synthetic_xy: XY) -> None:
    x_train, y_train, x_eval, _ = synthetic_xy
    params = {**FAST, "feature_fraction": 0.5, "bagging_fraction": 0.8}
    a = train_predict_single(base_features(x_train), y_train, base_features(x_eval), params, seed=0)
    b = train_predict_single(base_features(x_train), y_train, base_features(x_eval), params, seed=1)
    assert not np.allclose(a, b)


def test_train_predict_averages_seeds(synthetic_xy: XY) -> None:
    x_train, y_train, x_eval, _ = synthetic_xy
    xt, xe = base_features(x_train), base_features(x_eval)
    singles = [train_predict_single(xt, y_train, xe, FAST, seed=s) for s in (0, 1, 2)]
    averaged = train_predict(xt, y_train, xe, FAST, seeds=(0, 1, 2))
    assert np.allclose(averaged, np.mean(singles, axis=0))


def test_train_predict_handles_unseen_categories(synthetic_xy: XY) -> None:
    x_train, y_train, x_eval, _ = synthetic_xy
    x_eval = x_eval.copy()
    x_eval.loc[x_eval.index[:3], "ProductCD"] = "ZZ"
    p = train_predict(base_features(x_train), y_train, base_features(x_eval), FAST, seeds=(0,))
    assert np.all(np.isfinite(p))
