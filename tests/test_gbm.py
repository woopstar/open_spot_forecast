"""Tests for the histogram-based gradient boosting model (issue #14)."""

import time

import numpy as np
import pytest

from custom_components.open_spot_forecast.ml.gbm import (
    MAX_BINS,
    NumpyGradientBoosting,
    bin_column,
    bin_edges,
)


def _xor_data(rows: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Two uniform features and a target that is +1 where exactly one is "high".

    The x1 boundary sits off-centre, so a greedy root split on x0 already
    gains, and a depth-2 tree can then fit the interaction exactly.
    """
    rng = np.random.default_rng(0)
    X = rng.uniform(-1.0, 1.0, size=(rows, 2))
    y = np.where((X[:, 0] > 0.0) ^ (X[:, 1] > 0.5), 1.0, -1.0)
    return X, y


def _mae(model: NumpyGradientBoosting, X: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(model.predict(X) - y)))


# --- Tree depth and interactions --------------------------------------------------


def test_depth_two_trees_learn_xor_that_stumps_cannot() -> None:
    """An XOR interaction needs depth > 1: a sum of stumps is additive."""
    X, y = _xor_data()

    stumps = NumpyGradientBoosting(n_estimators=100, max_depth=1)
    stumps.fit(X, y)
    trees = NumpyGradientBoosting(n_estimators=100, max_depth=2)
    trees.fit(X, y)

    assert _mae(stumps, X, y) > 0.5
    assert _mae(trees, X, y) < 0.05


def test_trees_respect_depth_leaf_and_leaf_size_limits() -> None:
    """No tree is deeper than max_depth, wider than max_leaves or has tiny leaves."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(1500, 4))
    y = X[:, 0] * X[:, 1] + np.sin(X[:, 2]) + rng.normal(scale=0.1, size=1500)

    model = NumpyGradientBoosting(
        n_estimators=20, max_depth=3, max_leaves=6, min_samples_leaf=40
    )
    model.fit(X, y)

    assert len(model.trees) == 20
    for tree in model.trees:
        assert 1 <= tree.depth <= 3
        assert tree.n_leaves <= 6
        _, leaf_sizes = np.unique(tree.apply(X), return_counts=True)
        assert leaf_sizes.min() >= 40


def test_deeper_trees_are_grown_when_allowed() -> None:
    """With the default limits the trees are deeper than a single split."""
    X, y = _xor_data()

    model = NumpyGradientBoosting(n_estimators=5)
    model.fit(X, y)

    assert max(tree.depth for tree in model.trees) > 1
    assert max(tree.n_leaves for tree in model.trees) > 2


# --- Binning ---------------------------------------------------------------------


def test_continuous_feature_is_binned_into_at_most_max_bins() -> None:
    """A column with thousands of distinct values gets quantile bins, not one per value."""
    column = np.random.default_rng(2).normal(size=10_000)

    edges = bin_edges(column)
    bins = bin_column(column, edges)

    assert len(edges) <= MAX_BINS - 1
    assert int(bins.max()) < MAX_BINS
    # Quantile bins hold roughly equal numbers of rows
    counts = np.bincount(bins)
    assert counts.max() < 3 * counts.mean()


def test_low_cardinality_feature_gets_one_bin_per_value() -> None:
    """Hour-of-day style features keep every distinct value in its own bin."""
    column = np.tile(np.arange(24, dtype=float), 10)

    edges = bin_edges(column)
    bins = bin_column(column, edges)

    assert len(edges) == 23
    np.testing.assert_array_equal(bins, column.astype(int))


def test_missing_values_get_their_own_bin_and_do_not_move_edges() -> None:
    """NaN gets the bin after all value bins; infinities go to the outermost bins."""
    column = np.array([1.0, 2.0, np.nan, 3.0, np.inf, -np.inf])

    edges = bin_edges(column)
    bins = bin_column(column, edges)

    np.testing.assert_allclose(edges, [1.5, 2.5])
    assert bins.tolist() == [0, 1, 3, 2, 2, 0]


def test_all_missing_feature_is_never_split() -> None:
    """A column without a single known value cannot split, but does not break the fit."""
    X, y = _xor_data()
    X = np.column_stack([np.full(len(y), np.nan), X])

    model = NumpyGradientBoosting(n_estimators=50, max_depth=2)
    model.fit(X, y)

    assert all(0 not in tree.feature for tree in model.trees)
    assert _mae(model, X, y) < 0.05


# --- Missing values --------------------------------------------------------------


def test_nan_rows_learn_their_own_direction() -> None:
    """Missing inputs behave like the rows they resemble, not like zero.

    Missing x means a high price here. If NaN were replaced by 0 it would
    land on the low side of the 0.3 threshold and be predicted low.
    """
    rng = np.random.default_rng(3)
    X = rng.uniform(-1.0, 1.0, size=(2000, 1))
    y = np.where(X[:, 0] > 0.3, 2.0, 0.0)
    missing = rng.random(2000) < 0.2
    X[missing, 0] = np.nan
    y[missing] = 2.0

    model = NumpyGradientBoosting(n_estimators=100)
    model.fit(X, y)
    nan_row, zero_row, high_row = model.predict(np.array([[np.nan], [0.0], [0.8]]))

    assert nan_row == pytest.approx(2.0, abs=0.05)
    assert zero_row == pytest.approx(0.0, abs=0.05)
    assert high_row == pytest.approx(2.0, abs=0.05)


def test_nan_at_prediction_without_nan_in_training_follows_larger_child() -> None:
    """A split that never saw NaN sends it to the side with more training rows."""
    X = np.arange(100, dtype=float).reshape(-1, 1)
    y = np.where(X[:, 0] < 70, 0.0, 10.0)

    model = NumpyGradientBoosting(n_estimators=50, max_depth=1, min_samples_leaf=5)
    model.fit(X, y)

    assert model.trees[0].default_left[0]
    assert model.predict(np.array([[np.nan]]))[0] == pytest.approx(
        model.predict(np.array([[0.0]]))[0]
    )


# --- Fit / predict contract ------------------------------------------------------


def test_fit_is_deterministic() -> None:
    """Two fits on the same data give identical predictions."""
    X, y = _xor_data()

    first = NumpyGradientBoosting(n_estimators=30, random_state=7)
    first.fit(X, y)
    second = NumpyGradientBoosting(n_estimators=30, random_state=7)
    second.fit(X, y)

    np.testing.assert_array_equal(first.predict(X), second.predict(X))


def test_staged_predict_ends_at_predict() -> None:
    """staged_predict yields one prediction per tree, the last equal to predict."""
    X, y = _xor_data()
    model = NumpyGradientBoosting(n_estimators=12)
    model.fit(X, y)

    stages = list(model.staged_predict(X))

    assert len(stages) == 12
    np.testing.assert_allclose(stages[-1], model.predict(X))
    assert _mae_of(stages[-1], y) < _mae_of(stages[0], y)


def _mae_of(predicted: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(predicted - y)))


def test_get_params_builds_an_equivalent_unfitted_copy() -> None:
    """get_params round-trips every hyperparameter through the constructor."""
    model = NumpyGradientBoosting(
        n_estimators=7,
        learning_rate=0.3,
        random_state=1,
        max_depth=3,
        max_leaves=5,
        min_samples_leaf=4,
        l2_regularization=0.5,
    )

    copy = NumpyGradientBoosting(**model.get_params())

    assert copy.get_params() == model.get_params()
    assert copy.trees == []


def test_unfitted_model_predicts_zero() -> None:
    model = NumpyGradientBoosting()

    np.testing.assert_array_equal(model.predict(np.ones((3, 2))), np.zeros(3))


def test_constant_target_predicts_the_constant() -> None:
    """With nothing to learn, every tree is a single leaf."""
    X = np.random.default_rng(4).normal(size=(200, 3))
    y = np.full(200, 1.25)

    model = NumpyGradientBoosting(n_estimators=5)
    model.fit(X, y)

    assert all(tree.n_leaves == 1 for tree in model.trees)
    np.testing.assert_allclose(model.predict(X), 1.25)


@pytest.mark.parametrize(
    ("X", "y"),
    [
        (np.empty((0, 3)), np.empty(0)),
        (np.ones((4, 3)), np.ones(3)),
        (np.ones(4), np.ones(4)),
    ],
)
def test_fit_rejects_malformed_input(X: np.ndarray, y: np.ndarray) -> None:
    with pytest.raises(ValueError, match="non-empty 2-D"):
        NumpyGradientBoosting().fit(X, y)


def test_predict_rejects_wrong_feature_count() -> None:
    X, y = _xor_data()
    model = NumpyGradientBoosting(n_estimators=2)
    model.fit(X, y)

    with pytest.raises(ValueError, match="expected a 2-D array with 2 features"):
        model.predict(np.ones((3, 3)))


@pytest.mark.parametrize(
    "params",
    [{"max_depth": 0}, {"max_leaves": 1}, {"min_samples_leaf": 0}],
)
def test_constructor_rejects_invalid_tree_limits(params: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="max_depth"):
        NumpyGradientBoosting(**params)


# --- Performance -----------------------------------------------------------------


def test_training_time_smoke_180_days_by_40_features() -> None:
    """A 180-day × 96-slot × 40-feature fit stays fast.

    Guards against a return to per-unique-value threshold search, which is
    quadratic in the row count for continuous features. The PR records the
    full 200-tree timing; 50 trees keep the test quick.
    """
    rng = np.random.default_rng(5)
    rows, features = 180 * 96, 40
    X = rng.normal(size=(rows, features))
    y = X[:, 0] * X[:, 1] + np.sin(X[:, 2]) + rng.normal(scale=0.1, size=rows)

    model = NumpyGradientBoosting(n_estimators=50)
    started = time.perf_counter()
    model.fit(X, y)
    elapsed = time.perf_counter() - started

    assert elapsed < 20.0
    assert _mae(model, X, y) < float(np.mean(np.abs(y - y.mean())))


def test_staged_prediction_equals_a_model_with_fewer_trees() -> None:
    """Stage k is exactly what an n_estimators=k fit predicts (HPO relies on it)."""
    X, y = _xor_data()
    full = NumpyGradientBoosting(n_estimators=10, max_depth=3)
    full.fit(X, y)
    short = NumpyGradientBoosting(n_estimators=4, max_depth=3)
    short.fit(X, y)

    stages = list(full.staged_predict(X))

    np.testing.assert_allclose(stages[3], short.predict(X))
