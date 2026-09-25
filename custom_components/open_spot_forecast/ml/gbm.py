"""Histogram-based gradient boosting regressor in pure NumPy (the price model)."""

import logging
import time
from collections.abc import Iterator
from typing import Any

import numpy as np

_LOGGER = logging.getLogger(__name__)


# --- Histogram gradient boosting ------------------------------------------------
#
# LightGBM's key ideas in pure NumPy (LightGBM itself has no musllinux wheels, so
# it cannot ship on Home Assistant OS/Container): every feature is binned once
# into at most MAX_BINS quantile bins, trees are grown leaf-wise from gradient
# histograms (np.bincount), and missing values (NaN) get their own bin and a
# learned default direction at every split.

# Maximum number of non-missing bins per feature
MAX_BINS = 255

# A split must reduce the squared loss by more than this to be made
_MIN_SPLIT_GAIN = 1e-9

# Marks a leaf in HistogramTree.feature
_LEAF = -1


def bin_edges(column: np.ndarray) -> np.ndarray:
    """Return the upper bin edges of one feature column.

    A value ``v`` falls into bin ``searchsorted(edges, v, side="left")``, so bin
    ``k`` holds ``edges[k - 1] < v <= edges[k]`` and there are at most
    ``MAX_BINS`` non-missing bins. A column with at most ``MAX_BINS`` distinct
    values gets one bin per value (edges at the midpoints between them); a
    column with more gets quantile bins. NaN and infinite values do not move
    the edges: NaN gets its own bin (see ``bin_column``), ±inf the outermost.

    Args:
        column: One feature's training values.

    Returns:
        Sorted, unique edges (at most ``MAX_BINS - 1``).
    """
    finite = column[np.isfinite(column)]
    distinct = np.unique(finite)
    if distinct.size <= MAX_BINS:
        edges: np.ndarray = (distinct[:-1] + distinct[1:]) / 2
        return edges
    quantiles = np.linspace(0.0, 1.0, MAX_BINS + 1)[1:-1]
    return np.unique(np.quantile(finite, quantiles))


def bin_column(column: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Map one feature column to its bins.

    Values get bins ``0 … len(edges)``; NaN gets the missing-value bin
    ``len(edges) + 1``, after all value bins.
    """
    bins = np.searchsorted(edges, column, side="left")
    bins[np.isnan(column)] = edges.size + 1
    return bins


class HistogramTree:
    """A fitted regression tree stored as flat node arrays.

    Node 0 is the root. For an internal node ``i``, rows with
    ``x[feature[i]] <= threshold[i]`` go to ``left[i]`` and the others to
    ``right[i]``; rows where that feature is NaN follow ``default_left[i]``.
    Leaves have ``feature == -1`` and predict ``value`` (shrinkage included).
    """

    def __init__(
        self,
        feature: list[int],
        threshold: list[float],
        default_left: list[bool],
        left: list[int],
        right: list[int],
        value: list[float],
        depth: int,
    ) -> None:
        """Store the node arrays built by the tree grower."""
        self.feature = np.array(feature, dtype=np.intp)
        self.threshold = np.array(threshold, dtype=float)
        self.default_left = np.array(default_left, dtype=bool)
        self.left = np.array(left, dtype=np.intp)
        self.right = np.array(right, dtype=np.intp)
        self.value = np.array(value, dtype=float)
        self.depth = depth

    @property
    def n_leaves(self) -> int:
        """Return the number of leaves."""
        return int(np.count_nonzero(self.feature == _LEAF))

    def apply(self, X: np.ndarray) -> np.ndarray:
        """Return the leaf node index of every row, walking all rows one level at a time."""
        rows = np.arange(len(X))
        node = np.zeros(len(X), dtype=np.intp)
        for _ in range(self.depth):
            feature = self.feature[node]
            internal = feature != _LEAF
            x = X[rows, np.where(internal, feature, 0)]
            go_left = np.where(
                np.isnan(x), self.default_left[node], x <= self.threshold[node]
            )
            child = np.where(go_left, self.left[node], self.right[node])
            node = np.where(internal, child, node)
        return node

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the leaf value of every row."""
        leaf_values: np.ndarray = self.value[self.apply(X)]
        return leaf_values


class _Leaf:
    """A leaf of a tree being grown, with its rows, histograms and best split."""

    def __init__(
        self,
        node: int,
        rows: np.ndarray,
        hists: tuple[np.ndarray, np.ndarray],
        depth: int,
    ) -> None:
        self.node = node
        self.rows = rows
        self.grad_hist, self.count_hist = hists
        self.depth = depth
        # (gain, feature, histogram bin, default_left), or None if it cannot split
        self.split: tuple[float, int, int, bool] | None = None

    @property
    def gain(self) -> float:
        """Return the loss reduction of the best split (-inf if none)."""
        return self.split[0] if self.split is not None else -np.inf


class _TreeGrower:
    """Grows trees leaf-wise on one binned training matrix.

    All features share one flat histogram. Feature ``f`` owns the value bins
    ``offsets[f] … offsets[f] + sizes[f] - 1``, so a feature with 24 distinct
    values costs 24 bins, not ``MAX_BINS``. The missing-value bins follow all
    value bins, one per feature at ``n_value_bins + f``. The layout is built
    once per fit and shared by every tree. Each split histograms only the
    smaller child; the larger child's histogram is the parent's minus the
    smaller one's.
    """

    def __init__(
        self,
        X: np.ndarray,
        edges: list[np.ndarray],
        model: NumpyGradientBoosting,
    ) -> None:
        self.max_depth = model.max_depth
        self.max_leaves = model.max_leaves
        self.min_samples_leaf = model.min_samples_leaf
        self.l2 = model.l2_regularization
        self.n_rows, self.n_features = X.shape

        self.sizes = np.array([e.size + 1 for e in edges])
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)[:-1]])
        self.n_value_bins = int(self.sizes.sum())
        self.n_bins = self.n_value_bins + self.n_features
        # Feature of every value bin
        self.bin_feature = np.repeat(np.arange(self.n_features), self.sizes)
        # A split at value bin k sends x <= threshold[k] left; the last value
        # bin of a feature has no upper edge, so its threshold is +inf
        self.threshold = np.full(self.n_value_bins, np.inf)
        for f, feature_edges in enumerate(edges):
            start = self.offsets[f]
            self.threshold[start : start + feature_edges.size] = feature_edges
        # Histogram bin of every (row, feature)
        columns = []
        for f, feature_edges in enumerate(edges):
            bins = bin_column(X[:, f], feature_edges)
            missing = bins == feature_edges.size + 1
            columns.append(
                np.where(missing, self.n_value_bins + f, self.offsets[f] + bins)
            )
        self.index = np.column_stack(columns)
        self._all_rows = np.arange(self.n_rows)
        # The root's count histogram is the same for every tree
        self._root_counts = np.bincount(
            self.index.ravel(), minlength=self.n_bins
        ).astype(float)

    def _histograms(
        self, rows: np.ndarray, grad: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the gradient-sum and row-count histograms of ``rows``."""
        if rows.size == self.n_rows:
            index = self.index.ravel()
            counts = self._root_counts
        else:
            index = self.index[rows].ravel()
            counts = np.bincount(index, minlength=self.n_bins).astype(float)
        weights = np.repeat(grad[rows], self.n_features)
        return np.bincount(index, weights=weights, minlength=self.n_bins), counts

    def _cumulative(self, values: np.ndarray) -> np.ndarray:
        """Return per-feature cumulative sums of value-bin histograms (leaf × bin).

        Subtracting each feature's total at the start of the next feature
        makes one cumsum over all value bins restart at every feature.
        """
        restart = values.copy()
        totals = np.add.reduceat(values, self.offsets, axis=1)
        restart[:, self.offsets[1:]] -= totals[:, :-1]
        cumulative: np.ndarray = np.cumsum(restart, axis=1)
        return cumulative

    def _find_splits(self, leaves: list[_Leaf]) -> None:
        """Set the best split of every leaf that may still split.

        Every value-bin boundary of every feature of every leaf is scored at
        once from cumulative histograms: with the feature's NaN rows sent
        right, and, if the leaves have NaN rows, also sent left. The gain is
        the squared-loss reduction with L2 leaf regularization,
        ``G_L²/(n_L+λ) + G_R²/(n_R+λ) - G²/(n+λ)``.
        """
        growable = [
            leaf
            for leaf in leaves
            if leaf.depth < self.max_depth
            and leaf.rows.size >= 2 * self.min_samples_leaf
        ]
        if not growable:
            return
        grad = np.stack([leaf.grad_hist for leaf in growable])
        count = np.stack([leaf.count_hist for leaf in growable])
        values = self.n_value_bins

        cum_grad = self._cumulative(grad[:, :values])
        cum_count = self._cumulative(count[:, :values])
        nan_grad = grad[:, values:]
        nan_count = count[:, values:]
        total_grad = grad[:, : self.sizes[0]].sum(axis=1) + nan_grad[:, 0]
        total_grad = total_grad[:, np.newaxis]
        total_count = count[:, : self.sizes[0]].sum(axis=1) + nan_count[:, 0]
        total_count = total_count[:, np.newaxis]
        parent_score = total_grad**2 / (total_count + self.l2)

        candidates = [(cum_grad, cum_count)]
        if nan_count.any():
            candidates.append(
                (
                    cum_grad + np.repeat(nan_grad, self.sizes, axis=1),
                    cum_count + np.repeat(nan_count, self.sizes, axis=1),
                )
            )
        gains = []
        for left_grad, left_count in candidates:
            right_grad = total_grad - left_grad
            right_count = total_count - left_count
            valid = (left_count >= self.min_samples_leaf) & (
                right_count >= self.min_samples_leaf
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                score = left_grad**2 / (left_count + self.l2) + right_grad**2 / (
                    right_count + self.l2
                )
            gains.append(np.where(valid, score - parent_score, -np.inf))

        # Per leaf: [NaN-right candidates | NaN-left candidates]
        stacked = np.concatenate(gains, axis=1)
        best = np.argmax(stacked, axis=1)
        for i, leaf in enumerate(growable):
            gain = float(stacked[i, best[i]])
            if not gain > _MIN_SPLIT_GAIN:
                continue
            nan_left, split_bin = divmod(int(best[i]), values)
            feature = int(self.bin_feature[split_bin])
            default_left = bool(nan_left)
            if nan_count[i, feature] < 0.5:
                # No NaN seen here: send future NaN rows to the larger child
                left_rows = cum_count[i, split_bin]
                default_left = bool(left_rows >= total_count[i, 0] - left_rows)
            leaf.split = (gain, feature, split_bin, default_left)

    def grow(
        self, grad: np.ndarray, shrinkage: float
    ) -> tuple[HistogramTree, np.ndarray]:
        """Fit one tree to the gradients.

        Args:
            grad: Negative gradient (residual) of every training row.
            shrinkage: Learning rate, folded into the leaf values.

        Returns:
            The tree and its (shrunk) prediction for every training row.
        """
        feature: list[int] = [_LEAF]
        threshold: list[float] = [0.0]
        default_left: list[bool] = [False]
        left: list[int] = [_LEAF]
        right: list[int] = [_LEAF]
        depth_reached = 0

        root_rows = self._all_rows
        leaves = [_Leaf(0, root_rows, self._histograms(root_rows, grad), 0)]
        self._find_splits(leaves)

        while len(leaves) < self.max_leaves:
            # Leaf-wise growth: split the leaf whose best split gains the most
            parent = max(leaves, key=lambda leaf: leaf.gain)
            if parent.split is None:
                break
            _, split_feature, split_bin, split_default_left = parent.split

            # Value bins of the feature are ordered; its NaN bin is past them all
            column = self.index[parent.rows, split_feature]
            go_left = column <= split_bin
            if split_default_left:
                go_left |= column >= self.n_value_bins
            left_rows = parent.rows[go_left]
            right_rows = parent.rows[~go_left]

            # Histogram the smaller child; derive the larger by subtraction
            small_is_left = left_rows.size <= right_rows.size
            small = self._histograms(left_rows if small_is_left else right_rows, grad)
            large = (parent.grad_hist - small[0], parent.count_hist - small[1])

            node = parent.node
            left_node, right_node = len(feature), len(feature) + 1
            feature[node] = split_feature
            threshold[node] = float(self.threshold[split_bin])
            default_left[node] = split_default_left
            left[node], right[node] = left_node, right_node
            for _ in range(2):
                feature.append(_LEAF)
                threshold.append(0.0)
                default_left.append(False)
                left.append(_LEAF)
                right.append(_LEAF)

            depth = parent.depth + 1
            depth_reached = max(depth_reached, depth)
            children = [
                _Leaf(left_node, left_rows, small if small_is_left else large, depth),
                _Leaf(right_node, right_rows, large if small_is_left else small, depth),
            ]
            self._find_splits(children)
            leaves.remove(parent)
            leaves.extend(children)

        value = [0.0] * len(feature)
        train_prediction = np.empty(self.n_rows)
        for leaf in leaves:
            leaf_value = (
                shrinkage * float(grad[leaf.rows].sum()) / (leaf.rows.size + self.l2)
            )
            value[leaf.node] = leaf_value
            train_prediction[leaf.rows] = leaf_value

        tree = HistogramTree(
            feature, threshold, default_left, left, right, value, depth_reached
        )
        return tree, train_prediction


class NumpyGradientBoosting:
    """Histogram-based gradient boosting regressor (squared loss) in pure NumPy.

    Each feature is binned once into at most ``MAX_BINS`` quantile bins plus a
    bin for missing values. Trees are grown leaf-wise (best split first) up to
    ``max_leaves`` leaves and ``max_depth`` levels, so they can learn feature
    interactions that single-split trees cannot. NaN inputs are handled
    natively: every split learns which side missing values go to.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        random_state: int = 42,
        max_depth: int = 6,
        max_leaves: int = 31,
        min_samples_leaf: int = 20,
        l2_regularization: float = 1.0,
    ):
        """Store the hyperparameters; nothing is fitted yet.

        Args:
            n_estimators: Number of boosting rounds (trees).
            learning_rate: Shrinkage applied to every tree's output.
            random_state: Kept for a stable model signature; the fit is
                deterministic (no row or feature subsampling).
            max_depth: Maximum number of splits from the root to any leaf.
            max_leaves: Maximum number of leaves per tree.
            min_samples_leaf: Minimum number of training rows in a leaf.
            l2_regularization: L2 penalty on leaf values (λ).
        """
        if max_depth < 1 or max_leaves < 2 or min_samples_leaf < 1:
            raise ValueError(
                "max_depth must be >= 1, max_leaves >= 2 and min_samples_leaf >= 1"
            )
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.trees: list[HistogramTree] = []
        self.initial_prediction: float | None = None
        self.n_features: int | None = None

    def get_params(self) -> dict[str, Any]:
        """Return the constructor hyperparameters, e.g. to build an unfitted copy."""
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "random_state": self.random_state,
            "max_depth": self.max_depth,
            "max_leaves": self.max_leaves,
            "min_samples_leaf": self.min_samples_leaf,
            "l2_regularization": self.l2_regularization,
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model, replacing any previous fit.

        Args:
            X: Feature matrix (rows × features); NaN marks a missing value.
            y: Target of every row (finite).

        Raises:
            ValueError: If X and y are empty or their lengths differ.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2 or len(X) != len(y) or len(y) == 0:
            raise ValueError("X must be a non-empty 2-D array with one row per target")

        started = time.perf_counter()
        # Drop trees from an earlier fit so a retrain doesn't stack on top
        self.trees = []
        n_features = X.shape[1]
        self.n_features = n_features
        self.initial_prediction = float(np.mean(y))

        edges = [bin_edges(X[:, f]) for f in range(n_features)]
        grower = _TreeGrower(X, edges, self)
        predictions = np.full(len(y), self.initial_prediction)
        for _ in range(self.n_estimators):
            tree, update = grower.grow(y - predictions, self.learning_rate)
            predictions += update
            self.trees.append(tree)

        _LOGGER.debug(
            "Gradient boosting fitted: %d rows, %d features, %d trees, "
            "%.1f mean leaves, training MAE=%.4f, %.2f s",
            len(y),
            n_features,
            len(self.trees),
            float(np.mean([tree.n_leaves for tree in self.trees]))
            if self.trees
            else 0.0,
            float(np.mean(np.abs(y - predictions))),
            time.perf_counter() - started,
        )

    def staged_predict(self, X: np.ndarray) -> Iterator[np.ndarray]:
        """Yield the prediction after each tree (1, 2, …, ``n_estimators``).

        Lets hyperparameter search score every ``n_estimators`` up to the
        fitted one from a single fit.
        """
        X = self._check_input(X)
        predictions = np.full(len(X), self.initial_prediction or 0.0)
        for tree in self.trees:
            predictions = predictions + tree.predict(X)
            yield predictions

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the target of every row; an unfitted model predicts 0."""
        X = self._check_input(X)
        predictions = np.full(len(X), self.initial_prediction or 0.0)
        for tree in self.trees:
            predictions += tree.predict(X)
        return predictions

    def _check_input(self, X: np.ndarray) -> np.ndarray:
        """Return X as a float matrix with the fitted number of features."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or (
            self.n_features is not None and X.shape[1] != self.n_features
        ):
            raise ValueError(
                f"expected a 2-D array with {self.n_features} features, "
                f"got shape {X.shape}"
            )
        return X
