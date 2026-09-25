"""Pure numpy implementations of ML models for electricity price forecasting."""

import logging
from datetime import datetime, timedelta

import numpy as np

from homeassistant.util import dt as dt_util

from .gbm import NumpyGradientBoosting

_LOGGER = logging.getLogger(__name__)


class DecisionTree:
    """A simple decision tree with configurable depth."""

    def __init__(
        self, max_depth: int = 3, min_samples_split: int = 5, random_state: int = 42
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.tree: dict | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Build the decision tree."""
        np.random.seed(self.random_state)
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> dict:
        """Recursively build the tree."""
        n_samples = len(y)

        # Stopping conditions
        if (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            return {"value": np.mean(y)}

        # Find best split
        best_feature, best_threshold, best_loss = None, None, float("inf")
        n_features = X.shape[1]

        # Random feature subset (sqrt of features)
        n_features_to_try = max(1, int(np.sqrt(n_features)))
        feature_indices = np.random.choice(n_features, n_features_to_try, replace=False)

        for feature_idx in feature_indices:
            thresholds = np.unique(X[:, feature_idx])

            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) < 2 or np.sum(right_mask) < 2:
                    continue

                # Calculate MSE loss
                left_loss = np.var(y[left_mask]) * np.sum(left_mask)
                right_loss = np.var(y[right_mask]) * np.sum(right_mask)
                total_loss = left_loss + right_loss

                if total_loss < best_loss:
                    best_loss = total_loss
                    best_feature = feature_idx
                    best_threshold = threshold

        # If no good split found, return leaf
        if best_feature is None:
            return {"value": np.mean(y)}

        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        # Build subtrees
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return {
            "feature": best_feature,
            "threshold": best_threshold,
            "left": left_tree,
            "right": right_tree,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        # Ensure X doesn't contain None values
        X_safe = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if self.tree is None:
            return np.zeros(len(X_safe))
        return np.array([self._predict_single(x, self.tree) for x in X_safe])

    def _predict_single(self, x: np.ndarray, node: dict) -> float:
        """Predict for a single sample."""
        if "value" in node:
            return node["value"] if node["value"] is not None else 0.0

        # Handle None threshold (untrained node)
        if node.get("threshold") is None:
            # Return average of left and right values if available
            left_val = node.get("left", {}).get("value", 0)
            right_val = node.get("right", {}).get("value", 0)
            left_val = left_val if left_val is not None else 0.0
            right_val = right_val if right_val is not None else 0.0
            return (left_val + right_val) / 2

        # Ensure x doesn't contain None values
        x_safe = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        feature_idx = node["feature"]
        threshold = node["threshold"]

        # Safe comparison
        if x_safe[feature_idx] <= threshold:
            return self._predict_single(x_safe, node["left"])
        else:
            return self._predict_single(x_safe, node["right"])


class NumpyRandomForest:
    """Random Forest Regressor using only numpy."""

    def __init__(
        self,
        n_estimators: int = 50,
        max_depth: int = 5,
        min_samples_split: int = 5,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees: list[DecisionTree] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the random forest."""
        np.random.seed(self.random_state)
        n_samples = len(y)

        for i in range(self.n_estimators):
            # Bootstrap sampling
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]

            # Train a tree
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state + i,
            )
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions by averaging tree predictions."""
        # Ensure X doesn't contain None values
        X_safe = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        predictions = np.array([tree.predict(X_safe) for tree in self.trees])
        return np.mean(predictions, axis=0)


class FeatureEngineer:
    """Create features for electricity price prediction."""

    def __init__(self):
        self.feature_names = []

    def create_features(
        self,
        prices: list[float],
        weather_data: dict,
        timestamps: list[datetime] | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """Create feature matrix from prices and weather data."""
        if timestamps is None:
            timestamps = [
                dt_util.now() + timedelta(hours=i) for i in range(len(prices))
            ]

        features = []
        feature_names = []

        # Time features
        hours = np.array([t.hour for t in timestamps])
        days = np.array([t.weekday() for t in timestamps])

        # Cyclical encoding for hour
        hour_sin = np.sin(2 * np.pi * hours / 24)
        hour_cos = np.cos(2 * np.pi * hours / 24)

        # Cyclical encoding for day of week
        day_sin = np.sin(2 * np.pi * days / 7)
        day_cos = np.cos(2 * np.pi * days / 7)

        features.extend([hour_sin, hour_cos, day_sin, day_cos])
        feature_names.extend(["hour_sin", "hour_cos", "day_sin", "day_cos"])

        # Weekend indicator
        is_weekend = (days >= 5).astype(float)
        features.append(is_weekend)
        feature_names.append("is_weekend")

        # Weather features
        if "wind_speed" in weather_data and weather_data["wind_speed"] is not None:
            wind_speed = np.full(len(prices), weather_data["wind_speed"])
            features.append(wind_speed)
            feature_names.append("wind_speed")

        if "temperature" in weather_data and weather_data["temperature"] is not None:
            temperature = np.full(len(prices), weather_data["temperature"])
            features.append(temperature)
            feature_names.append("temperature")

        if "solar_power" in weather_data and weather_data["solar_power"] is not None:
            solar_power = np.full(len(prices), weather_data["solar_power"])
            features.append(solar_power)
            feature_names.append("solar_power")

        # Price lag features
        prices_array = np.array(prices)

        # Previous hour price
        lag_1 = np.roll(prices_array, 1)
        lag_1[0] = prices_array[0]
        features.append(lag_1)
        feature_names.append("price_lag_1")

        # 24-hour lag (same hour yesterday)
        lag_24 = np.roll(prices_array, 24)
        lag_24[:24] = prices_array[:24]
        features.append(lag_24)
        feature_names.append("price_lag_24")

        # Rolling statistics
        window = 6
        rolling_mean = np.array(
            [
                np.mean(prices_array[max(0, i - window) : i + 1])
                for i in range(len(prices_array))
            ]
        )
        features.append(rolling_mean)
        feature_names.append("price_rolling_mean_6")

        rolling_std = np.array(
            [
                np.std(prices_array[max(0, i - window) : i + 1])
                for i in range(len(prices_array))
            ]
        )
        features.append(rolling_std)
        feature_names.append("price_rolling_std_6")

        # Stack features
        X = np.column_stack(features)
        self.feature_names = feature_names

        return X, feature_names


class ElectricityPricePredictor:
    """Main predictor combining multiple models."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.feature_engineer = FeatureEngineer()

        # Ensemble of models
        self.gradient_boosting = NumpyGradientBoosting(
            n_estimators=100, learning_rate=0.1, random_state=random_state
        )

        self.random_forest = NumpyRandomForest(
            n_estimators=50, max_depth=5, random_state=random_state
        )

        self.is_trained = False

    def fit(
        self,
        prices: list[float],
        weather_data: dict,
        timestamps: list[datetime] | None = None,
    ) -> None:
        """Train the models."""
        # Create features
        X, feature_names = self.feature_engineer.create_features(
            prices, weather_data, timestamps
        )
        y = np.array(prices)

        # Train models
        self.gradient_boosting.fit(X, y)
        self.random_forest.fit(X, y)

        self.is_trained = True

    def predict(
        self,
        prices: list[float],
        weather_data: dict,
        timestamps: list[datetime] | None = None,
    ) -> np.ndarray:
        """Make predictions using ensemble."""
        if not self.is_trained:
            # Return simple baseline if not trained
            return np.full(len(prices), np.mean(prices) if prices else 0)

        # Create features
        X, _ = self.feature_engineer.create_features(prices, weather_data, timestamps)

        # Get predictions from both models
        gb_predictions = self.gradient_boosting.predict(X)
        rf_predictions = self.random_forest.predict(X)

        # Ensemble: weighted average (GB gets more weight)
        ensemble_predictions = 0.6 * gb_predictions + 0.4 * rf_predictions

        return ensemble_predictions
