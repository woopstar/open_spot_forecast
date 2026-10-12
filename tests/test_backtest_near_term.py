"""Tests for the backtest's near-term routing (--near-term-days, #135)."""

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from custom_components.open_spot_forecast.const import NEAR_TERM_REGIONS
from custom_components.open_spot_forecast.ml.gbm import NumpyGradientBoosting
from custom_components.open_spot_forecast.ml.price_lags import (
    NEAR_TERM_DAYS,
    NEAR_TERM_FEATURE,
)
from scripts import backtest
from scripts.backtest import (
    SLOT_SECONDS,
    BacktestConfig,
    CurrentModel,
    LightGbmReference,
    PriceSeries,
    build_models,
    history_for,
    local_midnight,
    target_slots,
)
from scripts.backtest_lags import LagConfig, NearTermModel

TZ = ZoneInfo("Europe/Copenhagen")


def _noisy_series(first: date, days: int, seed: int = 7) -> PriceSeries:
    """A weekly pattern plus noise on the 15-minute grid of ``days`` local days."""
    rng = np.random.default_rng(seed)
    starts = np.arange(
        local_midnight(first, TZ),
        local_midnight(first + timedelta(days=days), TZ),
        SLOT_SECONDS,
        dtype=np.int64,
    )
    moments = [datetime.fromtimestamp(start, TZ) for start in starts.tolist()]
    prices = [10 + m.weekday() + m.hour / 10 + m.minute / 1000 for m in moments]
    return PriceSeries(starts, np.array(prices) + rng.normal(0, 2, len(starts)))


def _config(origin: date, **kwargs: Any) -> BacktestConfig:
    return BacktestConfig(tz=TZ, first_origin=origin, last_origin=origin, **kwargs)


def _small_gbm() -> NumpyGradientBoosting:
    """A few shallow trees: enough to test the plumbing."""
    return NumpyGradientBoosting(n_estimators=10, min_samples_leaf=20, random_state=0)


@dataclass
class _Constant:
    """A forecaster that returns one price and records its target slots."""

    price: float
    name: str = "constant"
    targets: list[np.ndarray] = field(default_factory=list)

    def forecast(self, history: PriceSeries, targets: np.ndarray) -> np.ndarray:
        self.targets.append(targets)
        return np.full(targets.shape, self.price)


@pytest.mark.parametrize("days", [1, 2, 7])
def test_near_term_routing_gives_the_first_days_to_the_near_model(days: int) -> None:
    """Horizon days up to --near-term-days come from the lag model, the rest not."""
    origin = date(2026, 6, 15)
    series = _noisy_series(date(2026, 5, 20), 30)
    config = _config(origin, window_days=21, horizon_days=7)
    targets, horizon = target_slots(origin, config)
    plain, near = _Constant(1.0), _Constant(2.0)
    model = NearTermModel(plain, near, TZ, days, "routed")

    predicted = model.forecast(history_for(series, origin, config), targets)

    np.testing.assert_array_equal(predicted, np.where(horizon <= days, 2.0, 1.0))
    # Both models are asked for every target, so they share the origin's inputs
    np.testing.assert_array_equal(plain.targets[0], targets)
    np.testing.assert_array_equal(near.targets[0], targets)


def test_near_term_routing_counts_from_the_last_known_day() -> None:
    """A history that ends a day early makes every target a day older."""
    origin = date(2026, 6, 15)
    series = _noisy_series(date(2026, 5, 20), 30)
    config = _config(origin, window_days=21, horizon_days=3)
    targets, horizon = target_slots(origin, config)
    history = history_for(series, origin, config)
    short = history.between(
        int(history.starts.min()), local_midnight(origin - timedelta(days=1), TZ)
    )
    model = NearTermModel(_Constant(1.0), _Constant(2.0), TZ, 2, "routed")

    predicted = model.forecast(short, targets)

    np.testing.assert_array_equal(predicted, np.where(horizon <= 1, 2.0, 1.0))


def test_build_models_routes_both_gbm_rows_with_near_term_days() -> None:
    """The near model is the row's own model with the lagged price column."""
    pytest.importorskip("lightgbm")
    models, skipped = build_models(["naive", "current", "lightgbm"], TZ)
    assert [type(model) for model in models[1:]] == [CurrentModel, LightGbmReference]

    models, skipped = build_models(
        ["naive", "current", "lightgbm"], TZ, region="DK1", near_term_days=2
    )

    assert skipped == {}
    assert [model.name for model in models[1:]] == [
        "current (NumPy GBM, near-term 2d)",
        "lightgbm (reference, near-term 2d)",
    ]
    current, reference = models[1:]
    assert isinstance(current, NearTermModel) and current.days == 2
    assert isinstance(current.plain, CurrentModel)
    assert isinstance(current.near, CurrentModel)
    assert current.plain.lags is None and current.plain.region == "DK1"
    assert current.near.lags == LagConfig((NEAR_TERM_FEATURE,))
    assert isinstance(reference, NearTermModel) and reference.days == 2
    assert isinstance(reference.plain, LightGbmReference)
    assert isinstance(reference.near, LightGbmReference)
    assert reference.plain.lags is None
    assert reference.near.lags == LagConfig((NEAR_TERM_FEATURE,))


def test_near_term_model_equals_its_two_models_on_their_days() -> None:
    """Days 1-2 are the lag model's forecast, days 3-7 the plain model's."""
    origin = date(2026, 6, 15)
    series = _noisy_series(date(2026, 5, 20), 30)
    config = _config(origin, window_days=21, horizon_days=4)
    history = history_for(series, origin, config)
    targets, horizon = target_slots(origin, config)
    plain = CurrentModel(TZ, _small_gbm)
    near = CurrentModel(TZ, _small_gbm, lags=LagConfig((NEAR_TERM_FEATURE,)))

    routed = NearTermModel(plain, near, TZ, 2, "routed").forecast(history, targets)

    first = horizon <= 2
    np.testing.assert_array_equal(routed[first], near.forecast(history, targets)[first])
    np.testing.assert_array_equal(
        routed[~first], plain.forecast(history, targets)[~first]
    )
    assert not np.array_equal(routed, plain.forecast(history, targets))


def test_parse_args_near_term_days() -> None:
    """The default is the integration's per region; --lags is a one-model experiment."""
    assert {"DK1", "DK2"} == NEAR_TERM_REGIONS
    for region in ("DK1", "DK2"):
        args = backtest.parse_args(["--region", region])
        assert args.near_term_days == NEAR_TERM_DAYS
    assert backtest.parse_args(["--region", "SE3"]).near_term_days == 0
    args = backtest.parse_args(["--region", "SE3", "--near-term-days", "1"])
    assert args.near_term_days == 1
    assert backtest.parse_args(["--near-term-days", "0"]).near_term_days == 0
    assert backtest.parse_args(["--near-term-days", "1"]).near_term_days == 1
    assert backtest.parse_args(["--lags", "all"]).near_term_days == 0
    with pytest.raises(SystemExit):
        backtest.parse_args(["--near-term-days", "-1"])


def test_future_prices_cannot_change_the_near_term_forecast() -> None:
    """Poisoning every price at or after the cutoff leaves both models' rows alone."""
    origin = date(2026, 6, 15)
    clean = _noisy_series(date(2026, 5, 20), 30)
    future = clean.starts >= local_midnight(origin, TZ)
    config = _config(origin, window_days=21, horizon_days=3)
    targets, _ = target_slots(origin, config)

    forecasts = []
    for poison in (None, 1e6, math.nan):
        prices = clean.prices.copy()
        if poison is not None:
            prices[future] = poison
        history = history_for(PriceSeries(clean.starts, prices), origin, config)
        plain = CurrentModel(TZ, _small_gbm)
        near = CurrentModel(TZ, _small_gbm, lags=LagConfig((NEAR_TERM_FEATURE,)))
        model = NearTermModel(plain, near, TZ, NEAR_TERM_DAYS, "routed")
        forecasts.append(model.forecast(history, targets))

    assert np.all(np.isfinite(forecasts[0]))
    np.testing.assert_array_equal(forecasts[0], forecasts[1])
    np.testing.assert_array_equal(forecasts[0], forecasts[2])
