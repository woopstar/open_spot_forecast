"""p10/p90 prediction intervals per forecast slot (#280).

The band is the 10th-90th percentile of the raw model's past signed errors
per lead-time bucket and local hour (``ml/intervals.py``). It is never
clamped at 0, always contains the price, and takes the price's path through
``PriceOutput`` (tariff, hourly mean, unit, surcharge, VAT).
"""

from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from homeassistant.util import dt as dt_util

from custom_components.open_spot_forecast.const import PRICE_HIGH, PRICE_LOW
from custom_components.open_spot_forecast.forecast_attributes import (
    ATTRIBUTE_BUDGET_BYTES,
    attributes_size,
    compact_forecast,
    detailed_forecast,
    fit_compact,
)
from custom_components.open_spot_forecast.ml.intervals import (
    INTERVAL_QUANTILES,
    INTERVAL_WINDOW_DAYS,
    MIN_INTERVAL_DAYS,
    MIN_INTERVAL_SAMPLES,
    error_quantiles,
    group_quantiles,
    interval_bounds,
    interval_key,
)
from custom_components.open_spot_forecast.ml.lead_time import prediction_slot
from custom_components.open_spot_forecast.ml.predictor import SpotPricePredictor
from custom_components.open_spot_forecast.price_output import PriceOutput
from custom_components.open_spot_forecast.sensor import MLPredictionSensor
from custom_components.open_spot_forecast.services import forecast_response
from custom_components.open_spot_forecast.tariffs import TariffSchedule
from custom_components.open_spot_forecast.time_slots import (
    slot_start_in_day,
    slots_in_local_day,
)
from scripts.backtest_intervals import IntervalScore, format_intervals

pytestmark = pytest.mark.usefixtures("copenhagen_time_zone")

CPH = ZoneInfo("Europe/Copenhagen")
# Errors spread evenly over -0.5..0.5, and their band's quantiles
SPREAD = list(np.linspace(-0.5, 0.5, 101))
Q_LOW, Q_HIGH = (float(q) for q in np.quantile(SPREAD, INTERVAL_QUANTILES))


def _spread_rows(
    bucket: str, slot: int, scale: float = 1.0, days: int = 7
) -> list[tuple[str, str, int, float]]:
    """``(date, bucket, slot, error)`` rows of SPREAD × scale over ``days`` dates."""
    return [
        (f"2026-11-{1 + index % days:02d}", bucket, slot, error * scale)
        for index, error in enumerate(SPREAD)
    ]


def _store_spread(
    predictor: SpotPricePredictor,
    last: date,
    slot: int,
    bucket: str,
    scale: float = 1.0,
) -> None:
    """Store SPREAD × scale as one slot's errors over the days up to ``last``."""
    for index, error in enumerate(SPREAD):
        day = last - timedelta(days=index % MIN_INTERVAL_DAYS)
        predictor.storage.add_interval_errors(
            day.isoformat(), slot, {bucket: [error * scale]}
        )


def _hass(tmp_path: Path) -> Mock:
    hass = Mock()
    hass.config.path.return_value = str(tmp_path / ".storage")
    return hass


@pytest.fixture
def predictor(tmp_path: Path) -> Iterator[SpotPricePredictor]:
    predictor = SpotPricePredictor(_hass(tmp_path), "DK1")
    yield predictor
    predictor.storage.close()


# --- Quantiles and bounds ---------------------------------------------------------


def test_error_quantiles_need_enough_errors() -> None:
    assert error_quantiles([0.1] * (MIN_INTERVAL_SAMPLES - 1)) is None
    assert error_quantiles(SPREAD) == (pytest.approx(Q_LOW), pytest.approx(Q_HIGH))
    # Aimed at p10/p90 of the actual price, wider than the errors' own (#280)
    assert INTERVAL_QUANTILES[0] < 0.1 and INTERVAL_QUANTILES[1] > 0.9


def test_bounds_follow_from_the_errors_of_predicted_minus_actual() -> None:
    """An error is predicted - actual: a model that overpredicts has its
    actual prices below the prediction, so the band lies below it too."""
    low, high = interval_bounds(1.0, 1.0, (0.0, 0.3)) or (0.0, 0.0)

    assert low == pytest.approx(0.7)
    assert high == pytest.approx(1.0)


def test_bounds_contain_the_price_when_errors_are_one_sided() -> None:
    low, high = interval_bounds(1.0, 1.0, (0.1, 0.3)) or (0.0, 0.0)

    assert low == pytest.approx(0.7)
    # raw - q10 = 0.9 would be below the price: the band is widened to it
    assert high == pytest.approx(1.0)
    assert low <= 1.0 <= high


def test_bounds_are_not_clamped_at_zero() -> None:
    low, high = interval_bounds(-0.05, -0.05, (-0.4, 0.4)) or (0.0, 0.0)

    assert low == pytest.approx(-0.45)
    assert high == pytest.approx(0.35)


def test_bounds_are_around_the_raw_price_and_contain_the_served_price() -> None:
    """The errors are the raw model's (#203): the band is built around the
    prediction before its bias offset, and still contains the served price."""
    low, high = interval_bounds(0.8, 1.0, (-0.1, 0.1)) or (0.0, 0.0)

    assert low == pytest.approx(0.8)
    assert high == pytest.approx(1.1)


def test_no_bounds_without_quantiles() -> None:
    assert interval_bounds(1.0, 1.0, None) is None


def test_errors_are_grouped_per_bucket_and_local_hour() -> None:
    rows = (
        _spread_rows("day_1", 48)  # 12:00
        + [("2026-11-01", "day_1", 51, 1.0)] * 10  # 12:45, the same hour
        + _spread_rows("day_2", 48, scale=2.0)
        + _spread_rows("day_1", 0)[: MIN_INTERVAL_SAMPLES - 1]  # too few
        # Enough errors, but of too few days: one day's errors move together
        + _spread_rows("day_3", 48, days=MIN_INTERVAL_DAYS - 1)
    )

    quantiles = group_quantiles(rows)

    assert interval_key("day_1", 51) == ("day_1", 12)
    assert set(quantiles) == {("day_1", 12), ("day_2", 12)}
    assert quantiles[("day_2", 12)] == (
        pytest.approx(2 * Q_LOW),
        pytest.approx(2 * Q_HIGH),
    )
    # The 12:45 errors are in 12:00's hour
    assert quantiles[("day_1", 12)][1] > Q_HIGH


# --- Recording and storage --------------------------------------------------------


def _stored(
    slot: datetime, lead_hours: float, price: float, offset: float | None
) -> dict[str, Any]:
    return {
        "start": slot.isoformat(),
        "stored_at": (slot - timedelta(hours=lead_hours)).isoformat(),
        "price": price,
        "bias_offset": offset,
        "hour": slot.hour,
        "minute": slot.minute,
    }


def test_scored_slots_store_their_raw_errors_per_bucket(
    predictor: SpotPricePredictor,
) -> None:
    slot = dt_util.now().replace(hour=12, minute=30, second=0, microsecond=0)
    predictions = [
        _stored(slot, 5, 1.2, 0.1),  # raw 1.3: error +0.3
        _stored(slot, 30, 0.8, 0.0),  # error -0.2
        _stored(slot, 6, 9.0, None),  # before #203: offset unknown, left out
    ]

    predictor.record_lead_time_accuracy(predictions, 1.0)

    rows = sorted(predictor.storage.get_interval_errors("2000-01-01"))
    assert [row[:3] for row in rows] == [
        (slot.date().isoformat(), "day_1", 50),
        (slot.date().isoformat(), "day_2", 50),
    ]
    assert [error for *_, error in rows] == [pytest.approx(0.3), pytest.approx(-0.2)]


def test_prediction_slot_reads_the_stored_wall_clock(fall_back_day: date) -> None:
    """Both passes of the repeated hour of a fall-back day are 02:xx by wall
    clock, the slot the self-learning keys them by."""
    first = slot_start_in_day(fall_back_day, 8, CPH)  # 02:00, summer time
    second = slot_start_in_day(fall_back_day, 12, CPH)  # 02:00, winter time

    assert first.utcoffset() != second.utcoffset()
    assert prediction_slot({"start": first.isoformat()}) == 8
    assert prediction_slot({"start": second.isoformat()}) == 8
    assert prediction_slot({"start": "x", "hour": 2, "minute": 30}) == 10
    assert prediction_slot({"start": "not a time"}) is None


def test_errors_outside_the_window_are_pruned(predictor: SpotPricePredictor) -> None:
    today = date(2026, 11, 20)
    old = (today - timedelta(days=INTERVAL_WINDOW_DAYS)).isoformat()
    recent = (today - timedelta(days=INTERVAL_WINDOW_DAYS - 1)).isoformat()
    predictor.storage.add_interval_errors(old, 48, {"day_1": [5.0] * 100})
    _store_spread(predictor, today, 48, "day_1")
    assert recent < (today - timedelta(days=MIN_INTERVAL_DAYS)).isoformat()

    predictor.refresh_interval_quantiles(today)

    assert predictor.interval_quantiles == {
        ("day_1", 12): (pytest.approx(Q_LOW), pytest.approx(Q_HIGH))
    }
    assert len(predictor.storage.get_interval_errors("2000-01-01")) == len(SPREAD)


def test_clearing_the_learning_data_drops_the_errors(
    predictor: SpotPricePredictor,
) -> None:
    predictor.storage.add_interval_errors("2026-11-20", 48, {"day_1": SPREAD})

    predictor.storage.clear_all()

    assert predictor.storage.get_interval_errors("2000-01-01") == []


# --- Predictions ------------------------------------------------------------------


def _run(predictor: SpotPricePredictor, starts: list[datetime], price: float) -> None:
    predictor.price_model = MagicMock()
    predictor.price_model.predict.side_effect = lambda rows: np.full(len(rows), price)
    predictor.is_trained = True
    features = predictor._combine_features(
        [{"start": start.isoformat()} for start in starts], {}
    )
    predictor._generate_predictions(features, 2, 15)


def test_predictions_get_the_bounds_of_their_bucket_and_hour(
    predictor: SpotPricePredictor,
) -> None:
    now = dt_util.now().replace(minute=0, second=0, microsecond=0)
    starts = [now + timedelta(hours=5), now + timedelta(hours=29)]
    slot = starts[0].hour * 4
    _store_spread(predictor, now.date(), slot, "day_1")
    _store_spread(predictor, now.date(), slot, "day_2", scale=2.0)

    # A negative price keeps negative bounds (#280: never clamped at 0)
    _run(predictor, starts, -0.1)

    first, second = predictor.predictions
    assert first["price"] == pytest.approx(-0.1)
    assert first[PRICE_LOW] == pytest.approx(-0.1 - Q_HIGH)
    assert first[PRICE_HIGH] == pytest.approx(-0.1 - Q_LOW)
    assert first[PRICE_HIGH] > 0 > first[PRICE_LOW]
    # A day later: the day_2 errors, twice as wide
    assert second[PRICE_LOW] == pytest.approx(-0.1 - 2 * Q_HIGH)
    assert second[PRICE_HIGH] == pytest.approx(-0.1 - 2 * Q_LOW)
    for prediction in predictor.predictions:
        assert prediction[PRICE_LOW] <= prediction["price"] <= prediction[PRICE_HIGH]


def test_predictions_without_enough_errors_have_no_bounds(
    predictor: SpotPricePredictor,
) -> None:
    now = dt_util.now().replace(minute=0, second=0, microsecond=0)
    start = now + timedelta(hours=5)
    predictor.storage.add_interval_errors(
        now.date().isoformat(), start.hour * 4, {"day_1": [0.1] * 10}
    )

    _run(predictor, [start], 0.5)

    assert PRICE_LOW not in predictor.predictions[0]
    assert PRICE_HIGH not in predictor.predictions[0]


# --- PriceOutput, attributes and get_forecast ------------------------------------


def _slot_predictions(day: date, count: int) -> list[dict[str, Any]]:
    predictions = []
    for index in range(count):
        start = slot_start_in_day(day, index, CPH)
        price = -0.2 + 0.01 * index
        predictions.append(
            {
                "start": start.isoformat(),
                "end": slot_start_in_day(day, index + 1, CPH).isoformat(),
                "price": price,
                PRICE_LOW: price - 0.3,
                PRICE_HIGH: price + 0.5,
                "confidence": 0.8,
            }
        )
    return predictions


def test_bounds_take_the_prices_path_through_price_output() -> None:
    day = date(2026, 11, 24)
    predictions = _slot_predictions(day, 4)
    tariffs = TariffSchedule(
        {slot_start_in_day(day, i, CPH).astimezone(UTC): 0.6 for i in range(4)}
    )
    output = PriceOutput(vat=0.25, surcharge=0.1, precision=4)

    forecast = output.forecast(predictions, tariffs)

    for entry, prediction in zip(forecast, predictions, strict=True):
        for key in ("price", PRICE_LOW, PRICE_HIGH):
            assert entry[key] == pytest.approx(
                output.convert(prediction[key] + 0.6), abs=1e-4
            )
        assert entry[PRICE_LOW] <= entry["price"] <= entry[PRICE_HIGH]


def test_hourly_bounds_are_the_means_of_the_slots_bounds(
    dst_day: tuple[date, int],
) -> None:
    day, slots = dst_day
    predictions = _slot_predictions(day, slots_in_local_day(day, tz=CPH))
    # One slot of the last hour has no band: that hour has none
    del predictions[-1][PRICE_LOW]
    output = PriceOutput(vat=0.0, hourly_average=True, precision=6)

    forecast = output.forecast(predictions)

    assert len(forecast) == slots // 4
    first = forecast[0]
    mean = sum(p["price"] for p in predictions[:4]) / 4
    assert first["price"] == pytest.approx(mean, abs=1e-6)
    assert first[PRICE_LOW] == pytest.approx(mean - 0.3, abs=1e-6)
    assert first[PRICE_HIGH] == pytest.approx(mean + 0.5, abs=1e-6)
    assert PRICE_LOW not in forecast[-1]
    assert PRICE_HIGH in forecast[-1]


def test_confirmed_prices_have_no_bounds() -> None:
    forecast = PriceOutput().forecast(
        [{"start": "2026-11-24T00:00:00+01:00", "price": 0.5, "source": "actual"}]
    )

    assert PRICE_LOW not in forecast[0]
    assert PRICE_HIGH not in forecast[0]


def test_detailed_and_compact_attributes_carry_the_bounds() -> None:
    entries = PriceOutput().forecast(_slot_predictions(date(2026, 11, 24), 8))
    entries[0].pop(PRICE_LOW)
    entries[0].pop(PRICE_HIGH)

    detailed = detailed_forecast(entries, "DKK/kWh")
    compact = compact_forecast(entries, "DKK/kWh", 15)

    assert PRICE_LOW in detailed[1] and PRICE_HIGH in detailed[1]
    assert compact["l"][0] is None and compact["h"][0] is None
    assert compact["l"][1:] == [entry[PRICE_LOW] for entry in entries[1:]]
    assert compact["h"][1:] == [entry[PRICE_HIGH] for entry in entries[1:]]
    assert "l" not in compact_forecast(
        PriceOutput().forecast([{"start": "2026-11-24T00:00:00+01:00", "price": 0.5}]),
        "DKK/kWh",
        15,
    )


def test_compact_bounds_that_fit_are_kept() -> None:
    entries = PriceOutput().forecast(_slot_predictions(date(2026, 11, 24), 96))
    attributes = {"predictions": compact_forecast(entries, "DKK/kWh", 15)}

    fit_compact(attributes, "predictions")

    assert len(attributes["predictions"]["l"]) == 96
    assert attributes_size(attributes) <= ATTRIBUTE_BUDGET_BYTES


def test_compact_week_drops_the_bounds_before_any_hour() -> None:
    """A week of 15-minute slots with bounds is over the recorder's budget:
    the bounds go, the forecast keeps all its hours."""
    first = date(2026, 11, 23)
    predictions = [
        prediction
        for day in range(7)
        for prediction in _slot_predictions(first + timedelta(days=day), 96)
    ]
    for index, prediction in enumerate(predictions):
        # Prices with every decimal, the worst case for the size
        prediction["price"] = -1.3 + (index * 0.0371937) % 4.8
        prediction[PRICE_LOW] = prediction["price"] - 0.4123457
        prediction[PRICE_HIGH] = prediction["price"] + 0.5123457
    entries = PriceOutput(precision=4).forecast(predictions)
    attributes = {"predictions": compact_forecast(entries, "DKK/kWh", 15)}
    without = {
        "predictions": {
            key: value
            for key, value in attributes["predictions"].items()
            if key not in ("l", "h")
        }
    }
    assert attributes_size(attributes) > ATTRIBUTE_BUDGET_BYTES
    fit_compact(without, "predictions")

    fit_compact(attributes, "predictions")

    compact = attributes["predictions"]
    assert "l" not in compact and "h" not in compact
    assert compact["s"] == without["predictions"]["s"]
    assert attributes_size(attributes) <= ATTRIBUTE_BUDGET_BYTES


def test_get_forecast_returns_the_bounds() -> None:
    day = date(2026, 11, 24)
    predictions = _slot_predictions(day, 8)

    response = forecast_response(
        predictions,
        PriceOutput(vat=0.0, precision=4),
        "DKK",
        None,
        slot_start_in_day(day, 0, CPH),
        hours=1,
    )

    assert len(response["forecast"]) == 4
    for entry, prediction in zip(response["forecast"], predictions[:4], strict=True):
        assert entry[PRICE_LOW] == pytest.approx(prediction[PRICE_LOW], abs=1e-4)
        assert entry[PRICE_HIGH] == pytest.approx(prediction[PRICE_HIGH], abs=1e-4)


# --- Backtest coverage ------------------------------------------------------------


def _origin_arrays(origin: date, days: int) -> tuple[np.ndarray, np.ndarray]:
    starts, horizon = [], []
    for day in range(days):
        for index in range(slots_in_local_day(origin + timedelta(days=day), tz=CPH)):
            starts.append(
                int(
                    slot_start_in_day(
                        origin + timedelta(days=day), index, CPH
                    ).timestamp()
                )
            )
            horizon.append(day + 1)
    return np.asarray(starts), np.asarray(horizon)


def test_backtest_bands_cover_about_80_percent_of_actual_prices() -> None:
    """Errors drawn from one distribution: the band of the errors before each
    origin covers the share of the later actual prices its quantiles span
    (88 %: wider than p10/p90 on purpose, for errors that drift)."""
    rng = np.random.default_rng(280)
    score = IntervalScore(CPH)
    first = date(2026, 3, 1)
    for offset in range(60):
        origin = first + timedelta(days=offset)
        starts, horizon = _origin_arrays(origin, 2)
        predicted = rng.normal(5.0, 2.0, len(starts))
        actual = predicted - rng.normal(0.0, horizon.astype(float))
        score.add(origin, starts, horizon, actual, predicted)

    coverage = score.summary(2)

    for day, result in enumerate(coverage, start=1):
        assert result is not None
        low, high = INTERVAL_QUANTILES
        assert result.inside == pytest.approx(high - low, abs=0.03)
        assert result.below == pytest.approx(low, abs=0.02)
        # About 2 × 1.55 standard deviations, wider a day later
        assert result.width == pytest.approx(2 * 1.5548 * day, rel=0.1)
        # The first origins have no errors before them yet
        assert 0.3 < result.with_band < 1.0
    assert "| current | 1 |" in format_intervals({"current": score}, 2)


def test_backtest_bands_only_use_errors_known_at_the_origin() -> None:
    """A model that is exact until a day and then off by 10: the band of that
    day cannot know it yet."""
    score = IntervalScore(CPH)
    first = date(2026, 3, 1)
    for offset in range(40):
        origin = first + timedelta(days=offset)
        starts, horizon = _origin_arrays(origin, 1)
        predicted = np.full(len(starts), 5.0)
        error = 10.0 if offset >= 39 else 0.0
        score.add(origin, starts, horizon, predicted - error, predicted)

    (coverage,) = score.summary(1)

    assert coverage is not None
    # Only the last origin is off; its band was learned from exact days
    assert 0 < coverage.below < 0.1
    assert coverage.inside == pytest.approx(1 - coverage.below)


def test_the_forecast_sensor_converts_the_bounds_like_the_price() -> None:
    day = date(2026, 11, 24)
    predictor = MagicMock()
    predictor.predictions = _slot_predictions(day, 8)
    predictor.get_prediction_stats.return_value = {}
    output = PriceOutput(vat=0.25, surcharge=0.1, precision=4)
    sensor = MLPredictionSensor(
        Mock(),
        MagicMock(entry_id="test"),
        {"ml_predictor": predictor},
        "DKK",
        output,
        48,
    )

    with patch(
        "homeassistant.util.dt.utcnow",
        return_value=slot_start_in_day(day, 0, CPH).astimezone(UTC),
    ):
        sensor._state = sensor._build_state(dt_util.utcnow())
    entries = sensor.extra_state_attributes["predictions"]

    assert len(entries) == 8
    for entry, prediction in zip(entries, predictor.predictions, strict=True):
        assert entry[PRICE_LOW] == output.convert(prediction[PRICE_LOW])
        assert entry[PRICE_HIGH] == output.convert(prediction[PRICE_HIGH])
        assert entry[PRICE_LOW] <= entry["price"] <= entry[PRICE_HIGH]
