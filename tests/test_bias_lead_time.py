"""Bias offsets per slot and lead-time bucket (#118), from raw errors (#203).

An offset is the mean of the raw model's errors (stored price + stored
offset - actual) over the slot's last days in its lead-time bucket. The
simulations below replay what happens around a change of the bias's sign,
with the lag a real forecast has: an error is measured a lead time after the
forecast was made.
"""

import logging
import math
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pytest

from homeassistant.util import dt as dt_util

from custom_components.open_spot_forecast.const import (
    BIAS_FALLBACK_BUCKET,
    LEAD_TIME_BUCKETS,
)
from custom_components.open_spot_forecast.ml import bias
from custom_components.open_spot_forecast.ml.bias import (
    BIAS_WINDOW_DAYS,
    MIN_BIAS_SAMPLES,
    RAW_ERRORS_KEY,
    add_raw_errors,
    slot_offsets,
    window_offset,
)
from custom_components.open_spot_forecast.ml.bias_storage import (
    LEAD_TIME_BIAS_SCHEMA_VERSION,
    RAW_ERROR_BIAS_SCHEMA_VERSION,
)
from custom_components.open_spot_forecast.ml.lead_time import (
    prediction_bucket,
    raw_bucket_errors,
)
from custom_components.open_spot_forecast.ml.learning import (
    MAX_ERROR_SAMPLES,
    add_matched_errors,
    new_slot_metrics,
)
from custom_components.open_spot_forecast.ml.predictor import SpotPricePredictor
from custom_components.open_spot_forecast.ml.storage import LearningStorage

# The offsets are tested with the windows of #203: no bucket is corrected
# since #206 (``BIAS_WINDOW_DAYS`` above is the shipped, empty value)
pytestmark = pytest.mark.usefixtures("bias_windows")

SLOT = 48  # 12:00
BUCKETS = [bucket for bucket, _upper in LEAD_TIME_BUCKETS]
TODAY = date(2026, 10, 13)
# A lead time inside each bucket, and the days between a forecast and its slot
LEAD_HOURS = {"day_1": 20, "day_2": 44, "day_3": 68, "day_4_plus": 116}


def _hass(tmp_path: Path) -> Mock:
    hass = Mock()
    hass.config.path.return_value = str(tmp_path / ".storage")
    return hass


@pytest.fixture
def predictor(tmp_path: Path) -> Iterator[SpotPricePredictor]:
    predictor = SpotPricePredictor(_hass(tmp_path), "DK1")
    yield predictor
    predictor.storage.close()


def _metrics(windows: dict[str, dict[date, list[float]]]) -> dict:
    """A slot's metrics with the given raw errors per bucket and measured day."""
    metrics = new_slot_metrics()
    for bucket, days in windows.items():
        metrics[RAW_ERRORS_KEY][bucket] = {
            day.isoformat(): [len(errors), sum(errors)] for day, errors in days.items()
        }
        for errors in days.values():
            metrics["errors"] += errors
            metrics["abs_errors"] += [abs(e) for e in errors]
            metrics["pct_errors"] += [0.0] * len(errors)
            metrics["predictions"] += [1.0 + e for e in errors]
            metrics["actuals"] += [1.0] * len(errors)
            metrics["count"] += len(errors)
    return metrics


def _prediction(
    slot_start: datetime,
    lead_hours: float,
    price: float,
    bias_offset: float | None = 0.0,
) -> dict:
    stored_at = slot_start - timedelta(hours=lead_hours)
    return {
        "start": slot_start.isoformat(),
        "stored_at": stored_at.isoformat(),
        "price": price,
        "bias_offset": bias_offset,
    }


def _noon(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, 12, tzinfo=dt_util.DEFAULT_TIME_ZONE)


def _tomorrow_noon() -> datetime:
    return _noon(dt_util.now().date() + timedelta(days=1))


# --- The offset is the mean of the window's raw errors ---------------------------


def test_the_raw_error_adds_the_stored_offset_back() -> None:
    """A stored price had its offset subtracted: raw = price + offset - actual."""
    start = _noon(TODAY)
    predictions = [
        _prediction(start, 20, 1.5, bias_offset=-0.4),
        _prediction(start, 44, 1.2, bias_offset=0.0),
        _prediction(start, 44, 1.3, bias_offset=None),  # stored before #203
        _prediction(start, -1, 9.0, bias_offset=0.2),  # not a forecast
    ]

    errors = raw_bucket_errors(predictions, 1.0)

    assert errors == {
        "day_1": [(pytest.approx(0.1), pytest.approx(0.5))],
        "day_2": [(pytest.approx(0.2), pytest.approx(0.2))],
    }


def test_each_bucket_learns_its_own_offset(predictor: SpotPricePredictor) -> None:
    predictor.error_metrics = {
        SLOT: _metrics(
            {
                "day_1": {TODAY: [0.3] * 3},
                "day_2": {TODAY: [0.5, 0.7], TODAY - timedelta(days=1): [0.6]},
            }
        )
    }

    predictor._update_bias_correction(SLOT, TODAY)

    assert predictor.bias_correction == {
        SLOT: {"day_1": pytest.approx(0.3), "day_2": pytest.approx(0.6)}
    }


def test_the_offset_is_not_smoothed_towards_the_previous_one(
    predictor: SpotPricePredictor,
) -> None:
    """No moving average on top of the window: the old offset has no say (#203)."""
    predictor.bias_correction = {SLOT: {"day_1": -0.35, "day_2": -0.3}}
    predictor.error_metrics = {SLOT: _metrics({"day_1": {TODAY: [0.4] * 3}})}

    predictor._update_bias_correction(SLOT, TODAY)

    # day_2 has no errors in its window: its old offset is dropped, not kept
    assert predictor.bias_correction == {SLOT: {"day_1": pytest.approx(0.4)}}


def test_a_window_needs_three_samples(predictor: SpotPricePredictor) -> None:
    assert MIN_BIAS_SAMPLES == 3
    predictor.bias_correction = {SLOT: {"day_1": 0.2}}
    predictor.error_metrics = {SLOT: _metrics({"day_1": {TODAY: [1.0, 1.0]}})}

    predictor._update_bias_correction(SLOT, TODAY)

    assert predictor.bias_correction == {}


def test_the_window_is_the_last_days_with_errors() -> None:
    """Four measured days at most, none more than four days old."""
    assert bias.BIAS_WINDOW_DAYS["day_1"] == 4
    days = {
        (TODAY - timedelta(days=age)).isoformat(): [3, 3 * error]
        for age, error in enumerate([0.1, 0.2, 0.3, 0.4, 5.0])
    }

    # Today is measured: today and the three days before it
    assert window_offset(days, TODAY, 4) == pytest.approx(0.25)
    # Tomorrow before the slot is matched: the same four days
    assert window_offset(days, TODAY + timedelta(days=1), 4) == pytest.approx(0.25)
    # Two days on: today and the day before it are all that is recent
    assert window_offset(days, TODAY + timedelta(days=3), 4) == pytest.approx(0.15)
    assert window_offset(days, TODAY + timedelta(days=4), 4) == pytest.approx(0.1)
    # A day measured after ``today`` (a replay of older days) is not used
    assert window_offset(days, TODAY - timedelta(days=4), 4) == pytest.approx(5.0)


def test_a_bucket_without_recent_errors_has_no_offset(
    predictor: SpotPricePredictor,
) -> None:
    """E.g. ``day_1`` from 13:00 on: its prices are known before a forecast
    under 24 hours ahead is stored, so it never gets an error."""
    stale = TODAY - timedelta(days=bias.BIAS_WINDOW_DAYS["day_1"] + 1)
    predictor.bias_correction = {SLOT: {"day_1": -0.38, "day_2": 0.1}}
    predictor.error_metrics = {
        SLOT: _metrics({"day_1": {stale: [-0.4] * 9}, "day_2": {TODAY: [0.2] * 3}})
    }

    predictor._update_bias_correction(SLOT, TODAY)

    assert predictor.bias_correction == {SLOT: {"day_2": pytest.approx(0.2)}}
    assert predictor._bias_offset(SLOT, "day_1") is None
    assert predictor.apply_bias_correction(1.0, SLOT, "day_1") == pytest.approx(1.0)


def test_no_bucket_uses_another_buckets_offset(predictor: SpotPricePredictor) -> None:
    """Before #203 a bucket without an offset used ``day_1``'s, weeks old."""
    predictor.bias_correction = {SLOT: {BIAS_FALLBACK_BUCKET: 0.2}}

    assert predictor.apply_bias_correction(1.0, SLOT, "day_1") == pytest.approx(0.8)
    for bucket in ("day_2", "day_3", "day_4_plus"):
        assert predictor._bias_offset(SLOT, bucket) is None
        assert predictor.apply_bias_correction(1.0, SLOT, bucket) == pytest.approx(1.0)
    assert predictor.apply_bias_correction(1.0, SLOT + 1, "day_1") == pytest.approx(1.0)


def test_offsets_are_refreshed_before_they_are_used(
    predictor: SpotPricePredictor,
) -> None:
    """After days without a match the stored offsets are older than their window."""
    predictor.error_metrics = {SLOT: _metrics({"day_1": {TODAY: [0.3] * 3}})}
    predictor.bias_correction = {SLOT: {"day_1": 0.3}, SLOT + 1: {"day_2": 0.5}}

    predictor.refresh_bias_offsets(TODAY + timedelta(days=2))
    assert predictor.bias_correction == {SLOT: {"day_1": pytest.approx(0.3)}}

    predictor.refresh_bias_offsets(TODAY + timedelta(days=5))
    assert predictor.bias_correction == {}


def test_only_the_near_buckets_are_corrected() -> None:
    """Day 3 and day 4+ errors are 3-8 days old when they could be used."""
    assert set(bias.BIAS_WINDOW_DAYS) == {"day_1", "day_2"}
    start = _noon(TODAY)
    metrics = new_slot_metrics()
    predictions = [
        _prediction(start, LEAD_HOURS[bucket], 1.5) for bucket in BUCKETS for _ in "abc"
    ]

    add_matched_errors(metrics, predictions, 1.0, TODAY)

    assert set(metrics[RAW_ERRORS_KEY]) == {"day_1", "day_2"}
    assert slot_offsets(metrics, TODAY) == {
        "day_1": pytest.approx(0.5),
        "day_2": pytest.approx(0.5),
    }


# --- Simulations: a bias that stays, and one that changes sign --------------------


def _simulate(
    bias_by_day: list[float], noise: float = 0.0
) -> dict[str, dict[str, list[float]]]:
    """Replay one slot day by day; return each bucket's offsets and errors.

    Every day the slot's forecasts are matched: three per bucket, each made
    its lead time earlier with the offset the slot had on that day. The raw
    model is ``bias_by_day[day]`` off (plus noise), the same at every lead.

    Returns:
        ``{bucket: {"offset": offset after each day's match (NaN without),
        "raw": the day's mean |raw error|, "corrected": mean |stored error|}}``.
    """
    rng = np.random.default_rng(0)
    first = date(2026, 9, 1)
    actual = 1.0
    metrics = new_slot_metrics()
    offsets_after: list[dict[str, float]] = []
    result: dict[str, dict[str, list[float]]] = {
        bucket: {"offset": [], "raw": [], "corrected": []} for bucket in BUCKETS
    }
    for index, model_bias in enumerate(bias_by_day):
        day = first + timedelta(days=index)
        predictions = []
        for bucket, lead_hours in LEAD_HOURS.items():
            made = index - math.ceil(lead_hours / 24)
            offset = offsets_after[made].get(bucket, 0.0) if made >= 0 else 0.0
            raw_errors = model_bias + rng.normal(0.0, noise, 3)
            predictions += [
                _prediction(_noon(day), lead_hours, actual + raw - offset, offset)
                for raw in raw_errors.tolist()
            ]
            result[bucket]["raw"].append(float(np.mean(np.abs(raw_errors))))
            result[bucket]["corrected"].append(
                float(np.mean(np.abs(raw_errors - offset)))
            )
        add_matched_errors(metrics, predictions, actual, day)
        offsets_after.append(slot_offsets(metrics, day))
        for bucket in BUCKETS:
            result[bucket]["offset"].append(offsets_after[-1].get(bucket, math.nan))
    return result


@pytest.mark.parametrize("model_bias", [0.25, -0.4])
def test_with_a_constant_bias_the_offset_is_the_bias(model_bias: float) -> None:
    """It does not settle at half the bias: the errors it learns from are raw.

    An average of the errors left after the correction would (#15).
    """
    result = _simulate([model_bias] * 30, noise=0.02)

    for bucket, window_days in bias.BIAS_WINDOW_DAYS.items():
        offsets = result[bucket]["offset"]
        assert offsets[0] == pytest.approx(model_bias, abs=0.03)
        assert offsets[window_days:] == pytest.approx(
            [model_bias] * (30 - window_days), abs=0.02
        )
        # Once the forecasts carry the offset, the stored errors are noise
        lag = math.ceil(LEAD_HOURS[bucket] / 24)
        assert max(result[bucket]["corrected"][lag:]) < 0.05
        assert min(result[bucket]["raw"]) > abs(model_bias) - 0.05


def test_after_a_sign_change_the_offset_follows_within_its_window() -> None:
    """The offset has the new sign within the window's span, and is the new
    bias once the window holds only errors from after the change (#203).

    The offsets before #203 moved a tenth of the remaining error a day: ten
    days on they still had the old sign.
    """
    before, after, flip = -0.3, 0.3, 12
    result = _simulate([before] * flip + [after] * 12)

    for bucket, window_days in bias.BIAS_WINDOW_DAYS.items():
        offsets = result[bucket]["offset"]
        assert offsets[flip - 1] == pytest.approx(before)
        # Half the window is from after the change on its second day
        assert offsets[flip + 1] == pytest.approx(0.0, abs=1e-9)
        assert offsets[flip + 2] > 0
        assert offsets[flip + window_days - 1 :] == pytest.approx(
            [after] * (12 - window_days + 1)
        )
        # Over the whole run the correction removes more error than it adds
        assert sum(result[bucket]["corrected"]) < sum(result[bucket]["raw"])


def test_a_later_bucket_would_add_error_after_every_sign_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Why day 3 and day 4+ have no offset: their errors arrive too late.

    With the bias changing sign every 8 days (the DK1 export of 2026-10-12
    had runs of 4 and 8 days), day 1 and day 2 gain, and a day 4+ offset with
    the same window is on the wrong side of the change for 5 days each time.
    """
    monkeypatch.setattr(
        bias, "BIAS_WINDOW_DAYS", dict.fromkeys(BUCKETS, 4), raising=True
    )
    regime = [0.3] * 8 + [-0.3] * 8

    result = _simulate(regime * 4)

    def gain(bucket: str) -> float:
        return sum(result[bucket]["raw"]) - sum(result[bucket]["corrected"])

    assert gain("day_1") > gain("day_2") > 0
    assert gain("day_4_plus") < 0


# --- Applying the offset of the prediction's lead time ----------------------------


def test_the_offset_of_the_prediction_bucket_is_applied(
    predictor: SpotPricePredictor,
) -> None:
    predictor.bias_correction = {SLOT: {"day_1": 0.2, "day_2": -0.4}}

    assert predictor.apply_bias_correction(1.0, SLOT, "day_1") == pytest.approx(0.8)
    assert predictor.apply_bias_correction(1.0, SLOT, "day_2") == pytest.approx(1.4)
    # Additive and never clamped: a prediction can go below zero
    assert predictor.apply_bias_correction(0.1, SLOT, "day_1") == pytest.approx(-0.1)


@pytest.mark.parametrize(
    ("lead_hours", "bucket"),
    [
        (-0.5, "day_1"),
        (0.0, "day_1"),
        (5.0, "day_1"),
        (24.0, "day_2"),
        (47.9, "day_2"),
        (60.0, "day_3"),
        (200.0, "day_4_plus"),
    ],
)
def test_prediction_bucket_follows_the_lead_time(
    lead_hours: float, bucket: str
) -> None:
    now = dt_util.now()
    start = now + timedelta(hours=lead_hours)

    assert prediction_bucket(start.isoformat(), now) == bucket


def test_prediction_bucket_of_an_unreadable_start_is_the_fallback() -> None:
    assert prediction_bucket("None", dt_util.now()) == BIAS_FALLBACK_BUCKET


def test_generated_predictions_get_and_store_the_offset_of_their_lead_time(
    predictor: SpotPricePredictor,
) -> None:
    """A run predicts the same slot a day apart: each gets its bucket's offset,
    and the offset is stored with the prediction (#203)."""
    now = dt_util.now().replace(minute=0, second=0, microsecond=0)
    starts = [now + timedelta(hours=hours) for hours in (5, 29, 53)]
    slot = starts[0].hour * 4
    today = now.date()
    predictor.error_metrics = {
        slot: _metrics({"day_1": {today: [0.1] * 3}, "day_2": {today: [0.5] * 3}})
    }
    # An offset left from before: no errors behind it, so it is not applied
    predictor.bias_correction = {slot: {"day_1": 9.0, "day_3": 9.0}}
    predictor.price_model = MagicMock()
    predictor.price_model.predict.side_effect = lambda rows: np.zeros(len(rows))
    predictor.is_trained = True
    features = predictor._combine_features(
        [{"start": start.isoformat()} for start in starts], {}
    )

    predictor._generate_predictions(features, 3, 60)

    assert [p["price"] for p in predictor.predictions] == [
        pytest.approx(-0.1),
        pytest.approx(-0.5),
        pytest.approx(0.0),
    ]
    stored = [
        predictor.storage.find_predictions_for_timestamp(
            start.strftime("%Y-%m-%d"), start.hour, start.minute
        )[0]
        for start in starts
    ]
    assert [row["bias_offset"] for row in stored] == [
        pytest.approx(0.1),
        pytest.approx(0.5),
        pytest.approx(0.0),
    ]
    # price + bias_offset is the raw model output
    assert [row["price"] + row["bias_offset"] for row in stored] == pytest.approx(
        [0.0, 0.0, 0.0]
    )


# --- The learning loop -------------------------------------------------------------


def test_add_matched_errors_keeps_pooled_errors_and_raw_windows() -> None:
    start = _noon(TODAY)
    metrics = new_slot_metrics()
    predictions = [
        _prediction(start, 3, 1.1, bias_offset=0.2),
        _prediction(start, 6, 1.3, bias_offset=0.2),
        _prediction(start, 30, 1.4, bias_offset=-0.1),
        _prediction(start, 30, 1.6, bias_offset=None),  # stored before #203
        _prediction(start, -1, 2.0),  # stored after the slot started: not a forecast
    ]

    add_matched_errors(metrics, predictions, 1.0, TODAY)

    # The pooled errors are those of the stored (corrected) prices
    assert metrics["errors"] == pytest.approx([0.1, 0.3, 0.4, 0.6, 1.0])
    assert metrics["count"] == 5
    assert metrics[RAW_ERRORS_KEY] == {
        "day_1": {TODAY.isoformat(): [2, pytest.approx(0.3 + 0.5)]},
        "day_2": {TODAY.isoformat(): [1, pytest.approx(0.3)]},
    }


def test_the_windows_keep_their_last_days_and_the_pooled_lists_their_bound() -> None:
    metrics = new_slot_metrics()
    first = TODAY - timedelta(days=MAX_ERROR_SAMPLES + 9)

    for index in range(MAX_ERROR_SAMPLES + 10):
        day = first + timedelta(days=index)
        add_matched_errors(metrics, [_prediction(_noon(day), 3, 1.1)], 1.0, day)

    assert len(metrics["errors"]) == MAX_ERROR_SAMPLES
    assert metrics["count"] == MAX_ERROR_SAMPLES + 10
    assert sorted(metrics[RAW_ERRORS_KEY]["day_1"]) == [
        (TODAY - timedelta(days=age)).isoformat() for age in (3, 2, 1, 0)
    ]


def test_the_error_lists_of_before_are_dropped_on_the_next_match() -> None:
    """``bucket_errors`` held the errors left after the correction (#118, #129)."""
    metrics = new_slot_metrics()
    metrics["bucket_errors"] = {"day_2": [0.5] * 25}
    del metrics[RAW_ERRORS_KEY]  # metrics stored before #203

    add_raw_errors(metrics, [], 1.0, TODAY)
    assert slot_offsets(metrics, TODAY) == {}
    add_matched_errors(metrics, [_prediction(_noon(TODAY), 3, 1.1)], 1.0, TODAY)

    assert "bucket_errors" not in metrics
    assert metrics[RAW_ERRORS_KEY] == {
        "day_1": {TODAY.isoformat(): [1, pytest.approx(0.1)]}
    }


def test_learning_from_an_actual_price_updates_each_bucket(
    predictor: SpotPricePredictor,
) -> None:
    """End to end: stored offsets, raw errors, the new offsets and the sums."""
    start = _tomorrow_noon()
    for lead_hours, price, offset in [
        (3, 1.1, -0.2),
        (6, 1.1, -0.2),
        (9, 1.1, -0.2),
        (30, 1.4, 0.1),
        (33, 1.4, 0.1),
        (36, 1.4, 0.1),
        (60, 1.7, 0.0),
        (61, 1.7, 0.0),
        (62, 1.7, 0.0),
    ]:
        predictor.storage.insert_prediction(
            start=start.isoformat(),
            price=price,
            confidence=0.8,
            hour=start.hour,
            minute=start.minute,
            stored_at=(start - timedelta(hours=lead_hours)).isoformat(),
            bias_offset=offset,
        )

    assert predictor.learn_from_actual_price(start.isoformat(), 1.0)

    # day_1: 0.1 left after an offset of -0.2: the raw model was 0.1 low
    assert predictor.bias_correction == {
        SLOT: {"day_1": pytest.approx(-0.1), "day_2": pytest.approx(0.5)}
    }
    day = start.date().isoformat()
    with closing(sqlite3.connect(predictor.storage.db_path)) as conn:
        rows = conn.execute(
            "SELECT bucket, samples, sum_error, sum_abs_error, sum_raw_error,"
            " sum_raw_abs_error, sum_raw_sq_error FROM bias_accuracy"
            " WHERE date = ? ORDER BY bucket",
            (day,),
        ).fetchall()
    assert [row[:2] for row in rows] == [("day_1", 3), ("day_2", 3), ("day_3", 3)]
    # Stored error and |error|, then the raw model's error, |error| and error²
    assert rows[0][2:] == pytest.approx((0.3, 0.3, -0.3, 0.3, 0.03))
    assert rows[1][2:] == pytest.approx((1.2, 1.2, 1.5, 1.5, 0.75))
    assert rows[2][2:] == pytest.approx((2.1, 2.1, 2.1, 2.1, 1.47))


def test_no_bucket_is_corrected_since_206() -> None:
    """On 14 live DK1 days every trailing offset added error (#206).

    ``day_1`` MAE 2.21 with the 4-day offsets against 1.97 without, ``day_2``
    2.60 against 2.24 EUR ct/kWh (``docs/self_learning.md``).
    """
    assert BIAS_WINDOW_DAYS == {}


def test_the_offsets_of_an_upgraded_instance_are_dropped(
    predictor: SpotPricePredictor,
) -> None:
    """The offsets and windows an instance learned before #206 stop applying
    at the first refresh, and new errors are not kept for them."""
    predictor.bias_correction = {SLOT: {"day_1": 0.3, "day_2": -0.2}}
    predictor.error_metrics = {
        SLOT: _metrics({"day_1": {TODAY: [0.3] * 3}, "day_2": {TODAY: [-0.2] * 3}})
    }

    with patch.object(bias, "BIAS_WINDOW_DAYS", BIAS_WINDOW_DAYS):
        predictor.refresh_bias_offsets(TODAY)
        assert predictor.bias_correction == {}
        for bucket in BUCKETS:
            assert predictor.apply_bias_correction(1.0, SLOT, bucket) == (
                pytest.approx(1.0)
            )

        start = _noon(TODAY)
        predictions = [_prediction(start, LEAD_HOURS["day_1"], 1.5) for _ in "abc"]
        add_matched_errors(predictor.error_metrics[SLOT], predictions, 1.0, TODAY)
        assert predictor.error_metrics[SLOT][RAW_ERRORS_KEY] == {}


def test_a_prediction_stored_before_the_offsets_were_kept_is_handled(
    predictor: SpotPricePredictor,
) -> None:
    """Its error counts for the accuracy, not for the offset: the raw error
    is unknown without the offset that was subtracted."""
    start = _tomorrow_noon()
    for lead_hours in (3, 6, 9):
        predictor.storage.insert_prediction(
            start=start.isoformat(),
            price=1.3,
            confidence=0.8,
            hour=start.hour,
            minute=start.minute,
            stored_at=(start - timedelta(hours=lead_hours)).isoformat(),
        )

    assert predictor.learn_from_actual_price(start.isoformat(), 1.0)

    assert predictor.error_metrics[SLOT]["errors"] == pytest.approx([0.3] * 3)
    assert predictor.error_metrics[SLOT][RAW_ERRORS_KEY] == {}
    assert predictor.bias_correction == {}
    assert predictor.lead_time_accuracy["day_1"]["samples"] == 3
    with closing(sqlite3.connect(predictor.storage.db_path)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM bias_accuracy").fetchone() == (0,)


def test_learning_metrics_report_the_offsets_per_bucket(
    predictor: SpotPricePredictor,
) -> None:
    predictor.error_metrics = {
        SLOT: _metrics({"day_1": {TODAY: [0.1] * 3}}),
        50: _metrics({"day_2": {TODAY: [0.2] * 3}}),
    }
    predictor.bias_correction = {SLOT: {"day_1": 0.1, "day_2": 0.3}, 50: {"day_2": 0.5}}

    metrics = predictor.get_learning_metrics()

    assert metrics["bias_corrections"] == 2
    assert metrics["bias_offsets"] == {
        "day_1": {"slots": 1, "mean_offset": pytest.approx(0.1)},
        "day_2": {"slots": 2, "mean_offset": pytest.approx(0.4)},
        "day_3": {"slots": 0, "mean_offset": None},
        "day_4_plus": {"slots": 0, "mean_offset": None},
    }
    slot_metrics = metrics["hourly_metrics"][str(SLOT)]
    assert slot_metrics["bias_correction"] == pytest.approx(0.1)
    assert slot_metrics["bias_offsets"] == {
        "day_1": pytest.approx(0.1),
        "day_2": pytest.approx(0.3),
    }
    # A slot without a day_1 offset reports 0.0 as before, plus its buckets
    assert metrics["hourly_metrics"]["50"]["bias_correction"] == pytest.approx(0.0)
    assert metrics["hourly_metrics"]["50"]["bias_offsets"] == {
        "day_2": pytest.approx(0.5)
    }


@pytest.mark.asyncio
async def test_reset_learning_clears_the_offsets(predictor: SpotPricePredictor) -> None:
    predictor.bias_correction = {SLOT: {"day_1": 0.1}}

    with patch.object(
        predictor.storage, "async_clear_storage", AsyncMock(return_value=True)
    ):
        assert await predictor.reset_learning() is True

    assert predictor.bias_correction == {}


# --- Storage and migration ---------------------------------------------------------


def _bias_offsets(storage: LearningStorage) -> dict[int, dict[str, float]]:
    """Return the stored bias offsets through the bulk loader."""
    data = storage.load_all()
    assert data is not None
    offsets: dict[int, dict[str, float]] = data["bias_correction"]
    return offsets


def _schema_version(db_path: Path) -> int:
    with closing(sqlite3.connect(db_path)) as conn:
        row = conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
    return int(row[0])


def _insert(storage: LearningStorage, start: datetime, **extra: float) -> None:
    storage.insert_prediction(
        start=start.isoformat(),
        price=1.25,
        confidence=0.8,
        hour=start.hour,
        minute=start.minute,
        stored_at=(start - timedelta(hours=30)).isoformat(),
        **extra,
    )


def test_the_offset_of_a_prediction_survives_a_restart(tmp_path: Path) -> None:
    start = _noon(TODAY)
    storage = LearningStorage(_hass(tmp_path), "DK1")
    _insert(storage, start, bias_offset=-0.32)
    _insert(storage, start, bias_offset=0.0)
    storage.close()

    reopened = LearningStorage(_hass(tmp_path), "DK1")
    try:
        rows = reopened.find_predictions_for_timestamp(TODAY.isoformat(), 12, 0)
    finally:
        reopened.close()

    assert [row["bias_offset"] for row in rows] == pytest.approx([-0.32, 0.0])
    assert [row["price"] for row in rows] == pytest.approx([1.25, 1.25])


def test_upgrade_adds_the_offset_column_and_resets_the_lagging_offsets(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A v9 database: predictions without ``bias_offset``, EMA offsets."""
    start = _noon(TODAY)
    storage = LearningStorage(_hass(tmp_path), "DK1")
    db_path = storage.db_path
    _insert(storage, start)
    storage.save_all({"bias_correction": {SLOT: {"day_1": -0.38, "day_2": -0.32}}})
    storage.close()
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute("ALTER TABLE predictions DROP COLUMN bias_offset")
        conn.execute("DROP TABLE bias_accuracy")
        conn.execute("UPDATE meta SET value = '9' WHERE key = 'schema_version'")
        conn.commit()

    with caplog.at_level(logging.INFO):
        upgraded = LearningStorage(_hass(tmp_path), "DK1")
    try:
        assert _schema_version(db_path) == RAW_ERROR_BIAS_SCHEMA_VERSION == 10
        assert _bias_offsets(upgraded) == {}
        assert "reset 2 offsets" in caplog.text
        # The row of before is kept, without an offset; new rows have theirs
        _insert(upgraded, start, bias_offset=0.07)
        rows = upgraded.find_predictions_for_timestamp(TODAY.isoformat(), 12, 0)
        assert [row["bias_offset"] for row in rows] == [None, pytest.approx(0.07)]
        upgraded.add_bias_accuracy(TODAY.isoformat(), {"day_2": [(0.3, 0.2)]})
        # Offsets learned from now on are stored as before
        upgraded.save_all({"bias_correction": {SLOT: {"day_2": 0.3}}})
    finally:
        upgraded.close()

    caplog.clear()
    with caplog.at_level(logging.INFO):
        reopened = LearningStorage(_hass(tmp_path), "DK1")
    try:
        # The reset runs once
        assert _bias_offsets(reopened) == {SLOT: {"day_2": pytest.approx(0.3)}}
        assert "reset" not in caplog.text
    finally:
        reopened.close()


def test_upgrade_keeps_the_pooled_offsets_until_the_raw_error_reset(
    tmp_path: Path,
) -> None:
    """A v8 database (one offset per slot) goes through v9 and v10 in one start."""
    storage = LearningStorage(_hass(tmp_path), "DK1")
    db_path = storage.db_path
    storage.close()
    with closing(sqlite3.connect(db_path)) as conn:
        conn.executescript(
            "DROP TABLE bias_correction;"
            "CREATE TABLE bias_correction (hour INTEGER PRIMARY KEY, correction REAL NOT NULL);"
        )
        conn.executemany(
            "INSERT INTO bias_correction (hour, correction) VALUES (?, ?)",
            [(SLOT, 0.12), (SLOT + 1, -0.05)],
        )
        conn.execute("UPDATE meta SET value = '8' WHERE key = 'schema_version'")
        conn.commit()

    upgraded = LearningStorage(_hass(tmp_path), "DK1")
    try:
        assert _bias_offsets(upgraded) == {}
        assert _schema_version(db_path) == RAW_ERROR_BIAS_SCHEMA_VERSION
        with closing(sqlite3.connect(db_path)) as conn:
            columns = [c[1] for c in conn.execute("PRAGMA table_info(bias_correction)")]
        assert columns == ["hour", "bucket", "correction"]
    finally:
        upgraded.close()


def test_new_database_starts_at_the_raw_error_bias_schema(tmp_path: Path) -> None:
    storage = LearningStorage(_hass(tmp_path), "DK1")
    try:
        assert _schema_version(storage.db_path) == RAW_ERROR_BIAS_SCHEMA_VERSION
        assert RAW_ERROR_BIAS_SCHEMA_VERSION > LEAD_TIME_BIAS_SCHEMA_VERSION
        with closing(sqlite3.connect(storage.db_path)) as conn:
            columns = [c[1] for c in conn.execute("PRAGMA table_info(bias_correction)")]
            predictions = [c[1] for c in conn.execute("PRAGMA table_info(predictions)")]
        assert columns == ["hour", "bucket", "correction"]
        assert "bias_offset" in predictions
    finally:
        storage.close()


def test_every_lead_time_bucket_can_hold_an_offset(tmp_path: Path) -> None:
    storage = LearningStorage(_hass(tmp_path), "DK1")
    try:
        storage.save_all({"bias_correction": {SLOT: dict.fromkeys(BUCKETS, 0.1)}})
        assert set(_bias_offsets(storage)[SLOT]) == set(BUCKETS)
    finally:
        storage.close()


def test_bias_accuracy_adds_up_per_day_and_is_pruned_with_the_window(
    tmp_path: Path,
) -> None:
    storage = LearningStorage(_hass(tmp_path), "DK1")
    try:
        storage.add_bias_accuracy("2026-10-12", {"day_1": [(0.5, 0.1), (-0.3, -0.7)]})
        storage.add_bias_accuracy("2026-10-12", {"day_1": [(0.2, -0.2)], "day_2": []})
        storage.add_bias_accuracy("2026-09-01", {"day_2": [(0.4, 0.4)]})
        storage.add_bias_accuracy("2026-10-12", {})

        storage.delete_lead_time_accuracy_before("2026-10-01")

        with closing(sqlite3.connect(storage.db_path)) as conn:
            rows = conn.execute("SELECT * FROM bias_accuracy").fetchall()
        assert [row[:3] for row in rows] == [("2026-10-12", "day_1", 3)]
        # Stored errors 0.1, -0.7, -0.2; raw errors 0.5, -0.3, 0.2
        assert rows[0][3:] == pytest.approx((-0.8, 1.0, 0.54, 0.4, 1.0, 0.38))
    finally:
        storage.close()
