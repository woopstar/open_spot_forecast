"""The Predbat and evaluation sensors build their series once per data update (#295).

Each test compares the cached sensors with the uncached computation they
replaced (``_reference_*``), so the exposed values stay identical.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import UTC, date, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from zoneinfo import ZoneInfo

import pytest

from homeassistant.util import dt as dt_util

from custom_components.open_spot_forecast.const import (
    CONF_PREDBAT_TARIFF,
    EVALUATION_WINDOW_HOURS,
    RECORDER_MAX_ATTRIBUTES_BYTES,
)
from custom_components.open_spot_forecast.evaluation_sensor import (
    EXTRA_LEAD_TIME_ARRAYS,
    ForecastEvaluationSensor,
)
from custom_components.open_spot_forecast.forecast_attributes import (
    attributes_size,
    fit_entries,
)
from custom_components.open_spot_forecast.predbat_sensor import (
    DAY_TODAY,
    KIND_IMPORT,
    PredbatRateSensor,
    build_predbat_sensors,
)
from custom_components.open_spot_forecast.price_output import PriceOutput
from custom_components.open_spot_forecast.spot_prices import (
    known_until,
    with_known_prices,
)
from custom_components.open_spot_forecast.tariffs import TariffSchedule
from custom_components.open_spot_forecast.time_slots import (
    local_midnight,
    parse_utc,
    slot_start_in_day,
    slots_in_local_day,
)

CPH = ZoneInfo("Europe/Copenhagen")
DAY = date(2026, 9, 24)
NOW = datetime(2026, 9, 24, 10, 7, tzinfo=CPH)
VAT = 0.25
SURCHARGE = 0.1
TARIFF = 0.5
OUTPUT = PriceOutput(vat=VAT, surcharge=SURCHARGE, precision=4)
HOURLY = PriceOutput(vat=VAT, surcharge=SURCHARGE, precision=4, hourly_average=True)
UTC_NOW = NOW.replace(hour=14).astimezone(UTC)


# --- Fixture data (as in test_predbat_sensor and test_evaluation) --------------------


def _spot_data(tomorrow: bool = True) -> dict[str, Any]:
    """Confirmed spot prices, varying by slot, for today and tomorrow."""
    data: dict[str, Any] = {"day": DAY, "today": [], "tomorrow": []}
    for offset, key in enumerate(("today", "tomorrow")):
        if key == "tomorrow" and not tomorrow:
            continue
        day = DAY + timedelta(days=offset)
        count = slots_in_local_day(day)
        data[key] = [1.0 + offset + (i % 24) / 100 for i in range(count)]
        data[f"raw_{key}"] = [
            {"start": slot_start_in_day(day, i).isoformat()} for i in range(count)
        ]
    return data


def _predictions(first_day: date, days: int, first_slot: int = 0) -> list[dict]:
    """Model predictions, varying by slot, for ``days`` days from ``first_day``."""
    predictions = []
    for offset in range(days):
        day = first_day + timedelta(days=offset)
        for index in range(first_slot if offset == 0 else 0, slots_in_local_day(day)):
            predictions.append(
                {
                    "start": slot_start_in_day(day, index).isoformat(),
                    "end": slot_start_in_day(day, index + 1).isoformat(),
                    "price": 3.0 + (index % 96) / 100,
                    "confidence": 0.8,
                }
            )
    return predictions


def _slot_tariffs(spot_data: dict[str, Any]) -> TariffSchedule:
    """A tariff per slot that varies by time of day."""
    consumer = {
        **spot_data,
        "today": [p + TARIFF + (i % 8) / 10 for i, p in enumerate(spot_data["today"])],
        "tomorrow": [
            p + TARIFF + (i % 8) / 10 for i, p in enumerate(spot_data["tomorrow"])
        ],
    }
    return TariffSchedule.from_prices(consumer, spot_data)


def _rows(count: int, first: datetime) -> list[dict[str, Any]]:
    """Evaluated slots from ``first``: predicted 1.0 + n/100, actual 1.0."""
    return [
        {
            "start": (first + timedelta(minutes=15 * n)).astimezone(CPH).isoformat(),
            "end": (first + timedelta(minutes=15 * (n + 1)))
            .astimezone(CPH)
            .isoformat(),
            "predicted": 1.0 + (n % 10) / 100,
            "actual": 1.0,
            "lead_hours": 24.0,
        }
        for n in range(count)
    ]


# --- The uncached computation the sensors replaced ----------------------------------


def _reference_predbat(
    sensor: PredbatRateSensor, now: datetime
) -> tuple[float | None, dict[str, Any]]:
    """Return the state and attributes as the sensor built them before #295."""
    api_data = sensor.api_data
    output = sensor._output
    tariffs: TariffSchedule | None = None
    if sensor.kind == KIND_IMPORT:
        tariffs = api_data.get("tariffs") or (
            TariffSchedule.fixed(sensor._fixed_tariff)
            if abs(sensor._fixed_tariff) > 1e-9
            else None
        )
    ml_predictor = api_data.get("ml_predictor")
    predictions = ml_predictor.predictions if ml_predictor else None
    today = dt_util.as_local(now).date()
    tomorrow = local_midnight(today + timedelta(days=1))
    series = with_known_prices(
        api_data.get("spot_data"), predictions or [], local_midnight(today)
    )
    factor = sensor._series.factor
    entries = []
    for entry in output.forecast(series, tariffs):
        start = parse_utc(entry["start"])
        if start is None or (start < tomorrow) != (sensor.day == DAY_TODAY):
            continue
        entries.append(
            {
                "start": entry["start"],
                "end": entry["end"],
                "price": float(round(entry["price"] * factor, output.precision)),
            }
        )
    if sensor.day == DAY_TODAY:
        state = next(
            (
                entry["price"]
                for entry in entries
                if (start := parse_utc(entry["start"])) is not None
                and (end := parse_utc(entry["end"])) is not None
                and start <= now < end
            ),
            None,
        )
    else:
        state = entries[0]["price"] if entries else None
    key = f"prices_{sensor.day}"
    known_end = known_until(api_data.get("spot_data"))
    attrs: dict[str, Any] = {
        key: list(entries),
        "interval_minutes": output.interval_minutes,
        "known_until": dt_util.as_local(known_end).isoformat() if known_end else None,
        "includes_vat": sensor.kind == KIND_IMPORT,
        "includes_tariffs": bool(tariffs),
    }
    fit_entries(attrs, key, output.interval_minutes)
    return state, attrs


def _reference_evaluation(
    sensor: ForecastEvaluationSensor, now: datetime
) -> tuple[float | None, dict[str, Any]]:
    """Return the state and attributes as the sensor built them before #295."""
    predictor = sensor.api_data.get("ml_predictor")
    tariffs = sensor.api_data.get("tariffs")
    output = sensor.output
    since = now - timedelta(hours=EVALUATION_WINDOW_HOURS)
    window = [
        (start, row)
        for row in output.evaluation(predictor.evaluation if predictor else [], tariffs)
        if (start := parse_utc(row["start"])) is not None and start >= since
    ]
    errors = [row["predicted"] - row["actual"] for _, row in window]
    state = (
        float(round(sum(abs(e) for e in errors) / len(errors), output.precision))
        if errors
        else None
    )
    snapshots = predictor.evaluation_snapshots if predictor else {}
    extra = {}
    for name, target in EXTRA_LEAD_TIME_ARRAYS.items():
        predicted = {
            row["start"]: row["predicted"]
            for row in output.evaluation(snapshots.get(target, []), tariffs)
        }
        extra[name] = [predicted.get(row["start"]) for _, row in window]
    return state, {
        "interval_minutes": 15,
        "unit": output.unit(sensor.currency),
        "lead_hours": 24.0,
        "window_hours": EVALUATION_WINDOW_HOURS,
        "samples": len(window),
        "bias": (
            float(round(sum(errors) / len(errors), output.precision))
            if errors
            else None
        ),
        "s": [int(start.timestamp()) for start, _ in window],
        "t": [row["predicted"] for _, row in window],
        "a": [row["actual"] for _, row in window],
        **extra,
    }


# --- Helpers --------------------------------------------------------------------------


def _write(sensor: Any, now: datetime) -> tuple[float | None, dict[str, Any]]:
    """Read what one state write reads: the state, then the attributes."""
    with patch("homeassistant.util.dt.utcnow", return_value=now):
        return sensor.native_value, sensor.extra_state_attributes


def _api_data(
    predictions: list[dict[str, Any]] | None = None,
    spot_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    spot_data = _spot_data() if spot_data is None else spot_data
    predictor = MagicMock()
    predictor.predictions = (
        _predictions(DAY, 7, first_slot=40) if predictions is None else predictions
    )
    return {
        "spot_data": spot_data,
        "tariffs": _slot_tariffs(spot_data),
        "ml_predictor": predictor,
    }


def _predbat(
    api_data: dict[str, Any],
    output: PriceOutput = OUTPUT,
    fixed_tariff: float = 0.0,
) -> list[PredbatRateSensor]:
    entry = MagicMock(entry_id="test")
    entry.options = {CONF_PREDBAT_TARIFF: fixed_tariff}
    return build_predbat_sensors(MagicMock(), entry, api_data, "DKK", output)


def _assert_predbat_matches(sensors: list[PredbatRateSensor], now: datetime) -> None:
    for sensor in sensors:
        state, attrs = _write(sensor, now)
        expected_state, expected_attrs = _reference_predbat(sensor, now)
        if expected_state is None:
            assert state is None
        else:
            assert state == pytest.approx(expected_state)
        assert attrs == expected_attrs


@pytest.fixture
def forecast_builds() -> Iterator[list[int]]:
    """Count the ``PriceOutput.forecast`` calls (one per series build)."""
    builds: list[int] = []
    real = PriceOutput.forecast

    def counted(output: PriceOutput, *args: Any, **kwargs: Any) -> Any:
        builds.append(1)
        return real(output, *args, **kwargs)

    with patch.object(PriceOutput, "forecast", counted):
        yield builds


@pytest.fixture
def converted_rows() -> Iterator[list[int]]:
    """Count the rows ``PriceOutput.evaluation`` converts."""
    rows: list[int] = []
    real = PriceOutput.evaluation

    def counted(output: PriceOutput, items: Any, *args: Any, **kwargs: Any) -> Any:
        items = list(items)
        rows.append(len(items))
        return real(output, items, *args, **kwargs)

    with patch.object(PriceOutput, "evaluation", counted):
        yield rows


# --- Predbat --------------------------------------------------------------------------


@pytest.mark.usefixtures("copenhagen_time_zone")
@pytest.mark.parametrize("output", [OUTPUT, HOURLY], ids=["slots", "hourly"])
@pytest.mark.parametrize("hour", [0, 10, 14, 23])
def test_predbat_values_are_unchanged(output: PriceOutput, hour: int) -> None:
    now = NOW.replace(hour=hour).astimezone(UTC)
    _assert_predbat_matches(_predbat(_api_data(), output), now)


@pytest.mark.usefixtures("copenhagen_time_zone")
def test_predbat_values_are_unchanged_without_model_or_prices() -> None:
    no_model = _api_data()
    del no_model["ml_predictor"]
    _assert_predbat_matches(_predbat(no_model), UTC_NOW)
    _assert_predbat_matches(_predbat({}), UTC_NOW)
    # The fixed tariff applies without slot tariffs
    fixed = _api_data()
    fixed["tariffs"] = None
    _assert_predbat_matches(_predbat(fixed, fixed_tariff=0.2), UTC_NOW)


@pytest.mark.usefixtures("copenhagen_time_zone")
def test_one_build_per_kind_and_none_while_nothing_changes(
    forecast_builds: list[int],
) -> None:
    sensors = _predbat(_api_data())

    # One signal: every sensor reads its state and attributes
    for sensor in sensors:
        _write(sensor, UTC_NOW)
    assert len(forecast_builds) == 2

    # The next writes, later in the day, reuse the builds
    for sensor in sensors:
        _write(sensor, UTC_NOW + timedelta(hours=1))
    assert len(forecast_builds) == 2


@pytest.mark.usefixtures("copenhagen_time_zone")
@pytest.mark.parametrize(
    "change",
    ["predictions", "appended", "spot_data", "tariffs", "next_day"],
)
def test_a_data_update_gives_new_values(change: str) -> None:
    api_data = _api_data()
    sensors = _predbat(api_data)
    now = UTC_NOW
    for sensor in sensors:
        _write(sensor, now)

    if change == "predictions":
        api_data["ml_predictor"].predictions = [
            {**prediction, "price": prediction["price"] + 1.0}
            for prediction in api_data["ml_predictor"].predictions
        ]
    elif change == "appended":
        # A forecast run still filling the list
        predictions = api_data["ml_predictor"].predictions
        last = parse_utc(predictions[-1]["start"])
        assert last is not None
        start = last + timedelta(minutes=15)
        predictions.append(
            {
                "start": dt_util.as_local(start).isoformat(),
                "end": dt_util.as_local(start + timedelta(minutes=15)).isoformat(),
                "price": 9.0,
                "confidence": 0.8,
            }
        )
    elif change == "spot_data":
        spot_data = _spot_data(tomorrow=False)
        api_data["spot_data"] = spot_data
    elif change == "tariffs":
        consumer = {**api_data["spot_data"]}
        consumer["today"] = [p + 0.7 for p in consumer["today"]]
        consumer["tomorrow"] = [p + 0.7 for p in consumer["tomorrow"]]
        api_data["tariffs"] = TariffSchedule.from_prices(
            consumer, api_data["spot_data"]
        )
    else:
        now = UTC_NOW + timedelta(days=1)

    _assert_predbat_matches(sensors, now)


@pytest.mark.usefixtures("copenhagen_time_zone")
def test_the_returned_attributes_cannot_change_the_cache() -> None:
    sensor = _predbat(_api_data())[1]
    _, attrs = _write(sensor, UTC_NOW)
    expected = _reference_predbat(sensor, UTC_NOW)[1]

    attrs["prices_tomorrow"].clear()
    attrs["known_until"] = "changed"

    assert _write(sensor, UTC_NOW)[1] == expected


@pytest.mark.usefixtures("copenhagen_time_zone")
def test_predbat_attributes_fit_the_recorder() -> None:
    for sensor in _predbat(_api_data()):
        _, attrs = _write(sensor, UTC_NOW)
        recorded = {
            key: value
            for key, value in attrs.items()
            if key not in sensor._unrecorded_attributes
        }
        assert attributes_size(attrs) < RECORDER_MAX_ATTRIBUTES_BYTES
        assert attributes_size(recorded) < 1024


def _executor_hass() -> Mock:
    """A Home Assistant whose executor runs the job right away."""
    hass = Mock()

    async def run(func: Callable[..., Any], *args: Any) -> Any:
        return func(*args)

    hass.async_add_executor_job = AsyncMock(side_effect=run)
    return hass


@pytest.mark.asyncio
@pytest.mark.usefixtures("copenhagen_time_zone")
async def test_a_signal_builds_in_the_executor_then_writes(
    forecast_builds: list[int],
) -> None:
    sensors = _predbat(_api_data())
    hass = _executor_hass()
    writes = Mock()
    for sensor in sensors:
        sensor.hass = hass

    with patch("homeassistant.util.dt.utcnow", return_value=UTC_NOW):
        for sensor in sensors:
            with patch.object(sensor, "async_write_ha_state", writes):
                await sensor._async_refresh()
        # The state write finds the series built
        for sensor in sensors:
            _ = sensor.native_value, sensor.extra_state_attributes

    assert len(forecast_builds) == 2
    assert hass.async_add_executor_job.await_count == 4
    assert writes.call_count == 4


# --- Forecast evaluation --------------------------------------------------------------

EVAL_NOW = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)


def _evaluation_sensor(
    tariffs: TariffSchedule | None = None,
) -> ForecastEvaluationSensor:
    rows = _rows(7 * 96, EVAL_NOW - timedelta(days=7))
    twelve = [{**row, "predicted": row["predicted"] + 0.3} for row in rows[::2]]
    predictor = Mock(
        evaluation=rows,
        evaluation_snapshots={12.0: twelve, 48.0: rows[96 : 6 * 96]},
        is_trained=True,
    )
    return ForecastEvaluationSensor(
        Mock(),
        MagicMock(entry_id="test"),
        {"ml_predictor": predictor, "tariffs": tariffs},
        "DKK",
        PriceOutput(vat=0.25, precision=4),
    )


def _assert_evaluation_matches(sensor: ForecastEvaluationSensor, now: datetime) -> None:
    state, attrs = _write(sensor, now)
    expected_state, expected_attrs = _reference_evaluation(sensor, now)
    if expected_state is None:
        assert state is None
    else:
        assert state == pytest.approx(expected_state)
    assert attrs == expected_attrs


@pytest.mark.parametrize("tariffs", [None, TariffSchedule({}, default=0.4)])
@pytest.mark.parametrize("minutes", [0, 7, 15, 600])
def test_evaluation_values_are_unchanged(
    tariffs: TariffSchedule | None, minutes: int
) -> None:
    _assert_evaluation_matches(
        _evaluation_sensor(tariffs), EVAL_NOW + timedelta(minutes=minutes)
    )


def test_evaluation_without_data_is_unchanged() -> None:
    sensor = _evaluation_sensor()
    sensor.api_data.clear()
    _assert_evaluation_matches(sensor, EVAL_NOW)
    sensor.api_data["ml_predictor"] = Mock(evaluation=[], evaluation_snapshots={})
    _assert_evaluation_matches(sensor, EVAL_NOW)


def test_evaluation_converts_only_the_window_once(converted_rows: list[int]) -> None:
    sensor = _evaluation_sensor()

    _write(sensor, EVAL_NOW + timedelta(minutes=1))
    # The window's slots, then the other lead times' slots inside it
    assert sum(converted_rows) <= 3 * EVALUATION_WINDOW_HOURS * 4
    first = sum(converted_rows)

    # A write in the same slot converts nothing
    _write(sensor, EVAL_NOW + timedelta(minutes=14))
    assert sum(converted_rows) == first


@pytest.mark.parametrize("change", ["evaluation", "snapshots", "tariffs", "slot"])
def test_evaluation_update_gives_new_values(change: str) -> None:
    sensor = _evaluation_sensor()
    now = EVAL_NOW
    _write(sensor, now)
    predictor = sensor.api_data["ml_predictor"]

    if change == "evaluation":
        predictor.evaluation = [
            {**row, "actual": row["actual"] + 0.2} for row in predictor.evaluation
        ]
    elif change == "snapshots":
        predictor.evaluation_snapshots = {12.0: predictor.evaluation}
    elif change == "tariffs":
        sensor.api_data["tariffs"] = TariffSchedule({}, default=0.4)
    else:
        # The window moves on a slot; the oldest slot leaves it
        now = EVAL_NOW + timedelta(minutes=20)

    _assert_evaluation_matches(sensor, now)


def test_evaluation_attributes_fit_the_recorder() -> None:
    sensor = _evaluation_sensor()
    _, attrs = _write(sensor, EVAL_NOW)
    stored = {**attrs, "attribution": sensor.attribution, "friendly_name": "x"}
    assert attributes_size(stored) < RECORDER_MAX_ATTRIBUTES_BYTES


@pytest.mark.asyncio
async def test_evaluation_signal_builds_in_the_executor_then_writes(
    converted_rows: list[int],
) -> None:
    sensor = _evaluation_sensor()
    sensor.hass = _executor_hass()
    writes = Mock()

    with (
        patch("homeassistant.util.dt.utcnow", return_value=EVAL_NOW),
        patch.object(sensor, "async_write_ha_state", writes),
    ):
        await sensor._async_refresh()
        built = sum(converted_rows)
        _ = sensor.native_value, sensor.extra_state_attributes

    assert built > 0
    assert sum(converted_rows) == built
    writes.assert_called_once()
