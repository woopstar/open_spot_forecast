"""External price forecasts recorded and scored next to the model's own (#120).

Stromligning's forecast sensor and Energi Data Service's forecast attribute
are read at every forecast run, stored per (source, slot, stored_at) as raw
spot prices, and scored per lead-time bucket when self-learning scores the
slot. The model's own metrics and bias correction never see them.
"""

import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from zoneinfo import ZoneInfo

import pytest
import voluptuous as vol

from homeassistant.util import dt as dt_util

from custom_components.open_spot_forecast.accuracy_sensor import (
    LeadTimeAccuracySensor,
)
from custom_components.open_spot_forecast.config_flow import (
    OpenSpotForecastConfigFlow,
    OpenSpotForecastOptionsFlow,
)
from custom_components.open_spot_forecast.const import (
    CONF_EXTERNAL_FORECAST_SENSORS,
    CONF_REGION,
    EXTERNAL_MODEL_SOURCE,
    LEAD_TIME_WINDOW_DAYS,
)
from custom_components.open_spot_forecast.external_forecasts import (
    async_record_external_forecasts,
    external_spot_forecast,
)
from custom_components.open_spot_forecast.ml.predictor import SpotPricePredictor
from custom_components.open_spot_forecast.ml.storage import LearningStorage
from custom_components.open_spot_forecast.price_output import PriceOutput
from custom_components.open_spot_forecast.sensor_entities import SensorEntities
from custom_components.open_spot_forecast.sensor_reader import SensorReader
from custom_components.open_spot_forecast.tariffs import TariffSchedule
from custom_components.open_spot_forecast.time_slots import (
    slot_index_in_day,
    slots_in_local_day,
    utc_slot_key,
)

pytestmark = pytest.mark.usefixtures("copenhagen_time_zone")

CPH = ZoneInfo("Europe/Copenhagen")
# 12:00 UTC is 14:00 in Copenhagen
NOW = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
QUARTER = timedelta(minutes=15)
STROMLIGNING = "sensor.stromligning_forecasts_vat"
EDS = "sensor.energi_data_service"


def _hass(tmp_path: Path) -> Mock:
    hass = Mock()
    hass.config.path.return_value = str(tmp_path / ".storage")
    return hass


@pytest.fixture
def predictor(tmp_path: Path) -> Iterator[SpotPricePredictor]:
    predictor = SpotPricePredictor(_hass(tmp_path), "DK1")
    yield predictor
    predictor.storage.close()


def _reader(attributes: dict[str, Any] | None) -> SensorReader:
    hass = Mock()
    hass.states.get.return_value = (
        None if attributes is None else Mock(state="x", attributes=attributes)
    )
    return SensorReader(hass)


# --- Reading the sensors -------------------------------------------------------------


def test_stromlignings_forecast_is_read_from_prices() -> None:
    start = datetime(2026, 9, 26, 0, 0, tzinfo=CPH)
    reader = _reader(
        {
            "prices": [
                {"start": start, "end": start + QUARTER, "price": 1.5},
                {
                    "start": (start + QUARTER).isoformat(),
                    "end": (start + 2 * QUARTER).isoformat(),
                    "price": 2.5,
                },
            ]
        }
    )

    forecast = reader.read_external_forecast(STROMLIGNING)

    assert forecast == [
        (datetime(2026, 9, 25, 22, 0, tzinfo=UTC), pytest.approx(1.5)),
        (datetime(2026, 9, 25, 22, 15, tzinfo=UTC), pytest.approx(2.5)),
    ]
    states: Any = reader.hass.states
    states.get.assert_called_once_with(STROMLIGNING)


def test_energi_data_services_hourly_forecast_fills_its_slots() -> None:
    """``forecast`` items carry ``hour`` and no end: an hour is four slots."""
    first = datetime(2026, 9, 26, 22, 0, tzinfo=CPH)
    reader = _reader(
        {
            "forecast": [
                {"hour": first + timedelta(hours=n), "price": float(n)}
                for n in range(3)
            ]
        }
    )

    forecast = reader.read_external_forecast(EDS)

    # 22:00 and 23:00 of one local day, 00:00 of the next
    assert len(forecast) == 12
    assert [start for start, _ in forecast] == [
        first.astimezone(UTC) + n * QUARTER for n in range(12)
    ]
    assert [price for _, price in forecast] == pytest.approx(
        [0.0] * 4 + [1.0] * 4 + [2.0] * 4
    )


def test_prices_wins_over_forecast_and_bad_items_are_skipped() -> None:
    start = datetime(2026, 9, 26, 0, 0, tzinfo=CPH)
    reader = _reader(
        {
            "forecast": [{"hour": start, "price": 9.0}],
            "prices": [
                {"start": start, "end": start + QUARTER, "price": 1.0},
                {"start": "nonsense", "price": 2.0},
                {"start": start + QUARTER, "price": "n/a"},
                "junk",
            ],
        }
    )

    assert reader.read_external_forecast(STROMLIGNING) == [
        (start.astimezone(UTC), pytest.approx(1.0))
    ]


@pytest.mark.parametrize(
    "attributes", [None, {}, {"prices": "unavailable"}, {"forecast": []}]
)
def test_a_missing_sensor_or_forecast_reads_as_nothing(
    attributes: dict[str, Any] | None,
) -> None:
    assert _reader(attributes).read_external_forecast(EDS) == []
    assert _reader(attributes).read_external_forecast("") == []


# --- Converting to the model's unit --------------------------------------------------


@pytest.mark.parametrize(
    "output",
    [
        PriceOutput(),
        PriceOutput(vat=0.25),
        PriceOutput(vat=0.25, surcharge=0.1),
        PriceOutput(vat=0.25, surcharge=100.0, price_type="MWh", precision=6),
    ],
)
def test_to_spot_is_the_inverse_of_convert(output: PriceOutput) -> None:
    for price in (-0.25, 0.0, 0.8, 3.5):
        assert output.to_spot(output.convert(price)) == pytest.approx(price, abs=2e-3)


def test_an_external_forecast_becomes_the_raw_spot_price() -> None:
    """Shown like this entry's prices: (spot + tariff + surcharge) × (1 + VAT)."""
    output = PriceOutput(vat=0.25, surcharge=0.1)
    tariffs = TariffSchedule({NOW: 0.4, NOW + QUARTER: 0.6})
    forecast = [
        (NOW - QUARTER, 9.0),
        (NOW, (1.0 + 0.4 + 0.1) * 1.25),
        (NOW + QUARTER, (-0.2 + 0.6 + 0.1) * 1.25),
    ]

    rows = external_spot_forecast(forecast, output, tariffs, NOW)

    # The slot before the first predicted one has a confirmed price
    assert rows == [
        ("2026-09-24T12:00:00Z", pytest.approx(1.0)),
        ("2026-09-24T12:15:00Z", pytest.approx(-0.2)),
    ]
    assert external_spot_forecast(forecast[1:2], PriceOutput(vat=0.0), None, NOW) == [
        ("2026-09-24T12:00:00Z", pytest.approx(1.875))
    ]


# --- Recording at a forecast run -----------------------------------------------------


async def _record(
    entity_ids: tuple[str, ...],
    forecasts: dict[str, list[tuple[datetime, float]]],
    known_end: datetime | None = None,
) -> tuple[Mock, Mock, Mock]:
    hass = Mock()
    hass.async_add_executor_job = AsyncMock()
    reader = Mock()
    reader.read_external_forecast.side_effect = lambda entity_id: forecasts.get(
        entity_id, []
    )
    predictor = Mock()
    with patch("homeassistant.util.dt.utcnow", return_value=NOW):
        await async_record_external_forecasts(
            hass, reader, predictor, entity_ids, PriceOutput(vat=0.0), None, known_end
        )
    return hass, reader, predictor


@pytest.mark.asyncio
async def test_without_a_configured_sensor_nothing_is_read_or_stored() -> None:
    hass, reader, _ = await _record((), {})

    reader.read_external_forecast.assert_not_called()
    hass.async_add_executor_job.assert_not_awaited()


@pytest.mark.asyncio
async def test_each_sensor_is_stored_under_its_entity_id() -> None:
    known_end = NOW + 2 * QUARTER
    forecasts = {
        STROMLIGNING: [(NOW + n * QUARTER, 1.0 + n) for n in range(4)],
        # Nothing beyond the confirmed prices: left out
        EDS: [(NOW, 5.0)],
    }

    hass, _, predictor = await _record((STROMLIGNING, EDS, "sensor.gone"), forecasts)
    hass.async_add_executor_job.assert_awaited_once_with(
        predictor.store_external_forecasts,
        {
            STROMLIGNING: [
                (utc_slot_key(NOW + n * QUARTER), pytest.approx(1.0 + n))
                for n in range(4)
            ],
            EDS: [("2026-09-24T12:00:00Z", pytest.approx(5.0))],
        },
    )

    # The model's predictions start where the confirmed prices end; so do these
    hass, _, predictor = await _record((STROMLIGNING, EDS), forecasts, known_end)
    hass.async_add_executor_job.assert_awaited_once_with(
        predictor.store_external_forecasts,
        {
            STROMLIGNING: [
                ("2026-09-24T12:30:00Z", pytest.approx(3.0)),
                ("2026-09-24T12:45:00Z", pytest.approx(4.0)),
            ]
        },
    )


@pytest.mark.asyncio
async def test_sensors_without_a_forecast_store_nothing() -> None:
    hass, reader, _ = await _record((EDS,), {})

    reader.read_external_forecast.assert_called_once_with(EDS)
    hass.async_add_executor_job.assert_not_awaited()


# --- Storage -------------------------------------------------------------------------


def test_forecasts_are_stored_per_source_slot_and_reading(tmp_path: Path) -> None:
    storage = LearningStorage(_hass(tmp_path), "DK1")
    try:
        slot, later = "2026-09-25T10:00:00Z", "2026-09-25T10:15:00Z"
        storage.insert_external_forecasts(
            STROMLIGNING, "2026-09-24T06:00:00Z", [(slot, 1.0), (later, 2.0)]
        )
        storage.insert_external_forecasts(EDS, "2026-09-24T06:00:00Z", [(slot, 3.0)])
        storage.insert_external_forecasts(
            STROMLIGNING, "2026-09-24T12:00:00Z", [(slot, 1.5)]
        )
        # The same reading stored again replaces itself
        storage.insert_external_forecasts(
            STROMLIGNING, "2026-09-24T12:00:00Z", [(slot, 1.6)]
        )

        assert storage.count_external_forecasts() == 4
        assert storage.find_external_forecasts(slot) == [
            {
                "source": EDS,
                "start": slot,
                "stored_at": "2026-09-24T06:00:00Z",
                "price": pytest.approx(3.0),
            },
            {
                "source": STROMLIGNING,
                "start": slot,
                "stored_at": "2026-09-24T06:00:00Z",
                "price": pytest.approx(1.0),
            },
            {
                "source": STROMLIGNING,
                "start": slot,
                "stored_at": "2026-09-24T12:00:00Z",
                "price": pytest.approx(1.6),
            },
        ]
        assert storage.delete_external_forecasts(slot) == 3
        assert storage.find_external_forecasts(slot) == []
        assert (
            storage.delete_external_forecasts_stored_before("2026-09-24T06:00:00Z") == 0
        )
        assert (
            storage.delete_external_forecasts_stored_before("2026-09-24T06:15:00Z") == 1
        )
        assert storage.count_external_forecasts() == 0
    finally:
        storage.close()


def test_errors_are_summed_per_source_date_and_bucket(tmp_path: Path) -> None:
    storage = LearningStorage(_hass(tmp_path), "DK1")
    try:
        storage.add_external_errors("2026-09-23", EDS, {"day_1": [0.5]})
        storage.add_external_errors(
            "2026-09-24", EDS, {"day_1": [0.1, -0.3], "day_2": [], "day_3": [1.0]}
        )
        storage.add_external_errors("2026-09-24", EDS, {"day_1": [0.2]})
        storage.add_external_errors("2026-09-24", STROMLIGNING, {"day_1": [-1.0]})
        storage.add_external_errors("2026-09-24", STROMLIGNING, {})

        sums = storage.get_external_error_sums("2026-09-24")

        assert set(sums) == {EDS, STROMLIGNING}
        assert sums[EDS]["day_1"] == (
            3,
            pytest.approx(0.0),
            pytest.approx(0.6),
            pytest.approx(0.14),
        )
        assert set(sums[EDS]) == {"day_1", "day_3"}
        assert sums[STROMLIGNING]["day_1"][0] == 1
        assert storage.get_external_error_sums("2026-09-23")[EDS]["day_1"][0] == 4
        assert storage.delete_external_accuracy_before("2026-09-24") == 1
        storage.clear_all()
        assert storage.get_external_error_sums("2000-01-01") == {}
        assert storage.count_external_forecasts() == 0
    finally:
        storage.close()


def test_an_existing_database_gains_the_tables(tmp_path: Path) -> None:
    storage = LearningStorage(_hass(tmp_path), "DK1")
    path = storage.db_path
    storage.close()
    with sqlite3.connect(path) as conn:
        conn.executescript(
            "DROP TABLE external_forecasts; DROP TABLE external_accuracy;"
            "DROP TABLE external_slot_errors;"
        )
    conn.close()

    storage = LearningStorage(_hass(tmp_path), "DK1")
    try:
        assert storage.count_external_forecasts() == 0
        assert storage.get_external_error_sums("2000-01-01") == {}
        assert storage.get_external_slot_errors("") == []
    finally:
        storage.close()


def test_slot_errors_are_kept_per_slot_bucket_and_source(tmp_path: Path) -> None:
    storage = LearningStorage(_hass(tmp_path), "DK1")
    first, second = "2026-09-24T10:00:00Z", "2026-09-24T10:15:00Z"
    try:
        storage.upsert_external_slot_errors(
            first, [("day_1", EDS, 2, 0.25), ("day_1", EXTERNAL_MODEL_SOURCE, 3, -0.5)]
        )
        storage.upsert_external_slot_errors(second, [("day_2", EDS, 1, 1.0)])
        storage.upsert_external_slot_errors(second, [])
        # A slot scored again replaces its rows
        storage.upsert_external_slot_errors(first, [("day_1", EDS, 1, 0.75)])

        assert storage.get_external_slot_errors("") == [
            {
                "start": first,
                "bucket": "day_1",
                "source": EXTERNAL_MODEL_SOURCE,
                "samples": 3,
                "error": pytest.approx(-0.5),
            },
            {
                "start": first,
                "bucket": "day_1",
                "source": EDS,
                "samples": 1,
                "error": pytest.approx(0.75),
            },
            {
                "start": second,
                "bucket": "day_2",
                "source": EDS,
                "samples": 1,
                "error": pytest.approx(1.0),
            },
        ]
        assert [r["start"] for r in storage.get_external_slot_errors(second)] == [
            second
        ]
        assert storage.delete_external_slot_errors_before(second) == 2
        storage.clear_all()
        assert storage.get_external_slot_errors("") == []
    finally:
        storage.close()


# --- Storing and scoring in the predictor --------------------------------------------


def _slot(hours_ago: float = 1.0) -> datetime:
    """Return a recent local slot start."""
    moment = dt_util.now() - timedelta(hours=hours_ago)
    return moment.replace(minute=moment.minute // 15 * 15, second=0, microsecond=0)


def _external(
    predictor: SpotPricePredictor,
    source: str,
    slot: datetime,
    lead: float,
    price: float,
) -> None:
    predictor.storage.insert_external_forecasts(
        source,
        utc_slot_key(slot - timedelta(hours=lead)),
        [(utc_slot_key(slot), price)],
    )


def _own(predictor: SpotPricePredictor, slot: datetime, lead: float) -> None:
    predictor.storage.insert_prediction(
        start=slot.isoformat(),
        price=2.0,
        confidence=0.8,
        hour=slot.hour,
        minute=slot.minute,
        stored_at=(slot - timedelta(hours=lead)).isoformat(),
    )


def test_a_reading_is_stored_and_old_ones_are_pruned(
    predictor: SpotPricePredictor,
) -> None:
    predictor.max_history_days = 30
    old = NOW - timedelta(days=31)
    predictor.storage.insert_external_forecasts(
        EDS, utc_slot_key(old), [(utc_slot_key(old + timedelta(days=2)), 1.0)]
    )
    slot = utc_slot_key(NOW + timedelta(hours=30))

    with patch("homeassistant.util.dt.utcnow", return_value=NOW + timedelta(minutes=7)):
        predictor.store_external_forecasts({EDS: [(slot, 1.5)], STROMLIGNING: []})

    # Stored with the exact time of the reading, like a prediction
    assert predictor.storage.find_external_forecasts(slot) == [
        {
            "source": EDS,
            "start": slot,
            "stored_at": "2026-09-24T12:07:00+00:00",
            "price": pytest.approx(1.5),
        }
    ]
    assert predictor.storage.count_external_forecasts() == 1


def test_scoring_a_slot_scores_its_external_forecasts_per_lead_time(
    predictor: SpotPricePredictor,
) -> None:
    slot = _slot()
    _own(predictor, slot, 20)
    _external(predictor, EDS, slot, 20, 2.75)
    _external(predictor, EDS, slot, 30, 3.5)
    _external(predictor, EDS, slot, 50, 1.5)
    _external(predictor, STROMLIGNING, slot, 100, 2.25)
    # Another slot's forecast stays until that slot is scored
    _external(predictor, EDS, slot + QUARTER, 20, 9.0)

    assert predictor.learn_from_actual_price(slot.isoformat(), 2.5) is True

    eds = predictor.external_accuracy[EDS]
    assert eds["day_1"] == {
        "mae": pytest.approx(0.25),
        "rmse": pytest.approx(0.25),
        "bias": pytest.approx(0.25),
        "samples": 1,
    }
    assert eds["day_2"]["bias"] == pytest.approx(1.0)
    assert eds["day_3"]["bias"] == pytest.approx(-1.0)
    assert predictor.external_accuracy[STROMLIGNING] == {
        "day_4_plus": {
            "mae": pytest.approx(0.25),
            "rmse": pytest.approx(0.25),
            "bias": pytest.approx(-0.25),
            "samples": 1,
        }
    }
    assert predictor.storage.count_external_forecasts() == 1
    # Keyed by the slot's local date, like the model's own accuracy
    sums = predictor.storage.get_external_error_sums(slot.date().isoformat())
    assert sums[EDS]["day_1"][0] == 1


def test_scoring_a_slot_keeps_its_mean_errors_for_the_model_and_the_sources(
    predictor: SpotPricePredictor,
) -> None:
    """What a blend of the two would have scored needs both errors of a slot (#157)."""
    slot = _slot()
    # The model: 2.0 twice a day ahead and once two days ahead, actual 2.5
    _own(predictor, slot, 20)
    _own(predictor, slot, 22)
    _own(predictor, slot, 30)
    _external(predictor, EDS, slot, 20, 2.75)
    _external(predictor, EDS, slot, 22, 3.25)
    _external(predictor, STROMLIGNING, slot, 20, 2.25)
    _external(predictor, STROMLIGNING, slot, 100, 2.0)

    assert predictor.learn_from_actual_price(slot.isoformat(), 2.5) is True

    rows = {
        (row["bucket"], row["source"]): (row["samples"], row["error"])
        for row in predictor.storage.get_external_slot_errors("")
    }
    assert {row["start"] for row in predictor.storage.get_external_slot_errors("")} == {
        utc_slot_key(slot)
    }
    # The mean of a source's forecasts in the bucket; the model's own only in
    # the buckets a source has (day 2 has none, day 4+ has no prediction)
    assert rows == {
        ("day_1", EDS): (2, pytest.approx(0.5)),
        ("day_1", STROMLIGNING): (1, pytest.approx(-0.25)),
        ("day_1", EXTERNAL_MODEL_SOURCE): (2, pytest.approx(-0.5)),
        ("day_4_plus", STROMLIGNING): (1, pytest.approx(-0.5)),
    }


def test_slot_errors_outside_the_rolling_window_are_pruned(
    predictor: SpotPricePredictor,
) -> None:
    now = dt_util.utcnow()
    old = utc_slot_key(now - timedelta(days=LEAD_TIME_WINDOW_DAYS, minutes=15))
    kept = utc_slot_key(now - timedelta(days=LEAD_TIME_WINDOW_DAYS - 1))
    for start in (old, kept):
        predictor.storage.upsert_external_slot_errors(start, [("day_1", EDS, 1, 0.5)])

    predictor.refresh_external_accuracy()

    assert [row["start"] for row in predictor.storage.get_external_slot_errors("")] == [
        kept
    ]


def test_the_models_own_learning_is_unchanged_by_external_forecasts(
    tmp_path: Path,
) -> None:
    slot = _slot()
    learned = []
    for name, with_external in (("plain", False), ("external", True)):
        hass = Mock()
        hass.config.path.return_value = str(tmp_path / name)
        predictor = SpotPricePredictor(hass, "DK1")
        try:
            _own(predictor, slot, 20)
            _own(predictor, slot, 30)
            if with_external:
                _external(predictor, EDS, slot, 20, 9.0)
            assert predictor.learn_from_actual_price(slot.isoformat(), 2.5) is True
            learned.append(
                (
                    predictor.lead_time_accuracy,
                    predictor.bias_correction,
                    predictor.error_metrics,
                    predictor.evaluation,
                    predictor.volatility_mae,
                )
            )
        finally:
            predictor.storage.close()

    assert learned[0] == learned[1]
    assert learned[0][0]["day_1"]["bias"] == pytest.approx(-0.5)


def test_without_external_forecasts_nothing_is_recorded(
    predictor: SpotPricePredictor,
) -> None:
    slot = _slot()
    _own(predictor, slot, 20)

    assert predictor.learn_from_actual_price(slot.isoformat(), 2.5) is True

    assert predictor.external_accuracy == {}
    assert predictor.storage.get_external_error_sums("2000-01-01") == {}
    assert predictor.storage.get_external_slot_errors("") == []


def test_a_slot_the_model_did_not_predict_is_not_scored(
    predictor: SpotPricePredictor,
) -> None:
    """Sources are scored for the slots the model is scored for."""
    slot = _slot()
    _external(predictor, EDS, slot, 20, 3.0)

    assert predictor.learn_from_actual_price(slot.isoformat(), 2.5) is False

    assert predictor.external_accuracy == {}
    assert predictor.storage.count_external_forecasts() == 1


def test_a_reading_on_the_hour_lands_in_the_models_bucket(
    predictor: SpotPricePredictor,
) -> None:
    """The 6-hourly run stores both seconds after the hour: both are day 1."""
    slot = _slot()
    made = slot - timedelta(hours=24) + timedelta(seconds=20)
    predictor.storage.insert_prediction(
        start=slot.isoformat(),
        price=2.0,
        confidence=0.8,
        hour=slot.hour,
        minute=slot.minute,
        stored_at=made.isoformat(),
    )
    with patch(
        "homeassistant.util.dt.utcnow",
        return_value=(made + timedelta(seconds=5)).astimezone(UTC),
    ):
        predictor.store_external_forecasts({EDS: [(utc_slot_key(slot), 3.0)]})

    assert predictor.learn_from_actual_price(slot.isoformat(), 2.5) is True

    assert list(predictor.lead_time_accuracy) == ["day_1"]
    assert list(predictor.external_accuracy[EDS]) == ["day_1"]


def test_a_naive_slot_start_is_local_time(predictor: SpotPricePredictor) -> None:
    slot = _slot()
    _external(predictor, EDS, slot, 20, 3.0)

    predictor.record_external_accuracy(slot.replace(tzinfo=None), [], 2.5)

    assert predictor.external_accuracy[EDS]["day_1"]["samples"] == 1


def test_accuracy_outside_the_rolling_window_is_pruned(
    predictor: SpotPricePredictor,
) -> None:
    today = dt_util.now().date()
    inside = today - timedelta(days=LEAD_TIME_WINDOW_DAYS - 1)
    outside = today - timedelta(days=LEAD_TIME_WINDOW_DAYS)
    predictor.storage.add_external_errors(inside.isoformat(), EDS, {"day_1": [0.5]})
    predictor.storage.add_external_errors(outside.isoformat(), EDS, {"day_1": [9.0]})

    predictor.refresh_external_accuracy()

    assert predictor.external_accuracy[EDS]["day_1"] == {
        "mae": pytest.approx(0.5),
        "rmse": pytest.approx(0.5),
        "bias": pytest.approx(0.5),
        "samples": 1,
    }
    assert predictor.storage.get_external_error_sums("2000-01-01")[EDS]["day_1"][0] == 1


def test_catch_up_learning_scores_the_external_forecasts(
    predictor: SpotPricePredictor,
) -> None:
    slot = _slot()
    day = slot.date()
    prices: list[float | None] = [None] * slots_in_local_day(day)
    prices[slot_index_in_day(slot)] = 2.5
    predictor.price_history = [{"date": day.isoformat(), "prices": prices}]
    _own(predictor, slot, 25)
    _external(predictor, EDS, slot, 25, 3.0)

    assert predictor.catch_up_learning() == 1

    assert predictor.external_accuracy[EDS]["day_2"]["bias"] == pytest.approx(0.5)
    assert predictor.storage.count_external_forecasts() == 0


def test_storage_errors_never_stop_the_forecast_or_the_learning(
    predictor: SpotPricePredictor, caplog: pytest.LogCaptureFixture
) -> None:
    locked = sqlite3.OperationalError("locked")
    with patch.object(
        predictor.storage, "insert_external_forecasts", side_effect=locked
    ):
        predictor.store_external_forecasts({EDS: [("2026-09-24T12:00:00Z", 1.0)]})
    with patch.object(predictor.storage, "find_external_forecasts", side_effect=locked):
        predictor.record_external_accuracy(_slot(), [], 1.0)

    assert "Failed to store the external forecasts: locked" in caplog.text
    assert "Failed to record the external forecast accuracy: locked" in caplog.text


@pytest.mark.asyncio
async def test_reset_learning_clears_the_external_accuracy() -> None:
    predictor = SpotPricePredictor.__new__(SpotPricePredictor)
    predictor.storage = Mock()
    predictor.storage.async_clear_storage = AsyncMock(return_value=True)
    predictor.external_accuracy = {EDS: {}}

    await predictor.reset_learning()

    assert predictor.external_accuracy == {}


# --- The accuracy sensors ------------------------------------------------------------


def _accuracy_sensor(
    external: dict[str, Any], bucket: str = "day_1", metric: str = "mae"
) -> LeadTimeAccuracySensor:
    own = {"day_1": {"mae": 0.12, "rmse": 0.2, "bias": -0.03, "samples": 40}}
    predictor = Mock(lead_time_accuracy=own, external_accuracy=external)
    return LeadTimeAccuracySensor(
        Mock(),
        MagicMock(entry_id="test"),
        {"ml_predictor": predictor},
        "DKK",
        3,
        bucket,
        metric,
    )


def test_the_accuracy_sensors_show_each_source_in_their_bucket() -> None:
    external = {
        EDS: {
            "day_1": {"mae": 0.2, "rmse": 0.3, "bias": 0.05, "samples": 38},
            "day_2": {"mae": 0.4, "rmse": 0.5, "bias": 0.1, "samples": 30},
        },
        STROMLIGNING: {"day_3": {"mae": 0.6, "rmse": 0.7, "bias": 0.2, "samples": 9}},
    }

    attributes = _accuracy_sensor(external).extra_state_attributes

    assert attributes["samples"] == 40
    assert attributes["external"] == {
        EDS: {"mae": pytest.approx(0.2), "bias": pytest.approx(0.05), "samples": 38}
    }
    rmse = _accuracy_sensor(external, "day_3", "rmse").extra_state_attributes
    assert rmse["external"] == {
        STROMLIGNING: {
            "rmse": pytest.approx(0.7),
            "bias": pytest.approx(0.2),
            "samples": 9,
        }
    }


def test_without_external_forecasts_the_attributes_are_unchanged() -> None:
    assert _accuracy_sensor({}).extra_state_attributes == {
        "samples": 40,
        "bias": pytest.approx(-0.03),
        "window_days": LEAD_TIME_WINDOW_DAYS,
    }
    # No source has samples in this bucket
    assert (
        "external"
        not in _accuracy_sensor(
            {EDS: {"day_2": {"mae": 1, "rmse": 1, "bias": 1, "samples": 1}}}
        ).extra_state_attributes
    )
    sensor = LeadTimeAccuracySensor(
        Mock(), MagicMock(entry_id="test"), {}, "DKK", 3, "day_1", "mae"
    )
    assert "external" not in sensor.extra_state_attributes


# --- Configuration -------------------------------------------------------------------


def _entry(data: dict[str, Any], options: dict[str, Any]) -> MagicMock:
    entry = MagicMock()
    entry.data = {CONF_REGION: "DK1", **data}
    entry.options = options
    return entry


@pytest.mark.parametrize(
    ("data", "options", "expected"),
    [
        ({}, {}, ()),
        ({CONF_EXTERNAL_FORECAST_SENSORS: [EDS]}, {}, (EDS,)),
        (
            {CONF_EXTERNAL_FORECAST_SENSORS: [EDS]},
            {CONF_EXTERNAL_FORECAST_SENSORS: [STROMLIGNING, EDS]},
            (STROMLIGNING, EDS),
        ),
        # Emptied in the options: the setup's sensors do not come back
        (
            {CONF_EXTERNAL_FORECAST_SENSORS: [EDS]},
            {CONF_EXTERNAL_FORECAST_SENSORS: []},
            (),
        ),
        ({}, {CONF_EXTERNAL_FORECAST_SENSORS: EDS}, (EDS,)),
    ],
)
def test_the_configured_sensors_are_read_from_the_entry(
    data: dict[str, Any], options: dict[str, Any], expected: tuple[str, ...]
) -> None:
    sensors = SensorEntities.from_entry(_entry(data, options))

    assert sensors.external_forecasts == expected


def _field(schema: dict[Any, Any]) -> Any:
    return next(k for k in schema if str(k) == CONF_EXTERNAL_FORECAST_SENSORS)


@pytest.mark.asyncio
async def test_the_config_flow_offers_the_sensors_without_a_default() -> None:
    flow: Any = OpenSpotForecastConfigFlow()
    flow.hass = Mock()
    flow.async_show_form = Mock(return_value={"type": "show_form"})
    flow.async_create_entry = Mock(return_value={"type": "create_entry"})
    flow._data = {CONF_REGION: "DK1"}

    await flow.async_step_sensors()

    schema = flow.async_show_form.call_args.kwargs["data_schema"].schema
    key = _field(schema)
    assert isinstance(key, vol.Optional)
    assert key.default is vol.UNDEFINED
    assert schema[key].config["multiple"] is True
    assert schema[key].config["domain"] == ["sensor"]

    await flow.async_step_sensors({CONF_EXTERNAL_FORECAST_SENSORS: [EDS]})
    assert flow.async_create_entry.call_args.kwargs["data"] == {
        CONF_REGION: "DK1",
        CONF_EXTERNAL_FORECAST_SENSORS: [EDS],
    }


def _options_flow(data: dict[str, Any], options: dict[str, Any]) -> Any:
    hass = Mock()
    hass.config_entries.async_get_known_entry.return_value = _entry(data, options)
    flow: Any = OpenSpotForecastOptionsFlow()
    flow.hass = hass
    flow.handler = "test_entry"
    flow.async_show_form = Mock(return_value={"type": "show_form"})
    flow.async_create_entry = Mock(return_value={"type": "create_entry"})
    return flow


@pytest.mark.asyncio
async def test_the_options_flow_shows_and_changes_the_sensors() -> None:
    flow = _options_flow({CONF_EXTERNAL_FORECAST_SENSORS: [EDS]}, {})

    await flow.async_step_init()

    schema = flow.async_show_form.call_args.kwargs["data_schema"].schema
    key = _field(schema)
    assert isinstance(key, vol.Optional)
    assert key.description == {"suggested_value": [EDS]}
    assert schema[key].config["multiple"] is True

    await flow.async_step_init({CONF_EXTERNAL_FORECAST_SENSORS: [EDS, STROMLIGNING]})
    assert flow.async_create_entry.call_args.kwargs["data"] == {
        CONF_EXTERNAL_FORECAST_SENSORS: [EDS, STROMLIGNING]
    }


@pytest.mark.asyncio
async def test_emptying_the_sensors_in_the_options_stops_the_recording() -> None:
    """The frontend leaves an emptied picker out of the input."""
    flow = _options_flow({CONF_EXTERNAL_FORECAST_SENSORS: [EDS]}, {})

    await flow.async_step_init({"vat": 0.25})

    assert flow.async_create_entry.call_args.kwargs["data"] == {
        "vat": 0.25,
        CONF_EXTERNAL_FORECAST_SENSORS: [],
    }


@pytest.mark.asyncio
async def test_an_entry_without_sensors_keeps_its_options_as_they_were() -> None:
    flow = _options_flow({}, {})

    await flow.async_step_init()
    schema = flow.async_show_form.call_args.kwargs["data_schema"].schema
    assert _field(schema).description == {"suggested_value": []}

    await flow.async_step_init({"vat": 0.25})
    assert flow.async_create_entry.call_args.kwargs["data"] == {"vat": 0.25}
