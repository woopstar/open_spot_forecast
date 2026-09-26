"""The raw spot price series the ML model uses, and VAT applied once (issue #16)."""

import json
import logging
import sqlite3
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest

from homeassistant.util import dt as dt_util

from custom_components.open_spot_forecast.ml.spot_migration import (
    SPOT_PRICE_SCHEMA_VERSION,
)
from custom_components.open_spot_forecast.ml.storage import LearningStorage
from custom_components.open_spot_forecast.sensor import (
    MLPredictionSensor,
    SpotPriceSensor,
)
from custom_components.open_spot_forecast.sensor_reader import SensorReader
from custom_components.open_spot_forecast.spot_prices import (
    extract_latest_known_timestamp,
    ml_price_inputs,
)

VAT = 0.25
PRECISION = 4


def _items(day: datetime, prices: list[float]) -> list[dict[str, Any]]:
    return [
        {"price": price, "start": (day + timedelta(minutes=15 * i)).isoformat()}
        for i, price in enumerate(prices)
    ]


# --- Reading the spot series --------------------------------------------------------


def test_read_spot_prices_reads_today_and_tomorrow() -> None:
    """Today from the spot sensor, tomorrow from its binary sensor."""
    today = dt_util.start_of_local_day()
    tomorrow = today + timedelta(days=1)
    spot_today = [0.4 + i / 1000 for i in range(96)]
    spot_tomorrow = [0.6 + i / 1000 for i in range(96)]
    states = {
        "sensor.spot": Mock(
            state="0.4", attributes={"prices": _items(today, spot_today)}
        ),
        "binary_sensor.spot_tomorrow": Mock(
            state="on", attributes={"prices": _items(tomorrow, spot_tomorrow)}
        ),
    }
    hass = Mock()
    hass.states.get.side_effect = states.get

    spot = SensorReader(hass).read_spot_prices(
        "sensor.spot", "binary_sensor.spot_tomorrow"
    )

    assert spot["today"] == pytest.approx(spot_today)
    assert spot["tomorrow"] == pytest.approx(spot_tomorrow)
    assert len(spot["raw_tomorrow"]) == 96


def test_read_spot_prices_without_sensors_is_empty() -> None:
    hass = Mock()
    hass.states.get.return_value = None

    spot = SensorReader(hass).read_spot_prices(None, "binary_sensor.missing")

    assert spot == {"today": [], "tomorrow": [], "raw_today": [], "raw_tomorrow": []}


# --- ML price inputs ---------------------------------------------------------------


def test_ml_price_inputs_join_today_and_tomorrow_and_find_the_end() -> None:
    start = datetime(2026, 9, 25, 22, 0, tzinfo=UTC)
    spot = {
        "today": [0.1, 0.2],
        "tomorrow": [0.3],
        "raw_today": [{"start": start.isoformat()}],
        "raw_tomorrow": [{"start": (start + timedelta(minutes=15)).isoformat()}],
    }

    prices, end = ml_price_inputs(spot)

    assert prices == pytest.approx([0.1, 0.2, 0.3])
    assert end == start + timedelta(minutes=30)


@pytest.mark.parametrize("spot", [None, {}, {"today": [], "tomorrow": [0.3]}])
def test_ml_price_inputs_without_todays_spot_prices(spot: dict | None) -> None:
    """No spot prices, no model input: consumer prices are never substituted."""
    assert ml_price_inputs(spot) == ([], None)


def test_latest_known_timestamp_skips_unusable_items() -> None:
    """Only items with a parseable timestamp count; datetimes are accepted."""
    latest = datetime(2026, 9, 25, 10, 0, tzinfo=UTC)
    raw: list[Any] = [
        "not a dict",
        {"price": 1.0},
        {"start": 12345},
        {"start": "not a time"},
        {"start": "2026-13-45T99:00:00"},
        {"time": latest},
        {"timestamp": (latest - timedelta(hours=1)).isoformat()},
    ]

    assert extract_latest_known_timestamp(raw) == latest + timedelta(minutes=15)


def test_latest_known_timestamp_without_timestamps_is_none() -> None:
    assert extract_latest_known_timestamp([{"price": 1.0}]) is None


# --- VAT exactly once ----------------------------------------------------------------


def test_known_spot_price_gets_vat_exactly_once() -> None:
    """A raw spot price of 0.80 with 25 % VAT is exactly 1.00 everywhere."""
    start = datetime.now() + timedelta(hours=1)
    predictor = MagicMock()
    predictor.predictions = [
        {
            "start": start.isoformat(),
            "end": (start + timedelta(minutes=15)).isoformat(),
            "price": 0.80,
            "confidence": 0.9,
        }
    ]
    predictor.get_prediction_stats.return_value = {
        "min_price": 0.80,
        "max_price": 0.80,
        "mean_price": 0.80,
    }
    sensor = MLPredictionSensor(
        MagicMock(),
        MagicMock(entry_id="test"),
        {"ml_predictor": predictor},
        "DKK",
        VAT,
        PRECISION,
        "kWh",
    )

    attrs = sensor.extra_state_attributes

    assert sensor.native_value == pytest.approx(0.80 * 1.25)
    assert attrs["predictions"][0]["price"] == pytest.approx(1.0)
    for key in ("forecast_min", "forecast_max", "forecast_mean"):
        assert attrs[key] == pytest.approx(1.0)
    assert attrs["includes_vat"] is True
    assert attrs["includes_tariffs"] is False
    assert attrs["vat"] == pytest.approx(VAT)


# --- Upgrade: consumer-price learning data is discarded ------------------------------


def _hass(tmp_path: Path) -> Mock:
    hass = Mock()
    hass.config.path.return_value = str(tmp_path / ".storage")
    return hass


def test_upgrade_discards_consumer_price_history_and_logs_it(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Tariff-inclusive prices cannot become spot prices: they are dropped once."""
    storage = LearningStorage(_hass(tmp_path), "DK1")
    storage.save_all(
        {
            "prediction_history": [
                {"start": "2026-09-26T10:00:00+02:00", "price": 2.5, "confidence": 0.8}
            ],
            "error_metrics": {40: {"errors": [0.1], "count": 1}},
            "bias_correction": {40: 0.1},
        }
    )
    storage.insert_weather_snapshot(
        "2026-09-24T10:00:00+02:00", 12.0, 5.0, 180.0, 50.0, 70.0, None
    )
    db_path = storage.db_path
    storage.close()
    with closing(sqlite3.connect(db_path)) as conn:
        # A v5 database kept its (consumer) prices as JSON days
        conn.execute(
            "INSERT INTO price_history (date, prices) VALUES (?, ?)",
            ("2026-09-24", json.dumps([2.4] * 96)),
        )
        conn.execute("UPDATE meta SET value = '5' WHERE key = 'schema_version'")
        conn.commit()

    with caplog.at_level(logging.INFO):
        upgraded = LearningStorage(_hass(tmp_path), "DK1")
    try:
        data = upgraded.load_all()
        assert data is not None
        assert data["price_history"] == []
        assert data["error_metrics"] == {}
        assert data["bias_correction"] == {}
        assert upgraded.count_predictions() == 0
        assert len(upgraded.load_weather_history()) == 1
        assert "discarded 1 days of consumer price history" in caplog.text

        # New spot prices are kept from now on
        upgraded.save_all(
            {"price_history": [{"date": "2026-09-25", "prices": [0.5] * 96}]}
        )
    finally:
        upgraded.close()
    reopened = LearningStorage(_hass(tmp_path), "DK1")
    try:
        data = reopened.load_all()
        assert data is not None
        assert [entry["date"] for entry in data["price_history"]] == ["2026-09-25"]
    finally:
        reopened.close()


def test_new_database_starts_at_the_spot_price_schema(tmp_path: Path) -> None:
    storage = LearningStorage(_hass(tmp_path), "DK1")
    try:
        with closing(sqlite3.connect(storage.db_path)) as conn:
            row = conn.execute(
                "SELECT value FROM meta WHERE key = 'schema_version'"
            ).fetchone()
        assert int(row[0]) >= SPOT_PRICE_SCHEMA_VERSION
    finally:
        storage.close()


def test_current_price_sensor_is_labelled_all_in() -> None:
    """The displayed Stromligning price includes tariffs and VAT; the ML one does not."""
    sensor = SpotPriceSensor(
        MagicMock(),
        MagicMock(entry_id="test"),
        {"stromligning_data": {"current_price": 2.45, "today": [2.45]}},
        "DK1",
        "DKK",
        VAT,
        PRECISION,
        "kWh",
    )

    attrs = sensor.extra_state_attributes

    assert sensor.native_value == pytest.approx(2.45)
    assert attrs["includes_vat"] is True
    assert attrs["includes_tariffs"] is True
