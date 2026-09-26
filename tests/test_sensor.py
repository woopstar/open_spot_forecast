"""Tests for the Open Spot Forecast sensor platform."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch
from zoneinfo import ZoneInfo

import pytest

from homeassistant.util import dt as dt_util

from custom_components.open_spot_forecast.accuracy_sensor import (
    LeadTimeAccuracySensor,
)
from custom_components.open_spot_forecast.const import DOMAIN
from custom_components.open_spot_forecast.sensor import (
    LearningMetricsSensor,
    MLPredictionSensor,
    PredictionConfidenceSensor,
    SpotPriceSensor,
    TodayMaxSensor,
    TodayMeanSensor,
    TodayMinSensor,
    TomorrowMaxSensor,
    TomorrowMeanSensor,
    TomorrowMinSensor,
    async_setup_entry,
)

CPH = ZoneInfo("Europe/Copenhagen")
VAT = 0.25
PRECISION = 3
PRICE_TYPE = "kWh"


def _hass() -> Mock:
    """Return a mock Home Assistant whose ``data`` attribute is a real dict."""
    hass = Mock()
    hass.data = {}
    return hass


def _entry() -> MagicMock:
    """Return a mock config entry with realistic data/options."""
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.data = {"region": "DK1", "currency": "DKK"}
    entry.options = {"vat": VAT, "precision": PRECISION, "price_type": PRICE_TYPE}
    return entry


def _make_sensor(cls, api_data=None):
    """Instantiate a sensor class with a consistent mock environment."""
    if cls in (PredictionConfidenceSensor, LearningMetricsSensor):
        return cls(_hass(), _entry(), api_data or {})
    if cls is SpotPriceSensor:
        return cls(
            _hass(), _entry(), api_data or {}, "DK1", "DKK", VAT, PRECISION, PRICE_TYPE
        )
    return cls(_hass(), _entry(), api_data or {}, "DKK", VAT, PRECISION, PRICE_TYPE)


ALL_SENSOR_CLASSES = [
    SpotPriceSensor,
    TodayMinSensor,
    TodayMaxSensor,
    TodayMeanSensor,
    TomorrowMinSensor,
    TomorrowMaxSensor,
    TomorrowMeanSensor,
    MLPredictionSensor,
    PredictionConfidenceSensor,
    LearningMetricsSensor,
]


# --------------------------------------------------------------------------- #
# Module-level setup
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_async_setup_entry_adds_accuracy_sensors_with_ml():
    """With ML enabled, the ten sensors are followed by eight accuracy sensors."""
    hass = Mock()
    entry = MagicMock()
    entry.entry_id = "test"
    entry.data = {"region": "DK1", "currency": "DKK"}
    entry.options = {"vat": VAT, "precision": PRECISION, "price_type": PRICE_TYPE}
    api_data = {
        "stromligning_data": {"current_price": 1.0},
        "ml_predictor": MagicMock(),
        "last_update": "now",
    }
    hass.data = {DOMAIN: {"test": api_data}}
    async_add_entities = Mock()

    await async_setup_entry(hass, entry, async_add_entities)

    async_add_entities.assert_called_once()
    sensors, update = async_add_entities.call_args[0]
    assert update is True
    assert len(sensors) == 18
    assert isinstance(sensors[0], SpotPriceSensor)
    assert isinstance(sensors[9], LearningMetricsSensor)
    assert all(isinstance(s, LeadTimeAccuracySensor) for s in sensors[10:])


@pytest.mark.asyncio
async def test_async_setup_entry_uses_defaults_when_config_missing():
    """async_setup_entry falls back to constants when config keys are absent.

    Without an ML predictor, no accuracy sensors are created.
    """
    hass = Mock()
    entry = MagicMock()
    entry.entry_id = "test"
    entry.data = {}
    entry.options = {}
    hass.data = {DOMAIN: {"test": {}}}
    async_add_entities = Mock()

    await async_setup_entry(hass, entry, async_add_entities)

    sensors, _ = async_add_entities.call_args[0]
    assert len(sensors) == 10
    assert sensors[0].region == "DK1"
    assert sensors[0].currency == "DKK"
    assert sensors[0].vat == 0.25
    assert sensors[0].precision == 3
    assert sensors[0].price_type == "kWh"


# --------------------------------------------------------------------------- #
# Shared async lifecycle (async_added_to_hass + _handle_update)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.parametrize("cls", ALL_SENSOR_CLASSES)
async def test_async_lifecycle(cls):
    """Every sensor registers a dispatcher callback and writes state on update."""
    sensor = _make_sensor(cls)
    await sensor.async_added_to_hass()

    sensor.async_write_ha_state = Mock()
    await sensor._handle_update()
    sensor.async_write_ha_state.assert_called_once()


@pytest.mark.asyncio
async def test_learning_metrics_handle_update_invalidates_cache():
    """_handle_update resets the cached metrics so they are recomputed."""
    predictor = MagicMock()
    predictor.get_learning_metrics.return_value = {"total_samples": 5}
    sensor = LearningMetricsSensor(_hass(), _entry(), {"ml_predictor": predictor})
    await sensor.async_added_to_hass()

    assert sensor.native_value == 5
    assert sensor._cached_metrics is not None

    sensor.async_write_ha_state = Mock()
    await sensor._handle_update()
    assert sensor._cached_metrics is None
    sensor.async_write_ha_state.assert_called_once()


# --------------------------------------------------------------------------- #
# SpotPriceSensor
# --------------------------------------------------------------------------- #


def test_spot_price_native_value_stromligning():
    """Stromligning current price is used directly (already incl. VAT)."""
    api_data = {"stromligning_data": {"current_price": 100.0}}
    sensor = SpotPriceSensor(
        _hass(), _entry(), api_data, "DK1", "DKK", VAT, PRECISION, PRICE_TYPE
    )
    assert sensor.native_value == pytest.approx(100.0)


@pytest.mark.usefixtures("copenhagen_time_zone")
def test_spot_price_native_value_dayahead() -> None:
    """The day-ahead source shows the current slot's spot price with VAT (#27)."""
    # 10:20 local is slot 41 of the day
    now = datetime(2026, 9, 24, 10, 20, tzinfo=CPH)
    prices: list[float | None] = [float(slot) for slot in range(96)]
    prices[40] = None
    api_data = {
        "price_source": "dayahead",
        "prices_today": prices,
        # Stromligning data of an earlier configuration is not shown
        "stromligning_data": {"current_price": 100.0},
    }
    sensor = SpotPriceSensor(
        _hass(), _entry(), api_data, "DK1", "DKK", VAT, PRECISION, PRICE_TYPE
    )

    with patch("homeassistant.util.dt.now", return_value=now):
        assert sensor.native_value == pytest.approx(41.0)
        sensor.api_data["prices_today"] = prices[:40]
        assert sensor.native_value is None
    with patch("homeassistant.util.dt.now", return_value=now - timedelta(minutes=15)):
        sensor.api_data["prices_today"] = prices
        assert sensor.native_value is None


def test_spot_price_native_value_none():
    """No data yields None."""
    sensor = SpotPriceSensor(
        _hass(), _entry(), {}, "DK1", "DKK", VAT, PRECISION, PRICE_TYPE
    )
    assert sensor.native_value is None


def test_spot_price_attributes_dayahead() -> None:
    """Day-ahead prices are the spot price with VAT, without tariffs."""
    api_data = {
        "price_source": "dayahead",
        "prices_today": [1.0, 2.0],
        "prices_tomorrow": [3.0, 4.0],
    }
    sensor = SpotPriceSensor(
        _hass(), _entry(), api_data, "DK1", "DKK", VAT, PRECISION, PRICE_TYPE
    )

    attrs = sensor.extra_state_attributes

    assert attrs["today_prices"] == [1.0, 2.0]
    assert attrs["tomorrow_prices"] == [3.0, 4.0]
    assert attrs["price_source"] == "dayahead"
    assert attrs["includes_vat"] is True
    assert attrs["includes_tariffs"] is False


def test_spot_price_attributes_stromligning():
    """Attributes use Stromligning price lists when available."""
    api_data = {
        "stromligning_data": {"today": [1.0, 2.0], "tomorrow": [3.0]},
        "last_update": "2026-09-22T00:00:00",
    }
    sensor = SpotPriceSensor(
        _hass(), _entry(), api_data, "DK1", "DKK", VAT, PRECISION, PRICE_TYPE
    )

    attrs = sensor.extra_state_attributes

    assert attrs["region"] == "DK1"
    assert attrs["currency"] == "DKK"
    assert attrs["vat"] == 0.25
    assert attrs["last_update"] == "2026-09-22T00:00:00"
    assert attrs["today_prices"] == [1.0, 2.0]
    assert attrs["tomorrow_prices"] == [3.0]
    assert attrs["price_source"] == "stromligning"


# --------------------------------------------------------------------------- #
# Today / Tomorrow min / max / mean sensors
# --------------------------------------------------------------------------- #


def _min(values):
    return min(values)


def _max(values):
    return max(values)


def _mean(values):
    return sum(values) / len(values)


PRICE_STAT_SENSORS = [
    (TodayMinSensor, "today", _min),
    (TodayMaxSensor, "today", _max),
    (TodayMeanSensor, "today", _mean),
    (TomorrowMinSensor, "tomorrow", _min),
    (TomorrowMaxSensor, "tomorrow", _max),
    (TomorrowMeanSensor, "tomorrow", _mean),
]


@pytest.mark.parametrize("cls, list_key, reducer", PRICE_STAT_SENSORS)
def test_price_stat_native_value_stromligning(cls, list_key, reducer):
    """Stromligning list prices drive the aggregate directly."""
    prices = [10.0, 20.0, 30.0]
    sensor = cls(
        _hass(),
        _entry(),
        {"stromligning_data": {list_key: prices}},
        "DKK",
        VAT,
        PRECISION,
        PRICE_TYPE,
    )
    assert sensor.native_value == pytest.approx(round(reducer(prices), PRECISION))


@pytest.mark.parametrize("cls, list_key, reducer", PRICE_STAT_SENSORS)
def test_price_stat_native_value_dayahead(cls, list_key, reducer):
    """Day-ahead prices (with VAT, missing slots skipped) drive the aggregate."""
    prices = [10.0, None, 30.0]
    sensor = cls(
        _hass(),
        _entry(),
        {"price_source": "dayahead", f"prices_{list_key}": prices},
        "DKK",
        VAT,
        PRECISION,
        PRICE_TYPE,
    )
    assert sensor.native_value == pytest.approx(round(reducer([10.0, 30.0]), PRECISION))


@pytest.mark.parametrize("cls, list_key, reducer", PRICE_STAT_SENSORS)
def test_price_stat_native_value_none(cls, list_key, reducer):
    """No data yields None."""
    sensor = cls(_hass(), _entry(), {}, "DKK", VAT, PRECISION, PRICE_TYPE)
    assert sensor.native_value is None


@pytest.mark.parametrize("cls, list_key, reducer", PRICE_STAT_SENSORS)
def test_price_stat_native_value_empty_stromligning(cls, list_key, reducer):
    """An empty Stromligning list has no aggregate."""
    sensor = cls(
        _hass(),
        _entry(),
        {"stromligning_data": {list_key: []}},
        "DKK",
        VAT,
        PRECISION,
        PRICE_TYPE,
    )
    assert sensor.native_value is None


# --------------------------------------------------------------------------- #
# MLPredictionSensor
# --------------------------------------------------------------------------- #


def _prediction(start, price, confidence=0.8):
    end = start + timedelta(minutes=15)
    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "price": price,
        "confidence": confidence,
    }


def test_ml_prediction_native_value_future():
    """The next future prediction is selected."""
    future = dt_util.now() + timedelta(hours=1)
    predictor = MagicMock()
    predictor.predictions = [_prediction(future, 100.0)]
    sensor = MLPredictionSensor(
        _hass(),
        _entry(),
        {"ml_predictor": predictor},
        "DKK",
        VAT,
        PRECISION,
        PRICE_TYPE,
    )

    assert sensor.native_value == pytest.approx(100.0 * (1 + VAT))


def test_ml_prediction_native_value_is_none_when_every_prediction_is_past():
    """A stale forecast is not shown as the current price (#56)."""
    past = dt_util.now() - timedelta(hours=1)
    predictor = MagicMock()
    predictor.predictions = [_prediction(past, 80.0)]
    sensor = MLPredictionSensor(
        _hass(),
        _entry(),
        {"ml_predictor": predictor},
        "DKK",
        VAT,
        PRECISION,
        PRICE_TYPE,
    )

    assert sensor.native_value is None


def test_ml_prediction_native_value_skips_an_invalid_timestamp():
    """An unparseable timestamp is skipped; the next usable prediction is used."""
    predictor = MagicMock()
    predictor.predictions = [
        {"start": "not-a-date", "end": "x", "price": 60.0, "confidence": 0.6},
        _prediction(dt_util.now() + timedelta(hours=1), 65.0),
    ]
    sensor = MLPredictionSensor(
        _hass(),
        _entry(),
        {"ml_predictor": predictor},
        "DKK",
        VAT,
        PRECISION,
        PRICE_TYPE,
    )

    assert sensor.native_value == pytest.approx(65.0 * (1 + VAT))


def test_ml_prediction_native_value_missing_start_is_skipped():
    """A prediction with no start timestamp is never the state."""
    predictor = MagicMock()
    predictor.predictions = [
        {"start": None, "end": None, "price": 70.0, "confidence": 0.5}
    ]
    sensor = MLPredictionSensor(
        _hass(),
        _entry(),
        {"ml_predictor": predictor},
        "DKK",
        VAT,
        PRECISION,
        PRICE_TYPE,
    )

    assert sensor.native_value is None


def test_ml_prediction_native_value_price_none():
    """A future prediction with no price yields None."""
    future = dt_util.now() + timedelta(hours=1)
    predictor = MagicMock()
    predictor.predictions = [_prediction(future, None)]
    sensor = MLPredictionSensor(
        _hass(),
        _entry(),
        {"ml_predictor": predictor},
        "DKK",
        VAT,
        PRECISION,
        PRICE_TYPE,
    )

    assert sensor.native_value is None


def test_ml_prediction_native_value_no_predictions():
    """A predictor with no predictions yields None."""
    predictor = MagicMock()
    predictor.predictions = []
    sensor = MLPredictionSensor(
        _hass(),
        _entry(),
        {"ml_predictor": predictor},
        "DKK",
        VAT,
        PRECISION,
        PRICE_TYPE,
    )

    assert sensor.native_value is None


def test_ml_prediction_native_value_no_predictor():
    """No predictor yields None."""
    sensor = MLPredictionSensor(
        _hass(), _entry(), {}, "DKK", VAT, PRECISION, PRICE_TYPE
    )
    assert sensor.native_value is None


def test_ml_prediction_extra_attributes_full():
    """Predictions and stats are exposed with unit conversion."""
    predictor = MagicMock()
    predictor.predictions = [
        _prediction(dt_util.now() + timedelta(hours=1), 100.0, 0.8),
        _prediction(dt_util.now() + timedelta(hours=2), 200.0, 0.9),
    ]
    predictor.get_prediction_stats.return_value = {
        "min_price": 100.0,
        "max_price": 200.0,
        "mean_price": 150.0,
        "mean_confidence": 0.85,
        "total_predictions": 2,
        "is_ml_model": True,
        "training_samples": 10,
    }
    sensor = MLPredictionSensor(
        _hass(),
        _entry(),
        {"ml_predictor": predictor},
        "DKK",
        VAT,
        PRECISION,
        PRICE_TYPE,
    )

    attrs = sensor.extra_state_attributes

    # Predictions are raw spot prices in currency/kWh: VAT is added once
    conversion = 1 + VAT
    assert len(attrs["predictions"]) == 2
    assert attrs["predictions"][0]["price"] == round(100.0 * (1 + VAT), PRECISION)
    assert attrs["predictions"][0]["unit"] == "DKK/kWh"
    assert attrs["predictions"][0]["confidence"] == 0.8
    assert attrs["forecast_min"] == round(100.0 * conversion, PRECISION)
    assert attrs["forecast_max"] == round(200.0 * conversion, PRECISION)
    assert attrs["forecast_mean"] == round(150.0 * conversion, PRECISION)
    assert attrs["unit"] == "DKK/kWh"
    assert attrs["includes_vat"] is True
    assert attrs["includes_tariffs"] is False
    assert attrs["mean_confidence"] == 0.85
    assert attrs["total_predictions"] == 2
    assert attrs["is_ml_model"] is True
    assert attrs["training_samples"] == 10


def test_ml_prediction_extra_attributes_empty_stats():
    """Empty stats omit the forecast_* attributes."""
    predictor = MagicMock()
    predictor.predictions = [_prediction(dt_util.now() + timedelta(hours=1), 100.0)]
    predictor.get_prediction_stats.return_value = {}
    sensor = MLPredictionSensor(
        _hass(),
        _entry(),
        {"ml_predictor": predictor},
        "DKK",
        VAT,
        PRECISION,
        PRICE_TYPE,
    )

    attrs = sensor.extra_state_attributes

    assert len(attrs["predictions"]) == 1
    assert "forecast_min" not in attrs
    assert "forecast_max" not in attrs
    assert "forecast_mean" not in attrs
    assert "mean_confidence" not in attrs


def test_ml_prediction_extra_attributes_skips_missing_price():
    """Predictions without a price are skipped in the attribute list."""
    predictor = MagicMock()
    predictor.predictions = [
        _prediction(dt_util.now() + timedelta(hours=1), None),
        _prediction(dt_util.now() + timedelta(hours=2), 100.0),
    ]
    predictor.get_prediction_stats.return_value = {}
    sensor = MLPredictionSensor(
        _hass(),
        _entry(),
        {"ml_predictor": predictor},
        "DKK",
        VAT,
        PRECISION,
        PRICE_TYPE,
    )

    attrs = sensor.extra_state_attributes

    assert len(attrs["predictions"]) == 1
    assert attrs["predictions"][0]["price"] == round(100.0 * (1 + VAT), PRECISION)


def test_ml_prediction_extra_attributes_no_predictor():
    """No predictor yields empty attributes."""
    sensor = MLPredictionSensor(
        _hass(), _entry(), {}, "DKK", VAT, PRECISION, PRICE_TYPE
    )
    assert sensor.extra_state_attributes == {}


# --------------------------------------------------------------------------- #
# PredictionConfidenceSensor
# --------------------------------------------------------------------------- #


def test_prediction_confidence_native_value():
    """mean_confidence is scaled to a percentage."""
    predictor = MagicMock()
    predictor.get_prediction_stats.return_value = {"mean_confidence": 0.85}
    sensor = PredictionConfidenceSensor(_hass(), _entry(), {"ml_predictor": predictor})

    assert sensor.native_value == pytest.approx(85.0)


def test_prediction_confidence_native_value_missing_confidence():
    """Stats without mean_confidence yield None."""
    predictor = MagicMock()
    predictor.get_prediction_stats.return_value = {"unrelated": 1.0}
    sensor = PredictionConfidenceSensor(_hass(), _entry(), {"ml_predictor": predictor})

    assert sensor.native_value is None


def test_prediction_confidence_native_value_none():
    """No predictor yields None."""
    sensor = PredictionConfidenceSensor(_hass(), _entry(), {})
    assert sensor.native_value is None


# --------------------------------------------------------------------------- #
# LearningMetricsSensor
# --------------------------------------------------------------------------- #


def test_learning_metrics_full_attributes():
    """All metrics keys are surfaced as attributes."""
    predictor = MagicMock()
    predictor.get_learning_metrics.return_value = {
        "total_samples": 100,
        "status": "learning",
        "message": "ok",
        "is_learning": True,
        "mae": 1.5,
        "rmse": 2.0,
        "mean_bias": 0.1,
        "mean_pct_error": 0.05,
        "learning_confidence": 0.9,
        "slots_tracked": 96,
        "bias_corrections": 3,
        "pending_predictions": 4,
        "hourly_metrics": {"0": 1.0},
    }
    sensor = LearningMetricsSensor(_hass(), _entry(), {"ml_predictor": predictor})

    assert sensor.native_value == 100
    attrs = sensor.extra_state_attributes
    assert attrs["status"] == "learning"
    assert attrs["message"] == "ok"
    assert attrs["is_learning"] is True
    assert attrs["mae"] == 1.5
    assert attrs["rmse"] == 2.0
    assert attrs["mean_bias"] == 0.1
    assert attrs["mean_pct_error"] == 0.05
    assert attrs["learning_confidence"] == 0.9
    assert attrs["hours_tracked"] == 96
    assert attrs["bias_corrections"] == 3
    assert attrs["pending_predictions"] == 4
    assert attrs["hourly_metrics"] == {"0": 1.0}
    assert predictor.get_learning_metrics.call_count == 1


def test_learning_metrics_no_predictor():
    """No predictor yields None value and empty attributes."""
    sensor = LearningMetricsSensor(_hass(), _entry(), {})
    assert sensor.native_value is None
    assert sensor.extra_state_attributes == {}


def test_learning_metrics_empty_metrics():
    """Empty metrics yield None value and empty attributes."""
    predictor = MagicMock()
    predictor.get_learning_metrics.return_value = {}
    sensor = LearningMetricsSensor(_hass(), _entry(), {"ml_predictor": predictor})
    assert sensor.native_value is None
    assert sensor.extra_state_attributes == {}
