"""Tests for the Open Spot Forecast sensor platform."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock

import pytest

from custom_components.open_spot_forecast.accuracy_sensor import (
    LeadTimeAccuracySensor,
)
from custom_components.open_spot_forecast.const import DOMAIN, PRICE_IN
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
        "nordpool": MagicMock(),
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


def test_spot_price_native_value_nordpool():
    """Nordpool fallback converts MWh -> kWh and applies VAT."""
    nordpool = MagicMock()
    nordpool.get_current_price.return_value = 2000.0
    sensor = SpotPriceSensor(
        _hass(),
        _entry(),
        {"nordpool": nordpool},
        "DK1",
        "DKK",
        VAT,
        PRECISION,
        PRICE_TYPE,
    )
    expected = 2000.0 / PRICE_IN[PRICE_TYPE] * (1 + VAT)
    assert sensor.native_value == pytest.approx(expected)


def test_spot_price_native_value_none():
    """No data yields None."""
    sensor = SpotPriceSensor(
        _hass(), _entry(), {}, "DK1", "DKK", VAT, PRECISION, PRICE_TYPE
    )
    assert sensor.native_value is None


def test_spot_price_native_value_nordpool_returns_none():
    """Nordpool present but reporting no price yields None."""
    nordpool = MagicMock()
    nordpool.get_current_price.return_value = None
    sensor = SpotPriceSensor(
        _hass(),
        _entry(),
        {"nordpool": nordpool},
        "DK1",
        "DKK",
        VAT,
        PRECISION,
        PRICE_TYPE,
    )
    assert sensor.native_value is None


def test_spot_price_attributes_nordpool_fallback():
    """Attributes fall back to Nordpool lists when Stromligning is absent."""
    nordpool = MagicMock()
    nordpool.today = [1.0, 2.0]
    nordpool.tomorrow = [3.0, 4.0]
    sensor = SpotPriceSensor(
        _hass(),
        _entry(),
        {"nordpool": nordpool},
        "DK1",
        "DKK",
        VAT,
        PRECISION,
        PRICE_TYPE,
    )

    attrs = sensor.extra_state_attributes

    assert attrs["today_prices"] == [1.0, 2.0]
    assert attrs["tomorrow_prices"] == [3.0, 4.0]
    assert attrs["price_source"] == "nordpool"


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
    (TodayMinSensor, "get_today_stats", "min", "today", _min),
    (TodayMaxSensor, "get_today_stats", "max", "today", _max),
    (TodayMeanSensor, "get_today_stats", "mean", "today", _mean),
    (TomorrowMinSensor, "get_tomorrow_stats", "min", "tomorrow", _min),
    (TomorrowMaxSensor, "get_tomorrow_stats", "max", "tomorrow", _max),
    (TomorrowMeanSensor, "get_tomorrow_stats", "mean", "tomorrow", _mean),
]


@pytest.mark.parametrize(
    "cls, stats_method, stat_key, list_key, reducer", PRICE_STAT_SENSORS
)
def test_price_stat_native_value_stromligning(
    cls, stats_method, stat_key, list_key, reducer
):
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


@pytest.mark.parametrize(
    "cls, stats_method, stat_key, list_key, reducer", PRICE_STAT_SENSORS
)
def test_price_stat_native_value_nordpool(
    cls, stats_method, stat_key, list_key, reducer
):
    """Nordpool stats provide the aggregate with MWh -> kWh + VAT conversion."""
    nordpool = MagicMock()
    getattr(nordpool, stats_method).return_value = {stat_key: 2000.0}
    sensor = cls(
        _hass(), _entry(), {"nordpool": nordpool}, "DKK", VAT, PRECISION, PRICE_TYPE
    )

    expected = 2000.0 / PRICE_IN[PRICE_TYPE] * (1 + VAT)
    assert sensor.native_value == pytest.approx(expected)


@pytest.mark.parametrize(
    "cls, stats_method, stat_key, list_key, reducer", PRICE_STAT_SENSORS
)
def test_price_stat_native_value_none(cls, stats_method, stat_key, list_key, reducer):
    """No data yields None."""
    sensor = cls(_hass(), _entry(), {}, "DKK", VAT, PRECISION, PRICE_TYPE)
    assert sensor.native_value is None


@pytest.mark.parametrize(
    "cls, stats_method, stat_key, list_key, reducer", PRICE_STAT_SENSORS
)
def test_price_stat_native_value_empty_stromligning_falls_back_to_nordpool(
    cls, stats_method, stat_key, list_key, reducer
):
    """An empty Stromligning list falls through to the Nordpool stats."""
    nordpool = MagicMock()
    getattr(nordpool, stats_method).return_value = {stat_key: 4000.0}
    sensor = cls(
        _hass(),
        _entry(),
        {"stromligning_data": {list_key: []}, "nordpool": nordpool},
        "DKK",
        VAT,
        PRECISION,
        PRICE_TYPE,
    )

    expected = 4000.0 / PRICE_IN[PRICE_TYPE] * (1 + VAT)
    assert sensor.native_value == pytest.approx(expected)


@pytest.mark.parametrize(
    "cls, stats_method, stat_key, list_key, reducer", PRICE_STAT_SENSORS
)
def test_price_stat_native_value_nordpool_missing_stat(
    cls, stats_method, stat_key, list_key, reducer
):
    """Nordpool stats without the expected key yield None."""
    nordpool = MagicMock()
    getattr(nordpool, stats_method).return_value = {"unrelated": 1.0}
    sensor = cls(
        _hass(), _entry(), {"nordpool": nordpool}, "DKK", VAT, PRECISION, PRICE_TYPE
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
    future = datetime.now() + timedelta(hours=1)
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


def test_ml_prediction_native_value_falls_back_to_first():
    """When no future prediction exists, the first prediction is used."""
    past = datetime.now() - timedelta(hours=1)
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

    assert sensor.native_value == pytest.approx(80.0 * (1 + VAT))


def test_ml_prediction_native_value_invalid_timestamp_falls_back():
    """An unparseable timestamp is skipped, then first prediction is used."""
    predictor = MagicMock()
    predictor.predictions = [
        {"start": "not-a-date", "end": "x", "price": 60.0, "confidence": 0.6}
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

    assert sensor.native_value == pytest.approx(60.0 * (1 + VAT))


def test_ml_prediction_native_value_missing_start_falls_back():
    """A prediction with no start timestamp is skipped, then first is used."""
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

    assert sensor.native_value == pytest.approx(70.0 * (1 + VAT))


def test_ml_prediction_native_value_price_none():
    """A future prediction with no price yields None."""
    future = datetime.now() + timedelta(hours=1)
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
        _prediction(datetime.now() + timedelta(hours=1), 100.0, 0.8),
        _prediction(datetime.now() + timedelta(hours=2), 200.0, 0.9),
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

    conversion = 1 / PRICE_IN[PRICE_TYPE] * (1 + VAT)
    assert len(attrs["predictions"]) == 2
    assert attrs["predictions"][0]["price"] == round(100.0 * (1 + VAT), PRECISION)
    assert attrs["predictions"][0]["unit"] == "DKK/kWh"
    assert attrs["predictions"][0]["confidence"] == 0.8
    assert attrs["forecast_min"] == round(100.0 * conversion, PRECISION)
    assert attrs["forecast_max"] == round(200.0 * conversion, PRECISION)
    assert attrs["forecast_mean"] == round(150.0 * conversion, PRECISION)
    assert attrs["unit"] == "DKK/kWh"
    assert attrs["mean_confidence"] == 0.85
    assert attrs["total_predictions"] == 2
    assert attrs["is_ml_model"] is True
    assert attrs["training_samples"] == 10


def test_ml_prediction_extra_attributes_empty_stats():
    """Empty stats omit the forecast_* attributes."""
    predictor = MagicMock()
    predictor.predictions = [_prediction(datetime.now() + timedelta(hours=1), 100.0)]
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
        _prediction(datetime.now() + timedelta(hours=1), None),
        _prediction(datetime.now() + timedelta(hours=2), 100.0),
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
