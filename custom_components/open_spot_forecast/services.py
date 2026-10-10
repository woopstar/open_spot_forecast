"""The ``open_spot_forecast.get_forecast`` (#37) and ``reset_learning`` (#132) actions.

Entity attributes are capped by Home Assistant's 16 KB limit, so the sensor
shows at most 72 hours of the 7-day forecast. ``get_forecast`` returns the
whole forecast, or a window of it, as response data. Prices are converted
exactly as the sensor converts them (``PriceOutput``: tariffs, unit, surcharge,
VAT, hourly mean), so both always agree. With ``raw`` (#142) the prices are the
raw spot price per kWh instead (``PriceOutput.raw_spot()``, as the Predbat
export entities), e.g. for evcc's feed-in tariff.

``reset_learning`` deletes an entry's learning database through
``SpotPricePredictor.reset_learning()`` and refreshes the entities, so the
self-learning starts over without a restart or a manual file deletion.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from typing import Any

import voluptuous as vol

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import (
    HomeAssistant,
    ServiceCall,
    ServiceResponse,
    SupportsResponse,
    callback,
)
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError
from homeassistant.helpers import config_validation as cv, service
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.util import dt as dt_util, slugify as util_slugify

from .const import (
    CONF_INCLUDE_KNOWN_PRICES,
    DEFAULT_INCLUDE_KNOWN_PRICES,
    DOMAIN,
    EVALUATION_LEAD_HOURS,
    EVALUATION_LEAD_TIMES,
    UPDATE_SIGNAL,
)
from .price_output import PriceOutput
from .price_source import PriceSettings
from .spot_prices import known_until, with_known_prices
from .tariffs import TariffSchedule
from .time_slots import floor_to_slot, parse_utc

SERVICE_GET_FORECAST = "get_forecast"
SERVICE_RESET_LEARNING = "reset_learning"
ATTR_CONFIG_ENTRY_ID = "config_entry_id"
ATTR_START = "start"
ATTR_HOURS = "hours"
ATTR_HOURLY = "hourly"
ATTR_INCLUDE_KNOWN = "include_known"
ATTR_EVALUATION = "evaluation"
ATTR_TARGET_HOURS = "target_hours"
ATTR_RAW = "raw"

# The forecast reaches 7 days past the known prices: at most 9 days in all
MAX_FORECAST_HOURS = 9 * 24

GET_FORECAST_SCHEMA = vol.Schema(
    {
        # Optional with a single loaded entry
        vol.Optional(ATTR_CONFIG_ENTRY_ID): cv.string,
        vol.Optional(ATTR_START): cv.datetime,
        vol.Optional(ATTR_HOURS): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=MAX_FORECAST_HOURS)
        ),
        vol.Optional(ATTR_HOURLY): cv.boolean,
        vol.Optional(ATTR_INCLUDE_KNOWN): cv.boolean,
        vol.Optional(ATTR_EVALUATION, default=False): cv.boolean,
        # Which lead time's predictions the evaluation returns (#113)
        vol.Optional(ATTR_TARGET_HOURS, default=EVALUATION_LEAD_HOURS): vol.All(
            vol.Coerce(float), vol.In(EVALUATION_LEAD_TIMES)
        ),
        vol.Optional(ATTR_RAW, default=False): cv.boolean,
    }
)

RESET_LEARNING_SCHEMA = vol.Schema(
    {
        # Optional with a single loaded entry
        vol.Optional(ATTR_CONFIG_ENTRY_ID): cv.string,
    }
)


def forecast_response(
    predictions: list[dict[str, Any]],
    output: PriceOutput,
    currency: str,
    known_until: datetime | None,
    start: datetime,
    hours: int | None = None,
    tariffs: TariffSchedule | None = None,
) -> dict[str, Any]:
    """Build the action's response from the model's predictions.

    Args:
        predictions: The model's predictions (raw spot prices).
        output: How prices are exposed (the entry's, or with ``hourly`` set).
        currency: Currency of the prices.
        known_until: End of the last confirmed price slot, if any.
        start: First interval to return: the one containing this moment.
        hours: Length of the window in hours from that interval; all if None.
        tariffs: Each slot's tariff, added to its price (#107).

    Returns:
        ``known_until`` (ISO, local time, or None), ``unit``,
        ``interval_minutes`` and ``forecast``: ``{"start", "end", "price",
        "confidence"}`` entries in time order.
    """
    first = floor_to_slot(start, output.interval_minutes)
    last = first + timedelta(hours=hours) if hours is not None else None
    forecast = []
    for entry in output.forecast(predictions, tariffs):
        # Entries start on interval boundaries (slots, or hours if hourly)
        entry_start = parse_utc(entry["start"])
        if entry_start is None or entry_start < first:
            continue
        if last is not None and entry_start >= last:
            break
        forecast.append(entry)
    return {
        "known_until": (
            dt_util.as_local(known_until).isoformat() if known_until else None
        ),
        "unit": output.unit(currency),
        "interval_minutes": output.interval_minutes,
        "forecast": forecast,
    }


def _ml_predictor(hass: HomeAssistant, entry: ConfigEntry) -> Any:
    """Return the entry's predictor.

    Raises:
        ServiceValidationError: ML predictions are disabled for the entry.
    """
    ml_predictor = hass.data[DOMAIN][entry.entry_id].get("ml_predictor")
    if ml_predictor is None:
        raise ServiceValidationError(
            translation_domain=DOMAIN,
            translation_key="ml_prediction_disabled",
            translation_placeholders={"entry_title": entry.title},
        )
    return ml_predictor


async def _async_get_forecast(call: ServiceCall) -> ServiceResponse:
    """Return an entry's forecast (see ``forecast_response``)."""
    hass = call.hass
    entry = service.async_get_config_entry(
        hass, DOMAIN, call.data.get(ATTR_CONFIG_ENTRY_ID)
    )
    api_data = hass.data[DOMAIN][entry.entry_id]
    ml_predictor = _ml_predictor(hass, entry)
    settings = PriceSettings.from_entry(entry)
    output = settings.output
    tariffs = api_data.get("tariffs")
    if call.data[ATTR_RAW]:
        # The raw spot price per kWh, no tariff, surcharge or VAT (#142)
        output = output.raw_spot()
        tariffs = None
    if ATTR_HOURLY in call.data:
        output = replace(output, hourly_average=call.data[ATTR_HOURLY])
    start = call.data.get(ATTR_START)
    start = dt_util.as_utc(start) if start is not None else dt_util.utcnow()
    spot_data = api_data.get("spot_data")
    predictions = ml_predictor.predictions
    include_known = call.data.get(
        ATTR_INCLUDE_KNOWN,
        entry.options.get(CONF_INCLUDE_KNOWN_PRICES, DEFAULT_INCLUDE_KNOWN_PRICES),
    )
    if include_known:
        # Confirmed prices from start, then the predictions (#40)
        predictions = with_known_prices(spot_data, predictions, start)
    response = forecast_response(
        predictions,
        output,
        settings.currency,
        known_until(spot_data),
        start,
        call.data.get(ATTR_HOURS),
        tariffs,
    )
    if call.data[ATTR_EVALUATION]:
        # Every kept slot's day-ahead prediction (or the one kept at another
        # lead time, #113) next to its actual price (#36), per slot; the raw
        # spot price too with ``raw``
        target = call.data[ATTR_TARGET_HOURS]
        rows = (
            ml_predictor.evaluation
            if abs(target - EVALUATION_LEAD_HOURS) < 1e-9
            else ml_predictor.evaluation_snapshots.get(target, [])
        )
        response["evaluation"] = output.evaluation(rows, tariffs)
    return response


async def _async_reset_learning(call: ServiceCall) -> None:
    """Delete an entry's learning database and refresh its entities.

    Raises:
        HomeAssistantError: The database could not be cleared.
    """
    hass = call.hass
    entry = service.async_get_config_entry(
        hass, DOMAIN, call.data.get(ATTR_CONFIG_ENTRY_ID)
    )
    ml_predictor = _ml_predictor(hass, entry)
    if not await ml_predictor.reset_learning():
        raise HomeAssistantError(
            translation_domain=DOMAIN,
            translation_key="reset_learning_failed",
            translation_placeholders={"entry_title": entry.title},
        )
    # The learning and accuracy entities read the predictor's state directly
    async_dispatcher_send(hass, util_slugify(UPDATE_SIGNAL))


@callback
def async_setup_services(hass: HomeAssistant) -> None:
    """Register the integration's actions (once, not per config entry)."""
    hass.services.async_register(
        DOMAIN,
        SERVICE_GET_FORECAST,
        _async_get_forecast,
        schema=GET_FORECAST_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_RESET_LEARNING,
        _async_reset_learning,
        schema=RESET_LEARNING_SCHEMA,
    )
