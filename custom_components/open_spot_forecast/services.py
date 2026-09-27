"""The ``open_spot_forecast.get_forecast`` action (#37).

Entity attributes are capped by Home Assistant's 16 KB limit, so the sensor
shows at most 72 hours of the 7-day forecast. The action returns the whole
forecast, or a window of it, as response data. Prices are converted exactly
as the sensor converts them (``PriceOutput``: unit, surcharge, VAT, hourly
mean), so both always agree.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from typing import Any

import voluptuous as vol

from homeassistant.core import (
    HomeAssistant,
    ServiceCall,
    ServiceResponse,
    SupportsResponse,
    callback,
)
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers import config_validation as cv, service
from homeassistant.util import dt as dt_util

from .const import CONF_INCLUDE_KNOWN_PRICES, DEFAULT_INCLUDE_KNOWN_PRICES, DOMAIN
from .price_output import PriceOutput
from .price_source import PriceSettings
from .spot_prices import known_until, with_known_prices
from .time_slots import floor_to_slot, parse_utc

SERVICE_GET_FORECAST = "get_forecast"
ATTR_CONFIG_ENTRY_ID = "config_entry_id"
ATTR_START = "start"
ATTR_HOURS = "hours"
ATTR_HOURLY = "hourly"
ATTR_INCLUDE_KNOWN = "include_known"

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
    }
)


def forecast_response(
    predictions: list[dict[str, Any]],
    output: PriceOutput,
    currency: str,
    known_until: datetime | None,
    start: datetime,
    hours: int | None = None,
) -> dict[str, Any]:
    """Build the action's response from the model's predictions.

    Args:
        predictions: The model's predictions (raw spot prices).
        output: How prices are exposed (the entry's, or with ``hourly`` set).
        currency: Currency of the prices.
        known_until: End of the last confirmed price slot, if any.
        start: First interval to return: the one containing this moment.
        hours: Length of the window in hours from that interval; all if None.

    Returns:
        ``known_until`` (ISO, local time, or None), ``unit``,
        ``interval_minutes`` and ``forecast``: ``{"start", "end", "price",
        "confidence"}`` entries in time order.
    """
    first = floor_to_slot(start, output.interval_minutes)
    last = first + timedelta(hours=hours) if hours is not None else None
    forecast = []
    for entry in output.forecast(predictions):
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


async def _async_get_forecast(call: ServiceCall) -> ServiceResponse:
    """Return an entry's forecast (see ``forecast_response``)."""
    hass = call.hass
    entry = service.async_get_config_entry(
        hass, DOMAIN, call.data.get(ATTR_CONFIG_ENTRY_ID)
    )
    api_data = hass.data[DOMAIN][entry.entry_id]
    ml_predictor = api_data.get("ml_predictor")
    if ml_predictor is None:
        raise ServiceValidationError(
            translation_domain=DOMAIN,
            translation_key="ml_prediction_disabled",
            translation_placeholders={"entry_title": entry.title},
        )
    settings = PriceSettings.from_entry(entry)
    output = settings.output
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
    return forecast_response(
        predictions,
        output,
        settings.currency,
        known_until(spot_data),
        start,
        call.data.get(ATTR_HOURS),
    )


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
