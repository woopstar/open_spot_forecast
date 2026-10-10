"""Read the configured external price forecasts and hand them to the model (#120).

A forecast run records, next to the model's own predictions, what each
configured sensor (Stromligning's forecast sensor, Energi Data Service's
forecast attribute) shows for the slots the model predicts, so self-learning
can score every source per lead time (``ml/external_forecasts.py``).

**Unit.** An external forecast is read as its sensor shows it and is taken
to be in the terms of this entry's own prices, as it would be on a shared
chart: the configured unit, with the slot's tariff, the surcharge and VAT.
It is converted back to the raw spot price in currency/kWh with the inverse
of ``PriceOutput`` and the slot's tariff, the exact reverse of how the
model's forecast is exposed, so every source's errors are in the model's
unit. A source in other terms (e.g. without VAT while this entry adds it)
shows as a constant bias in its accuracy.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from .ml.predictor import SpotPricePredictor
from .price_output import PriceOutput
from .sensor_reader import SensorReader
from .tariffs import TariffSchedule
from .time_slots import first_prediction_slot, utc_slot_key

_LOGGER = logging.getLogger(__name__)


def external_spot_forecast(
    forecast: Sequence[tuple[datetime, float]],
    output: PriceOutput,
    tariffs: TariffSchedule | None,
    first: datetime,
) -> list[tuple[str, float]]:
    """Return an external forecast as raw spot prices for the slots from ``first``.

    Args:
        forecast: ``(slot start, price as shown)`` per 15-minute slot
            (``SensorReader.read_external_forecast``).
        output: How this entry exposes prices; its inverse is applied.
        tariffs: Each slot's tariff, subtracted (#107); None without one.
        first: The first slot the model predicts: earlier slots have a
            confirmed price, so what a source shows for them is no forecast.

    Returns:
        ``(slot's UTC key, raw spot price in currency/kWh)`` rows.
    """
    return [
        (
            utc_slot_key(start),
            output.to_spot(price) - (tariffs.at(start) if tariffs else 0.0),
        )
        for start, price in forecast
        if start >= first
    ]


async def async_record_external_forecasts(
    hass: HomeAssistant,
    sensor_reader: SensorReader,
    ml_predictor: SpotPricePredictor,
    entity_ids: Sequence[str],
    output: PriceOutput,
    tariffs: TariffSchedule | None,
    known_data_end_time: datetime | None,
) -> None:
    """Store what the configured sensors forecast for the slots the model predicts.

    Each source is stored under its entity id. Nothing is read or stored
    without a configured sensor.

    Args:
        hass: Home Assistant instance.
        sensor_reader: Reads the sensors.
        ml_predictor: The model, whose database keeps the forecasts.
        entity_ids: The configured external forecast sensors.
        output: How this entry exposes prices.
        tariffs: Each slot's tariff, if any.
        known_data_end_time: End of the confirmed prices, where the model's
            predictions start.
    """
    if not entity_ids:
        return
    first = first_prediction_slot(dt_util.utcnow(), known_data_end_time)
    forecasts = {
        entity_id: rows
        for entity_id in entity_ids
        if (
            rows := external_spot_forecast(
                sensor_reader.read_external_forecast(entity_id), output, tariffs, first
            )
        )
    }
    _LOGGER.debug(
        "Recording external forecasts: %s",
        {entity_id: len(rows) for entity_id, rows in forecasts.items()},
    )
    if forecasts:
        await hass.async_add_executor_job(
            ml_predictor.store_external_forecasts, forecasts
        )
