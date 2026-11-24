"""Diagnostic sensor: the day-ahead forecast next to the actual prices (#36).

Once a slot's actual price is known its prediction leaves the forecast
sensor. This sensor keeps, for the last ``EVALUATION_WINDOW_HOURS``, the
prediction made closest to ``EVALUATION_LEAD_HOURS`` before each slot next
to the slot's actual price, as compact parallel arrays (see
``forecast_attributes``), so a dashboard can chart predicted against actual.
The predictions kept at the other ``EVALUATION_LEAD_TIMES`` (#113) are further
arrays aligned with them (``t12``, ``t48``; None where there is none).
Its state is the mean absolute error over that window. Prices are converted
like every exposed price (``PriceOutput``: unit, surcharge, VAT), with the
slot's tariff added to both (#107), so the error stays the spot price's.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from homeassistant.components.sensor import SensorEntity, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.util import dt as dt_util, slugify as util_slugify

from .attribution import ModelAttributionMixin
from .const import (
    DOMAIN,
    EVALUATION_LEAD_HOURS,
    EVALUATION_LEAD_TIMES,
    EVALUATION_WINDOW_HOURS,
    UPDATE_SIGNAL,
)
from .price_output import PriceOutput
from .tariffs import TariffSchedule
from .time_slots import SLOT_MINUTES, parse_utc

# The other lead times a prediction is kept for (#113), by attribute name
EXTRA_LEAD_TIME_ARRAYS: dict[str, float] = {
    f"t{target:g}": target
    for target in EVALUATION_LEAD_TIMES
    if abs(target - EVALUATION_LEAD_HOURS) > 1e-9
}


@dataclass(frozen=True, slots=True)
class _Parsed:
    """The kept day-ahead slots with parsed starts, and what they were built from."""

    evaluation: list[dict[str, Any]]
    count: int
    snapshots: dict[float, list[dict[str, Any]]]
    tariffs: TariffSchedule | None
    slots: list[tuple[datetime, dict[str, Any]]]


@dataclass(frozen=True, slots=True)
class _Window:
    """The converted slots of one window, and the values derived from them."""

    parsed: _Parsed
    count: int
    slots: list[tuple[datetime, dict[str, Any]]]
    extra: dict[str, list[float | None]]
    mae: float | None
    bias: float | None


class ForecastEvaluationSensor(ModelAttributionMixin, SensorEntity):
    """Mean absolute error of the day-ahead forecast, with the series behind it."""

    _attr_has_entity_name = True
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_translation_key = "forecast_evaluation"
    _attr_icon = "mdi:chart-bell-curve-cumulative"

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        api_data: dict[str, Any],
        currency: str,
        output: PriceOutput,
    ) -> None:
        """Initialize the sensor."""
        self.hass = hass
        self.api_data = api_data
        self.currency = currency
        self.output = output
        self._attr_unique_id = util_slugify(
            f"{DOMAIN}_{entry.entry_id}_forecast_evaluation"
        )
        self._attr_native_unit_of_measurement = output.unit(currency)
        self._attr_suggested_display_precision = output.precision
        self._attr_device_info = DeviceInfo(identifiers={(DOMAIN, entry.entry_id)})
        # The parsed slots and the converted window (see _window, #295)
        self._parsed_cache: _Parsed | None = None
        self._window_cache: _Window | None = None

    async def async_added_to_hass(self) -> None:
        """Refresh the state after every self-learning update."""
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass, util_slugify(UPDATE_SIGNAL), self._async_refresh
            )
        )

    async def _async_refresh(self) -> None:
        """Build the window off the event loop, then write the state (#295)."""
        await self.hass.async_add_executor_job(self._window)
        self.async_write_ha_state()

    def _parsed(self) -> _Parsed:
        """Return the kept day-ahead slots with their parsed starts.

        Parsed once per ``evaluation`` list, snapshots and tariff schedule:
        learning reloads both series as new objects (``refresh_evaluation``)
        and a price update builds a new schedule, so a new one gives new
        values (#295).
        """
        ml_predictor = self.api_data.get("ml_predictor")
        evaluation = ml_predictor.evaluation if ml_predictor else []
        snapshots = ml_predictor.evaluation_snapshots if ml_predictor else {}
        tariffs = self.api_data.get("tariffs")
        cached = self._parsed_cache
        if (
            cached is not None
            and cached.evaluation is evaluation
            and cached.count == len(evaluation)
            and cached.snapshots is snapshots
            and cached.tariffs is tariffs
        ):
            return cached
        self._parsed_cache = _Parsed(
            evaluation,
            len(evaluation),
            snapshots,
            tariffs,
            [
                (start, row)
                for row in evaluation
                if (start := parse_utc(row.get("start"))) is not None
            ],
        )
        return self._parsed_cache

    def _window(self) -> _Window:
        """Return the evaluated slots of the last window, prices converted.

        Only the window's slots are converted, and only once per window: a
        state write used to convert every kept slot of every lead time
        twice, half a second on a Raspberry Pi (#295). The window's slots
        are those starting at or after ``since``; a later ``since`` keeps a
        subset, so the same number of slots is the same window.
        """
        parsed = self._parsed()
        since = dt_util.utcnow() - timedelta(hours=EVALUATION_WINDOW_HOURS)
        selected = [(start, row) for start, row in parsed.slots if start >= since]
        cached = self._window_cache
        if (
            cached is not None
            and cached.parsed is parsed
            and cached.count == len(selected)
        ):
            return cached
        converted = self.output.evaluation([row for _, row in selected], parsed.tariffs)
        window = [
            (start, row) for (start, _), row in zip(selected, converted, strict=True)
        ]
        errors = [row["predicted"] - row["actual"] for _, row in window]
        self._window_cache = _Window(
            parsed,
            len(selected),
            window,
            self._extra_predictions(window, parsed),
            (
                float(
                    round(
                        sum(abs(error) for error in errors) / len(errors),
                        self.output.precision,
                    )
                )
                if errors
                else None
            ),
            (
                float(round(sum(errors) / len(errors), self.output.precision))
                if errors
                else None
            ),
        )
        return self._window_cache

    @property
    def native_value(self) -> float | None:
        """Return the mean absolute error over the window, None without slots."""
        return self._window().mae

    def _extra_predictions(
        self, window: list[tuple[datetime, dict[str, Any]]], parsed: _Parsed
    ) -> dict[str, list[float | None]]:
        """Return the other lead times' predictions, one per slot of the window."""
        starts = {row["start"] for _, row in window}
        arrays: dict[str, list[float | None]] = {}
        for name, target in EXTRA_LEAD_TIME_ARRAYS.items():
            rows = [
                row
                for row in parsed.snapshots.get(target, [])
                if row.get("start") in starts
            ]
            predicted = {
                row["start"]: row["predicted"]
                for row in self.output.evaluation(rows, parsed.tariffs)
            }
            arrays[name] = [predicted.get(row["start"]) for _, row in window]
        return arrays

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return the predicted and actual series as compact parallel arrays."""
        window = self._window()
        slots = window.slots
        return {
            "interval_minutes": SLOT_MINUTES,
            "unit": self.output.unit(self.currency),
            "lead_hours": EVALUATION_LEAD_HOURS,
            "window_hours": EVALUATION_WINDOW_HOURS,
            "samples": len(slots),
            # Mean signed error: positive = the forecast was too high
            "bias": window.bias,
            "s": [int(start.timestamp()) for start, _ in slots],
            "t": [row["predicted"] for _, row in slots],
            "a": [row["actual"] for _, row in slots],
            **{name: list(values) for name, values in window.extra.items()},
        }
