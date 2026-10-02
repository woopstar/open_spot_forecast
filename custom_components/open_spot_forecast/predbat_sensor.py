"""Predbat-compatible rate entities (#124).

`Predbat <https://springfall2008.github.io/batpred/>`_ reads Danish rates
from the Stromligning integration's entities: the "today" entity's
``prices_today`` attribute and the "tomorrow" entity's ``prices_tomorrow``,
each a list of ``{"start", "end", "price"}``. It multiplies the prices by
100 (to øre) only if the entity's unit contains ``kr/``, and keeps every
item up to ``forecast_days + 1`` days ahead, so the tomorrow entity may
carry several days. Stromligning's entities hold two days at most; OSF's
forecast reaches 7 days.

With the option on, an entry has four such sensors, one per ``apps.yaml``
key. The import price is what the user pays: the confirmed prices, then
the forecast, with each slot's tariff (``TariffSchedule``, or the fixed
tariff option without one), the surcharge and VAT, always per kWh. The
export price is the raw spot price excl. VAT: the confirmed spot prices,
then the model's raw output. Today runs from local midnight; tomorrow
holds tomorrow and the following days, trimmed to the attribute budget.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from homeassistant.components.sensor import SensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.util import dt as dt_util, slugify as util_slugify

from .attribution import ModelAttributionMixin
from .const import (
    CONF_PREDBAT_TARIFF,
    DEFAULT_PREDBAT_TARIFF,
    DOMAIN,
    PREDBAT_KRONE_CURRENCIES,
    PREDBAT_UNIT_CENT,
    PREDBAT_UNIT_KRONE,
    UPDATE_SIGNAL,
    UPDATE_SIGNAL_FORECAST,
)
from .forecast_attributes import fit_entries
from .price_output import PriceOutput
from .spot_prices import known_until, with_known_prices
from .tariffs import TariffSchedule
from .time_slots import local_midnight, parse_utc

KIND_IMPORT = "import"
KIND_EXPORT = "export"
DAY_TODAY = "today"
DAY_TOMORROW = "tomorrow"


def predbat_unit(currency: str) -> tuple[str, float]:
    """Return the unit of the Predbat entities and the factor from currency/kWh.

    Predbat wants øre or cents per kWh. A krone currency is exposed as is
    with ``kr/kWh``, which Predbat scales by 100; any other currency is
    exposed in cents with a unit Predbat does not scale.

    Args:
        currency: The entry's currency code.

    Returns:
        The unit string and the factor applied to each price.
    """
    if currency in PREDBAT_KRONE_CURRENCIES:
        return PREDBAT_UNIT_KRONE, 1.0
    return PREDBAT_UNIT_CENT, 100.0


def current_entry(
    entries: list[dict[str, Any]], now: datetime
) -> dict[str, Any] | None:
    """Return the entry whose interval contains ``now``, or None."""
    for entry in entries:
        start, end = parse_utc(entry["start"]), parse_utc(entry["end"])
        if start is not None and end is not None and start <= now < end:
            return entry
    return None


def build_predbat_sensors(
    hass: HomeAssistant,
    entry: ConfigEntry,
    api_data: dict[str, Any],
    currency: str,
    output: PriceOutput,
) -> list[PredbatRateSensor]:
    """Return the four Predbat rate sensors of a config entry."""
    fixed_tariff = float(entry.options.get(CONF_PREDBAT_TARIFF, DEFAULT_PREDBAT_TARIFF))
    return [
        PredbatRateSensor(
            hass, entry, api_data, currency, output, kind, day, fixed_tariff
        )
        for kind in (KIND_IMPORT, KIND_EXPORT)
        for day in (DAY_TODAY, DAY_TOMORROW)
    ]


class PredbatRateSensor(ModelAttributionMixin, SensorEntity):
    """One of Predbat's import/export, today/tomorrow rate entities."""

    _attr_has_entity_name = True
    # The series are live data for Predbat, too large to record (#103)
    _unrecorded_attributes = frozenset({"prices_today", "prices_tomorrow"})

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        api_data: dict[str, Any],
        currency: str,
        output: PriceOutput,
        kind: str,
        day: str,
        fixed_tariff: float = DEFAULT_PREDBAT_TARIFF,
    ) -> None:
        """Initialize the sensor.

        Args:
            hass: Home Assistant.
            entry: The config entry.
            api_data: The entry's integration data.
            currency: The entry's currency code.
            output: How the entry exposes prices (unit, surcharge, VAT, ...).
            kind: ``import`` (consumer price) or ``export`` (raw spot price).
            day: ``today`` or ``tomorrow`` (and the following days).
            fixed_tariff: Tariff per kWh excl. VAT added to the import
                price when no slot tariff is known.
        """
        self.hass = hass
        self.api_data = api_data
        self.kind = kind
        self.day = day
        self._fixed_tariff = fixed_tariff
        # Import: per kWh with the surcharge and VAT; export: the raw spot price
        self._output = output.per_kwh() if kind == KIND_IMPORT else output.raw_spot()
        self._attr_native_unit_of_measurement, self._factor = predbat_unit(currency)
        self._attr_translation_key = f"predbat_{kind}_{day}"
        self._attr_unique_id = util_slugify(
            f"{DOMAIN}_{entry.entry_id}_predbat_{kind}_{day}"
        )
        self._attr_icon = (
            "mdi:transmission-tower-import"
            if kind == KIND_IMPORT
            else "mdi:transmission-tower-export"
        )
        self._attr_suggested_display_precision = output.precision
        self._attr_device_info = {"identifiers": {(DOMAIN, entry.entry_id)}}

    async def async_added_to_hass(self) -> None:
        """Refresh after every price update (the state) and every forecast."""
        for signal in (UPDATE_SIGNAL, UPDATE_SIGNAL_FORECAST):
            self.async_on_remove(
                async_dispatcher_connect(
                    self.hass, util_slugify(signal), self.async_write_ha_state
                )
            )

    def _tariffs(self) -> TariffSchedule | None:
        """Return the import price's tariffs: the slot schedule, else the fixed one."""
        if self.kind != KIND_IMPORT:
            return None
        tariffs: TariffSchedule | None = self.api_data.get("tariffs")
        if tariffs:
            return tariffs
        if abs(self._fixed_tariff) > 1e-9:
            return TariffSchedule.fixed(self._fixed_tariff)
        return None

    def _series(self, now: datetime) -> list[dict[str, Any]]:
        """Return this sensor's days: confirmed prices, then the forecast.

        Args:
            now: The current time, timezone-aware.

        Returns:
            ``{"start", "end", "price"}`` per interval in local ISO time,
            from today's local midnight (``today``) or from tomorrow's
            (``tomorrow``, open-ended), prices in the entity's unit.
        """
        ml_predictor = self.api_data.get("ml_predictor")
        predictions = ml_predictor.predictions if ml_predictor else None
        today = dt_util.as_local(now).date()
        midnight = local_midnight(today)
        tomorrow = local_midnight(today + timedelta(days=1))
        series = with_known_prices(
            self.api_data.get("spot_data"), predictions or [], midnight
        )
        entries = []
        for entry in self._output.forecast(series, self._tariffs()):
            start = parse_utc(entry["start"])
            if start is None:
                continue
            if (start < tomorrow) != (self.day == DAY_TODAY):
                continue
            entries.append(
                {
                    "start": entry["start"],
                    "end": entry["end"],
                    "price": float(
                        round(entry["price"] * self._factor, self._output.precision)
                    ),
                }
            )
        return entries

    @property
    def native_value(self) -> float | None:
        """Return the current interval's price (today), or tomorrow's first."""
        now = dt_util.utcnow()
        entries = self._series(now)
        if self.day == DAY_TODAY:
            entry = current_entry(entries, now)
            return entry["price"] if entry else None
        return entries[0]["price"] if entries else None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return the series Predbat reads, within the attribute budget."""
        key = f"prices_{self.day}"
        known_end = known_until(self.api_data.get("spot_data"))
        attrs: dict[str, Any] = {
            key: self._series(dt_util.utcnow()),
            "interval_minutes": self._output.interval_minutes,
            "known_until": (
                dt_util.as_local(known_end).isoformat() if known_end else None
            ),
            "includes_vat": self.kind == KIND_IMPORT,
            "includes_tariffs": bool(self._tariffs()),
        }
        fit_entries(attrs, key, self._output.interval_minutes)
        return attrs
