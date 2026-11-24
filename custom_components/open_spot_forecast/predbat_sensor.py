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

import asyncio
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from homeassistant.components.sensor import SensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
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


def build_predbat_sensors(
    hass: HomeAssistant,
    entry: ConfigEntry,
    api_data: dict[str, Any],
    currency: str,
    output: PriceOutput,
) -> list[PredbatRateSensor]:
    """Return the four Predbat rate sensors of a config entry.

    The today and tomorrow sensor of a kind share one ``PredbatSeries``,
    so a price or forecast update builds each kind's series once (#295).
    """
    fixed_tariff = float(entry.options.get(CONF_PREDBAT_TARIFF, DEFAULT_PREDBAT_TARIFF))
    sensors: list[PredbatRateSensor] = []
    for kind in (KIND_IMPORT, KIND_EXPORT):
        series = PredbatSeries(api_data, currency, output, kind, fixed_tariff)
        sensors.extend(
            PredbatRateSensor(
                hass, entry, api_data, currency, output, kind, day, fixed_tariff, series
            )
            for day in (DAY_TODAY, DAY_TOMORROW)
        )
    return sensors


# One exposed interval: its UTC start and end, and the ``{"start", "end",
# "price"}`` entry Predbat reads
type Rate = tuple[datetime, datetime | None, dict[str, Any]]


@dataclass(frozen=True, slots=True)
class RateDays:
    """One build of a kind's series: today's rates and the following days'."""

    key: tuple[Any, ...]
    today: list[Rate]
    tomorrow: list[Rate]
    known_until: str | None
    includes_tariffs: bool


class PredbatSeries:
    """The confirmed prices, then the forecast, of one Predbat rate kind.

    Building the series converts the confirmed prices and the whole 7-day
    forecast; doing it in every read of every sensor blocked the event loop
    for over half a second on a Raspberry Pi (#295, like #179). It is built
    once per forecast (the ``predictions`` list and its length), spot price
    read (``spot_data``), tariff schedule and local day: the updater
    replaces each of them with a new object, so a new one gives new values.
    The today and tomorrow sensor of a kind share one instance.
    """

    def __init__(
        self,
        api_data: dict[str, Any],
        currency: str,
        output: PriceOutput,
        kind: str,
        fixed_tariff: float = DEFAULT_PREDBAT_TARIFF,
    ) -> None:
        """Initialize the series.

        Args:
            api_data: The entry's integration data.
            currency: The entry's currency code.
            output: How the entry exposes prices (unit, surcharge, VAT, ...).
            kind: ``import`` (consumer price) or ``export`` (raw spot price).
            fixed_tariff: Tariff per kWh excl. VAT added to the import
                price when no slot tariff is known.
        """
        self.api_data = api_data
        self.kind = kind
        # Import: per kWh with the surcharge and VAT; export: the raw spot price
        self.output = output.per_kwh() if kind == KIND_IMPORT else output.raw_spot()
        self.factor = predbat_unit(currency)[1]
        self._fixed = (
            TariffSchedule.fixed(fixed_tariff)
            if kind == KIND_IMPORT and abs(fixed_tariff) > 1e-9
            else None
        )
        self._days: RateDays | None = None
        # One build at a time: the other day's sensor then finds it built
        self._lock = asyncio.Lock()

    def tariffs(self) -> TariffSchedule | None:
        """Return the import price's tariffs: the slot schedule, else the fixed one."""
        if self.kind != KIND_IMPORT:
            return None
        tariffs: TariffSchedule | None = self.api_data.get("tariffs")
        if tariffs:
            return tariffs
        return self._fixed

    async def async_build(self, hass: HomeAssistant) -> None:
        """Build the series in the executor, so the state write finds it built.

        Args:
            hass: Home Assistant.
        """
        async with self._lock:
            await hass.async_add_executor_job(self.days, dt_util.utcnow())

    def days(self, now: datetime) -> RateDays:
        """Return the series split at local midnight, built once per input.

        Args:
            now: The current time, timezone-aware.

        Returns:
            Today's rates from local midnight and the rates from tomorrow's
            local midnight on (open-ended), prices in the entity's unit,
            with ``known_until`` and whether the prices include tariffs.
        """
        ml_predictor = self.api_data.get("ml_predictor")
        predictions = ml_predictor.predictions if ml_predictor else None
        spot_data = self.api_data.get("spot_data")
        tariffs = self.tariffs()
        today = dt_util.as_local(now).date()
        key = (
            predictions,
            len(predictions or ()),
            spot_data,
            tariffs,
            today,
        )
        cached = self._days
        if cached is not None and _same_key(cached.key, key):
            return cached
        self._days = self._build(key, predictions or [], spot_data, tariffs, today)
        return self._days

    def _build(
        self,
        key: tuple[Any, ...],
        predictions: list[dict[str, Any]],
        spot_data: dict[str, Any] | None,
        tariffs: TariffSchedule | None,
        today: date,
    ) -> RateDays:
        """Convert the confirmed prices and the forecast from local midnight."""
        midnight = local_midnight(today)
        tomorrow = local_midnight(today + timedelta(days=1))
        series = with_known_prices(spot_data, predictions, midnight)
        days: dict[bool, list[Rate]] = {True: [], False: []}
        for entry in self.output.forecast(series, tariffs):
            start = parse_utc(entry["start"])
            if start is None:
                continue
            is_today = start < tomorrow
            days[is_today].append(
                (
                    start,
                    # Only today's state looks for the interval containing now
                    parse_utc(entry["end"]) if is_today else None,
                    {
                        "start": entry["start"],
                        "end": entry["end"],
                        "price": float(
                            round(entry["price"] * self.factor, self.output.precision)
                        ),
                    },
                )
            )
        known_end = known_until(spot_data)
        return RateDays(
            key=key,
            today=days[True],
            tomorrow=days[False],
            known_until=dt_util.as_local(known_end).isoformat() if known_end else None,
            includes_tariffs=bool(tariffs),
        )


def _same_key(cached: tuple[Any, ...], key: tuple[Any, ...]) -> bool:
    """Return whether two series keys name the same inputs.

    The objects are compared by identity (a new forecast, read or schedule
    is a new object, and an equal one is not worth comparing slot by slot);
    the predictions' length and the day by value.
    """
    return (
        cached[0] is key[0]
        and cached[1] == key[1]
        and cached[2] is key[2]
        and cached[3] is key[3]
        and cached[4] == key[4]
    )


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
        series: PredbatSeries | None = None,
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
            series: The kind's series, shared with the other day's sensor;
                a new one if None.
        """
        self.hass = hass
        self.api_data = api_data
        self.kind = kind
        self.day = day
        self._fixed_tariff = fixed_tariff
        self._series = series or PredbatSeries(
            api_data, currency, output, kind, fixed_tariff
        )
        self._output = self._series.output
        # The attributes of the last build of the series (see _attributes)
        self._attrs_cache: tuple[RateDays, dict[str, Any]] | None = None
        self._attr_native_unit_of_measurement = predbat_unit(currency)[0]
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
        self._attr_device_info = DeviceInfo(identifiers={(DOMAIN, entry.entry_id)})

    async def async_added_to_hass(self) -> None:
        """Refresh after every price update (the state) and every forecast."""
        for signal in (UPDATE_SIGNAL, UPDATE_SIGNAL_FORECAST):
            self.async_on_remove(
                async_dispatcher_connect(
                    self.hass, util_slugify(signal), self._async_refresh
                )
            )

    async def _async_refresh(self) -> None:
        """Build the series off the event loop, then write the state (#295)."""
        await self._series.async_build(self.hass)
        self.async_write_ha_state()

    def _rates(self, days: RateDays) -> list[Rate]:
        """Return this sensor's day of ``days``."""
        return days.today if self.day == DAY_TODAY else days.tomorrow

    @property
    def native_value(self) -> float | None:
        """Return the current interval's price (today), or tomorrow's first."""
        now = dt_util.utcnow()
        rates = self._rates(self._series.days(now))
        if self.day == DAY_TODAY:
            return next(
                (
                    entry["price"]
                    for start, end, entry in rates
                    if end is not None and start <= now < end
                ),
                None,
            )
        return rates[0][2]["price"] if rates else None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return the series Predbat reads, within the attribute budget.

        Built and trimmed once per build of the series (#295).
        """
        days = self._series.days(dt_util.utcnow())
        cached = self._attrs_cache
        if cached is None or cached[0] is not days:
            key = f"prices_{self.day}"
            attrs: dict[str, Any] = {
                key: [entry for _, _, entry in self._rates(days)],
                "interval_minutes": self._output.interval_minutes,
                "known_until": days.known_until,
                "includes_vat": self.kind == KIND_IMPORT,
                "includes_tariffs": days.includes_tariffs,
            }
            fit_entries(attrs, key, self._output.interval_minutes)
            cached = self._attrs_cache = (days, attrs)
        attrs = cached[1]
        key = f"prices_{self.day}"
        # A copy, so a caller changing it cannot change the cache
        return {**attrs, key: list(attrs[key])}
