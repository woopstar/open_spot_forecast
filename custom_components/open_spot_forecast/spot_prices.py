"""The raw spot price series behind the ML forecast (#16).

The model trains on, learns from and predicts the day-ahead spot price excl.
VAT and tariffs, in the configured currency per kWh: read by
``SensorReader.read_spot_prices()`` from Stromligning's spot sensors, or
built from stored day-ahead prices (EUR/MWh, #27) by ``dayahead_spot_data()``.
VAT is applied once, in the sensor layer. Stromligning's all-in consumer
price (tariffs, fees and VAT included) is only displayed; the model never
sees it.
"""

from collections.abc import Callable, Iterable
from datetime import date, datetime, timedelta
from typing import Any

from homeassistant.util import dt as dt_util

from .const import PRICE_IN
from .price_series import align_to_grid, is_invalid_price_series
from .time_slots import parse_utc


def extract_latest_known_timestamp(
    raw_data_list: list, interval_minutes: int = 15
) -> datetime | None:
    """Find the end time of the latest known price from raw sensor data.

    Returns the timestamp after the last known interval (UTC-aware), i.e.
    the point from which we should start predicting.
    """
    if not raw_data_list:
        return None

    latest = None
    for item in raw_data_list:
        if not isinstance(item, dict):
            continue
        ts = (
            item.get("timestamp")
            or item.get("time")
            or item.get("start")
            or item.get("end")
        )
        if ts is None:
            continue
        try:
            if isinstance(ts, str):
                dt = dt_util.parse_datetime(ts)
            elif isinstance(ts, datetime):
                dt = ts
            else:
                continue
            if dt is None:
                continue
            # Normalize to UTC (naive timestamps are assumed to be in HA's
            # local time zone) so comparisons are consistent.
            dt = dt_util.as_utc(dt)
            if latest is None or dt > latest:
                latest = dt
        except ValueError, TypeError:
            continue

    if latest is not None:
        # Return the start of the next interval after known data
        return latest + timedelta(minutes=interval_minutes)
    return None


def ml_price_inputs(
    spot_data: dict | None,
) -> tuple[list[float | None], datetime | None]:
    """Return the model's known prices and where they end.

    Args:
        spot_data: ``SensorReader.read_spot_prices()`` result, or None.

    Returns:
        Today's then tomorrow's raw spot prices (one per 15-min slot, None for
        a missing slot), and the end of the last known slot (UTC), from which
        predictions start. ``([], None)`` without spot prices.
    """
    if not spot_data or not spot_data.get("today"):
        return [], None
    prices = list(spot_data["today"]) + list(spot_data.get("tomorrow") or [])
    raw = list(spot_data.get("raw_today") or []) + list(
        spot_data.get("raw_tomorrow") or []
    )
    return prices, extract_latest_known_timestamp(raw)


def _samples_by_day(
    rows: Iterable[dict[str, Any]],
) -> dict[date, list[tuple[datetime, float]]]:
    """Group ``dayahead_prices`` rows (EUR/MWh) by local calendar day."""
    days: dict[date, list[tuple[datetime, float]]] = {}
    for row in rows:
        start = parse_utc(row.get("timestamp"))
        if start is not None and row.get("price") is not None:
            day = dt_util.as_local(start).date()
            days.setdefault(day, []).append((start, float(row["price"])))
    return days


def _day_prices(
    samples: list[tuple[datetime, float]], day: date, rate: float
) -> list[float | None]:
    """Return a day's prices in currency/kWh, or ``[]`` if invalid."""
    prices = align_to_grid(
        ((start, None, price * rate / PRICE_IN["kWh"]) for start, price in samples),
        day,
    )
    return [] if is_invalid_price_series(prices) else prices


def dayahead_prices_by_day(
    rows: Iterable[dict[str, Any]], rate: Callable[[date], float | None]
) -> dict[date, list[float | None]]:
    """Return every local day's spot prices in currency/kWh excl. VAT.

    Args:
        rows: ``dayahead_prices`` rows (``timestamp``, ``price`` in EUR/MWh).
        rate: EUR to the configured currency for a day (None: no rate yet).

    Returns:
        One price per 15-minute slot from local midnight (None if missing)
        by day, for the days with valid prices and a rate.
    """
    days: dict[date, list[float | None]] = {}
    for day, samples in _samples_by_day(rows).items():
        factor = rate(day)
        prices = _day_prices(samples, day, factor) if factor else []
        if prices:
            days[day] = prices
    return days


def dayahead_spot_data(
    rows: Iterable[dict[str, Any]],
    rate: Callable[[date], float | None],
    today: date,
) -> dict[str, list]:
    """Return today's and tomorrow's day-ahead prices as ``read_spot_prices()`` does.

    Args:
        rows: ``dayahead_prices`` rows covering today and tomorrow.
        rate: EUR to the configured currency for a day (None: no rate yet).
        today: Today's local date.

    Returns:
        ``today``/``tomorrow`` (currency/kWh excl. VAT, one per slot) and
        ``raw_today``/``raw_tomorrow`` (``start`` of each priced slot).
    """
    result: dict[str, list] = {
        "today": [],
        "tomorrow": [],
        "raw_today": [],
        "raw_tomorrow": [],
    }
    by_day = _samples_by_day(rows)
    for key, day in (("today", today), ("tomorrow", today + timedelta(days=1))):
        samples = by_day.get(day, [])
        factor = rate(day)
        prices = _day_prices(samples, day, factor) if samples and factor else []
        if prices:
            result[key] = prices
            result[f"raw_{key}"] = [
                {"start": start.isoformat()} for start, _ in samples
            ]
    return result
