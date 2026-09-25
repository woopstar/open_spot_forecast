"""The raw spot price series behind the ML forecast (#16).

The model trains on, learns from and predicts the day-ahead spot price excl.
VAT and tariffs, read by ``SensorReader.read_spot_prices()``. VAT is applied
once, in the sensor layer. Stromligning's all-in consumer price (tariffs, fees
and VAT included) is only displayed; the model never sees it.
"""

from datetime import datetime, timedelta

from homeassistant.util import dt as dt_util


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
