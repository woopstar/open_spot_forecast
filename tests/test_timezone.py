"""Tests for timezone normalization in the forecast pipeline."""

from datetime import UTC, datetime

from custom_components.open_spot_forecast.spot_prices import (
    extract_latest_known_timestamp,
)


def test_extract_latest_known_timestamp_normalizes_aware_to_utc():
    """Aware source timestamps are normalized to UTC, not stripped to naive."""
    raw = [
        {"start": "2026-09-23T22:00:00Z"},
        {"start": "2026-09-23T23:00:00Z"},
    ]

    result = extract_latest_known_timestamp(raw, interval_minutes=15)

    # Latest is 23:00 UTC; the next 15-min interval starts at 23:15 UTC.
    assert result == datetime(2026, 9, 23, 23, 15, tzinfo=UTC)
    assert result.tzinfo is not None


def test_extract_latest_known_timestamp_handles_naive_timestamps():
    """Naive timestamps are treated as local (UTC in the test env)."""
    raw = [{"start": "2026-09-23T23:00:00"}]

    result = extract_latest_known_timestamp(raw, interval_minutes=15)

    assert result == datetime(2026, 9, 23, 23, 15, tzinfo=UTC)
    assert result.tzinfo is not None


def test_extract_latest_known_timestamp_returns_none_when_empty():
    """An empty list yields None."""
    assert extract_latest_known_timestamp([], interval_minutes=15) is None
