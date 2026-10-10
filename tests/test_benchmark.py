"""Tests for the dev-only external-forecast benchmark (scripts/benchmark.py, #115).

No network: payloads have the shape recorded from each source on 2026-10-10
with made-up prices (Smartere Elforbrug's forecasts may not be redistributed),
and ``urlopen`` is replaced where a request is made.
"""

import io
import json
import math
import sqlite3
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Mapping
from datetime import UTC, date, datetime, time, timedelta, timezone
from email.message import Message
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from scripts import benchmark, benchmark_sources as sources
from scripts.backtest import (
    CT_PER_KWH_PER_EUR_PER_MWH,
    SLOT_SECONDS,
    HorizonScore,
    PriceSeries,
    local_midnight,
)
from scripts.benchmark import (
    NAIVE,
    CollectOptions,
    LeadScore,
    ReportConfig,
    cheapest_window,
    collect,
    cutoff_seconds,
    day_hours,
    forecast_hourly,
    format_report,
    hourly_means,
    rank_correlation,
    report_units,
    run_benchmark,
    select_origin,
)
from scripts.benchmark_sources import (
    BenchmarkStore,
    Forecast,
    SourceUnavailable,
    backfill_smartere,
    collect_carnot,
    collect_epex,
    collect_osf,
    collect_smartere,
    eur_per_unit,
    load_osf_export,
    parse_carnot,
    parse_epex,
    parse_osf,
    parse_smartere,
    smartere_commits,
)

CPH = ZoneInfo("Europe/Copenhagen")
DKK_PER_EUR = 7.46038
NOW = datetime(2026, 10, 10, 9, 30, tzinfo=CPH)

# The shape of prognose.json: local creation time, UTC hours, EUR/MWh
SMARTERE: dict[str, Any] = {
    "Forecast created": "2026-10-10T14:43:06",
    "DK1": [
        {"Time": "2026-10-10T13:00:00", "Price": 100.0},
        {"Time": "2026-10-10T14:00:00", "Price": 120.0},
    ],
    "DK2": [{"Time": "2026-10-10T13:00:00", "Price": 90.0}],
}
# The shape of Carnot's get_predict answer: naive UTC, DKK/MWh
CARNOT: dict[str, Any] = {
    "predictions": [
        {"utctime": "2026-10-11T00:00:00", "prediction": 746.038},
        {"utctime": "2026-10-11T01:00:00", "prediction": -74.6038},
    ]
}
# The shape of EpexPredictor's /prices answer with unit=EUR_PER_MWH
EPEX: dict[str, Any] = {
    "prices": [
        {"startsAt": "2026-10-10T16:15:00Z", "total": 100.0},
        {"startsAt": "2026-10-10T16:30:00Z", "total": 110.5},
    ],
    "knownUntil": "2026-10-11T22:00:00Z",
}
# The shape of Home Assistant's answer to get_forecast?return_response (raw)
OSF: dict[str, Any] = {
    "changed_states": [],
    "service_response": {
        "known_until": "2026-10-12T00:00:00+02:00",
        "unit": "DKK/kWh",
        "interval_minutes": 15,
        "forecast": [
            {
                "start": "2026-10-12T00:00:00+02:00",
                "end": "2026-10-12T00:15:00+02:00",
                "price": 0.746038,
                "confidence": 0.8,
            },
            {
                "start": "2026-10-12T00:15:00+02:00",
                "end": "2026-10-12T00:30:00+02:00",
                "price": -0.0746038,
                "confidence": 0.8,
            },
        ],
    },
}


def _ts(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    second: int = 0,
    tz: ZoneInfo | timezone = UTC,
) -> int:
    return int(datetime(year, month, day, hour, minute, second, tzinfo=tz).timestamp())


@pytest.fixture
def store(tmp_path: Path) -> Iterator[BenchmarkStore]:
    store = BenchmarkStore(tmp_path / "benchmark" / "DK1.db")
    yield store
    store.close()


# --- Parsers ----------------------------------------------------------------------


def test_smartere_times_are_utc_and_its_origin_is_danish_local_time() -> None:
    forecast = parse_smartere(SMARTERE, "DK1")

    assert forecast.source == "smartere"
    # 14:43:06 in Copenhagen (CEST) is 12:43:06 UTC
    assert forecast.origin == _ts(2026, 10, 10, 12, 43, 6)
    assert forecast.points == (
        (_ts(2026, 10, 10, 13), pytest.approx(100.0)),
        (_ts(2026, 10, 10, 14), pytest.approx(120.0)),
    )
    assert parse_smartere(SMARTERE, "DK2").points == (
        (_ts(2026, 10, 10, 13), pytest.approx(90.0)),
    )


def test_smartere_falls_back_to_the_commit_time() -> None:
    payload = {
        key: value for key, value in SMARTERE.items() if key != "Forecast created"
    }

    assert parse_smartere(payload, "DK1", committed=1234).origin == 1234
    with pytest.raises(ValueError, match="Forecast created"):
        parse_smartere(payload, "DK1")
    with pytest.raises(ValueError, match="no DK1 prices"):
        parse_smartere({"Forecast created": "2026-10-10T14:43:06"}, "DK1")


def test_carnot_is_converted_from_dkk_per_mwh() -> None:
    forecast = parse_carnot(CARNOT, origin=99)

    assert (forecast.source, forecast.origin) == ("carnot", 99)
    assert forecast.points == (
        (_ts(2026, 10, 11, 0), pytest.approx(100.0)),
        (_ts(2026, 10, 11, 1), pytest.approx(-10.0)),
    )
    with pytest.raises(ValueError, match="no predictions"):
        parse_carnot({"predictions": []}, 1)


def test_epex_prices_keep_their_quarter_hours() -> None:
    forecast = parse_epex(EPEX, origin=7)

    assert forecast.points == (
        (_ts(2026, 10, 10, 16, 15), pytest.approx(100.0)),
        (_ts(2026, 10, 10, 16, 30), pytest.approx(110.5)),
    )
    with pytest.raises(ValueError, match="no prices"):
        parse_epex({}, 1)


def test_osf_raw_spot_per_kwh_becomes_eur_per_mwh() -> None:
    forecast = parse_osf(OSF, origin=5)

    assert forecast.source == "osf"
    assert forecast.points == (
        (_ts(2026, 10, 11, 22, 0), pytest.approx(100.0)),
        (_ts(2026, 10, 11, 22, 15), pytest.approx(-10.0)),
    )
    # The action's own response (no REST wrapper), in EUR
    eur = {
        "unit": "EUR/kWh",
        "forecast": [{"start": "2026-10-12T00:00:00Z", "price": 0.1}],
    }
    assert parse_osf(eur, 1).points[0][1] == pytest.approx(100.0)
    with pytest.raises(ValueError, match="per kWh"):
        parse_osf({"unit": "DKK/MWh", "forecast": [{"start": "x", "price": 1}]}, 1)
    with pytest.raises(ValueError, match="no forecast"):
        parse_osf({"unit": "DKK/kWh", "forecast": []}, 1)


def test_every_source_lands_in_eur_ct_per_kwh() -> None:
    """100 EUR/MWh, 746.038 DKK/MWh and 0.746038 DKK/kWh are all 10 ct/kWh."""
    forecasts = [
        parse_smartere(SMARTERE, "DK1"),
        parse_carnot(CARNOT, 1),
        parse_epex(EPEX, 1),
        parse_osf(OSF, 1),
    ]

    for forecast in forecasts:
        first = forecast.points[0][1] * CT_PER_KWH_PER_EUR_PER_MWH
        assert first == pytest.approx(10.0), forecast.source
    assert eur_per_unit("EUR") == pytest.approx(1.0)
    assert eur_per_unit("DKK") == pytest.approx(1 / DKK_PER_EUR)
    with pytest.raises(ValueError, match="SEK"):
        eur_per_unit("SEK")


# --- Storage ----------------------------------------------------------------------


def test_storing_the_same_forecast_twice_adds_nothing(store: BenchmarkStore) -> None:
    forecast = parse_smartere(SMARTERE, "DK1")

    assert store.add(forecast) == 2
    assert store.add(forecast) == 0
    # A later publication of the same slots is another forecast
    later = Forecast("smartere", forecast.origin + 3600, forecast.points)
    assert store.add(later) == 2
    store.add(parse_epex(EPEX, 7))

    assert store.count() == 6
    assert store.count("smartere") == 4
    loaded = store.forecasts(["smartere"])
    assert list(loaded) == ["smartere"]
    assert list(loaded["smartere"]) == [forecast.origin, forecast.origin + 3600]
    assert loaded["smartere"][forecast.origin] == dict(forecast.points)
    assert not store.is_backfilled("smartere", "abc")
    store.mark_backfilled("smartere", "abc")
    store.mark_backfilled("smartere", "abc")
    assert store.is_backfilled("smartere", "abc")


# --- Collecting -------------------------------------------------------------------


Fetch = Callable[[str, Mapping[str, str] | None, bytes | None], Any]


def _fetcher(answers: Mapping[str, Any], calls: list[tuple[Any, ...]]) -> Fetch:
    """Return a fetch that answers by URL prefix and records its calls."""

    def fetch(url: str, headers: Mapping[str, str] | None, data: bytes | None) -> Any:
        calls.append((url, dict(headers or {}), data))
        for prefix, answer in answers.items():
            if url.startswith(prefix):
                if isinstance(answer, Exception):
                    raise answer
                return answer(url) if callable(answer) else answer
        raise AssertionError(f"unexpected request: {url}")

    return fetch


def test_each_source_is_requested_as_documented() -> None:
    calls: list[tuple[Any, ...]] = []
    fetch = _fetcher(
        {
            "https://raw.githubusercontent.com/": SMARTERE,
            sources.CARNOT_API: CARNOT,
            sources.EPEX_API: EPEX,
            "http://ha.local:8123/": OSF,
        },
        calls,
    )
    environ = {
        "CARNOT_API_KEY": "carnot-secret",
        "CARNOT_USER": "me@example.com",
        "HA_URL": "http://ha.local:8123/",
        "HA_TOKEN": "ha-secret",
        "OSF_CONFIG_ENTRY_ID": "entry-1",
    }

    assert collect_smartere("DK1", fetch).origin == _ts(2026, 10, 10, 12, 43, 6)
    assert collect_carnot("DK1", NOW, fetch, environ).origin == int(NOW.timestamp())
    assert collect_epex("DK2", NOW, fetch).source == "epex"
    assert collect_osf(NOW, fetch, environ).source == "osf"

    smartere, carnot, epex, osf = calls
    assert smartere[0] == (
        "https://raw.githubusercontent.com/solmoller/Spotprisprognose/main/prognose.json"
    )
    # Credentials travel as headers, never in the URL
    assert carnot[0] == (
        f"{sources.CARNOT_API}?region=dk1&energysource=spotprice&daysahead=7"
    )
    assert carnot[1] == {"apikey": "carnot-secret", "username": "me@example.com"}
    assert epex[0] == (
        f"{sources.EPEX_API}?region=DK2&unit=EUR_PER_MWH&hours=168&timezone=UTC"
    )
    assert osf[0] == (
        "http://ha.local:8123/api/services/open_spot_forecast/get_forecast"
        "?return_response"
    )
    assert osf[1]["Authorization"] == "Bearer ha-secret"
    assert json.loads(osf[2]) == {
        "raw": True,
        "hourly": False,
        "include_known": False,
        "config_entry_id": "entry-1",
    }
    assert all("secret" not in call[0] for call in calls)


def test_sources_without_credentials_are_unavailable() -> None:
    fetch = _fetcher({}, [])

    with pytest.raises(SourceUnavailable, match="CARNOT_API_KEY"):
        collect_carnot("DK1", NOW, fetch, {"CARNOT_USER": "me@example.com"})
    with pytest.raises(SourceUnavailable, match="HA_TOKEN"):
        collect_osf(NOW, fetch, {"HA_URL": "http://ha.local:8123"})


def test_collect_is_idempotent_and_keeps_secrets_out(
    store: BenchmarkStore,
) -> None:
    calls: list[tuple[Any, ...]] = []
    fetch = _fetcher(
        {
            "https://raw.githubusercontent.com/": SMARTERE,
            sources.CARNOT_API: CARNOT,
            sources.EPEX_API: urllib.error.URLError("no route"),
            "http://ha.local:8123": OSF,
        },
        calls,
    )
    environ = {
        "CARNOT_API_KEY": "carnot-secret",
        "CARNOT_USER": "me@example.com",
        "HA_URL": "http://ha.local:8123",
        "HA_TOKEN": "ha-secret",
    }
    options = CollectOptions("DK1", ["smartere", "carnot", "epex", "osf"])
    out = io.StringIO()

    assert collect(options, store, NOW, fetch, environ, out) == 0
    first = store.count()
    # The same moment again: one forecast per source and origin, nothing new
    assert collect(options, store, NOW, fetch, environ, out) == 0
    # An hour later nothing has changed: the forecasts fetched now are the
    # ones already stored, not new ones
    later = NOW + timedelta(hours=1)
    assert collect(options, store, later, fetch, environ, out) == 0
    assert "carnot: unchanged since 2026-10-10 07:30Z" in out.getvalue()
    assert "osf: unchanged since 2026-10-10 07:30Z" in out.getvalue()

    assert first == store.count() == 6
    assert store.count("carnot") == store.count("osf") == store.count("smartere") == 2
    printed = out.getvalue()
    assert "smartere: 2 prices published 2026-10-10 12:43Z (2 new)" in printed
    assert "smartere: 2 prices published 2026-10-10 12:43Z (0 new)" in printed
    assert "epex: failed, connection failed (no route)" in printed
    store.close()
    stored = store.path.read_bytes()
    for secret in ("carnot-secret", "me@example.com", "ha-secret"):
        assert secret not in printed
        assert secret.encode() not in stored


def test_a_changed_forecast_fetched_later_is_a_new_one(store: BenchmarkStore) -> None:
    answers: dict[str, Any] = {sources.EPEX_API: EPEX}
    fetch = _fetcher(answers, [])
    options = CollectOptions("DK1", ["epex"])
    out = io.StringIO()

    collect(options, store, NOW, fetch, {}, out)
    changed = {"prices": [{**EPEX["prices"][0], "total": 101.0}, EPEX["prices"][1]]}
    answers[sources.EPEX_API] = changed
    collect(options, store, NOW + timedelta(hours=6), fetch, {}, out)
    # One price fewer is a change too
    answers[sources.EPEX_API] = {"prices": changed["prices"][:1]}
    collect(options, store, NOW + timedelta(hours=12), fetch, {}, out)

    assert list(store.forecasts(["epex"])["epex"]) == [
        int((NOW + timedelta(hours=hours)).timestamp()) for hours in (0, 6, 12)
    ]
    assert "unchanged" not in out.getvalue()


def test_collect_skips_missing_credentials_and_reports_total_failure(
    store: BenchmarkStore,
) -> None:
    out = io.StringIO()
    fetch = _fetcher({sources.EPEX_API: {"prices": []}}, [])

    skipped = CollectOptions("DK1", ["carnot", "osf"])
    assert collect(skipped, store, NOW, fetch, {}, out) == 0
    failing = CollectOptions("DK1", ["epex"])
    assert collect(failing, store, NOW, fetch, {}, out) == 1

    assert (
        "carnot: skipped, CARNOT_API_KEY and CARNOT_USER are not set" in out.getvalue()
    )
    assert "osf: skipped, HA_URL and HA_TOKEN are not set" in out.getvalue()
    assert (
        "epex: failed, ValueError: EpexPredictor returned no prices" in out.getvalue()
    )
    assert store.count() == 0


def test_an_http_error_is_reported_without_its_url(store: BenchmarkStore) -> None:
    error = urllib.error.HTTPError(
        "https://x/?token=secret", 401, "Unauthorized", Message(), None
    )
    out = io.StringIO()
    fetch = _fetcher({sources.CARNOT_API: error}, [])
    environ = {"CARNOT_API_KEY": "k", "CARNOT_USER": "u"}

    assert (
        collect(CollectOptions("DK1", ["carnot"]), store, NOW, fetch, environ, out) == 1
    )

    assert out.getvalue() == "carnot: failed, HTTP 401\n"


class _FakeResponse(io.BytesIO):
    """Minimal context-manager response for urlopen."""

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def test_requests_go_through_the_backtests_http_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Headers and a body reach urlopen; a 429 is waited out and retried."""
    requests: list[urllib.request.Request] = []
    headers = Message()
    headers["Retry-After"] = "1"
    answers: list[object] = [
        urllib.error.HTTPError("https://x", 429, "slow down", headers, None),
        _FakeResponse(json.dumps(CARNOT).encode()),
    ]

    def urlopen(request: urllib.request.Request, **kwargs: object) -> object:
        requests.append(request)
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr("scripts.backtest.time.sleep", lambda _seconds: None)

    payload = sources.fetch_json("https://x/predict", {"apikey": "k"}, b"{}")

    assert payload == CARNOT
    assert len(requests) == 2
    assert requests[1].get_header("Apikey") == "k"
    assert requests[1].data == b"{}"
    assert requests[1].get_method() == "POST"
    assert requests[1].get_header("User-agent") == "open-spot-forecast-backtest"


# --- Backfilling Smartere Elforbrug from its git history --------------------------


def _commit(sha: str, when: str) -> dict[str, Any]:
    return {"sha": sha, "commit": {"committer": {"name": "x", "date": when}}}


def _version(created: str) -> dict[str, Any]:
    return {**SMARTERE, "Forecast created": created}


def test_past_forecasts_are_rebuilt_from_the_commit_history(
    store: BenchmarkStore,
) -> None:
    calls: list[tuple[Any, ...]] = []
    files = {
        "aaa": _version("2026-10-09T06:23:00"),
        "bbb": _version("2026-10-09T12:02:00"),
        # A commit whose file cannot be read is skipped, and remembered
        "ccc": {"oops": True},
    }
    commits = [
        _commit("ccc", "2026-10-09T18:23:10Z"),
        _commit("bbb", "2026-10-09T10:02:40Z"),
        _commit("aaa", "2026-10-09T04:23:30Z"),
    ]
    fetch = _fetcher(
        {
            "https://api.github.com/": commits,
            "https://raw.githubusercontent.com/": lambda url: files[url.split("/")[5]],
        },
        calls,
    )
    since = datetime(2026, 10, 9, 0, 0, tzinfo=CPH)

    assert backfill_smartere(
        store, "DK1", since, fetch, {"GITHUB_TOKEN": "gh-secret"}
    ) == (
        2,
        None,
    )

    # Each with its original origin: the file's creation time, local
    assert list(store.forecasts(["smartere"])["smartere"]) == [
        _ts(2026, 10, 9, 4, 23),
        _ts(2026, 10, 9, 10, 2),
    ]
    api = calls[0]
    assert api[0] == (
        "https://api.github.com/repos/solmoller/Spotprisprognose/commits"
        "?path=prognose.json&since=2026-10-08T22%3A00%3A00Z&per_page=100&page=1"
    )
    # The token goes to the GitHub API only, as a header
    assert api[1]["Authorization"] == "Bearer gh-secret"
    assert all(call[1] == {} for call in calls[1:])
    assert calls[1][0].endswith("/Spotprisprognose/ccc/prognose.json")

    # A second run fetches the commit list again, and no file
    calls.clear()
    assert backfill_smartere(store, "DK1", since, fetch, {}) == (0, None)
    assert [call[0].split("?")[0] for call in calls] == [
        "https://api.github.com/repos/solmoller/Spotprisprognose/commits"
    ]
    assert "Authorization" not in calls[0][1]


def test_the_commit_list_is_read_page_by_page() -> None:
    pages = {
        1: [_commit(f"sha{n}", "2026-10-09T04:00:00Z") for n in range(100)],
        2: [_commit("last", "2026-10-08T04:00:00Z")],
    }
    calls: list[tuple[Any, ...]] = []
    fetch = _fetcher(
        {"https://api.github.com/": lambda url: pages[int(url.rsplit("=", 1)[1])]},
        calls,
    )

    commits = list(smartere_commits(NOW, fetch, {}))

    assert len(commits) == 101
    assert commits[-1] == ("last", _ts(2026, 10, 8, 4))
    assert len(calls) == 2
    with pytest.raises(TypeError, match="GitHub commits API"):
        list(smartere_commits(NOW, _fetcher({"https://": {"message": "x"}}, []), {}))


@pytest.mark.parametrize("code", [403, 429])
def test_a_rate_limit_stops_the_backfill_and_it_resumes(
    store: BenchmarkStore, code: int
) -> None:
    limited = urllib.error.HTTPError("https://x", code, "rate limit", Message(), None)
    answers: dict[str, Any] = {"aaa": _version("2026-10-09T06:23:00"), "bbb": limited}

    def raw(url: str) -> Any:
        answer = answers[url.split("/")[5]]
        if isinstance(answer, Exception):
            raise answer
        return answer

    commits = [
        _commit("aaa", "2026-10-09T04:23:30Z"),
        _commit("bbb", "2026-10-09T10:02:40Z"),
    ]
    fetch = _fetcher(
        {"https://api.github.com/": commits, "https://raw.githubusercontent.com/": raw},
        [],
    )

    stored, stopped = backfill_smartere(store, "DK1", NOW, fetch, {})

    assert stored == 1
    assert stopped is not None
    assert f"HTTP {code}" in stopped
    assert "GITHUB_TOKEN" in stopped
    # Later the limit is lifted: only the missing commit is fetched
    answers["bbb"] = _version("2026-10-09T12:02:00")
    assert backfill_smartere(store, "DK1", NOW, fetch, {}) == (1, None)
    assert len(store.forecasts(["smartere"])["smartere"]) == 2


def test_another_http_error_ends_the_backfill(store: BenchmarkStore) -> None:
    broken = urllib.error.HTTPError("https://x", 404, "gone", Message(), None)
    fetch = _fetcher({"https://api.github.com/": broken}, [])

    with pytest.raises(urllib.error.HTTPError):
        backfill_smartere(store, "DK1", NOW, fetch, {})


def test_collect_backfills_before_the_current_forecast(store: BenchmarkStore) -> None:
    limited = urllib.error.HTTPError("https://x", 403, "rate limit", Message(), None)
    out = io.StringIO()
    fetch = _fetcher(
        {
            "https://api.github.com/": limited,
            "https://raw.githubusercontent.com/": SMARTERE,
        },
        [],
    )
    options = CollectOptions("DK1", ["smartere"], backfill_days=30)

    assert collect(options, store, NOW, fetch, {}, out) == 0

    assert out.getvalue().splitlines() == [
        "smartere: 0 past forecast(s) backfilled",
        "smartere: backfill stopped, GitHub rate limit (HTTP 403); run again later, "
        "or set GITHUB_TOKEN",
        "smartere: 2 prices published 2026-10-10 12:43Z (2 new)",
    ]


# --- OSF from a learning-database export ------------------------------------------


def test_an_exports_pending_predictions_are_grouped_per_forecast_run(
    tmp_path: Path, store: BenchmarkStore
) -> None:
    export = tmp_path / "export.db"
    with sqlite3.connect(export) as connection:
        connection.execute(
            "CREATE TABLE predictions (id INTEGER PRIMARY KEY, start TEXT, price REAL,"
            " stored_at TEXT)"
        )
        connection.executemany(
            "INSERT INTO predictions (start, price, stored_at) VALUES (?, ?, ?)",
            [
                # One run: rows stored seconds apart
                ("2026-10-12T00:00:00+02:00", 0.746038, "2026-10-10T06:00:05+02:00"),
                ("2026-10-12T00:15:00+02:00", 1.492076, "2026-10-10T06:00:09+02:00"),
                # The next run, six hours later; a naive time is local
                ("2026-10-12T00:00:00+02:00", 0.373019, "2026-10-10T12:00:03"),
            ],
        )
    connection.close()

    forecasts = load_osf_export(export, "DKK", CPH)

    assert [(f.source, f.origin) for f in forecasts] == [
        ("osf", _ts(2026, 10, 10, 4, 0, 5)),
        ("osf", _ts(2026, 10, 10, 10, 0, 3)),
    ]
    assert forecasts[0].points == (
        (_ts(2026, 10, 11, 22, 0), pytest.approx(100.0)),
        (_ts(2026, 10, 11, 22, 15), pytest.approx(200.0)),
    )
    assert forecasts[1].points == ((_ts(2026, 10, 11, 22, 0), pytest.approx(50.0)),)
    out = io.StringIO()
    options = CollectOptions("DK1", ["osf"], osf_db=export, currency="DKK")
    assert collect(options, store, NOW, _fetcher({}, []), {}, out) == 0
    assert store.count("osf") == 3
    with pytest.raises(FileNotFoundError):
        load_osf_export(tmp_path / "missing.db", "DKK", CPH)
    missing = CollectOptions("DK1", ["osf"], osf_db=tmp_path / "missing.db")
    assert collect(missing, store, NOW, _fetcher({}, []), {}, out) == 1


# --- Selecting a forecast: the origin rule -----------------------------------------


def test_the_cutoff_is_local_time_on_the_day_lead_days_before() -> None:
    config = ReportConfig(tz=CPH)

    assert cutoff_seconds(date(2026, 10, 20), 1, config) == _ts(
        2026, 10, 19, 12, tz=CPH
    )
    assert cutoff_seconds(date(2026, 10, 20), 7, config) == _ts(
        2026, 10, 13, 12, tz=CPH
    )
    # Across the change to winter time the cutoff stays 12:00 local
    assert cutoff_seconds(date(2026, 10, 26), 2, config) == _ts(2026, 10, 24, 10)
    early = ReportConfig(tz=CPH, cutoff=time(10, 30))
    assert cutoff_seconds(date(2026, 10, 20), 1, early) == _ts(2026, 10, 19, 8, 30)


def test_a_forecast_published_at_or_after_the_cutoff_is_never_used() -> None:
    cutoff = _ts(2026, 10, 19, 12, tz=CPH)
    origins = [cutoff - 86400, cutoff - 1, cutoff, cutoff + 1800]

    assert select_origin(origins, cutoff) == cutoff - 1
    assert select_origin(origins[2:], cutoff) is None
    assert select_origin([], cutoff) is None
    assert select_origin(origins, cutoff + 1) == cutoff


# --- Hourly alignment ---------------------------------------------------------------


def test_quarter_hours_are_averaged_and_hourly_prices_kept() -> None:
    hour = _ts(2026, 10, 20, 10)
    quarters = {
        hour + n * SLOT_SECONDS: price for n, price in enumerate([80, 100, 100, 120])
    }

    assert hourly_means(quarters) == {hour: pytest.approx(100.0)}
    assert hourly_means({hour: 90.0, hour + 3600: 110.0}) == {
        hour: pytest.approx(90.0),
        hour + 3600: pytest.approx(110.0),
    }
    # A forecast is in EUR/MWh, the comparison in ct/kWh
    values = forecast_hourly(quarters, np.array([hour]))
    assert values is not None
    assert values == pytest.approx([10.0])
    assert forecast_hourly(quarters, np.array([hour, hour + 3600])) is None


@pytest.mark.parametrize(
    ("day", "hours"),
    [(date(2026, 3, 29), 23), (date(2026, 10, 20), 24), (date(2026, 10, 25), 25)],
)
def test_a_local_day_has_23_24_or_25_hours(day: date, hours: int) -> None:
    starts = day_hours(day, CPH)

    assert len(starts) == hours
    assert starts[0] == local_midnight(day, CPH)
    assert np.all(np.diff(starts) == 3600)


# --- Metrics ----------------------------------------------------------------------


def test_the_cheapest_window_and_the_rank_correlation() -> None:
    actual = np.array([5.0, 4.0, 1.0, 1.0, 2.0, 6.0, 7.0, 8.0])

    assert cheapest_window(actual) == 2
    assert cheapest_window(actual[::-1].copy()) == 3
    assert rank_correlation(actual, actual * 2 + 1) == pytest.approx(1.0)
    assert rank_correlation(actual, -actual) == pytest.approx(-1.0)
    # Ties share their mean rank; a constant forecast has no ranking
    assert rank_correlation(
        np.array([1.0, 2.0, 2.0, 3.0]), np.array([1.0, 2.0, 2.0, 3.0])
    ) == (pytest.approx(1.0))
    assert math.isnan(rank_correlation(actual, np.ones(actual.size)))


def test_a_lead_score_uses_the_backtests_daily_definitions() -> None:
    actual = np.arange(24, dtype=float)
    days = [actual + 1.0, actual - 3.0]
    score, reference = LeadScore(), HorizonScore()

    for predicted in days:
        score.add_day(actual, predicted)
        reference.add_day(actual, predicted)

    assert score.days == 2
    assert score.errors.summary() == pytest.approx(reference.summary())
    # Mean of the daily MAEs, root of the mean daily MSE
    assert score.errors.summary() == pytest.approx((2.0, math.sqrt((1 + 9) / 2)))
    assert score.bias() == pytest.approx(-1.0)
    assert score.overlap_share() == pytest.approx(1.0)
    assert score.correlation() == pytest.approx(1.0)
    empty = LeadScore()
    assert math.isnan(empty.bias())
    assert math.isnan(empty.overlap_share())
    assert math.isnan(empty.correlation())


def test_a_missed_cheap_window_and_a_flat_forecast_are_counted() -> None:
    actual = np.concatenate([np.full(3, 1.0), np.full(21, 5.0)])
    late = np.concatenate([np.full(21, 5.0), np.full(3, 1.0)])
    score = LeadScore()

    score.add_day(actual, late)
    score.add_day(actual, np.full(24, 3.0))

    # The flat forecast's window is the first, as the actual's: an overlap
    assert score.overlap_share() == pytest.approx(0.5)
    # Only the first day has a ranking
    assert len(score.correlations) == 1


# --- The report -------------------------------------------------------------------


def _actual(first: date, last: date) -> PriceSeries:
    """Actual prices in ct/kWh: 10 + the local hour + half the day of the month."""
    starts = np.arange(
        local_midnight(first, CPH),
        local_midnight(last + timedelta(days=1), CPH),
        SLOT_SECONDS,
        dtype=np.int64,
    )
    prices = []
    for start in starts.tolist():
        local = datetime.fromtimestamp(start, CPH)
        prices.append(10.0 + local.hour + local.day * 0.5)
    return PriceSeries(starts, np.array(prices))


def _hourly_forecast(
    series: PriceSeries, day: date, offset_ct: float
) -> dict[int, float]:
    """An hourly forecast of ``day``: the actual price plus ``offset_ct``, EUR/MWh."""
    hours = day_hours(day, CPH)
    actual = series.prices_at(hours)
    return {
        int(hour): float(price + offset_ct) / CT_PER_KWH_PER_EUR_PER_MWH
        for hour, price in zip(hours, actual, strict=True)
    }


SERIES = _actual(date(2026, 10, 10), date(2026, 10, 26))
T1, T2 = date(2026, 10, 20), date(2026, 10, 21)
BEFORE = _ts(2026, 10, 19, 10, tz=CPH)
AFTER = _ts(2026, 10, 19, 13, tz=CPH)


def _collected() -> dict[str, dict[int, dict[int, float]]]:
    # Published before the cutoff on the 19th: 1 ct too high for the 20th, 2 for the 21st
    before = _hourly_forecast(SERIES, T1, 1.0) | _hourly_forecast(SERIES, T2, 2.0)
    # Published after it (the 20th's prices are public by then): exactly right
    after = _hourly_forecast(SERIES, T1, 0.0) | _hourly_forecast(SERIES, T2, 0.0)
    # A 15-minute source for the 20th: wrong per quarter, right per hour
    quarters: dict[int, float] = {}
    for hour, price in _hourly_forecast(SERIES, T1, 0.0).items():
        for n, error in enumerate((-20.0, 0.0, 0.0, 20.0)):
            quarters[hour + n * SLOT_SECONDS] = price + error
    return {
        "smartere": {BEFORE: before, AFTER: after},
        "epex": {_ts(2026, 10, 19, 11, tz=CPH): quarters},
    }


def test_sources_are_scored_per_lead_with_the_origin_rule() -> None:
    result = run_benchmark(
        _collected(), SERIES, ["smartere", "epex"], ReportConfig(tz=CPH)
    )

    assert result.sources == ["smartere", "epex", NAIVE]
    assert (result.first_day, result.last_day) == (T1, T2)
    smartere = result.available["smartere"]
    # The 20th at lead 1: the forecast from after the cutoff is not used (1 ct
    # off, not 0). The 21st at lead 1 may use it: it predates the 20th's cutoff
    assert smartere[1].errors.daily_mae == pytest.approx([1.0, 0.0])
    assert smartere[1].errors.summary() == pytest.approx((0.5, math.sqrt(0.5)))
    assert smartere[1].bias() == pytest.approx(0.5)
    # The 21st at lead 2: only the forecast from before the 19th's cutoff
    assert smartere[2].errors.daily_mae == pytest.approx([2.0])
    assert smartere[3].days == 0
    # Quarter-hour errors of ±2 ct average out per hour
    epex = result.available["epex"]
    assert epex[1].errors.daily_mae == pytest.approx([0.0])
    assert epex[2].days == 0
    # The naive row: the same slot a week earlier, 7 × 0.5 ct lower
    naive = result.available[NAIVE]
    assert naive[1].days == 2
    assert naive[1].errors.summary() == pytest.approx((3.5, 3.5))
    assert naive[1].bias() == pytest.approx(-3.5)
    assert naive[2].days == 1
    # Paired: only the 20th at lead 1 has every source
    assert {name: score[1].days for name, score in result.paired.items()} == {
        "smartere": 1,
        "epex": 1,
        NAIVE: 1,
    }
    assert result.paired["smartere"][1].errors.daily_mae == pytest.approx([1.0])
    assert all(score[2].days == 0 for score in result.paired.values())


def test_a_dst_day_is_scored_on_its_25_hours() -> None:
    day = date(2026, 10, 25)
    forecast = _hourly_forecast(SERIES, day, 1.5)
    assert len(forecast) == 25
    collected = {"carnot": {_ts(2026, 10, 24, 9, tz=CPH): forecast}}

    result = run_benchmark(collected, SERIES, ["carnot"], ReportConfig(tz=CPH))

    score = result.available["carnot"][1]
    assert score.errors.summary() == pytest.approx((1.5, 1.5))
    assert score.days == 1
    # Without the repeated hour the day is incomplete and not scored
    forecast.pop(sorted(forecast)[2])
    partial = run_benchmark(collected, SERIES, ["carnot"], ReportConfig(tz=CPH))
    assert partial.available["carnot"][1].days == 0
    assert partial.first_day is None


def test_days_without_actual_prices_or_before_since_are_left_out() -> None:
    collected = _collected()
    # The actual prices end before the 21st
    short = SERIES.between(int(SERIES.starts[0]), local_midnight(T2, CPH))

    result = run_benchmark(collected, short, ["smartere"], ReportConfig(tz=CPH))
    assert result.available["smartere"][1].days == 1
    assert result.last_day == T1
    later = run_benchmark(
        collected, SERIES, ["smartere"], ReportConfig(tz=CPH, since=T2)
    )
    assert later.first_day == T2
    assert later.available["smartere"][1].errors.daily_mae == pytest.approx([0.0])
    nothing = run_benchmark({}, SERIES, ["smartere"], ReportConfig(tz=CPH))
    assert nothing.first_day is None
    empty = PriceSeries.from_points([], [])
    assert (
        run_benchmark(collected, empty, ["smartere"], ReportConfig(tz=CPH)).first_day
        is None
    )


def test_the_report_has_the_paired_all_days_and_minima_tables() -> None:
    config = ReportConfig(tz=CPH)
    result = run_benchmark(_collected(), SERIES, ["smartere", "epex"], config)

    report = format_report(result, "DK1", config, report_units("EUR"))

    assert report.startswith("# Benchmark against external forecasts: DK1\n")
    assert "Target days 2026-10-20 to 2026-10-21 (local)" in report
    assert "published before 12:00 local time" in report
    assert "errors in EUR ct/kWh" in report
    paired, available, minima = report.split("\n## ")[1:]
    assert paired.startswith("Paired days (every source has a forecast)")
    assert "| Lead | Source | Days | MAE | RMSE | Bias |" in paired
    assert "| 1d | smartere | 1 | 1.00 | 1.00 | 1.00 |" in paired
    assert "| 1d | epex | 1 | 0.00 | 0.00 | 0.00 |" in paired
    assert f"| 1d | {NAIVE} | 1 | 3.50 | 3.50 | -3.50 |" in paired
    assert "| 2d |" not in paired
    assert available.startswith("All available days")
    assert "| 1d | smartere | 2 | 0.50 | 0.71 | 0.50 |" in available
    assert "| 2d | smartere | 1 | 2.00 | 2.00 | 2.00 |" in available
    assert f"| 2d | {NAIVE} | 1 | 3.50 | 3.50 | -3.50 |" in available
    assert minima.startswith("Finding the cheap hours")
    assert (
        "| Lead | Source | Days | Cheapest 3 h overlap | Rank correlation |" in minima
    )
    assert "| 1d | smartere | 2 | 100% | 1.00 |" in minima
    # In DKK the same errors are hundredths of a krone
    dkk = format_report(result, "DK1", config, report_units("DKK"))
    assert "| 1d | smartere | 1 | 7.46 | 7.46 | 7.46 |" in dkk
    assert "DKK 1/100 per kWh" in dkk


def test_the_report_says_when_nothing_can_be_scored_or_paired() -> None:
    config = ReportConfig(tz=CPH)
    nothing = run_benchmark({}, SERIES, ["smartere"], config)
    assert "No forecast can be scored yet" in format_report(
        nothing, "DK1", config, report_units("EUR")
    )
    # carnot was never collected: no day has every source
    unpaired = run_benchmark(_collected(), SERIES, ["smartere", "carnot"], config)
    report = format_report(unpaired, "DK1", config, report_units("EUR"))
    assert "No day has every source." in report
    assert "| 1d | smartere | 2 |" in report
    assert "carnot" not in report.split("## All available days")[1]


# --- CLI --------------------------------------------------------------------------


def test_main_collects_and_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in ("CARNOT_API_KEY", "CARNOT_USER", "HA_URL", "HA_TOKEN", "GITHUB_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        benchmark,
        "fetch_json",
        _fetcher({"https://raw.githubusercontent.com/": SMARTERE}, calls),
    )
    cache = ["--cache-dir", str(tmp_path)]

    assert (
        benchmark.main(
            ["collect", "--source", "smartere,carnot", *cache], now=lambda: NOW
        )
        == 0
    )

    out = capsys.readouterr().out
    assert "smartere: 2 prices published 2026-10-10 12:43Z (2 new)" in out
    assert "carnot: skipped" in out
    assert (tmp_path / "benchmark" / "DK1.db").is_file()

    # Report: seed a scoreable forecast, and serve the actual prices offline
    store = BenchmarkStore(tmp_path / "benchmark" / "DK1.db")
    for origin, points in _collected()["smartere"].items():
        store.add(Forecast("smartere", origin, tuple(points.items())))
    store.close()
    requested: list[tuple[Any, ...]] = []

    def load(region: str, first: date, last: date, cache_dir: Path) -> PriceSeries:
        requested.append((region, first, last, cache_dir))
        return SERIES

    monkeypatch.setattr(benchmark, "load_energy_charts_prices", load)
    today = datetime(2026, 10, 27, 8, 0, tzinfo=CPH)

    assert (
        benchmark.main(
            ["report", "--sources", "smartere", "--since", "2026-10-20", *cache],
            now=lambda: today,
        )
        == 0
    )

    report = capsys.readouterr().out
    assert "| 1d | smartere | 2 | 0.50 | 0.71 | 0.50 |" in report
    # Actual prices from nine days before the first forecast slot, for the naive row
    assert requested == [
        ("DK1", date(2026, 10, 1), date(2026, 10, 21), tmp_path / "backtest")
    ]


def test_report_without_forecasts_needs_no_prices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def load(*args: object) -> PriceSeries:
        raise AssertionError("no request without forecasts")

    monkeypatch.setattr(benchmark, "load_energy_charts_prices", load)

    assert (
        benchmark.main(["report", "--region", "DK2", "--cache-dir", str(tmp_path)]) == 0
    )

    assert "No forecast can be scored yet" in capsys.readouterr().out
    assert (tmp_path / "benchmark" / "DK2.db").is_file()


def test_arguments_are_validated(capsys: pytest.CaptureFixture[str]) -> None:
    args = benchmark.parse_args(
        [
            "report",
            "--sources",
            "epex, osf,epex",
            "--cutoff",
            "10:30",
            "--currency",
            "DKK",
        ]
    )
    assert args.sources == ["epex", "osf"]
    assert args.cutoff == time(10, 30)
    assert benchmark.parse_args(["collect"]).sources == list(sources.SOURCES)
    assert benchmark.parse_args(["collect", "--backfill-days", "7"]).backfill_days == 7
    for bad in (
        ["report", "--sources", "nordpool"],
        ["report", "--sources", ","],
        ["report", "--cutoff", "noon"],
        ["collect", "--region", "DE"],
        [],
    ):
        with pytest.raises(SystemExit):
            benchmark.parse_args(bad)
    assert "unknown source(s) nordpool" in capsys.readouterr().err


def test_the_quality_script_passes_arguments_to_the_benchmark() -> None:
    script = (Path(__file__).parent.parent / "scripts" / "quality.sh").read_text()

    assert 'python -m scripts.benchmark "${@:2}"' in script
