"""Tests for the dev-only rolling backtest (scripts/backtest.py, issue #34)."""

import io
import json
import math
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from email.message import Message
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from custom_components.open_spot_forecast.ml.numpy_models import NumpyGradientBoosting
from scripts import backtest
from scripts.backtest import (
    SLOT_SECONDS,
    BacktestConfig,
    CurrentModel,
    Forecaster,
    HorizonScore,
    LightGbmReference,
    NaiveLastWeek,
    PriceSeries,
    build_models,
    format_report,
    history_for,
    load_energy_charts_prices,
    local_midnight,
    mae,
    merge_series,
    mse,
    parse_energy_charts,
    rmse,
    run_backtest,
    target_slots,
)

TZ = ZoneInfo("Europe/Copenhagen")


def _series(first: date, days: int, price: Callable[[datetime], float]) -> PriceSeries:
    """Return a 15-minute series of ``days`` local days starting at ``first``."""
    starts = np.arange(
        local_midnight(first, TZ),
        local_midnight(first + timedelta(days=days), TZ),
        SLOT_SECONDS,
        dtype=np.int64,
    )
    prices = [price(datetime.fromtimestamp(start, TZ)) for start in starts.tolist()]
    return PriceSeries(starts, np.array(prices))


def _weekly(moment: datetime) -> float:
    """Price that repeats exactly every local week."""
    return 10 + moment.weekday() + moment.hour / 10 + moment.minute / 1000


def _noisy_series(first: date, days: int, seed: int = 7) -> PriceSeries:
    """Weekly pattern plus noise, so no model can forecast it perfectly."""
    rng = np.random.default_rng(seed)
    series = _series(first, days, _weekly)
    return PriceSeries(series.starts, series.prices + rng.normal(0, 2, len(series)))


def _config(origin: date, last: date | None = None, **kwargs: int) -> BacktestConfig:
    """Return a backtest config with a short training window."""
    kwargs.setdefault("window_days", 14)
    return BacktestConfig(
        tz=TZ, first_origin=origin, last_origin=last or origin, **kwargs
    )


@dataclass
class _Recorder:
    """Wrap a model, recording what it is given."""

    model: Forecaster
    calls: list[tuple[PriceSeries, np.ndarray]] = field(default_factory=list)

    @property
    def name(self) -> str:
        """Report the wrapped model's name."""
        return self.model.name

    def forecast(self, history: PriceSeries, targets: np.ndarray) -> np.ndarray:
        """Record the call and delegate to the wrapped model."""
        self.calls.append((history, targets))
        return self.model.forecast(history, targets)


def _small_gbm() -> NumpyGradientBoosting:
    """A fast stand-in for the production model with the same code path."""
    return NumpyGradientBoosting(n_estimators=10, learning_rate=0.1, random_state=42)


# --- Metric math ----------------------------------------------------------------


def test_metrics_known_values():
    """MAE, MSE and RMSE match hand-computed values."""
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    predicted = np.array([2.0, 2.0, 1.0, 4.0])  # errors 1, 0, -2, 0

    assert mae(actual, predicted) == pytest.approx(0.75)
    assert mse(actual, predicted) == pytest.approx(1.25)
    assert rmse(actual, predicted) == pytest.approx(math.sqrt(1.25))


def test_metrics_perfect_forecast_is_zero():
    """A perfect forecast scores zero error."""
    values = np.array([-1.5, 0.0, 3.25])

    assert mae(values, values) == pytest.approx(0.0)
    assert rmse(values, values) == pytest.approx(0.0)


def test_rmse_never_below_mae():
    """RMSE >= MAE for any error vector."""
    rng = np.random.default_rng(0)
    actual = rng.normal(5, 3, 500)
    predicted = actual + rng.standard_t(2, 500)

    assert rmse(actual, predicted) >= mae(actual, predicted)


def test_metrics_empty_input_is_nan():
    """Empty input gives NaN rather than a misleading zero."""
    empty = np.array([])

    assert math.isnan(mae(empty, empty))
    assert math.isnan(rmse(empty, empty))


def test_metrics_reject_shape_mismatch():
    """Mismatched arrays are an error, not a silent broadcast."""
    with pytest.raises(ValueError, match="shape mismatch"):
        mae(np.array([1.0, 2.0]), np.array([1.0]))


def test_horizon_score_averages_daily_metrics_like_epexpredictor():
    """MAE is the mean of daily MAEs and RMSE the root of the mean daily MSE."""
    score = HorizonScore()
    score.add_day(np.zeros(2), np.array([1.0, -1.0]))  # MAE 1, MSE 1
    score.add_day(np.zeros(4), np.full(4, 3.0))  # MAE 3, MSE 9

    mean_mae, root_mean_mse = score.summary()

    # Pooling all six slots would give MAE 14/6; per-day averaging gives 2.
    assert mean_mae == pytest.approx(2.0)
    assert root_mean_mse == pytest.approx(math.sqrt(5.0))


def test_horizon_score_ignores_missing_actuals():
    """Slots without an actual are skipped; a day without any is not scored."""
    score = HorizonScore()
    score.add_day(np.array([1.0, math.nan]), np.array([2.0, 100.0]))
    score.add_day(np.array([math.nan, math.nan]), np.array([5.0, 5.0]))

    assert score.daily_mae == pytest.approx([1.0])
    assert score.summary()[0] == pytest.approx(1.0)


def test_horizon_score_without_days_is_nan():
    """An unscored horizon reports NaN."""
    assert all(math.isnan(value) for value in HorizonScore().summary())


# --- Cutoff enforcement ----------------------------------------------------------


def test_between_excludes_the_cutoff_slot_and_later():
    """The slot starting exactly at the cutoff is already in the future."""
    series = _series(date(2026, 6, 1), 3, _weekly)
    cutoff = local_midnight(date(2026, 6, 2), TZ)

    visible = series.between(int(series.starts[0]), cutoff)

    assert int(visible.starts.max()) == cutoff - SLOT_SECONDS
    assert cutoff not in visible.starts
    assert len(visible) == 96


def test_history_is_a_read_only_copy():
    """Models get copies they cannot write to or use to reach later prices."""
    series = _series(date(2026, 6, 1), 10, _weekly)

    history = history_for(series, date(2026, 6, 9), _config(date(2026, 6, 9)))

    assert history.prices.base is None
    assert history.starts.base is None
    with pytest.raises(ValueError):
        history.prices[0] = 0.0


def test_history_for_spans_window_up_to_cutoff():
    """History covers exactly [origin - window_days, origin) in local time."""
    series = _series(date(2026, 5, 1), 60, _weekly)
    origin = date(2026, 6, 15)

    history = history_for(series, origin, _config(origin, window_days=14))

    assert int(history.starts.min()) == local_midnight(date(2026, 6, 1), TZ)
    assert int(history.starts.max()) == local_midnight(origin, TZ) - SLOT_SECONDS
    assert len(history) == 14 * 96


def test_run_backtest_never_passes_prices_at_or_after_the_cutoff():
    """Across every origin, each model only receives pre-cutoff history."""
    series = _noisy_series(date(2026, 5, 1), 40)
    recorder = _Recorder(NaiveLastWeek(TZ))
    config = _config(date(2026, 5, 20), date(2026, 6, 5), window_days=14)

    result = run_backtest(series, [recorder], config)

    assert result.origins == len(recorder.calls) == 17
    for history, targets in recorder.calls:
        cutoff = int(targets.min())
        assert int(history.starts.max()) < cutoff
        assert int(history.starts.min()) >= cutoff - 14 * 86400 - 3600


@pytest.mark.parametrize(
    "make_model",
    [
        pytest.param(lambda: NaiveLastWeek(TZ), id="naive"),
        pytest.param(lambda: CurrentModel(TZ, _small_gbm), id="current-small"),
        pytest.param(lambda: CurrentModel(TZ), id="current-production"),
        pytest.param(lambda: LightGbmReference(TZ), id="lightgbm"),
    ],
)
def test_future_prices_cannot_change_the_forecast(make_model):
    """Poisoning every price at or after the cutoff leaves forecasts unchanged.

    If training or feature building could see anything past the cutoff, the
    1e6 prices (or NaN) would change the forecast.
    """
    if isinstance(make_model(), LightGbmReference):
        pytest.importorskip("lightgbm")
    origin = date(2026, 6, 15)
    clean = _noisy_series(date(2026, 5, 20), 30)
    future = clean.starts >= local_midnight(origin, TZ)
    config = _config(origin, window_days=21)
    targets, _ = target_slots(origin, config)

    forecasts = []
    for poison in (None, 1e6, math.nan):
        prices = clean.prices.copy()
        if poison is not None:
            prices[future] = poison
        history = history_for(PriceSeries(clean.starts, prices), origin, config)
        forecasts.append(make_model().forecast(history, targets))

    assert np.all(np.isfinite(forecasts[0]))
    np.testing.assert_array_equal(forecasts[0], forecasts[1])
    np.testing.assert_array_equal(forecasts[0], forecasts[2])


# --- Horizons, naive baseline, runner ---------------------------------------------


@pytest.mark.parametrize(
    ("origin", "day_one_slots"),
    [(date(2026, 6, 15), 96), (date(2026, 3, 29), 92), (date(2025, 10, 26), 100)],
)
def test_target_slots_follow_the_local_calendar(origin, day_one_slots):
    """Horizon days are local calendar days, including DST-length days."""
    targets, horizon = target_slots(origin, _config(origin))

    assert int(targets[0]) == local_midnight(origin, TZ)
    assert int(np.sum(horizon == 1)) == day_one_slots
    assert sorted(set(horizon.tolist())) == [1, 2, 3]
    assert int(targets[-1]) + SLOT_SECONDS == local_midnight(
        origin + timedelta(days=3), TZ
    )


def test_naive_repeats_the_local_slot_across_a_dst_change():
    """Last week's 08:00 is used for this week's 08:00 even if the offset moved."""
    series = _series(date(2026, 3, 20), 20, _weekly)
    origin = date(2026, 3, 30)  # Monday after the 2026-03-29 spring-forward
    targets, _ = target_slots(origin, _config(origin))
    target = int(targets[8 * 4])  # 08:00 local
    history = history_for(series, origin, _config(origin))

    (predicted,) = NaiveLastWeek(TZ).forecast(history, np.array([target]))

    week_ago = datetime(2026, 3, 23, 8, tzinfo=TZ)
    assert predicted == pytest.approx(_weekly(week_ago))


def test_naive_scores_zero_on_a_weekly_pattern():
    """An exactly weekly price is forecast perfectly at every horizon."""
    series = _series(date(2026, 5, 1), 50, _weekly)
    config = _config(date(2026, 5, 20), date(2026, 6, 10))

    result = run_backtest(series, [NaiveLastWeek(TZ)], config)

    assert result.origins == 22
    for score in result.scores[NaiveLastWeek.name]:
        assert len(score.daily_mae) == 22
        assert score.summary() == pytest.approx((0.0, 0.0))


def test_run_backtest_skips_origins_without_enough_history():
    """Origins with under a week of history are not scored."""
    series = _series(date(2026, 6, 1), 20, _weekly)
    config = _config(date(2026, 6, 3), date(2026, 6, 10))

    result = run_backtest(series, [NaiveLastWeek(TZ)], config)

    assert result.origins == 3  # 2026-06-08 .. 2026-06-10


def test_run_backtest_skips_origins_without_actuals():
    """Origins whose forecast days have no actual prices are not scored."""
    series = _series(date(2026, 6, 1), 14, _weekly)
    config = _config(date(2026, 6, 15), date(2026, 6, 20))

    assert run_backtest(series, [NaiveLastWeek(TZ)], config).origins == 0


def test_run_backtest_rejects_a_wrongly_shaped_forecast() -> None:
    """A model returning the wrong number of slots is a hard error."""

    @dataclass
    class Broken:
        name: str = "broken"

        def forecast(self, history: PriceSeries, targets: np.ndarray) -> np.ndarray:
            return np.zeros(3)

    series = _series(date(2026, 6, 1), 20, _weekly)
    with pytest.raises(ValueError, match="broken returned"):
        run_backtest(series, [Broken()], _config(date(2026, 6, 15)))


def test_current_model_uses_the_production_factory_by_default():
    """The backtest's current model is the integration's configured GBM."""
    model = CurrentModel(TZ).model_factory()

    assert (model.n_estimators, model.learning_rate) == (200, pytest.approx(0.1))


def test_current_model_learns_the_daily_shape():
    """The NumPy GBM beats a flat forecast on a strongly daily-shaped price."""
    series = _noisy_series(date(2026, 5, 1), 40)
    config = _config(date(2026, 6, 1), date(2026, 6, 3), window_days=21)

    result = run_backtest(series, [CurrentModel(TZ)], config)

    current_mae = result.scores[CurrentModel.name][0].summary()[0]
    assert current_mae < float(np.std(series.prices))


# --- Model selection and report ----------------------------------------------------


def test_build_models_selects_by_name():
    """Model names map to the three documented models."""
    models, skipped = build_models(["naive", "current"], TZ)

    assert [type(model) for model in models] == [NaiveLastWeek, CurrentModel]
    assert skipped == {}


def test_build_models_skips_lightgbm_when_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without lightgbm the reference row is reported as skipped, not an error."""
    monkeypatch.setattr(backtest, "_lightgbm_available", lambda: False)

    models, skipped = build_models(["lightgbm"], TZ)

    assert models == []
    assert "requirements_backtest.txt" in skipped[LightGbmReference.name]


def test_build_models_rejects_unknown_names():
    """A typo in --models fails loudly."""
    with pytest.raises(ValueError, match="unknown model"):
        build_models(["prophet"], TZ)


def test_format_report_renders_markdown_table():
    """The report has one row per model and a MAE/RMSE column per horizon."""
    series = _series(date(2026, 5, 1), 30, _weekly)
    config = _config(date(2026, 5, 20), date(2026, 5, 22))
    result = run_backtest(series, [NaiveLastWeek(TZ)], config)
    result.skipped_models[LightGbmReference.name] = "lightgbm not installed"

    report = format_report(result, "DK1").splitlines()

    assert "DK1" in report[0]
    assert "3 scored" in report[0]
    assert report[2] == (
        "| Model | 1d MAE | 1d RMSE | 2d MAE | 2d RMSE | 3d MAE | 3d RMSE |"
    )
    assert report[4] == (
        "| naive (same slot last week) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |"
    )
    assert report[5].startswith("| lightgbm (reference) | n/a |")
    assert report[-1] == "lightgbm (reference): skipped, lightgbm not installed."


# --- energy-charts data ------------------------------------------------------------


def _payload(starts: list[int], prices: list[float | None]) -> dict:
    """Return a minimal energy-charts /price response."""
    return {"unix_seconds": starts, "price": prices, "unit": "EUR / MWh"}


def test_parse_energy_charts_expands_hourly_prices_and_converts_units():
    """Hourly EUR/MWh prices fill four quarter-hours each, in ct/kWh."""
    t0 = 1_759_269_600
    # One hourly price, then two 15-minute prices (as at DK1's 2025-10-01 switch).
    payload = _payload([t0 - 3600, t0, t0 + 900], [100.0, 50.0, 60.0])

    series = parse_energy_charts(payload)

    assert series.starts.tolist() == [
        t0 - 3600,
        t0 - 2700,
        t0 - 1800,
        t0 - 900,
        t0,
        t0 + 900,
    ]
    assert series.prices.tolist() == pytest.approx([10.0] * 4 + [5.0, 6.0])


def test_parse_energy_charts_drops_nulls_without_stretching_neighbours():
    """A missing quarter-hour stays missing."""
    t0 = 1_780_000_200
    series = parse_energy_charts(
        _payload([t0, t0 + 900, t0 + 1800], [10.0, None, 30.0])
    )

    assert series.starts.tolist() == [t0, t0 + 1800]
    assert series.prices.tolist() == pytest.approx([1.0, 3.0])


def test_parse_energy_charts_rejects_unexpected_payloads():
    """Unknown units or misaligned arrays are errors."""
    with pytest.raises(ValueError, match="unit"):
        parse_energy_charts({"unix_seconds": [], "price": [], "unit": "EUR / kWh"})
    with pytest.raises(ValueError, match="differ in length"):
        parse_energy_charts(_payload([1, 2], [1.0]))


def test_merge_series_keeps_first_price_for_duplicates():
    """Overlapping month chunks do not duplicate slots."""
    first = PriceSeries(np.array([0, 900]), np.array([1.0, 2.0]))
    second = PriceSeries(np.array([900, 1800]), np.array([9.0, 3.0]))

    merged = merge_series([first, second])

    assert merged.starts.tolist() == [0, 900, 1800]
    assert merged.prices.tolist() == pytest.approx([1.0, 2.0, 3.0])
    assert len(merge_series([])) == 0


def test_price_series_rejects_unsorted_input():
    """Out-of-order slots would break the cutoff search."""
    with pytest.raises(ValueError, match="strictly increasing"):
        PriceSeries(np.array([900, 0]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="equal length"):
        PriceSeries(np.array([0, 900]), np.array([1.0]))


def test_prices_at_returns_nan_for_missing_slots():
    """Scoring looks up actuals by slot and marks gaps as NaN."""
    series = PriceSeries(np.array([0, 900, 2700]), np.array([1.0, 2.0, 4.0]))

    found = series.prices_at(np.array([900, 1800, 2700, 3600]))

    assert found[[0, 2]].tolist() == pytest.approx([2.0, 4.0])
    assert np.isnan(found[[1, 3]]).all()
    assert np.isnan(merge_series([]).prices_at(np.array([0]))).all()


def test_load_energy_charts_prices_caches_complete_months(tmp_path: Path) -> None:
    """Past months are fetched once and then served from the cache."""
    requested: list[str] = []

    def fetch(url: str) -> dict:
        requested.append(url)
        month_start = local_midnight(
            date.fromisoformat(url.split("start=")[1][:10]), TZ
        )
        return _payload([month_start], [100.0])

    def load(region: str, last: date) -> PriceSeries:
        return load_energy_charts_prices(
            region, date(2026, 4, 20), last, tmp_path, fetch, today=date(2026, 6, 15)
        )

    first = load("DK1", date(2026, 6, 2))
    again = load("DE", date(2026, 5, 2))
    load("DK1", date(2026, 6, 2))

    assert len(first) == 3  # one price per month
    assert "bzn=DK1" in requested[0] and "bzn=DE-LU" in requested[3]
    # DK1: April, May, June; DE: April, May; the DK1 re-run only refetches June.
    assert len(requested) == 6
    assert "start=2026-06-01" in requested[-1]
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "energy_charts_DE-LU_2026-04.json",
        "energy_charts_DE-LU_2026-05.json",
        "energy_charts_DK1_2026-04.json",
        "energy_charts_DK1_2026-05.json",
    ]
    assert len(again) == 2


class _FakeResponse(io.BytesIO):
    """Minimal context-manager response for urlopen."""

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _http_error(code: int, retry_after: str | None) -> urllib.error.HTTPError:
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError("https://x", code, "err", headers, None)


def test_http_get_json_waits_out_rate_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    """HTTP 429 is retried after the server's Retry-After delay."""
    responses: list[object] = [
        _http_error(429, "22"),
        _http_error(429, "soon"),
        _FakeResponse(json.dumps({"ok": 1}).encode()),
    ]

    def urlopen(request: urllib.request.Request, **kwargs: object) -> object:
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    waits: list[float] = []

    assert backtest._http_get_json("https://x", sleep=waits.append) == {"ok": 1}
    assert waits == pytest.approx([22.0, 30.0])


def test_http_get_json_gives_up_on_other_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-rate-limit errors propagate immediately."""

    def urlopen(request: urllib.request.Request, **kwargs: object) -> object:
        raise _http_error(500, None)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)

    with pytest.raises(urllib.error.HTTPError):
        backtest._http_get_json("https://x", sleep=lambda _: None)


# --- CLI -----------------------------------------------------------------------------


def test_main_prints_and_writes_the_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI runs end to end on injected data and writes the markdown table."""
    series = _series(date(2026, 5, 1), 40, _weekly)
    loaded: list[tuple[date, date]] = []

    def fake_load(region: str, first: date, last: date, *_: object) -> PriceSeries:
        loaded.append((first, last))
        return series

    monkeypatch.setattr(backtest, "load_energy_charts_prices", fake_load)
    output = tmp_path / "report.md"

    exit_code = backtest.main(
        [
            "--start=2026-05-25",
            "--end=2026-05-27",
            "--window-days=14",
            "--models=naive",
            f"--output={output}",
        ]
    )

    assert exit_code == 0
    assert loaded == [(date(2026, 5, 10), date(2026, 5, 30))]
    assert "| naive (same slot last week) | 0.00 |" in capsys.readouterr().out
    assert output.read_text(encoding="utf-8").startswith("Rolling backtest, DK1")


@pytest.mark.parametrize(
    "argv",
    [
        ["--window-days=3"],
        ["--step-days=0"],
        ["--start=2026-06-02", "--end=2026-06-01"],
    ],
)
def test_main_rejects_invalid_arguments(argv):
    """Nonsensical ranges fail before any data is fetched."""
    with pytest.raises(SystemExit):
        backtest.main(argv)
