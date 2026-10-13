"""The backtest's per-day output (``--daily``, #202).

The report's mean hides a few very wrong days: the gas price lifted DK1's
forecast by 7 ct/kWh on 2026-10-12 while the yearly mean barely moved.
``--daily`` writes every scored day's MAE and bias, and adds the bias and
the worst days per horizon to the report.
"""

import csv
import math
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from scripts import backtest
from scripts.backtest import (
    SLOT_SECONDS,
    BacktestConfig,
    HorizonScore,
    PriceSeries,
    daily_rows,
    format_tail,
    local_midnight,
    run_backtest,
)

TZ = ZoneInfo("Europe/Copenhagen")
FIRST = date(2026, 5, 1)
# The one day the model under test gets wrong, and by how much
BAD_DAY = date(2026, 5, 22)
BAD_ERROR = 7.0


def _flat_series(days: int, price: float = 5.0) -> PriceSeries:
    """Return ``days`` local days of one price from ``FIRST``."""
    starts = np.arange(
        local_midnight(FIRST, TZ),
        local_midnight(FIRST + timedelta(days=days), TZ),
        SLOT_SECONDS,
        dtype=np.int64,
    )
    return PriceSeries(starts, np.full(starts.shape, price))


class WrongOnOneDay:
    """Forecasts the last known price, ``BAD_ERROR`` too high on ``BAD_DAY``."""

    name = "wrong on one day"

    def forecast(self, history: PriceSeries, targets: np.ndarray) -> np.ndarray:
        """Return the last price, lifted for the slots of ``BAD_DAY``."""
        predicted = np.full(targets.shape, float(history.prices[-1]))
        for i, start in enumerate(targets.tolist()):
            if datetime.fromtimestamp(start, TZ).date() == BAD_DAY:
                predicted[i] += BAD_ERROR
        return predicted


def _result() -> backtest.BacktestResult:
    config = BacktestConfig(
        tz=TZ,
        first_origin=date(2026, 5, 20),
        last_origin=date(2026, 5, 23),
        window_days=14,
        horizon_days=2,
    )
    return run_backtest(_flat_series(30), [WrongOnOneDay()], config)


def test_horizon_score_keeps_each_days_bias_and_date() -> None:
    """The bias is the mean of predicted - actual over the known slots."""
    score = HorizonScore()
    score.add_day(np.array([1.0, 3.0]), np.array([2.0, 2.0]), date(2026, 5, 1))
    score.add_day(np.array([1.0, math.nan]), np.array([4.0, 100.0]), date(2026, 5, 2))
    score.add_day(np.array([math.nan]), np.array([1.0]), date(2026, 5, 3))

    assert score.daily_mae == pytest.approx([1.0, 3.0])
    assert score.daily_bias == pytest.approx([0.0, 3.0])
    assert score.days == [date(2026, 5, 1), date(2026, 5, 2)]


def test_daily_rows_name_the_day_a_forecast_was_wrong() -> None:
    """One row per model, origin and horizon; the bad day shows at both horizons."""
    rows = daily_rows(_result())

    assert rows[0] == ["model", "origin", "day", "horizon", "mae", "bias"]
    body = rows[1:]
    # 4 origins x 2 horizon days
    assert len(body) == 8
    bad = [row for row in body if float(row[4]) > 1.0]
    assert [(row[1], row[2], row[3]) for row in bad] == [
        ("2026-05-22", "2026-05-22", "1"),
        ("2026-05-21", "2026-05-22", "2"),
    ]
    for row in bad:
        assert float(row[4]) == pytest.approx(BAD_ERROR)
        assert float(row[5]) == pytest.approx(BAD_ERROR)
    assert all(row[0] == WrongOnOneDay.name for row in body)


def test_the_tail_shows_what_the_mean_hides() -> None:
    """One bad day in four: a mean of a quarter of it, a maximum of all of it."""
    result = _result()
    mean_mae = result.scores[WrongOnOneDay.name][0].summary()[0]
    assert mean_mae == pytest.approx(BAD_ERROR / 4)

    lines = format_tail(result).splitlines()

    assert lines[2] == (
        "| Model | 1d bias | 1d P95 | 1d max | 2d bias | 2d P95 | 2d max |"
    )
    cells = [cell.strip() for cell in lines[4].strip("|").split("|")]
    assert cells[0] == WrongOnOneDay.name
    assert cells[1] == "+1.75"
    assert float(cells[2]) == pytest.approx(np.percentile([0, 0, 7, 0], 95), abs=0.01)
    assert cells[3] == "7.00"


def test_an_unscored_horizon_has_no_tail() -> None:
    config = BacktestConfig(
        tz=TZ, first_origin=FIRST, last_origin=FIRST, window_days=14, horizon_days=1
    )
    result = run_backtest(_flat_series(30), [WrongOnOneDay()], config)

    # No history before the first day: nothing is scored
    assert daily_rows(result) == [["model", "origin", "day", "horizon", "mae", "bias"]]
    assert format_tail(result).splitlines()[4] == (
        "| wrong on one day | nan | nan | nan |"
    )


def test_main_writes_the_daily_file_and_extends_the_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        backtest, "load_energy_charts_prices", lambda *_: _flat_series(30)
    )
    daily = tmp_path / "daily.csv"
    output = tmp_path / "report.md"

    exit_code = backtest.main(
        [
            "--start=2026-05-20",
            "--end=2026-05-23",
            "--window-days=14",
            "--models=naive",
            "--weather=none",
            f"--daily={daily}",
            f"--output={output}",
        ]
    )

    assert exit_code == 0
    with daily.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["model", "origin", "day", "horizon", "mae", "bias"]
    # 4 origins x 3 horizon days
    assert len(rows) == 13
    assert {row[2] for row in rows[1:]} == {f"2026-05-{day}" for day in range(20, 26)}
    report = capsys.readouterr().out
    assert "| Model | 1d bias | 1d P95 | 1d max |" in report
    assert "1d bias" in output.read_text(encoding="utf-8")
