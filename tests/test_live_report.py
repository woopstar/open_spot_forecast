"""Tests for the dev-only live-accuracy report (scripts/live_report.py, #94)."""

import hashlib
import math
import sqlite3
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import pytest

from custom_components.open_spot_forecast.ml.storage import LearningStorage
from scripts import live_report
from scripts.live_report import (
    Units,
    bucket_stats,
    error_stats,
    evaluation_stats,
    format_report,
    load_live_data,
    training_times,
    units_from_args,
    weekly_stats,
)

DAY_1 = {
    "2026-09-14": [0.1, -0.3],
    "2026-09-15": [0.2],
    "2026-09-21": [-0.4, 0.4, 0.0, 0.2],
}
DAY_2 = {"2026-09-15": [0.5, -0.5]}


def _export(tmp_path: Path) -> Path:
    """Build a small learning DB through LearningStorage and return its path."""
    hass = Mock()
    hass.config.path.return_value = str(tmp_path / ".storage")
    storage = LearningStorage(hass, "DK1")
    try:
        for day, errors in DAY_1.items():
            storage.add_lead_time_errors(day, {"day_1": errors})
        for day, errors in DAY_2.items():
            storage.add_lead_time_errors(day, {"day_2": errors})
        # 2026-09-21 22:00Z is 2026-09-22 00:00 in Copenhagen
        storage.upsert_evaluation("2026-09-21T21:45:00Z", 1.0, 1.5, 24.0)
        storage.upsert_evaluation("2026-09-21T22:00:00Z", 2.0, 1.0, 23.0)
        storage.upsert_evaluation("2026-09-21T22:15:00Z", 1.0, 1.0, 25.0)
        # The snapshots at another lead time (#113), for two of the slots
        storage.upsert_evaluation("2026-09-21T22:00:00Z", 1.5, 1.0, 11.0, 12.0)
        storage.upsert_evaluation("2026-09-21T22:15:00Z", 0.5, 1.0, 13.0, 12.0)
        storage.save_meta_dict(
            {
                "holdout_mae": 0.25,
                "holdout_rmse": 0.4,
                "holdout_trained_at": "2026-09-28T04:10:18+00:00",
                "training_samples": 5760,
                "hpo_max_depth": 6,
                "hpo_best_mae": 0.2,
            }
        )
        path = storage.db_path
    finally:
        storage.close()
    return path


@pytest.fixture
def export(tmp_path: Path) -> Iterator[Path]:
    """Return a synthetic export path."""
    yield _export(tmp_path)


def _snapshot(path: Path) -> dict[str, str]:
    """Hash every file next to the export (the DB and any WAL/SHM)."""
    return {
        file.name: hashlib.sha256(file.read_bytes()).hexdigest()
        for file in sorted(path.parent.iterdir())
    }


def _expected(errors: list[float]) -> tuple[float, float, float]:
    """Return the pooled MAE, RMSE and bias of errors."""
    n = len(errors)
    return (
        sum(abs(e) for e in errors) / n,
        math.sqrt(sum(e * e for e in errors) / n),
        sum(errors) / n,
    )


def test_bucket_stats_pool_all_retained_days(export: Path) -> None:
    stats = bucket_stats(load_live_data(export))

    assert list(stats) == ["day_1", "day_2"]
    all_day_1 = [e for errors in DAY_1.values() for e in errors]
    mae, rmse, bias = _expected(all_day_1)
    day_1 = stats["day_1"]
    assert day_1.samples == len(all_day_1)
    assert day_1.days == 3
    assert day_1.mae == pytest.approx(mae)
    assert day_1.rmse == pytest.approx(rmse)
    assert day_1.bias == pytest.approx(bias)
    # The backtest's method: mean of daily MAEs, root of the mean daily MSE
    daily = [_expected(errors) for errors in DAY_1.values()]
    assert day_1.daily_mae == pytest.approx(sum(d[0] for d in daily) / 3)
    assert day_1.daily_rmse == pytest.approx(
        math.sqrt(sum(d[1] ** 2 for d in daily) / 3)
    )
    assert stats["day_2"].mae == pytest.approx(0.5)
    assert stats["day_2"].bias == pytest.approx(0.0)


def test_weekly_stats_group_by_iso_week(export: Path) -> None:
    weekly = weekly_stats(load_live_data(export))

    assert [str(monday) for monday in weekly] == ["2026-09-14", "2026-09-21"]
    assert weekly[min(weekly)]["day_1"].samples == 3
    assert set(weekly[max(weekly)]) == {"day_1"}


def test_evaluation_stats_group_by_local_day(export: Path) -> None:
    rows = load_live_data(export).evaluation
    stats = evaluation_stats(rows, ZoneInfo("Europe/Copenhagen"))

    assert stats is not None
    assert stats.samples == 3
    assert stats.days == 2
    assert stats.mae == pytest.approx(0.5)
    assert stats.bias == pytest.approx(1 / 6)
    # Local days: {-0.5} and {1.0, 0.0}
    assert stats.daily_mae == pytest.approx((0.5 + 0.5) / 2)


def test_since_leaves_out_older_slots(
    export: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = load_live_data(export, date(2026, 9, 21), ZoneInfo("Europe/Copenhagen"))

    assert {day for day, _ in data.lead_time} == {date(2026, 9, 21)}
    assert bucket_stats(data)["day_1"].samples == 4
    # 2026-09-21T21:45Z is still 21/9 locally; all three rows stay
    assert len(data.evaluation) == 3
    later = load_live_data(export, date(2026, 9, 22), ZoneInfo("Europe/Copenhagen"))
    assert later.lead_time == {}
    assert [row.timestamp for row in later.evaluation] == [
        "2026-09-21T22:00:00Z",
        "2026-09-21T22:15:00Z",
    ]

    assert live_report.main([str(export), "--since", "2026-09-21"]) == 0
    out = capsys.readouterr().out
    assert "Slots from 2026-09-21 on only." in out
    assert "| day_1 | 4 | 1 |" in out


def test_error_stats_without_samples_is_none() -> None:
    assert error_stats([]) is None
    assert error_stats([(0, 0.0, 0.0, 0.0)]) is None


def test_report_is_read_only(export: Path) -> None:
    before = _snapshot(export)

    load_live_data(export)
    assert live_report.main([str(export), "--currency", "DKK"]) == 0

    assert _snapshot(export) == before


def test_report_contents_in_model_unit(export: Path) -> None:
    report = format_report(load_live_data(export), Units())

    assert "currency/kWh (model unit)" in report
    assert "2026-09-14 to 2026-09-21 (local slot dates), 3 day(s)" in report
    assert "**Only 3 day(s)**" in report
    assert "| day_1 | 7 | 3 |" in report
    assert "### Per week" in report
    assert "2026-09-21T21:45:00Z to 2026-09-21T22:15:00Z (UTC slots), 3 slot(s)" in (
        report
    )
    assert "MAE 0.250, RMSE 0.400" in report
    assert "Training samples: 5760" in report
    assert "`hpo_max_depth` = 6" in report
    assert "`hpo_best_mae` = 0.200" in report
    assert "Training time" not in report


def test_eur_conversion(export: Path) -> None:
    dkk = units_from_args(live_report.parse_args([str(export), "--currency", "DKK"]))
    assert dkk.factor == pytest.approx(100 / 7.46038)
    eur = units_from_args(live_report.parse_args([str(export), "--currency", "EUR"]))
    assert eur.factor == pytest.approx(100)
    rate = units_from_args(
        live_report.parse_args([str(export), "--eur-per-unit", "0.1"])
    )
    assert rate.factor == pytest.approx(10)
    assert rate.label == "EUR ct/kWh"

    report = format_report(load_live_data(export), rate)
    assert "MAE 2.50, RMSE 4.00" in report
    assert "| day_2 | 2 | 1 | 5.00 | 5.00 | 0.00 |" in report


def test_invalid_rate_and_missing_export(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="positive"):
        live_report.main([str(tmp_path / "x.db"), "--eur-per-unit", "0"])
    with pytest.raises(SystemExit, match="No such export"):
        live_report.main([str(tmp_path / "missing.db")])
    assert not (tmp_path / "missing.db").exists()


def test_training_times_from_log(export: Path, tmp_path: Path) -> None:
    log = tmp_path / "home-assistant.log"
    log.write_text(
        "INFO ML model trained in 24.3 s: holdout MAE=0.27, RMSE=0.41\n"
        "INFO something else\n"
        "INFO ML model trained in 31.0 s: holdout MAE=0.26, RMSE=0.40\n"
        "INFO ML model trained in 28 s: holdout MAE=0.26, RMSE=0.40\n",
        encoding="utf-8",
    )
    assert training_times(log.read_text(encoding="utf-8")) == [24.3, 31.0, 28.0]

    report = format_report(load_live_data(export), Units(), times=[24.3, 31.0, 28.0])
    assert "(3 trainings): median 28.0 s, min 24.3 s, max 31.0 s" in report
    assert "no `ML model trained" in format_report(
        load_live_data(export), Units(), times=[]
    )


def test_main_prints_report(
    export: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = tmp_path / "ha.log"
    log.write_text("ML model trained in 3.4 s: holdout MAE=0.2\n", encoding="utf-8")

    assert live_report.main([str(export), "--region", "DK2", "--log", str(log)]) == 0

    out = capsys.readouterr().out
    assert out.startswith("# Live accuracy: open_spot_forecast_DK1_learning.db (DK2)")
    assert "median 3.4 s" in out


def test_empty_database(tmp_path: Path) -> None:
    hass = Mock()
    hass.config.path.return_value = str(tmp_path / ".storage")
    storage = LearningStorage(hass, "DK1")
    path = storage.db_path
    storage.close()

    report = format_report(load_live_data(path), Units())

    assert "Lead-time accuracy: no rows." in report
    assert "No lead-time accuracy samples." in report
    assert "No evaluation rows." in report
    assert "Holdout: none stored" in report
    assert "Per week" not in report


def test_the_other_lead_times_are_reported_next_to_the_evaluation(
    export: Path,
) -> None:
    data = load_live_data(export)

    # The evaluation stays the day-ahead series
    assert len(data.evaluation) == 3
    assert [row.lead_hours for row in data.snapshots[12.0]] == pytest.approx(
        [11.0, 13.0]
    )
    report = format_report(data, Units())
    assert "| evaluation | 3 | 2 |" in report
    assert "| 12 h snapshot | 2 | 1 | 0.500 | 0.500 | 0.000 |" in report
    assert "Mean lead time: 24.0 h." in report
    assert "Mean lead time of the 12 h snapshots: 12.0 h." in report
    later = load_live_data(export, since=date(2026, 9, 23))
    assert later.snapshots == {}


def test_an_export_from_before_the_lead_times_reads_as_day_ahead(
    export: Path,
) -> None:
    with sqlite3.connect(export) as conn:
        conn.executescript(
            """DROP TABLE evaluation;
               CREATE TABLE evaluation (
                   timestamp   TEXT    PRIMARY KEY,
                   predicted   REAL    NOT NULL,
                   actual      REAL    NOT NULL,
                   lead_hours  REAL    NOT NULL
               );
               INSERT INTO evaluation VALUES ('2026-09-21T22:00:00Z', 2.0, 1.0, 23.0);
            """
        )
    conn.close()

    data = load_live_data(export)

    assert [row.predicted for row in data.evaluation] == pytest.approx([2.0])
    assert data.snapshots == {}
    assert "snapshot" not in format_report(data, Units())
