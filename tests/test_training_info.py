"""The training duration and the model options are kept in meta (#178)."""

import json
from collections.abc import Callable, Iterator
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from homeassistant.util import dt as dt_util

from custom_components.open_spot_forecast.ml.gbm import NumpyGradientBoosting
from custom_components.open_spot_forecast.ml.predictor import SpotPricePredictor
from custom_components.open_spot_forecast.ml.storage import LearningStorage
from custom_components.open_spot_forecast.ml.training_info import (
    TRAINING_TIME_META_KEYS,
)
from custom_components.open_spot_forecast.sensor import LearningMetricsSensor
from scripts.live_report import Units, format_report, load_live_data


def _hass(tmp_path: Path) -> Mock:
    async def run_inline(func: Callable[..., Any], *args: Any) -> Any:
        return func(*args)

    hass = Mock()
    hass.config.path.return_value = str(tmp_path / ".storage")
    hass.async_add_executor_job = run_inline
    hass.async_create_background_task = lambda coro, _name: coro.close()
    return hass


@pytest.fixture
def make_predictor(
    tmp_path: Path,
) -> Iterator[Callable[..., SpotPricePredictor]]:
    """Return a factory of predictors sharing one database (a restart)."""
    created: list[SpotPricePredictor] = []

    def make(region: str = "DK1", **options: Any) -> SpotPricePredictor:
        predictor = SpotPricePredictor(_hass(tmp_path), region, **options)
        predictor.price_model = NumpyGradientBoosting(n_estimators=3)
        today = dt_util.now().date()
        predictor.price_history = [
            {
                "date": (today - timedelta(days=3 - n)).isoformat(),
                "prices": [0.3 + 0.01 * (slot // 4) for slot in range(96)],
            }
            for n in range(3)
        ]
        created.append(predictor)
        return predictor

    yield make
    for predictor in created:
        predictor.storage.close()


def _meta(predictor: SpotPricePredictor) -> dict[str, Any]:
    data = predictor.storage.load_all()
    assert data is not None
    return data


def test_a_training_stores_its_duration_and_the_model_options(
    make_predictor: Callable[..., SpotPricePredictor],
) -> None:
    predictor = make_predictor()
    before = dt_util.utcnow()

    predictor._train_models()

    assert predictor.training_seconds is not None and predictor.training_seconds > 0
    assert predictor.stage1_seconds is None
    assert predictor.model_options_since is not None
    assert predictor.model_options_since >= before
    meta = _meta(predictor)
    assert float(meta["training_seconds"]) == pytest.approx(
        predictor.training_seconds, abs=0.01
    )
    assert "stage1_seconds" not in meta
    assert json.loads(meta["model_options"]) == {
        "cross_border": False,
        "near_term": True,
        "outages": True,
        "training_days": predictor.max_history_days,
    }
    assert meta["model_options_since"] == predictor.model_options_since.isoformat()


def test_the_cross_border_stage_1_time_is_stored_with_that_model(
    make_predictor: Callable[..., SpotPricePredictor],
) -> None:
    predictor = make_predictor(cross_border=True)

    predictor._train_models()

    assert predictor.stage1_seconds is not None
    assert predictor.training_seconds is not None
    assert 0 <= predictor.stage1_seconds <= predictor.training_seconds
    meta = _meta(predictor)
    assert float(meta["stage1_seconds"]) == pytest.approx(
        predictor.stage1_seconds, abs=0.01
    )
    assert json.loads(meta["model_options"])["cross_border"] is True


def test_a_region_without_the_optional_parts_says_so(
    make_predictor: Callable[..., SpotPricePredictor],
) -> None:
    predictor = make_predictor("SE3", training_days=60)

    assert predictor.model_options() == {
        "cross_border": False,
        "training_days": 60,
        "near_term": False,
        "outages": False,
    }


@pytest.mark.asyncio
async def test_a_restart_keeps_the_date_and_a_changed_option_moves_it(
    make_predictor: Callable[..., SpotPricePredictor],
) -> None:
    first = make_predictor()
    first._train_models()
    since = first.model_options_since
    assert since is not None
    await first.save_learning_data()

    # A restart with the same options, then another training
    second = make_predictor()
    await second._load_learning_data()
    assert second.model_options_since == since
    assert second.training_seconds == pytest.approx(first.training_seconds, abs=0.01)
    later = since + timedelta(hours=6)
    with patch("homeassistant.util.dt.utcnow", return_value=later):
        second._train_models()
    assert second.model_options_since == since

    # The cross-border model is switched on: the options are new from now on
    third = make_predictor(cross_border=True)
    await third._load_learning_data()
    assert third.model_options_since == since
    with patch("homeassistant.util.dt.utcnow", return_value=later):
        third._train_models()
    assert third.model_options_since == later
    assert _meta(third)["model_options_since"] == later.isoformat()


def test_a_failed_training_clears_the_durations_and_keeps_the_options(
    make_predictor: Callable[..., SpotPricePredictor],
) -> None:
    predictor = make_predictor()
    predictor._train_models()
    since = predictor.model_options_since

    with patch.object(predictor.price_model, "fit", side_effect=RuntimeError("no fit")):
        predictor._train_models()

    assert predictor.training_seconds is None
    assert predictor.stage1_seconds is None
    assert predictor.model_options_since == since
    meta = _meta(predictor)
    assert not any(key in meta for key in TRAINING_TIME_META_KEYS)
    assert "model_options" in meta


def test_unreadable_stored_values_are_treated_as_missing(
    make_predictor: Callable[..., SpotPricePredictor],
) -> None:
    predictor = make_predictor()

    predictor._restore_training_info(
        {
            "training_seconds": "soon",
            "model_options": "{}",
            "model_options_since": "yesterday",
        }
    )

    assert predictor.training_seconds is None
    assert predictor.model_options_since is None
    assert predictor.training_info()["model_options_since"] is None


def test_a_storage_error_does_not_fail_the_training(
    make_predictor: Callable[..., SpotPricePredictor],
    caplog: pytest.LogCaptureFixture,
) -> None:
    predictor = make_predictor()

    with patch.object(
        predictor.storage, "save_meta_dict", side_effect=OSError("disk full")
    ):
        predictor._train_models()

    assert predictor.is_trained is True
    assert "Could not store the training information" in caplog.text


def test_the_learning_metrics_sensor_shows_them(
    make_predictor: Callable[..., SpotPricePredictor],
) -> None:
    predictor = make_predictor()
    predictor._train_models()
    sensor = LearningMetricsSensor(
        MagicMock(), MagicMock(entry_id="test"), {"ml_predictor": predictor}
    )

    attributes = sensor.extra_state_attributes

    assert attributes["training_seconds"] == predictor.training_seconds
    assert attributes["stage1_seconds"] is None
    assert attributes["model_options"]["near_term"] is True
    assert attributes["model_options_since"] == (
        predictor.model_options_since.isoformat()
        if predictor.model_options_since
        else None
    )


# --- The live report -----------------------------------------------------------------


def _export(tmp_path: Path, meta: dict[str, Any]) -> Path:
    hass = Mock()
    hass.config.path.return_value = str(tmp_path / ".storage")
    storage = LearningStorage(hass, "DK1")
    try:
        for day in ("2026-10-12", "2026-10-13", "2026-10-20"):
            storage.add_lead_time_errors(day, {"day_1": [0.1, -0.2]})
        storage.save_meta_dict(meta)
        return Path(storage.db_path)
    finally:
        storage.close()


OPTIONS = json.dumps(
    {"cross_border": True, "near_term": True, "outages": True, "training_days": 180}
)


def test_the_report_prints_the_training_time_and_the_options(tmp_path: Path) -> None:
    export = _export(
        tmp_path,
        {
            "training_seconds": 61.37,
            "stage1_seconds": 48.2,
            "model_options": OPTIONS,
            "model_options_since": "2026-10-11T04:10:00+00:00",
        },
    )

    report = format_report(load_live_data(export), Units())

    assert "Latest training took 61.4 s, 48.2 s of it the cross-border stage 1." in (
        report
    )
    assert "Model options since 2026-10-11T04:10:00+00:00: `cross_border` = True" in (
        report
    )
    assert "`training_days` = 180" in report
    assert "inside the reported period" not in report


def test_the_report_warns_when_the_options_changed_inside_the_period(
    tmp_path: Path,
) -> None:
    export = _export(
        tmp_path,
        {
            "training_seconds": 5.0,
            "model_options": OPTIONS,
            "model_options_since": "2026-10-13T12:00:00+00:00",
        },
    )

    report = format_report(load_live_data(export), Units())
    later = format_report(load_live_data(export, since=date(2026, 10, 14)), Units())

    assert "**The model options changed on 2026-10-13 (UTC), inside the" in report
    assert "`--since 2026-10-14`" in report
    assert "Latest training took 5.0 s." in report
    assert "inside the reported period" not in later


def test_an_export_without_the_keys_reports_as_before(tmp_path: Path) -> None:
    export = _export(tmp_path, {"model_options": "not json"})

    report = format_report(load_live_data(export), Units())

    assert "Latest training took" not in report
    assert "Model options" not in report
    assert "inside the reported period" not in report
