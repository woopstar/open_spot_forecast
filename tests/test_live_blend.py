"""What a blend of the model with the external forecasts would have scored (#157).

``scripts/live_blend.py`` scores the model, every source, the equal-weight mean
and the inverse-MSE weighted mean over the same slots, from the per-slot errors
of an export; ``scripts/live_report.py`` prints the table.
"""

import math
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import pytest

from custom_components.open_spot_forecast.const import EXTERNAL_MODEL_SOURCE
from custom_components.open_spot_forecast.ml.storage import LearningStorage
from custom_components.open_spot_forecast.time_slots import (
    UTC_KEY_FORMAT,
    utc_slot_key,
)
from scripts.live_blend import (
    EQUAL_BLEND,
    INVERSE_MSE_BLEND,
    MIN_BLEND_SLOTS,
    SlotErrors,
    blend_stats,
    inverse_mse_weights,
)
from scripts.live_report import Units, format_report, load_live_data

EDS = "sensor.energi_data_service"
STROMLIGNING = "sensor.stromligning_forecasts_vat"
CPH = ZoneInfo("Europe/Copenhagen")
# Midnight of 2026-09-01 in Copenhagen, the report's time zone for DK1
FIRST = datetime(2026, 8, 31, 22, tzinfo=UTC)


def _day_of(start: str) -> date:
    """Return the Copenhagen date of a UTC slot key."""
    moment = datetime.strptime(start, UTC_KEY_FORMAT).replace(tzinfo=UTC)
    return moment.astimezone(CPH).date()


def _slots(days: int, model: float, source: float, per_day: int = 96) -> SlotErrors:
    """Return ``days`` of day-1 errors: the model's and one source's, alternating sign."""
    errors: SlotErrors = {}
    for index in range(days * per_day):
        start = FIRST + timedelta(days=index // per_day, minutes=15 * (index % per_day))
        sign = 1 if index % 2 == 0 else -1
        errors[(utc_slot_key(start), "day_1")] = {
            EXTERNAL_MODEL_SOURCE: sign * model,
            EDS: -sign * source,
        }
    return errors


def test_the_weights_are_proportional_to_the_inverse_mse() -> None:
    # MSE 1 and 4: weights 4/5 and 1/5
    weights = inverse_mse_weights([MIN_BLEND_SLOTS * 1.0, MIN_BLEND_SLOTS * 4.0], 288)

    assert weights == pytest.approx([0.8, 0.2])
    assert inverse_mse_weights([1.0, 1.0], MIN_BLEND_SLOTS - 1) is None


def test_a_member_without_an_error_does_not_divide_by_zero() -> None:
    weights = inverse_mse_weights([0.0, MIN_BLEND_SLOTS * 1.0], MIN_BLEND_SLOTS)

    assert weights is not None
    assert weights[0] == pytest.approx(1.0)
    assert sum(weights) == pytest.approx(1.0)


def test_members_and_blends_are_scored_over_the_same_slots() -> None:
    errors: SlotErrors = {
        ("2026-09-01T10:00:00Z", "day_1"): {EXTERNAL_MODEL_SOURCE: 0.5, EDS: -0.25},
        ("2026-09-02T10:00:00Z", "day_1"): {EXTERNAL_MODEL_SOURCE: -1.0, EDS: 0.5},
        # Not paired: the source has no forecast, or the model no prediction
        ("2026-09-02T10:15:00Z", "day_1"): {EXTERNAL_MODEL_SOURCE: 9.0},
        ("2026-09-02T10:30:00Z", "day_1"): {EDS: 9.0},
        # Without a source there is nothing to blend
        ("2026-09-02T10:00:00Z", "day_2"): {EXTERNAL_MODEL_SOURCE: 1.0},
    }

    blends = blend_stats(errors, _day_of)

    assert list(blends) == ["day_1"]
    blend = blends["day_1"]
    assert list(blend.rows) == [
        EXTERNAL_MODEL_SOURCE,
        EDS,
        EQUAL_BLEND,
        INVERSE_MSE_BLEND,
    ]
    model = blend.rows[EXTERNAL_MODEL_SOURCE]
    assert (model.slots, model.days) == (2, 2)
    assert model.mae == pytest.approx(0.75)
    assert model.rmse == pytest.approx(math.sqrt((0.25 + 1.0) / 2))
    assert model.bias == pytest.approx(-0.25)
    assert blend.rows[EDS].mae == pytest.approx(0.375)
    # The mean of the two members, slot by slot: 0.125 and -0.25
    equal = blend.rows[EQUAL_BLEND]
    assert equal.mae == pytest.approx(0.1875)
    assert equal.bias == pytest.approx(-0.0625)
    # Too few earlier slots for weights: the model alone
    assert blend.rows[INVERSE_MSE_BLEND] == model
    assert blend.weighted_slots == 0
    assert blend.weights is None


def test_the_inverse_mse_weights_of_a_day_come_from_the_earlier_days() -> None:
    # The model is off by 2, the source by 1, with opposite signs
    blend = blend_stats(_slots(days=5, model=2.0, source=1.0), _day_of)["day_1"]

    assert blend.rows[EXTERNAL_MODEL_SOURCE].mae == pytest.approx(2.0)
    assert blend.rows[EDS].mae == pytest.approx(1.0)
    assert blend.rows[EQUAL_BLEND].mae == pytest.approx(0.5)
    # MSE 4 and 1: weights 1/5 and 4/5, a blend error of 2/5 - 4/5 = -2/5.
    # The first three days have no weights yet and are the model's
    assert blend.weights == pytest.approx({EXTERNAL_MODEL_SOURCE: 0.2, EDS: 0.8})
    assert blend.weighted_slots == 2 * 96
    assert blend.rows[INVERSE_MSE_BLEND].slots == 5 * 96
    assert blend.rows[INVERSE_MSE_BLEND].mae == pytest.approx((3 * 2.0 + 2 * 0.4) / 5)


def test_a_days_own_errors_never_weight_that_day() -> None:
    """One day short of the minimum: every day is still the model alone."""
    blend = blend_stats(_slots(days=3, model=2.0, source=1.0), _day_of)["day_1"]

    assert blend.weighted_slots == 0
    assert blend.rows[INVERSE_MSE_BLEND] == blend.rows[EXTERNAL_MODEL_SOURCE]
    # The weights a blend would use from now on
    assert blend.weights == pytest.approx({EXTERNAL_MODEL_SOURCE: 0.2, EDS: 0.8})


def test_every_source_of_a_bucket_has_to_have_the_slot() -> None:
    errors = _slots(days=1, model=1.0, source=1.0, per_day=4)
    first = next(iter(errors))
    errors[first][STROMLIGNING] = 0.5

    blend = blend_stats(errors, _day_of)["day_1"]

    assert list(blend.rows) == [
        EXTERNAL_MODEL_SOURCE,
        EDS,
        STROMLIGNING,
        EQUAL_BLEND,
        INVERSE_MSE_BLEND,
    ]
    assert {stats.slots for stats in blend.rows.values()} == {1}
    # (1.0 - 1.0 + 0.5) / 3
    assert blend.rows[EQUAL_BLEND].bias == pytest.approx(0.5 / 3)


def _export(tmp_path: Path, errors: SlotErrors) -> Path:
    hass = Mock()
    hass.config.path.return_value = str(tmp_path / ".storage")
    storage = LearningStorage(hass, "DK1")
    try:
        for (start, bucket), by_source in errors.items():
            storage.upsert_external_slot_errors(
                start,
                [(bucket, source, 1, error) for source, error in by_source.items()],
            )
        return storage.db_path
    finally:
        storage.close()


def test_the_report_prints_the_blend_table_per_bucket(tmp_path: Path) -> None:
    errors = _slots(days=5, model=2.0, source=1.0)
    errors[("2026-09-03T10:00:00Z", "day_3")] = {EXTERNAL_MODEL_SOURCE: 1.0, EDS: 3.0}
    data = load_live_data(_export(tmp_path, errors))

    report = format_report(data, Units())

    section = report[report.index("## Blend with the external forecasts") :]
    assert "| Bucket | Forecast | Slots | Days | MAE | RMSE | Bias |" in section
    rows = [line for line in section.splitlines() if line.startswith("| day_")]
    assert [row.split(" | ")[:2] for row in rows] == [
        ["| day_1", "open_spot_forecast"],
        ["| day_1", EDS],
        ["| day_1", EQUAL_BLEND],
        ["| day_1", INVERSE_MSE_BLEND],
        ["| day_3", "open_spot_forecast"],
        ["| day_3", EDS],
        ["| day_3", EQUAL_BLEND],
        ["| day_3", INVERSE_MSE_BLEND],
    ]
    assert "| day_1 | open_spot_forecast | 480 | 5 | 2.000 | 2.000 | 0.000 |" in section
    assert f"| day_1 | {EQUAL_BLEND} | 480 | 5 | 0.500 |" in section
    assert (
        f"- `day_1`: inverse-MSE weights over every slot: open_spot_forecast 0.20, "
        f"{EDS} 0.80; 192 slot(s) were scored with weights."
    ) in section
    assert "- `day_3`: too few slots for inverse-MSE weights." in section
    assert "**Only 5 day(s)**: decide on a blend from 14+ days" in section
    # The training section still follows
    assert "## Training" in section


def test_since_leaves_out_the_older_slots(tmp_path: Path) -> None:
    path = _export(tmp_path, _slots(days=2, model=2.0, source=1.0))

    data = load_live_data(path, since=date(2026, 9, 2), tz=CPH)

    assert len(data.slot_errors) == 96
    # Midnight of 2026-09-02 in Copenhagen
    assert min(start for start, _ in data.slot_errors) == "2026-09-01T22:00:00Z"


def test_fourteen_days_are_enough_to_decide(tmp_path: Path) -> None:
    data = load_live_data(
        _export(tmp_path, _slots(days=14, model=1.0, source=1.0, per_day=4))
    )

    assert "**Only" not in format_report(data, Units())


def test_without_slot_errors_the_report_has_no_blend_table(tmp_path: Path) -> None:
    data = load_live_data(_export(tmp_path, {}))

    assert data.slot_errors == {}
    assert "Blend with" not in format_report(data, Units())
