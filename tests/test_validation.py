"""The integration validates with the library Home Assistant uses (#195)."""

import importlib
import sys
from unittest.mock import patch

import pytest

from homeassistant.helpers import config_validation as cv

from custom_components.open_spot_forecast import config_flow, services, validation
from custom_components.open_spot_forecast.validation import vol


def test_the_integration_builds_its_schemas_with_one_library() -> None:
    """Config flow and services take ``vol`` from the same module."""
    assert config_flow.vol is validation.vol
    assert services.vol is validation.vol
    assert isinstance(services.GET_FORECAST_SCHEMA, vol.Schema)


def test_it_is_the_library_home_assistant_validates_with() -> None:
    """A schema mixes its markers with Home Assistant's own validators."""
    schema = vol.Schema(
        {
            vol.Required("name"): cv.string,
            vol.Optional("hours", default=24): vol.All(
                vol.Coerce(int), vol.Range(min=1, max=168)
            ),
            vol.Optional("unit"): vol.In(["kWh", "MWh"]),
        }
    )

    assert schema({"name": "DK1", "hours": "48"}) == {"name": "DK1", "hours": 48}
    assert schema({"name": "DK1"})["hours"] == 24
    for bad in ({"hours": 5}, {"name": "DK1", "hours": 0}, {"name": "x", "unit": "J"}):
        with pytest.raises(vol.Invalid):
            schema(bad)


def test_without_probatio_it_falls_back_to_voluptuous() -> None:
    """A Home Assistant from before probatio has voluptuous only."""
    try:
        # None in sys.modules makes the import raise ImportError
        with patch.dict(sys.modules, {"probatio": None}):
            fallback = importlib.reload(validation).vol

        # Whatever ``import voluptuous`` gives: the real package on an older
        # Home Assistant, the alias Home Assistant installs from 2026.10 on
        assert fallback is sys.modules["voluptuous"]
        assert fallback.Schema({fallback.Required("a"): int})({"a": 1}) == {"a": 1}
    finally:
        importlib.reload(validation)

    assert validation.vol.Schema is not None
