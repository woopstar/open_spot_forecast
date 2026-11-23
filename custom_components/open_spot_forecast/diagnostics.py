"""Config entry diagnostics (#276).

Home Assistant offers a config entry's diagnostics as a download on the
integration's page. OSF's show what a bug report needs and the log does not:

* ``entry``: the config entry's data and options, the ENTSO-E key redacted;
* ``sources``: every source's health (``source_health.py``): last attempt,
  last success, age, last error and how far its data reaches;
* ``forecast``: when the forecast was generated and whether it is stale
  (``forecast_freshness``, #277), and how many predictions it has;
* ``model``: whether the model is trained, its holdout metrics and its
  training info (duration, options, last rejected retrain, #178, #275);
* ``row_counts``: the rows of every table of the learning database.

No API key, URL or query parameter appears in the output: the key is
redacted from the entry and scrubbed from every text, and the source errors
never carry a URL (``describe_error``).
"""

from __future__ import annotations

from typing import Any

from homeassistant.components.diagnostics import async_redact_data
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from .const import CONF_ENTSOE_API_KEY, DOMAIN
from .forecast_attributes import forecast_freshness

# Config entry keys whose values never leave the installation
TO_REDACT = {CONF_ENTSOE_API_KEY}
REDACTED = "**REDACTED**"


def _scrub(value: Any, secrets: set[str]) -> Any:
    """Replace every secret inside the strings of a JSON-ready value."""
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, REDACTED)
        return value
    if isinstance(value, dict):
        return {key: _scrub(item, secrets) for key, item in value.items()}
    if isinstance(value, list):
        return [_scrub(item, secrets) for item in value]
    return value


def _model(ml_predictor: Any) -> dict[str, Any] | None:
    """Return the model's training state, None with ML prediction disabled."""
    if ml_predictor is None:
        return None
    return {
        "is_trained": bool(ml_predictor.is_trained),
        **ml_predictor.holdout_metrics(),
        **ml_predictor.training_info(),
    }


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: ConfigEntry
) -> dict[str, Any]:
    """Return the diagnostics of a config entry.

    Args:
        hass: Home Assistant instance.
        entry: The config entry; its data is read from ``hass.data``, and
            an entry that is not loaded shows its configuration only.

    Returns:
        ``entry``, ``sources``, ``forecast``, ``model`` and ``row_counts``
        (see the module docstring), JSON-ready.
    """
    api_data: dict[str, Any] = hass.data.get(DOMAIN, {}).get(entry.entry_id) or {}
    now = dt_util.utcnow()
    ml_predictor = api_data.get("ml_predictor")
    storage = ml_predictor.storage if ml_predictor else api_data.get("price_storage")
    row_counts: dict[str, int] = (
        await hass.async_add_executor_job(storage.table_row_counts) if storage else {}
    )
    health = api_data.get("source_health")
    diagnostics = {
        "entry": {
            "version": entry.version,
            "data": async_redact_data(dict(entry.data), TO_REDACT),
            "options": async_redact_data(dict(entry.options), TO_REDACT),
        },
        "sources": health.summary(now) if health else {},
        "forecast": {
            **forecast_freshness(api_data, now),
            "predictions": len(api_data.get("ml_predictions") or []),
            "tomorrow_available": bool(api_data.get("tomorrow_available")),
        },
        "model": _model(ml_predictor),
        "row_counts": row_counts,
    }
    secrets = {
        str(value)
        for source in (entry.data, entry.options)
        if (value := source.get(CONF_ENTSOE_API_KEY))
    }
    scrubbed: dict[str, Any] = _scrub(diagnostics, secrets)
    return scrubbed
