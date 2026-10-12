"""What the latest training took and which options shaped the model (#178).

A live learning-database export is read weeks later to judge an option (the
cross-border model, the near-term model). Two facts it needs were only in
the Home Assistant log, at INFO, or in the owner's memory: how long a
training takes on the installation's hardware, and since when the model has
been what it is now. Both are kept in ``meta`` next to the holdout metrics:

* ``training_seconds`` — the duration of the latest successful training
  (the value of the ``ML model trained in … s`` log line), and
  ``stage1_seconds`` — the part the cross-border stage-1 fits took (only
  with that model). Deleted after a failed training.
* ``model_options`` — what shapes the model, as JSON: the cross-border
  model, the training window, the near-term model, the outage features.
* ``model_options_since`` — when (UTC) the stored options last changed: the
  first training with them. A restart does not move it; switching an option
  does. It is the ``--since`` date of ``scripts/live_report.py``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from homeassistant.util import dt as dt_util

from ..const import NEAR_TERM_REGIONS, UMM_REGIONS
from .base import PredictorBase
from .features import optional_float

_LOGGER = logging.getLogger(__name__)

# meta keys of the latest training's durations (deleted after a failed one)
TRAINING_TIME_META_KEYS = ("training_seconds", "stage1_seconds")


class TrainingInfoMixin(PredictorBase):
    """Training duration and model options, kept in ``meta``.

    Designed to be mixed into SpotPricePredictor — all attributes
    referenced via self are provided by the owning class.
    """

    def model_options(self) -> dict[str, Any]:
        """Return what shapes the model on this installation.

        Returns:
            ``cross_border`` (the two-stage model, #29), ``training_days``
            (the training window, #24), ``near_term`` (the near-term model,
            #135) and ``outages`` (the outage features, #123).
        """
        return {
            "cross_border": self.cross_border is not None,
            "training_days": self.max_history_days,
            "near_term": self.region in NEAR_TERM_REGIONS,
            "outages": self.region in UMM_REGIONS,
        }

    def _store_training_info(self, seconds: float | None) -> None:
        """Keep a training's duration and the model options, and persist them.

        Args:
            seconds: How long the training took; None after a failed
                training, which clears the durations and leaves the options
                (the model in use, if any, is still the earlier one's).
        """
        self.training_seconds = seconds
        cross = self.cross_border
        self.stage1_seconds = (
            cross.fit_seconds if seconds is not None and cross is not None else None
        )
        try:
            if seconds is None:
                self.storage.delete_meta_keys(TRAINING_TIME_META_KEYS)
                return
            options = json.dumps(self.model_options(), sort_keys=True)
            if options != self._model_options_stored or (
                self.model_options_since is None
            ):
                self._model_options_stored = options
                self.model_options_since = dt_util.utcnow()
            meta: dict[str, Any] = {
                "training_seconds": round(seconds, 2),
                "model_options": options,
                "model_options_since": self.model_options_since.isoformat(),
            }
            if self.stage1_seconds is None:
                self.storage.delete_meta_keys(("stage1_seconds",))
            else:
                meta["stage1_seconds"] = round(self.stage1_seconds, 2)
            self.storage.save_meta_dict(meta)
        except Exception as err:
            _LOGGER.warning("Could not store the training information: %s", err)

    def _restore_training_info(self, data: dict[str, Any]) -> None:
        """Restore the durations and the stored options from loaded meta data."""
        self.training_seconds = optional_float(data.get("training_seconds"))
        self.stage1_seconds = optional_float(data.get("stage1_seconds"))
        stored = data.get("model_options")
        since = dt_util.parse_datetime(str(data.get("model_options_since", "")))
        if isinstance(stored, str) and since is not None:
            self._model_options_stored = stored
            self.model_options_since = dt_util.as_utc(since)
        else:
            self._model_options_stored = None
            self.model_options_since = None

    def training_info(self) -> dict[str, Any]:
        """Return the latest training's duration and the model options.

        ``training_seconds`` / ``stage1_seconds`` are None before a
        successful training (and the latter without the cross-border
        model); ``model_options_since`` is None before the first one.
        """
        since: datetime | None = self.model_options_since
        return {
            "training_seconds": self.training_seconds,
            "stage1_seconds": self.stage1_seconds,
            "model_options": self.model_options(),
            "model_options_since": since.isoformat() if since else None,
        }
