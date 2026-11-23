"""The non-spot part of the consumer price, per 15-minute slot (#107).

Stromligning's consumer price excl. VAT is the spot price plus the
supplier's surcharge, the electricity tax, Energinet's net and system
tariffs and the grid company's time-of-use tariff. Its sensors publish the
consumer and the spot price per slot for today and tomorrow, so their
difference is every non-spot component of a slot, excl. VAT. That is the
tariff added to the forecast; its separate tariff sensors only report the
current value.

The model never sees a tariff (#16): ``PriceOutput`` adds it to the
predicted spot price when a price is exposed. A slot past the published
prices takes the latest known day's tariff at the same local time of day:
Danish tariffs follow a fixed daily schedule. Known dated changes of the
national charges (Energinet's net and system tariffs and the electricity
tax, e.g. on 1 January) are added on top (#278): the slot's charges minus
those of the known slot its tariff was taken from (``NationalCharges``).
Without them the schedule is unchanged.
"""

from __future__ import annotations

from bisect import bisect_right
from datetime import UTC, datetime, time
from typing import TYPE_CHECKING

from homeassistant.util import dt as dt_util

from .spot_prices import slot_prices

if TYPE_CHECKING:
    from .api.tariff_changes import NationalCharges


class TariffSchedule:
    """Tariff (consumer − spot price, excl. VAT) by slot and by local time of day."""

    def __init__(
        self,
        by_slot: dict[datetime, float],
        default: float = 0.0,
        changes: NationalCharges | None = None,
    ) -> None:
        """Initialize from the known tariffs, keyed by UTC slot start.

        Args:
            by_slot: The known tariffs by UTC slot start.
            default: The tariff of every slot when none is known (#124).
            changes: Dated national charges (#278), added as a change
                to the slots past the known ones; None adds nothing.
        """
        self._by_slot = by_slot
        self._default = default
        self._changes = changes
        # The latest known slot at each local time of day
        self._by_time: dict[time, datetime] = {}
        for start in sorted(by_slot):
            self._by_time[dt_util.as_local(start).time()] = start
        self._times = sorted(self._by_time)

    @classmethod
    def from_prices(
        cls,
        consumer_data: dict | None,
        spot_data: dict | None,
        changes: NationalCharges | None = None,
    ) -> TariffSchedule:
        """Return the schedule of the slots where both prices are known.

        Args:
            consumer_data: Consumer prices excl. VAT
                (``read_stromligning_sensor()``-shaped: ``today``,
                ``tomorrow``, ``day``).
            spot_data: Spot prices excl. VAT (``read_spot_prices()``).
            changes: Dated national charges for the later slots (#278).

        Returns:
            The schedule; empty (falsy) without a slot known in both.
        """
        spot = {
            start.astimezone(UTC): price for start, _, price in slot_prices(spot_data)
        }
        return cls(
            {
                utc: consumer - spot[utc]
                for start, _, consumer in slot_prices(consumer_data)
                if (utc := start.astimezone(UTC)) in spot
            },
            changes=changes,
        )

    @classmethod
    def fixed(cls, tariff: float) -> TariffSchedule:
        """Return a schedule that gives every slot the same tariff (#124)."""
        return cls({}, default=tariff)

    def __bool__(self) -> bool:
        """Return whether any tariff is known (or a fixed one is set)."""
        return bool(self._by_slot) or abs(self._default) > 1e-9

    def __len__(self) -> int:
        """Return the number of slots with a known tariff."""
        return len(self._by_slot)

    def at(self, start: datetime) -> float:
        """Return the tariff of the slot starting at ``start``.

        Args:
            start: Timezone-aware slot start.

        Returns:
            The slot's own tariff if it is known; else the latest known
            day's tariff at the same local time of day (or the latest
            earlier time of day, for an hour a DST change skipped), plus
            the change in the national charges since that slot when both
            are known (#278); the default (0.0 unless fixed) if no tariff
            is known.
        """
        utc = start.astimezone(UTC)
        if utc in self._by_slot:
            return self._by_slot[utc]
        if not self._times:
            return self._default
        local = dt_util.as_local(utc).time()
        source = self._by_time.get(local)
        if source is None:
            source = self._by_time[self._times[bisect_right(self._times, local) - 1]]
        tariff = self._by_slot[source]
        if self._changes:
            now, then = self._changes.total_at(utc), self._changes.total_at(source)
            if now is not None and then is not None:
                tariff += now - then
        return tariff
