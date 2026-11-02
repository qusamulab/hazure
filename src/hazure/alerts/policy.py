"""Causal notification decisions from a stream of detector labels."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import timedelta
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from hazure._core import Persistent, TimeSeries, parse_duration
from hazure._core.missing import as_flags
from hazure.events.interval import _timestamp_to_ns

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import NDArray

__all__ = ["AlertPolicy"]


class AlertPolicy(Persistent):
    """Notify once per sustained anomaly, with a cooldown per column.

    Parameters
    ----------
    min_consecutive
        Number of consecutive anomalous observations required before notifying.
        Normal or unknown observations break the run.
    cooldown
        Minimum time between notifications, as a duration or integer UTC
        nanoseconds. Zero disables the cooldown. An anomaly that persists during
        cooldown can notify when the cooldown expires.

    Notes
    -----
    Output is a notification pulse, not an anomaly interval: 1 only on the row
    that opens a notification, 0 on other known rows, and NaN on unknown rows.
    A sustained anomaly notifies once, even after cooldown expires. It must end
    before another notification can open. Cooldown is measured from the previous
    notification, independently for each column.

    :meth:`apply` starts fresh and leaves this policy's state unchanged.
    :meth:`update` and :meth:`update_many` carry state across calls. Persist a
    running policy with :meth:`to_dict` and resume it with :meth:`from_dict`.
    These methods return decisions; they do not send messages.

    Examples
    --------
    >>> ts = TimeSeries.from_arrays(np.arange(8), [1, 0, 1, 1, 1, 0, 1, 1])
    >>> AlertPolicy(min_consecutive=2).apply(ts).values.ravel()
    array([0., 0., 0., 1., 0., 0., 0., 1.])
    """

    _persisted: ClassVar[tuple[str, ...]] = (
        "_columns",
        "_counts",
        "_active",
        "_last_alert",
        "_last_time",
    )

    def __init__(
        self,
        min_consecutive: int = 1,
        cooldown: int | str | timedelta | np.timedelta64 = 0,
    ) -> None:
        _check_consecutive(min_consecutive)
        duration = _cooldown_ns(cooldown)
        self.min_consecutive = min_consecutive
        # Keep readable string parameters; timedelta objects are normalised so
        # every supported constructor argument also survives JSON storage.
        self.cooldown: int | str = cooldown if isinstance(cooldown, str) else duration
        self.reset()

    def reset(self) -> AlertPolicy:
        """Clear notification history and return this policy."""
        self._columns: tuple[str, ...] | None = None
        self._counts: dict[str, int] = {}
        self._active: dict[str, bool] = {}
        self._last_alert: dict[str, int] = {}
        self._last_time: int | None = None
        return self

    def apply(self, labels: Any) -> Any:
        """Filter a whole label series from a fresh state, preserving its backend.

        Parameters
        ----------
        labels
            Label series or frame accepted by :class:`~hazure.TimeSeries`.

        Returns
        -------
        Any
            Notification pulses in the same flavour as the input.
        """
        return self.clone().update_many(labels)

    def update(
        self, time: Any, labels: float | Mapping[str, float]
    ) -> float | dict[str, float]:
        """Consume one timestamp and scalar label or column-to-label mapping.

        Timestamps must strictly increase. After the first observation, a mapping
        must have exactly the same columns; a scalar requires a single column.
        Unknown labels reset continuity and remain unknown in the output.
        """
        _check_consecutive(self.min_consecutive)
        cooldown = _cooldown_ns(self.cooldown)
        stamp = _timestamp_to_ns(time)
        if stamp == np.iinfo(np.int64).min:
            raise ValueError("Notification timestamps must not be missing.")
        if self._last_time is not None and stamp <= self._last_time:
            raise ValueError("Notification timestamps must strictly increase.")
        if isinstance(labels, Mapping):
            if not labels or not _valid_names(labels):
                raise ValueError("Label mappings need non-empty string column names.")
            columns = self._columns or tuple(labels)
            if set(columns) != set(labels):
                raise ValueError(
                    f"Expected label columns {list(columns)}, got {list(labels)}."
                )
            row = np.asarray([labels[name] for name in columns], dtype=np.float64)
        else:
            columns = self._columns or ("value",)
            if len(columns) != 1:
                raise ValueError("A multi-column policy needs a mapping of labels.")
            row = np.asarray([labels], dtype=np.float64)
        if row.shape != (len(columns),):
            raise ValueError("Each column needs one scalar label.")
        result = self._advance(stamp, columns, row, cooldown)
        return result[columns[0]] if not isinstance(labels, Mapping) else result

    def update_many(self, labels: Any) -> Any:
        """Consume a batch in time order, carrying state into the next call.

        The whole time axis and column schema are validated before updating
        state. Unlike general ingestion, unsorted and duplicate times are
        rejected instead of sorted or dropped.
        """
        _check_consecutive(self.min_consecutive)
        cooldown = _cooldown_ns(self.cooldown)
        ts = TimeSeries.from_any(labels, sort=False, drop_duplicates=False)
        if not _valid_names(ts.columns):
            raise ValueError("Label columns need non-empty string names.")
        if np.any(ts.time == np.iinfo(np.int64).min):
            raise ValueError("Notification timestamps must not be missing.")
        if ts.n_rows and np.any(np.diff(ts.time) <= 0):
            raise ValueError("Notification timestamps must strictly increase.")
        if ts.n_rows and self._last_time is not None and ts.time[0] <= self._last_time:
            raise ValueError("Notification timestamps must strictly increase.")
        columns = self._columns or ts.columns
        if set(columns) != set(ts.columns):
            raise ValueError(
                f"Expected label columns {list(columns)}, got {list(ts.columns)}."
            )
        ordered = ts.select(columns)
        out = np.empty_like(ordered.values)
        for i, stamp in enumerate(ordered.time):
            row = self._advance(int(stamp), columns, ordered.values[i], cooldown)
            out[i] = [row[name] for name in columns]
        result = ordered.wrap(out).select(ts.columns)
        return result.to_native()

    def _advance(
        self,
        stamp: int,
        columns: tuple[str, ...],
        row: NDArray[np.float64],
        cooldown: int,
    ) -> dict[str, float]:
        """Advance the single state machine shared by batch and live processing."""
        flags = as_flags(row)
        result: dict[str, float] = {}
        for i, name in enumerate(columns):
            if not flags[i]:
                self._counts[name] = 0
                self._active[name] = False
                result[name] = np.nan if np.isnan(row[i]) else 0.0
                continue
            count = min(self._counts.get(name, 0) + 1, self.min_consecutive)
            self._counts[name] = count
            previous = self._last_alert.get(name)
            ready = previous is None or stamp - previous >= cooldown
            fire = (
                count >= self.min_consecutive
                and not self._active.get(name, False)
                and ready
            )
            result[name] = float(fire)
            if fire:
                self._last_alert[name] = stamp
                self._active[name] = True
        self._columns = columns
        self._last_time = stamp
        return result


def _check_consecutive(value: Any) -> None:
    """Require a positive integer run length, excluding bools."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("min_consecutive must be an int.")
    if value < 1:
        raise ValueError("min_consecutive must be at least 1.")


def _valid_names(names: Iterable[object]) -> bool:
    """Validate names also supplied by callers outside typed Python."""
    return all(isinstance(name, str) and bool(name) for name in names)


def _cooldown_ns(value: Any) -> int:
    """Read a non-negative cooldown without interpreting calendar durations."""
    if isinstance(value, bool) or not isinstance(
        value, int | str | timedelta | np.timedelta64
    ):
        raise TypeError("cooldown must be integer nanoseconds or a duration.")
    duration = value if isinstance(value, int) else parse_duration(value)
    if duration < 0:
        raise ValueError("cooldown must not be negative.")
    return duration
