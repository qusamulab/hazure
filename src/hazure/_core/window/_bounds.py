"""Duration parsing and positional or temporal window geometry."""

from __future__ import annotations

import re
from datetime import timedelta
from typing import TYPE_CHECKING, Final, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


Closed = Literal["right", "left", "both", "neither"]


Window = int | str | np.timedelta64 | timedelta


_DURATION_PATTERN: Final = re.compile(
    r"^\s*(?P<count>\d+(?:\.\d+)?)?\s*(?P<unit>[A-Za-zµ]+)\s*$"
)


# Fixed-length units only; calendar units do not have a constant duration.
_DURATION_UNITS: Final[dict[str, int]] = {
    "ns": 1,
    "nanosecond": 1,
    "nanoseconds": 1,
    "us": 1_000,
    "µs": 1_000,
    "microsecond": 1_000,
    "microseconds": 1_000,
    "ms": 1_000_000,
    "millisecond": 1_000_000,
    "milliseconds": 1_000_000,
    "s": 1_000_000_000,
    "sec": 1_000_000_000,
    "secs": 1_000_000_000,
    "second": 1_000_000_000,
    "seconds": 1_000_000_000,
    "m": 60 * 1_000_000_000,
    "min": 60 * 1_000_000_000,
    "mins": 60 * 1_000_000_000,
    "minute": 60 * 1_000_000_000,
    "minutes": 60 * 1_000_000_000,
    "h": 3_600 * 1_000_000_000,
    "hr": 3_600 * 1_000_000_000,
    "hour": 3_600 * 1_000_000_000,
    "hours": 3_600 * 1_000_000_000,
    "d": 86_400 * 1_000_000_000,
    "day": 86_400 * 1_000_000_000,
    "days": 86_400 * 1_000_000_000,
    "w": 7 * 86_400 * 1_000_000_000,
    "week": 7 * 86_400 * 1_000_000_000,
    "weeks": 7 * 86_400 * 1_000_000_000,
}


_CALENDAR_UNITS: Final = frozenset(
    {"month", "months", "mo", "y", "year", "years", "q", "quarter", "quarters", "b"}
)


def parse_duration(spec: str | np.timedelta64 | timedelta) -> int:
    """Convert a duration to nanoseconds.

    Parameters
    ----------
    spec
        A string such as ``"7d"``, ``"30min"``, ``"1h30min"`` is *not*
        supported — use a single unit — or a ``timedelta`` /
        ``numpy.timedelta64``.

    Returns
    -------
    int
        Length of the duration in nanoseconds.

    Raises
    ------
    ValueError
        The string is unparseable, names a calendar unit, or is not positive.

    Examples
    --------
    >>> parse_duration("2h")
    7200000000000
    >>> parse_duration("500ms")
    500000000
    """
    if isinstance(spec, np.timedelta64):
        return int(spec.astype("timedelta64[ns]").astype(np.int64))
    if isinstance(spec, timedelta):
        return int(spec / timedelta(microseconds=1)) * 1_000

    match = _DURATION_PATTERN.match(spec)
    if match is None:
        msg = (
            f"Cannot read {spec!r} as a duration. Use a number followed by a "
            f"unit, e.g. '7d', '30min', '500ms'."
        )
        raise ValueError(msg)

    unit = match["unit"]
    # Case matters for exactly one unit: "M" is a calendar month, "m" is
    # minutes. Rule that out before folding case, so "5M" cannot quietly become
    # five minutes.
    if unit == "M":
        msg = (
            f"Unit 'M' in {spec!r} is a calendar unit (month), not a fixed "
            f"duration. Use 'm' for minutes, or days/hours."
        )
        raise ValueError(msg)
    key = unit.lower()
    if key in _CALENDAR_UNITS:
        msg = (
            f"Unit {unit!r} in {spec!r} is a calendar unit, not a fixed "
            f"duration, so a window of it would vary in length. Use days or "
            f"hours instead."
        )
        raise ValueError(msg)
    if key not in _DURATION_UNITS:
        msg = (
            f"Unknown duration unit {unit!r} in {spec!r}. Known units: "
            f"ns, us, ms, s, m/min, h, d, w."
        )
        raise ValueError(msg)

    count = float(match["count"]) if match["count"] is not None else 1.0
    nanoseconds = round(count * _DURATION_UNITS[key])
    if nanoseconds <= 0:
        msg = f"Duration {spec!r} must be positive."
        raise ValueError(msg)
    return nanoseconds


def window_bounds(
    time: NDArray[np.int64] | None,
    window: Window,
    n_rows: int,
    *,
    center: bool = False,
    closed: Closed | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Resolve a window spec into per-row ``[start, stop)`` positions.

    Parameters
    ----------
    time
        UTC nanoseconds, strictly increasing. Required for duration windows and
        ignored for integer ones.
    window
        A positive integer count of observations, or a duration.
    n_rows
        Length of the series.
    center
        Centre the window on each row rather than trailing it. A centred
        duration window spans ``(t - w/2, t + w/2]`` and ignores ``closed``,
        matching pandas.
    closed
        Which endpoints to include. Defaults to ``"right"``.

    Returns
    -------
    tuple of numpy.ndarray
        ``(start, stop)``, each of length ``n_rows``, with
        ``0 <= start <= stop <= n_rows``.

    Raises
    ------
    ValueError
        The window is not positive, ``closed`` is unknown, or a duration window
        was requested without a time axis.
    """
    if closed is None:
        closed = "right"
    if closed not in ("right", "left", "both", "neither"):
        msg = (
            f"Unknown closed={closed!r}; expected 'right', 'left', 'both' or 'neither'."
        )
        raise ValueError(msg)

    if isinstance(window, bool):
        # bool is an int subclass, so this would otherwise mean "a window of 1".
        msg = f"A window must be a count or a duration, not {window!r}."
        raise TypeError(msg)
    if isinstance(window, int):
        if window < 1:
            msg = f"An integer window must be at least 1, got {window}."
            raise ValueError(msg)
        return _bounds_by_position(n_rows, window, center=center, closed=closed)

    span = parse_duration(window)
    if time is None:
        msg = (
            f"A duration window ({window!r}) needs a time axis; pass one, or "
            f"use an integer number of observations."
        )
        raise ValueError(msg)
    if time.shape[0] != n_rows:
        msg = f"Time axis has {time.shape[0]} entries but n_rows is {n_rows}."
        raise ValueError(msg)
    return _bounds_by_time(time, span, center=center, closed=closed)


def _bounds_by_position(
    n_rows: int, window: int, *, center: bool, closed: Closed
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Positional bounds, matching pandas' fixed-window indexer."""
    # pandas puts the extra observation of an even centred window on the left.
    offset = (window - 1) // 2 if center else 0
    stop = np.arange(1 + offset, n_rows + 1 + offset, dtype=np.int64)
    start = stop - window
    if closed == "left":
        start -= 1
        stop -= 1
    elif closed == "both":
        start -= 1
    elif closed == "neither":
        stop -= 1
    np.clip(start, 0, n_rows, out=start)
    np.clip(stop, 0, n_rows, out=stop)
    return start, np.maximum(stop, start)


def _bounds_by_time(
    time: NDArray[np.int64], span: int, *, center: bool, closed: Closed
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Bounds from timestamps, so gaps in the data shrink the window.

    ``searchsorted`` turns each row's time bounds into positions in one
    vectorised pass, which is what makes irregular sampling cost nothing extra.
    """
    if center:
        # pandas centres a duration window symmetrically and disregards
        # `closed`; mirroring that keeps results comparable.
        half = span // 2
        start = np.searchsorted(time, time - half, side="right")
        stop = np.searchsorted(time, time + half, side="right")
        return start.astype(np.int64), stop.astype(np.int64)

    # Including the left endpoint means the first equal timestamp is inside the
    # window, which is what side="left" finds.
    if closed in ("left", "both"):
        start = np.searchsorted(time, time - span, side="left")
    else:
        start = np.searchsorted(time, time - span, side="right")
    if closed in ("right", "both"):
        stop = np.arange(1, time.shape[0] + 1, dtype=np.int64)
    else:
        stop = np.searchsorted(time, time, side="left").astype(np.int64)
    return start.astype(np.int64), np.maximum(stop, start.astype(np.int64))


def default_min_periods(window: Window, closed: Closed | None = None) -> int:
    """Return pandas' default ``min_periods`` for a window spec.

    An integer window defaults to requiring a full window; a duration window
    defaults to a single observation, because its width legitimately varies.

    Parameters
    ----------
    window
        The window spec.
    closed
        Boundary rule, which changes how many observations a full integer
        window holds.

    Returns
    -------
    int
        The default minimum number of observations.
    """
    if isinstance(window, int) and not isinstance(window, bool):
        if closed == "both":
            return window + 1
        if closed == "neither":
            return max(window - 1, 1)
        return window
    return 1
