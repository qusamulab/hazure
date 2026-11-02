"""Single and adjacent-window operations over the bounds and reduction layers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np

from hazure._core.window._aggregate import aggregate_windows
from hazure._core.window._bounds import (
    Closed,
    Window,
    default_min_periods,
    window_bounds,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


_T = TypeVar("_T")


def rolling(
    values: NDArray[np.float64],
    window: Window,
    agg: str = "mean",
    *,
    time: NDArray[np.int64] | None = None,
    center: bool = False,
    min_periods: int | None = None,
    closed: Closed | None = None,
    q: float | None = None,
) -> NDArray[np.float64]:
    """Roll a window over a series and aggregate it.

    Mirrors ``pandas.Series.rolling(...).agg()``, extended with statistics
    pandas exposes only via ``apply`` (``nnz``, ``nunique``, ``iqr``, ``idr``)
    and available for duration windows on irregular data.

    Parameters
    ----------
    values
        1-D float array. NaN marks a missing observation and is skipped.
    window
        Observations (``int``) or duration (``"7d"``, ``timedelta``).
    agg
        A name from :data:`AGGREGATIONS`.
    time
        UTC nanoseconds. Required for duration windows.
    center
        Centre the window instead of trailing it.
    min_periods
        Minimum non-missing observations for a result. Defaults to the full
        window for integer windows and to 1 for duration windows, as in pandas.
    closed
        Which endpoints to include; defaults to ``"right"``. Ignored when
        ``center`` is set on a duration window, matching pandas.
    q
        Quantile in ``[0, 1]``, required when ``agg="quantile"``.

    Returns
    -------
    numpy.ndarray
        Float array the same length as ``values``.

    Notes
    -----
    Unlike pandas, ``agg="count"`` respects ``min_periods``: pandas returns the
    observation count even when it falls below ``min_periods``, which makes
    ``count`` behave unlike every other statistic.

    Examples
    --------
    >>> import numpy as np
    >>> rolling(np.array([1.0, 2.0, 3.0, 4.0]), 2, "mean")
    array([nan, 1.5, 2.5, 3.5])
    """
    series = np.asarray(values, dtype=np.float64)
    if series.ndim != 1:
        msg = f"rolling expects a 1-D array, got {series.ndim}-D."
        raise ValueError(msg)

    start, stop = window_bounds(
        time, window, series.shape[0], center=center, closed=closed
    )
    resolved = (
        default_min_periods(window, closed) if min_periods is None else min_periods
    )
    return aggregate_windows(series, start, stop, agg, min_periods=resolved, q=q)


def double_rolling(
    values: NDArray[np.float64],
    window: Window | tuple[Window, Window],
    agg: str | tuple[str, str] = "median",
    *,
    time: NDArray[np.int64] | None = None,
    diff: Literal["l1", "l2", "diff", "rel_diff", "abs_rel_diff"] = "l1",
    min_periods: int | tuple[int | None, int | None] | None = None,
    q: float | None = None,
) -> NDArray[np.float64]:
    """Compare the window before each point with the window after it.

    This is the primitive behind spike and level-shift detection: aggregate the
    recent past and the near future separately, then measure how far apart they
    are. A large gap means the series changed character at that point.

    The left window covers observations strictly before each row; the right
    window covers the row itself and what follows. Both are expressed directly
    as bounds, so unlike pandas there is no need to reverse the series or build
    a mirrored index to get a forward-looking window.

    Parameters
    ----------
    values
        1-D float array.
    window
        One spec for both sides, or ``(left, right)``. Asymmetric windows are
        how spike detection differs from level-shift detection: a long left
        window characterises "normal", a short right window catches a blip.
    agg
        One statistic for both sides, or ``(left, right)``. Median by default,
        for robustness against the very outliers being detected. ``"std"``,
        ``"iqr"`` or ``"idr"`` turn this into volatility-shift detection.
    time
        UTC nanoseconds. Required for duration windows.
    diff
        How to compare the two sides. ``"l1"`` and ``"l2"`` give the unsigned
        magnitude, ``"diff"`` the signed ``right - left``, ``"rel_diff"`` that
        divided by the left value, and ``"abs_rel_diff"`` its magnitude.
    min_periods
        One value for both sides, or ``(left, right)``.
    q
        Quantile for ``agg="quantile"``.

    Returns
    -------
    numpy.ndarray
        Float array the same length as ``values``. NaN where either side lacked
        enough observations.

    Raises
    ------
    ValueError
        ``diff`` is unknown.

    Examples
    --------
    A step change shows up as a spike in the output:

    >>> import numpy as np
    >>> series = np.array([0.0, 0.0, 0.0, 5.0, 5.0, 5.0])
    >>> double_rolling(series, 2, "mean", diff="diff")
    array([nan, nan, 2.5, 5. , 2.5, nan])
    """
    series = np.asarray(values, dtype=np.float64)
    if series.ndim != 1:
        msg = f"double_rolling expects a 1-D array, got {series.ndim}-D."
        raise ValueError(msg)
    if diff not in ("l1", "l2", "diff", "rel_diff", "abs_rel_diff"):
        msg = (
            f"Unknown diff={diff!r}; expected 'l1', 'l2', 'diff', 'rel_diff' "
            f"or 'abs_rel_diff'."
        )
        raise ValueError(msg)

    # The annotations are load-bearing: the argument types are already unions,
    # and without them `_as_pair`'s type variable widens to `object`.
    windows: tuple[Window, Window] = _as_pair(window)
    left_window, right_window = windows
    aggs: tuple[str, str] = _as_pair(agg)
    left_agg, right_agg = aggs
    mins: tuple[int | None, int | None] = _as_pair(min_periods)
    left_min, right_min = mins
    n_rows = series.shape[0]
    del window, agg, min_periods  # only the expanded pairs are used below

    # The left window trails and stops short of the current row: closed="left"
    # gives exactly `[i - w, i)`. The right window leads and includes the row.
    # Reversing the series turns that leading window back into a trailing one,
    # which is the only place the symmetry needs help.
    left_start, left_stop = window_bounds(time, left_window, n_rows, closed="left")
    left = aggregate_windows(
        series,
        left_start,
        left_stop,
        left_agg,
        min_periods=_resolve_min(left_min, left_window, "left"),
        q=q,
    )

    reversed_time = None if time is None else -time[::-1]
    right_start, right_stop = window_bounds(
        reversed_time, right_window, n_rows, closed="right"
    )
    right = aggregate_windows(
        series[::-1],
        right_start,
        right_stop,
        right_agg,
        min_periods=_resolve_min(right_min, right_window, "right"),
        q=q,
    )[::-1]

    return _compare(left, right, diff)


def _compare(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
    diff: Literal["l1", "l2", "diff", "rel_diff", "abs_rel_diff"],
) -> NDArray[np.float64]:
    """Measure how far the right window's summary sits from the left one's."""
    if diff == "l1":
        return np.abs(right - left)
    if diff == "l2":
        return np.sqrt((right - left) ** 2)
    if diff == "diff":
        return right - left
    # A left window summarising to zero has no scale to be relative to, so the
    # division is left to produce inf/NaN rather than being special-cased.
    with np.errstate(invalid="ignore", divide="ignore"):
        relative = (right - left) / left
    return relative if diff == "rel_diff" else np.abs(relative)


def _as_pair(spec: _T | tuple[_T, _T]) -> tuple[_T, _T]:
    """Expand a single value into a pair, or pass a 2-tuple through unchanged."""
    if isinstance(spec, tuple):
        if len(spec) != 2:
            msg = f"Expected one value or a 2-tuple, got {len(spec)} items."
            raise ValueError(msg)
        return spec
    return spec, spec


def _resolve_min(min_periods: int | None, window: Window, closed: Closed) -> int:
    """Fall back to the pandas default when ``min_periods`` was not given."""
    if min_periods is None:
        return default_min_periods(window, closed)
    return int(min_periods)
