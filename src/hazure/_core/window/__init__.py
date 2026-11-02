"""Rolling window statistics on numpy arrays.

hazure computes these itself rather than delegating, for three reasons. narwhals,
the dataframe-agnostic layer at the I/O boundary, offers only
``rolling_mean``/``sum``/``std``/``var`` over integer windows — no median, no
quantile, no duration windows — and those omissions are exactly what robust
detection leans on. Owning the engine keeps the core free of heavier
dependencies. And two adjacent windows, the primitive behind spike and
level-shift detection, fall out symmetrically from explicit bounds, where a
dataframe API needs the series reversed and its index mirrored to look forwards.

Semantics follow ``pandas.Series.rolling`` so results are directly comparable,
including the ``closed`` boundary rules and the fact that a centred duration
window ignores ``closed``. The one deliberate divergence is ``count``: pandas
exempts it from ``min_periods``, hazure does not. See :func:`rolling`.

Everything routes through :func:`window_bounds`, which turns a window spec into
per-row ``[start, stop)`` positions, and :func:`aggregate_windows`, which
applies a statistic to those bounds. Aggregations are computed on a padded
matrix of windows, evaluated in row chunks so peak memory stays bounded
regardless of window width.
"""

from __future__ import annotations

from hazure._core.window._aggregate import AGGREGATIONS, aggregate_windows
from hazure._core.window._bounds import Closed, Window, parse_duration, window_bounds
from hazure._core.window._bounds import default_min_periods as default_min_periods
from hazure._core.window._rolling import double_rolling, rolling

__all__ = [
    "AGGREGATIONS",
    "Closed",
    "Window",
    "aggregate_windows",
    "double_rolling",
    "parse_duration",
    "rolling",
    "window_bounds",
]
