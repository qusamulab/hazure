"""The dataframe boundary.

Everything hazure computes happens on plain numpy arrays. ``TimeSeries`` is the
single place where a caller's pandas / polars / pyarrow object is turned into
those arrays, and the single place where results are turned back into whatever
the caller handed us.

Keeping that translation in one class buys three things that the design depends
on:

* **The time axis is data, not metadata.** Internally there is no index — just a
  sorted ``int64`` array of UTC nanoseconds. Combining two series is an explicit
  join on that array rather than an implicit pandas index alignment, so polars
  and pandas cannot silently disagree.
* **Sampling frequency is state on the object.** It is inferred once, on ingest,
  and then carried. Nothing downstream re-derives it from whatever frame it
  happens to be holding, which is what makes ``to_events`` and friends behave
  identically on every backend.
* **Round-tripping is total.** Time unit, time zone, and whether the timestamps
  arrived on an index or in a column are all remembered, so ``to_native``
  reproduces the caller's shape instead of an approximation of it.
"""

from __future__ import annotations

from hazure._core.series._model import TimeSeries
from hazure._core.series._native import NO_BACKEND as NO_BACKEND
from hazure._core.series._native import Origin as Origin

__all__ = ["TimeSeries"]


# Preserve import paths used by existing pickles and introspection.
TimeSeries.__module__ = __name__
Origin.__module__ = __name__
