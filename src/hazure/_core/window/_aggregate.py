"""Memory-bounded reductions over explicit numpy window bounds."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Final

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


AGGREGATIONS: Final = frozenset(
    {
        "count",
        "nnz",
        "nunique",
        "sum",
        "mean",
        "median",
        "min",
        "max",
        "std",
        "var",
        "skew",
        "kurt",
        "quantile",
        "iqr",
        "idr",
    }
)


# Cap the padded window matrix at 8M float64 elements (64 MiB).
_MAX_BLOCK_ELEMENTS: Final = 8_000_000


def aggregate_windows(
    values: NDArray[np.float64],
    start: NDArray[np.int64],
    stop: NDArray[np.int64],
    agg: str,
    *,
    min_periods: int = 1,
    q: float | None = None,
) -> NDArray[np.float64]:
    """Apply a statistic to explicit window bounds.

    Missing values are skipped, as in pandas, and a row whose window holds
    fewer than ``min_periods`` observations yields NaN.

    Parameters
    ----------
    values
        1-D float array.
    start, stop
        Per-row window bounds, half-open. Same length as the result.
    agg
        A name from :data:`AGGREGATIONS`.
    min_periods
        Minimum number of non-missing observations for a result.
    q
        Quantile in ``[0, 1]``. Required when ``agg="quantile"``.

    Returns
    -------
    numpy.ndarray
        Float array of length ``len(start)``.

    Raises
    ------
    ValueError
        The aggregation is unknown, the bounds disagree in length, or ``q`` is
        missing or out of range.
    """
    if agg not in AGGREGATIONS:
        msg = f"Unknown aggregation {agg!r}. Available: {sorted(AGGREGATIONS)}."
        raise ValueError(msg)
    if start.shape != stop.shape:
        msg = f"start has shape {start.shape} but stop has {stop.shape}."
        raise ValueError(msg)
    if agg == "quantile":
        if q is None:
            msg = "agg='quantile' needs q=... in [0, 1]."
            raise ValueError(msg)
        if not 0.0 <= q <= 1.0:
            msg = f"q must lie in [0, 1], got {q}."
            raise ValueError(msg)

    n_rows = start.shape[0]
    out = np.full(n_rows, np.nan, dtype=np.float64)
    if n_rows == 0:
        return out

    widths = stop - start
    max_width = int(widths.max()) if widths.size else 0
    if max_width == 0:
        return out

    # Bound peak memory rather than window width: one chunk of the padded
    # matrix at a time.
    rows_per_chunk = max(1, _MAX_BLOCK_ELEMENTS // max_width)
    offsets = np.arange(max_width, dtype=np.int64)
    last = values.shape[0] - 1

    for lo in range(0, n_rows, rows_per_chunk):
        hi = min(lo + rows_per_chunk, n_rows)
        chunk_start = start[lo:hi]
        chunk_stop = stop[lo:hi]
        index = chunk_start[:, None] + offsets[None, :]
        inside = index < chunk_stop[:, None]
        # Clip before gathering so out-of-window slots read a valid address;
        # `inside` masks whatever they picked up.
        block = np.where(inside, values[np.minimum(index, last)], np.nan)
        out[lo:hi] = _reduce_block(block, agg, min_periods=min_periods, q=q)

    return out


def _reduce_block(
    block: NDArray[np.float64], agg: str, *, min_periods: int, q: float | None
) -> NDArray[np.float64]:
    """Reduce a padded ``(rows, width)`` window matrix to one value per row."""
    present = ~np.isnan(block)
    counts = present.sum(axis=1)
    enough = counts >= max(min_periods, 1)

    if agg == "count":
        # Deliberate divergence from pandas, which exempts count() from
        # min_periods; being consistent here is less surprising.
        return np.where(enough, counts.astype(np.float64), np.nan)
    if agg == "nnz":
        nonzero = (present & (block != 0.0)).sum(axis=1)
        return np.where(enough, nonzero.astype(np.float64), np.nan)
    if agg == "nunique":
        return np.where(enough, _nunique(block), np.nan)

    # numpy warns on all-NaN slices and on the degenerate arithmetic that
    # short windows produce. Both are expected: `enough` already decides which
    # rows are reportable.
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        result = _statistic(block, agg, counts=counts, q=q)

    return np.where(enough, result, np.nan)


def _statistic(
    block: NDArray[np.float64],
    agg: str,
    *,
    counts: NDArray[np.int_],
    q: float | None,
) -> NDArray[np.float64]:
    """Compute one statistic across the rows of a padded window matrix.

    The result is assigned to a typed local before returning because numpy's
    ``nan*`` reductions are annotated loosely enough to erase the dtype.
    """
    result: NDArray[np.float64]
    if agg == "sum":
        # A window of only NaN sums to 0 in numpy; `enough` discards those rows.
        result = np.nansum(block, axis=1)
    elif agg == "mean":
        result = np.nanmean(block, axis=1)
    elif agg == "median":
        result = np.nanmedian(block, axis=1)
    elif agg == "min":
        result = np.nanmin(block, axis=1)
    elif agg == "max":
        result = np.nanmax(block, axis=1)
    elif agg in ("var", "std"):
        # ddof=1 to match pandas; a single observation gives NaN, as it should.
        variance: NDArray[np.float64] = np.nanvar(block, axis=1, ddof=1)
        result = variance if agg == "var" else np.sqrt(variance)
    elif agg == "quantile":
        assert q is not None
        result = np.nanquantile(block, q, axis=1)
    elif agg in ("iqr", "idr"):
        edges = (0.25, 0.75) if agg == "iqr" else (0.1, 0.9)
        spread: NDArray[np.float64] = np.nanquantile(block, edges, axis=1)
        result = spread[1] - spread[0]
    else:
        result = _shape_statistic(block, agg, counts=counts)
    return result


def _shape_statistic(
    block: NDArray[np.float64], agg: str, *, counts: NDArray[np.int_]
) -> NDArray[np.float64]:
    """Compute bias-corrected skewness or excess kurtosis, as pandas defines them."""
    n = counts.astype(np.float64)
    centre: NDArray[np.float64] = np.nanmean(block, axis=1, keepdims=True)
    deviation = block - centre
    m2: NDArray[np.float64] = np.nanmean(deviation**2, axis=1)
    # A window with no spread has no shape; report NaN rather than dividing by
    # zero, as pandas does.
    m2 = np.where(m2 > 0.0, m2, np.nan)

    if agg == "skew":
        m3: NDArray[np.float64] = np.nanmean(deviation**3, axis=1)
        skew = np.sqrt(n * (n - 1.0)) * m3 / ((n - 2.0) * m2**1.5)
        return np.where(n >= 3, skew, np.nan)

    m4: NDArray[np.float64] = np.nanmean(deviation**4, axis=1)
    numerator = (n * n - 1.0) * m4 / (m2 * m2) - 3.0 * (n - 1.0) ** 2
    kurt = numerator / ((n - 2.0) * (n - 3.0))
    return np.where(n >= 4, kurt, np.nan)


def _nunique(block: NDArray[np.float64]) -> NDArray[np.float64]:
    """Count distinct non-missing values per row.

    Sorting pushes NaN to the end of each row, so distinctness reduces to
    counting neighbours that differ within the valid prefix.
    """
    ordered = np.sort(block, axis=1)
    valid = ~np.isnan(ordered)
    fresh = np.empty_like(valid)
    fresh[:, 0] = valid[:, 0]
    if ordered.shape[1] > 1:
        fresh[:, 1:] = valid[:, 1:] & (ordered[:, 1:] != ordered[:, :-1])
    distinct: NDArray[np.float64] = fresh.sum(axis=1).astype(np.float64)
    return distinct
