"""Candidate grids and per-column input handling shared by calibration policies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from hazure._core import TimeSeries
from hazure.events import Events

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray

    from hazure.calibration._result import Calibration


def _check_candidates(candidates: Any) -> None:
    """Reject a search grid too coarse to be a search.

    Parameters
    ----------
    candidates
        The value passed to a calibrator.

    Raises
    ------
    TypeError
        It is not an int.
    ValueError
        It is below 2.
    """
    if not isinstance(candidates, int) or isinstance(candidates, bool):
        msg = f"candidates must be an int, got {type(candidates).__name__}."
        raise TypeError(msg)
    if candidates < 2:
        msg = (
            f"candidates must be at least 2, got {candidates}; with one candidate "
            f"there is nothing to choose between."
        )
        raise ValueError(msg)


def _grid(values: NDArray[np.float64], candidates: int) -> list[float]:
    """Place cut-offs at quantiles of the scores, packed towards the top.

    Parameters
    ----------
    values
        One score column, ``NaN`` included.
    candidates
        Roughly how many cut-offs to produce. The result may be shorter, since
        cut-offs that coincide are one cut-off.

    Returns
    -------
    list of float
        Distinct cut-offs in increasing order. The largest score is always among
        them, so the grid always contains a fence that flags nothing.

    Raises
    ------
    ValueError
        Every score is ``NaN``, so there is nothing to place a fence among.

    Notes
    -----
    Half the candidates go on evenly spaced quantiles, which is what a metric
    being maximised wants — the peak can be anywhere. The other half approach the
    maximum geometrically, which is what an alert budget needs, and it needs it
    badly: a hundred evenly spaced quantiles of ten thousand scores leave the top
    percentile as a single interval, and every fence a monitor would actually
    accept is inside it. The geometric half resolves down to the individual
    largest scores, so a budget of one alert a week is reachable rather than
    rounded up to "flag nothing".

    When there are fewer distinct scores than candidates asked for, every one of
    them is a candidate and the search is exhaustive.
    """
    observed = values[~np.isnan(values)]
    if observed.size == 0:
        msg = (
            "Every score is NaN, so there is no cut-off to choose. Check that the "
            "scorer was given enough history to fill its windows."
        )
        raise ValueError(msg)

    distinct = np.unique(observed)
    if distinct.size <= candidates:
        return [float(value) for value in distinct]

    half = candidates // 2
    even = np.linspace(0.0, 1.0, candidates - half)
    # Deep enough to isolate the single largest score, so no achievable fence is
    # out of reach; 0.5 at the shallow end, where the even half takes over.
    tail = 1.0 - np.logspace(np.log10(0.5), np.log10(1.0 / observed.size), half)
    quantiles = np.unique(np.concatenate([even, tail]))
    return [float(value) for value in np.unique(np.quantile(observed, quantiles))]


def _flag(values: NDArray[np.float64], cut: float) -> NDArray[np.float64]:
    """Label scores strictly above ``cut``, leaving missing scores unknown."""
    labels = (values > cut).astype(np.float64)
    labels[np.isnan(values)] = np.nan
    return labels


def _pairs(y_true: Any, scores: Any) -> Iterator[tuple[str | None, Any, TimeSeries]]:
    """Yield one ``(name, ground truth, score column)`` triple per column to search.

    Parameters
    ----------
    y_true
        Ground truth in any accepted form, or None when there is none.
    scores
        The score series or frame.

    Yields
    ------
    tuple
        The column name (None when there is only one column), the ground truth
        narrowed to it, and the column itself.
    """
    ts = TimeSeries.from_any(scores)
    if ts.n_columns == 1:
        yield None, y_true, ts
        return
    for name in ts.columns:
        yield name, _narrow(y_true, name, ts.columns), ts.select(name)


def _narrow(y_true: Any, name: str, columns: tuple[str, ...]) -> Any:
    """Pick out the ground truth belonging to one score column.

    Parameters
    ----------
    y_true
        Ground truth in any accepted form, or None.
    name
        The score column being calibrated.
    columns
        Every score column, for the error message.

    Returns
    -------
    object
        The ground truth for ``name``.

    Raises
    ------
    ValueError
        There is no ground truth for this column.
    """
    if y_true is None or _is_events(y_true):
        # One set of intervals stands for the whole frame: several scores of the
        # same underlying incidents is the ordinary case.
        return y_true
    if isinstance(y_true, dict):
        if name not in y_true:
            msg = (
                f"y_true has no entry for score column {name!r}; it holds "
                f"{sorted(y_true)}. Every column of {list(columns)} needs one."
            )
            raise ValueError(msg)
        return y_true[name]

    truth = TimeSeries.from_any(y_true)
    if truth.n_columns == 1:
        return truth
    if name in truth.columns:
        return truth.select(name)
    msg = (
        f"y_true carries {list(truth.columns)} and has no column {name!r} to "
        f"match the scores {list(columns)} against. Pass one label series, a "
        f"frame with the same column names, or a dict keyed by them."
    )
    raise ValueError(msg)


def _is_events(value: Any) -> bool:
    """Report whether a ground truth is interval-based rather than per-sample."""
    return isinstance(value, Events | list | tuple)


def _unwrap(
    results: dict[str | None, Calibration],
) -> Calibration | dict[str, Calibration]:
    """Return one calibration for a single column, or a dict keyed by column."""
    if len(results) == 1 and None in results:
        return results[None]
    return {str(name): value for name, value in results.items()}
