"""Threshold search against labelled samples or events."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast

from hazure.calibration._common import (
    _check_candidates,
    _flag,
    _grid,
    _is_events,
    _pairs,
    _unwrap,
)
from hazure.calibration._result import Calibration
from hazure.evaluation import f1_score, iou, precision, recall
from hazure.events import to_events

if TYPE_CHECKING:
    from collections.abc import Callable

    from hazure._core import TimeSeries


# Each registered objective is maximised, so delay metrics do not belong here.
_METRICS: dict[str, Callable[..., Any]] = {
    "f1": f1_score,
    "iou": iou,
    "precision": precision,
    "recall": recall,
}


def tune_threshold(
    y_true: Any,
    scores: Any,
    *,
    metric: str | Callable[..., Any] = "f1",
    candidates: int = 200,
    events: bool = True,
    **options: Any,
) -> Calibration | dict[str, Calibration]:
    """Find the cut-off that scores best against labelled history.

    Parameters
    ----------
    y_true
        Ground truth: a label series, an :class:`~hazure.events.Events`, a list of
        intervals, or a dict of those keyed by score column. A single ground
        truth given against several score columns is used for all of them.
    scores
        Continuous scores from any :class:`~hazure.Scorer`, on the same time
        axis as ``y_true``. Higher means more anomalous.
    metric
        What to maximise: ``"f1"``, ``"iou"``, ``"precision"``, ``"recall"``, or
        any callable taking ``(y_true, y_pred)`` and returning a float. Callables
        receive whatever ``events`` selected, so a custom one has to accept the
        same kind of object.
    candidates
        Roughly how many cut-offs to try. They are placed at quantiles of the
        scores rather than at evenly spaced values, half evenly and half packed
        towards the largest score, so the search resolves the extreme tail where
        a usable fence normally sits.
    events
        Score by interval rather than by sample. This is the default because it
        is the question a monitor is judged on — an outage caught late is caught,
        and a hundred alerts inside one incident are one page. Turn it off to
        weight every sample equally.
    **options
        Passed through to the metric, e.g. ``thresh=0.8`` to demand that 80% of
        an event's duration be covered before it counts as caught.

    Returns
    -------
    Calibration or dict of Calibration
        The chosen cut-off and the curve behind it, or one per score column.

    Raises
    ------
    KeyError
        ``metric`` names something not in the registry.
    TypeError
        ``metric`` is neither a name nor a callable.
    ValueError
        ``candidates`` is below 2, the scores hold nothing but ``NaN``, or
        ``y_true`` cannot be matched up with the score columns.

    See Also
    --------
    budget_threshold : The same question without labels, answered from an alert
        budget instead.

    Notes
    -----
    Ties go to the higher cut-off. A flat top to the curve is common — the metric
    cannot tell apart two fences with no scores between them — and of two equally
    good fences the higher one raises fewer alerts, so it is the one that survives
    contact with people.

    A metric that comes back undefined, as precision does when nothing at all is
    flagged, is recorded in the curve as ``nan`` and never chosen.

    None of the four metrics counts alerts, and event-based precision in particular
    does not punish a fragmented one: two alerts inside the same true event are two
    justified alerts. So a cut-off with a perfect F1 can still page twice for one
    incident. Check :func:`~hazure.events.to_events` on the result, or debounce with
    :func:`~hazure.events.expand_events`, or calibrate with :func:`budget_threshold`
    instead, which counts alerts by construction.

    The cut-off this finds is fitted to ``y_true``, and quoting the same metric at
    the same cut-off on the same data would be reporting a training score. Tune on
    one fold and measure on the next;
    :func:`~hazure.evaluation.split_train_test` exists for that.

    Examples
    --------
    A series with two planted outages, scored by deviation from the median, and a
    cut-off chosen against the intervals somebody wrote down:

    >>> import numpy as np
    >>> import pandas as pd
    >>> from hazure.scorers import DeviationScorer
    >>> index = pd.date_range("2024-01-01", periods=480, freq="h", name="time")
    >>> rng = np.random.default_rng(0)
    >>> values = pd.Series(rng.normal(0, 1, 480), index=index, name="x")
    >>> values.iloc[100:106] += 6.0
    >>> values.iloc[300:304] += 5.0
    >>> truth = pd.Series(0.0, index=index)
    >>> truth.iloc[100:106] = 1.0
    >>> truth.iloc[300:304] = 1.0
    >>> scores = DeviationScorer().fit_score(values)
    >>> best = tune_threshold(truth, scores)
    >>> best
    Calibration(cut_off=3.96769, event f1=1, over 200 candidates)

    Both outages are caught and nothing else is flagged — but the fence cuts one of
    them into two alerts, and an event-based F1 of 1.0 does not notice, because both
    fragments lie inside a true event and so both count as justified:

    >>> labels = best.threshold.apply(scores)
    >>> to_events(labels).n_events
    3

    That is worth knowing before trusting the number. Read the alert count next to
    it, or debounce with :func:`~hazure.events.expand_events` first.
    """
    _check_candidates(candidates)
    measure = _resolve_metric(metric)
    label = f"{'event' if events else 'sample'} {_name_of(metric)}"

    results = {
        name: _tune_column(truth, column, measure, candidates, events, label, options)
        for name, truth, column in _pairs(y_true, scores)
    }
    return _unwrap(results)


def _tune_column(
    y_true: Any,
    scores: TimeSeries,
    measure: Callable[..., Any],
    candidates: int,
    events: bool,
    label: str,
    options: dict[str, Any],
) -> Calibration:
    """Search one score column for the cut-off that maximises a metric.

    Parameters
    ----------
    y_true
        Ground truth for this column.
    scores
        The column, as a univariate series.
    measure
        The metric to maximise.
    candidates
        How many cut-offs to try.
    events
        Whether to score by interval rather than by sample.
    label
        Human-readable name of the objective.
    options
        Extra keyword arguments for the metric.

    Returns
    -------
    Calibration
        The best cut-off found, and the curve.
    """
    truth = to_events(y_true) if events and not _is_events(y_true) else y_true
    if events and isinstance(truth, dict):
        msg = (
            f"y_true expanded into several anomaly types {sorted(truth)} for one "
            f"score column. Pass a single label series, an Events, or a dict "
            f"keyed by score column name."
        )
        raise ValueError(msg)

    grid = _grid(scores.values[:, 0], candidates)
    curve: list[tuple[float, float]] = []
    best_cut, best_score = math.nan, -math.inf
    for cut in grid:
        guess = scores.wrap(_flag(scores.values[:, 0], cut))
        value = float(measure(truth, to_events(guess) if events else guess, **options))
        curve.append((cut, value))
        # Ties go to the higher cut-off, and the grid ascends, so >= suffices.
        if math.isfinite(value) and value >= best_score:
            best_cut, best_score = cut, value

    if not math.isfinite(best_score):
        best_score = math.nan
    return Calibration(best_cut, best_score, label, tuple(curve))


def _resolve_metric(metric: Any) -> Callable[..., Any]:
    """Look up a metric by name, or accept a callable.

    Parameters
    ----------
    metric
        A name from the registry, or a callable.

    Returns
    -------
    callable
        The metric.

    Raises
    ------
    KeyError
        The name is not in the registry.
    TypeError
        It is neither a name nor callable.
    """
    if callable(metric):
        return cast("Callable[..., Any]", metric)
    if isinstance(metric, str):
        if metric not in _METRICS:
            msg = (
                f"{metric!r} is not a metric tune_threshold knows; it accepts "
                f"{sorted(_METRICS)}, or any callable taking (y_true, y_pred)."
            )
            raise KeyError(msg)
        return _METRICS[metric]
    msg = f"metric must be a name or a callable, got {type(metric).__name__}."
    raise TypeError(msg)


def _name_of(metric: str | Callable[..., Any]) -> str:
    """Name a metric for :attr:`Calibration.objective`."""
    if isinstance(metric, str):
        return metric
    return getattr(metric, "__name__", "metric")
