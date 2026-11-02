"""Threshold search within an alert-rate budget."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hazure._core import parse_duration
from hazure.calibration._common import _check_candidates, _flag, _grid, _pairs, _unwrap
from hazure.calibration._result import Calibration
from hazure.events import Events, expand_events, to_events

if TYPE_CHECKING:
    from datetime import timedelta

    import numpy as np

    from hazure._core import TimeSeries


def budget_threshold(
    scores: Any,
    *,
    alerts: float = 1.0,
    per: str | timedelta | np.timedelta64 = "7d",
    gap: int | str | timedelta | np.timedelta64 | None = None,
    candidates: int = 200,
) -> Calibration | dict[str, Calibration]:
    """Find the most sensitive cut-off that stays inside an alert budget.

    No labels needed, which is the point. What is known instead is how much
    attention exists: one page a week, five a day. This lowers the fence as far as
    that allows and no further.

    Parameters
    ----------
    scores
        Continuous scores from any :class:`~hazure.Scorer`, over a stretch of
        history long enough to contain several budget periods. Higher means more
        anomalous.
    alerts
        How many alerts are affordable per ``per``. May be fractional — ``0.5``
        with ``per="7d"`` is one a fortnight.
    per
        The period the budget is expressed over: ``"7d"``, ``"1d"``, ``"1h"``, a
        :class:`~datetime.timedelta`, or a :class:`numpy.timedelta64`.
    gap
        Alerts closer together than this are one alert. Without it a flagged
        stretch broken by a single quiet sample counts twice, which overstates the
        rate a monitor would actually produce, since real ones debounce. A
        duration, or an ``int`` of nanoseconds.
    candidates
        Roughly how many cut-offs to try, placed at quantiles of the scores and
        packed towards the largest of them — an alert budget of one a week lives in
        the top fraction of a percent, which an evenly spaced grid cannot resolve.

    Returns
    -------
    Calibration or dict of Calibration
        The chosen cut-off, the alert rate it realises, and the whole rate curve,
        or one per score column. :attr:`Calibration.score` is alerts per ``per``.

    Raises
    ------
    ValueError
        ``alerts`` is not positive, ``per`` is not a positive duration,
        ``candidates`` is below 2, the scores hold nothing but ``NaN``, or the
        series is too short to measure a rate over.

    See Also
    --------
    tune_threshold : The same question when there is labelled history to aim at.
    hazure.events.expand_events : What ``gap`` is doing, available on its own.

    Notes
    -----
    The search descends. It starts at the highest candidate — which flags nothing
    and so costs nothing — and lowers the fence one candidate at a time, stopping
    at the first cut-off that breaks the budget and taking the last one that did
    not.

    Descending rather than scanning for the lowest cut-off that happens to fit is
    not a detail. Alert *count* is not monotone in the cut-off: raising a fence
    removes flagged samples, and removing one from the middle of a flagged stretch
    splits one alert into two. Taken far enough that reverses completely — a fence
    at the 1st percentile flags almost every sample, which merges into a single
    enormous alert and satisfies any budget you like. Descending until the budget
    breaks never reaches that region, because the count passes through the budget
    on the way down long before the alerts begin to merge.

    The rate is measured over the span the scores cover, so it is an estimate from
    one sample of history and inherits everything that was unusual about that
    history. A budget met over a quiet fortnight will be missed in a busy one.

    Examples
    --------
    Three weeks of a metric with a handful of excursions in it, and a budget of one
    alert a week:

    >>> import numpy as np
    >>> import pandas as pd
    >>> from hazure.scorers import DeviationScorer
    >>> index = pd.date_range("2024-01-01", periods=24 * 21, freq="h", name="time")
    >>> rng = np.random.default_rng(0)
    >>> values = pd.Series(rng.normal(0, 1, len(index)), index=index, name="rps")
    >>> for start in (40, 150, 300, 380, 460):
    ...     values.iloc[start : start + 3] += 5.0
    >>> scores = DeviationScorer().fit_score(values)
    >>> chosen = budget_threshold(scores, alerts=1, per="7d")
    >>> chosen
    Calibration(cut_off=2.88988, alerts per 7d=1, over 200 candidates)

    Three alerts over three weeks is the budget, met exactly:

    >>> from hazure.events import to_events
    >>> to_events(chosen.threshold.apply(scores)).n_events
    3

    Ask for more attention and the fence comes down:

    >>> generous = budget_threshold(scores, alerts=5, per="7d")
    >>> generous.cut_off < chosen.cut_off
    True
    """
    _check_alerts(alerts)
    _check_candidates(candidates)

    period = parse_duration(per)
    margin = _as_nanoseconds(gap)
    label = f"alerts per {per if isinstance(per, str) else _render_duration(period)}"

    results = {
        name: _budget_column(column, alerts, period, margin, candidates, label)
        for name, _, column in _pairs(None, scores)
    }
    return _unwrap(results)


def _budget_column(
    scores: TimeSeries,
    alerts: float,
    period: int,
    margin: int,
    candidates: int,
    label: str,
) -> Calibration:
    """Search one score column for the lowest cut-off inside an alert budget.

    Parameters
    ----------
    scores
        The column, as a univariate series.
    alerts
        Alerts affordable per ``period``.
    period
        The budget period, in nanoseconds.
    margin
        Alerts closer than this are merged, in nanoseconds.
    candidates
        How many cut-offs to try.
    label
        Human-readable name of the objective.

    Returns
    -------
    Calibration
        The chosen cut-off, the rate it realises, and the curve.

    Raises
    ------
    ValueError
        The series covers no time, so no rate can be measured over it.
    """
    span = _span(scores)
    periods = span / period
    allowed = alerts * periods

    grid = _grid(scores.values[:, 0], candidates)
    curve = [
        (cut, _alert_count(scores, cut, margin) / periods)
        for cut in grid  # ascending, as Calibration.curve promises
    ]

    # Descend from the fence that flags nothing, and stop where the budget breaks.
    chosen, realised = curve[-1]
    for cut, rate in reversed(curve):
        if rate * periods > allowed:
            break
        chosen, realised = cut, rate
    return Calibration(chosen, realised, label, tuple(curve))


def _alert_count(scores: TimeSeries, cut: float, margin: int) -> int:
    """Count the alerts a cut-off would have raised over this series.

    Parameters
    ----------
    scores
        A univariate score series.
    cut
        The cut-off; scores strictly above it are flagged.
    margin
        Alerts closer than this are one alert, in nanoseconds.

    Returns
    -------
    int
        Number of distinct alerts.
    """
    events = to_events(scores.wrap(_flag(scores.values[:, 0], cut)))
    if not isinstance(events, Events):  # pragma: no cover - the input is univariate
        msg = "Scoring a budget needs one score column at a time."
        raise TypeError(msg)
    if margin > 0:
        # Widening each alert by the margin and re-merging is exactly debouncing:
        # two alerts within the margin now overlap, and overlapping events merge.
        widened = expand_events(events, after=margin)
        events = widened if isinstance(widened, Events) else events
    return events.n_events


def _span(scores: TimeSeries) -> float:
    """Return the duration the series covers, in nanoseconds.

    Parameters
    ----------
    scores
        The series.

    Returns
    -------
    float
        Its span. A regular series covers one further sampling interval than its
        first and last timestamps are apart, because the last sample stands for
        the period it opens.

    Raises
    ------
    ValueError
        The series covers no time.
    """
    if scores.n_rows < 2:
        msg = (
            f"A rate cannot be measured over {scores.n_rows} observation(s). Pass "
            f"scores covering several budget periods."
        )
        raise ValueError(msg)
    span = int(scores.time[-1] - scores.time[0]) + (scores.freq or 0)
    if span <= 0:  # pragma: no cover - a sorted axis of 2+ rows always spans
        msg = "The scores cover no time, so no alert rate can be measured."
        raise ValueError(msg)
    return float(span)


def _check_alerts(alerts: Any) -> None:
    """Reject an alert budget no fence could be calibrated against.

    Parameters
    ----------
    alerts
        The value passed to :func:`budget_threshold`.

    Raises
    ------
    TypeError
        It is not a number.
    ValueError
        It is not positive.
    """
    if not isinstance(alerts, float | int) or isinstance(alerts, bool):
        msg = f"alerts must be a number, got {type(alerts).__name__}."
        raise TypeError(msg)
    if alerts <= 0.0:
        msg = (
            f"alerts must be positive, got {alerts}. A budget of zero alerts is "
            f"met by any fence high enough to flag nothing, which is not a "
            f"calibration of anything."
        )
        raise ValueError(msg)


def _as_nanoseconds(gap: Any) -> int:
    """Read the debounce margin as nanoseconds, accepting a bare int as already so.

    Parameters
    ----------
    gap
        A duration, an int of nanoseconds, or None for no debouncing.

    Returns
    -------
    int
        The margin in nanoseconds; 0 for None.
    """
    if gap is None:
        return 0
    if isinstance(gap, int) and not isinstance(gap, bool):
        return gap
    return parse_duration(gap)


def _render_duration(nanoseconds: int) -> str:
    """Render a duration for :attr:`Calibration.objective`."""
    for unit, size in (("d", 86_400), ("h", 3_600), ("min", 60), ("s", 1)):
        scale = size * 1_000_000_000
        if nanoseconds >= scale and nanoseconds % scale == 0:
            return f"{nanoseconds // scale}{unit}"
    return f"{nanoseconds}ns"
