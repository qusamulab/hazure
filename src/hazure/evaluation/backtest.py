"""Fit fresh detectors on time-ordered folds and compare held-out predictions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from hazure._core import Component, Origin, TimeSeries
from hazure._core.missing import as_flags
from hazure.alerts import AlertPolicy
from hazure.evaluation.delay import detection_delay
from hazure.evaluation.metrics import f1_score, iou, precision, recall
from hazure.evaluation.split import split_train_test
from hazure.events import Events, to_events, to_labels
from hazure.streaming import Stream

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["BacktestFold", "BacktestResult", "backtest"]

_Metric = float | dict[str, float]
_Truth = TimeSeries | Events | dict[str, TimeSeries | Events]


@dataclass(frozen=True)
class BacktestFold:
    """One model's held-out predictions and metrics for one fold.

    Attributes
    ----------
    model, fold
        Model name and zero-based fold number.
    n_train
        Number of observations used to fit this fold's fresh model.
    labels
        Held-out labels, always a backend-independent TimeSeries.
    metrics
        Alert counts and, when truth was supplied, precision, recall, F1, IoU
        and detection delay in nanoseconds. Multi-column results are keyed by
        output column. Undefined metrics remain NaN.
    notifications
        Notification pulses when an alert policy was supplied, otherwise None.
    """

    model: str
    fold: int
    n_train: int
    labels: TimeSeries
    metrics: dict[str, _Metric]
    notifications: TimeSeries | None = None


@dataclass(frozen=True)
class BacktestResult:
    """All held-out folds, with dataframe-independent comparison tables."""

    folds: tuple[BacktestFold, ...]

    def to_records(self) -> list[dict[str, Any]]:
        """Return one table row per model, fold and output column."""
        rows: list[dict[str, Any]] = []
        for result in self.folds:
            for column in result.labels.columns:
                row: dict[str, Any] = {
                    "model": result.model,
                    "fold": result.fold,
                    "column": column,
                    "n_train": result.n_train,
                    "n_test": result.labels.n_rows,
                }
                row.update(
                    {
                        metric: value[column] if isinstance(value, dict) else value
                        for metric, value in result.metrics.items()
                    }
                )
                rows.append(row)
        return rows

    def summary(self) -> list[dict[str, Any]]:
        """Compare models using unweighted means of defined fold metrics.

        Counts are summed. NaN metric values are excluded from the mean; an
        entirely undefined metric stays NaN. These are fold summaries, not pooled
        sample metrics. Modes with overlapping test blocks count those blocks
        separately, so their summed alert counts include repeated observations.
        """
        groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in self.to_records():
            groups.setdefault((row["model"], row["column"]), []).append(row)
        summaries: list[dict[str, Any]] = []
        metadata = {"model", "fold", "column", "n_train", "n_test"}
        for (model, column), rows in groups.items():
            summary: dict[str, Any] = {
                "model": model,
                "column": column,
                "n_folds": len(rows),
            }
            for metric in (key for key in rows[0] if key not in metadata):
                values = np.asarray([row[metric] for row in rows], dtype=np.float64)
                present = values[np.isfinite(values)]
                summary[metric] = (
                    float(values.sum())
                    if metric in ("n_alerts", "n_notifications")
                    else float(present.mean())
                    if present.size
                    else np.nan
                )
            summaries.append(summary)
        return summaries


def backtest(
    models: Component | Mapping[str, Component],
    data: Any,
    y_true: Any = None,
    *,
    mode: int = 3,
    n_splits: int = 3,
    train_ratio: float = 0.7,
    events: bool = True,
    history: int | str | None = None,
    alert_policy: AlertPolicy | None = None,
) -> BacktestResult:
    """Compare detectors on chronological training and test folds.

    Parameters
    ----------
    models
        A label-producing component, or a mapping of names to components.
        Each fold fits a fresh clone. Supplied models, fitted state and policies
        are left unchanged. Scorers must first be paired with a threshold.
    data
        Time series accepted by :class:`~hazure.TimeSeries`.
    y_true
        Optional labels on exactly the data's time axis, Events, or a mapping
        from output column to either. One truth column or one Events is shared
        across prediction columns; otherwise column names must match.
    mode, n_splits, train_ratio
        Arguments to :func:`split_train_test`. The default uses expanding
        training windows and disjoint held-out blocks.
    events
        Score events rather than individual samples. Interval truth is clipped
        to each test block's sampled time axis before scoring.
    history
        None evaluates each test block in batch. A row count or duration instead
        primes a :class:`~hazure.Stream` from training data and evaluates samples
        as they arrive. Its buffer checks reject forward-looking components.
    alert_policy
        Optional notification policy. A fresh policy per fold produces pulses
        and notification counts; detection metrics still score raw labels.

    Returns
    -------
    BacktestResult
        Predictions and metrics per fold, with ``to_records()`` and ``summary()``
        tables requiring no dataframe library.

    Notes
    -----
    Training never uses the held-out block. Batch evaluation can use context
    within that block, including future observations for offline algorithms;
    use ``history`` for online evaluation. Batch rolling warmup remains unknown.
    Notification history starts fresh at each fold's test boundary.

    Examples
    --------
    >>> from hazure.detectors import limits
    >>> ts = TimeSeries.from_arrays(np.arange(12), [0] * 7 + [9, 0, 9, 0, 0])
    >>> result = backtest({"limit": limits(high=5)}, ts, n_splits=2)
    >>> [row["n_alerts"] for row in result.to_records()]
    [1.0, 1.0]
    """
    named = _models(models)
    if not isinstance(events, bool):
        raise TypeError("events must be a bool.")
    if alert_policy is not None and not isinstance(alert_policy, AlertPolicy):
        raise TypeError("alert_policy must be an AlertPolicy.")
    ts = TimeSeries.from_any(data)
    truth = None if y_true is None else _prepare_truth(y_true, ts)
    # Keep split_train_test's slices canonical, including their original freq.
    canonical = replace(ts, origin=Origin.default())
    splits = split_train_test(canonical, mode, n_splits, train_ratio)
    results: list[BacktestFold] = []
    for name, template in named.items():
        for fold, (train, test) in enumerate(splits):
            fitted = template.clone().fit(train)
            if history is None:
                predicted = fitted.run(test)
            else:
                predicted = TimeSeries.from_any(
                    Stream(fitted, history).prime(train).update_many(test)
                )
            if not np.array_equal(predicted.time, test.time):
                raise ValueError(f"Model {name!r} changed the held-out time axis.")
            metrics = _metrics(truth, predicted, events)
            notifications = None
            if alert_policy is not None:
                notifications = TimeSeries.from_any(alert_policy.apply(predicted))
                metrics["n_notifications"] = _by_column(
                    notifications,
                    lambda column: float(as_flags(column.values[:, 0]).sum()),
                )
            results.append(
                BacktestFold(
                    name, fold, train.n_rows, predicted, metrics, notifications
                )
            )
    return BacktestResult(tuple(results))


def _models(models: Any) -> dict[str, Component]:
    """Require a non-empty collection of named label-producing components."""
    if isinstance(models, Component):
        models = {type(models).__name__: models}
    if not isinstance(models, Mapping) or not models:
        raise TypeError("models must be a component or a non-empty mapping.")
    for name, model in models.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Model names must be non-empty strings.")
        if not isinstance(model, Component) or model.output_kind != "labels":
            raise TypeError(
                f"Model {name!r} must produce labels; pair scorers with a threshold."
            )
    return dict(models)


def _prepare_truth(value: Any, data: TimeSeries) -> _Truth:
    """Validate global truth once, before any model is fitted."""
    if isinstance(value, Mapping):
        if not value or any(not isinstance(name, str) for name in value):
            raise ValueError("Truth mappings must name output columns.")
        prepared: dict[str, TimeSeries | Events] = {}
        for name, item in value.items():
            if isinstance(item, Mapping):
                raise TypeError("Truth mappings cannot be nested.")
            resolved = _prepare_truth(item, data)
            if isinstance(resolved, dict):
                raise TypeError("Truth mappings cannot be nested.")
            prepared[name] = resolved
        return prepared
    if isinstance(value, Events | list | tuple):
        return Events.from_any(value)
    truth = TimeSeries.from_any(value)
    if not np.array_equal(truth.time, data.time):
        raise ValueError("Truth labels must share the data's time axis.")
    return truth


def _fold_truth(truth: _Truth, prediction: TimeSeries) -> TimeSeries:
    """Align truth to a test block and its output columns without future labels."""
    if isinstance(truth, dict):
        if set(truth) != set(prediction.columns):
            raise ValueError("Truth mapping keys must match prediction columns.")
        columns = [
            _fold_truth(truth[name], prediction.select(name)).values[:, 0]
            for name in prediction.columns
        ]
        return prediction.wrap(np.column_stack(columns))
    if isinstance(truth, Events):
        values = TimeSeries.from_any(to_labels(truth, prediction)).values
    else:
        positions = np.searchsorted(truth.time, prediction.time)
        values = truth.values[positions]
        if truth.n_columns > 1:
            if set(truth.columns) != set(prediction.columns):
                raise ValueError("Truth and prediction columns must match.")
            values = truth.select(prediction.columns).values[positions]
    if values.shape[1] == 1 and prediction.n_columns > 1:
        values = np.repeat(values, prediction.n_columns, axis=1)
    return prediction.wrap(values)


def _metrics(
    truth: _Truth | None, prediction: TimeSeries, events: bool
) -> dict[str, _Metric]:
    """Score raw labels and count their distinct anomalous intervals."""
    result: dict[str, _Metric] = {"n_alerts": _by_column(prediction, _count_alerts)}
    if truth is None:
        return result
    actual = _fold_truth(truth, prediction)
    left, right = (
        (to_events(actual), to_events(prediction)) if events else (actual, prediction)
    )
    for name, measure in (
        ("precision", precision),
        ("recall", recall),
        ("f1", f1_score),
        ("iou", iou),
        ("detection_delay_ns", detection_delay),
    ):
        result[name] = measure(left, right)
    return result


def _by_column(series: TimeSeries, measure: Callable[[TimeSeries], float]) -> _Metric:
    """Use the metric convention: one scalar, or a dict for several columns."""
    values = {name: measure(series.select(name)) for name in series.columns}
    return next(iter(values.values())) if series.n_columns == 1 else values


def _count_alerts(column: TimeSeries) -> float:
    """Count intervals from exactly one output column."""
    events = to_events(column)
    if not isinstance(events, Events):
        raise TypeError("Alert counts need one label column at a time.")
    return float(events.n_events)
