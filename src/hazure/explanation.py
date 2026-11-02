"""Inspect the actual computations behind a fitted model's verdict."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from hazure._core import Component, Detector, TimeSeries
from hazure._core.detector import _FRAME_LABEL
from hazure._core.fanout import check_columns, join_all
from hazure.compose import Graph, Pipeline
from hazure.events.interval import _timestamp_to_ns
from hazure.scorers import AsScorer
from hazure.transformers import PcaColumnError, PcaReconstructionError

if TYPE_CHECKING:
    from hazure._core.component import OutputKind
    from hazure.compose import Node

__all__ = ["Explanation", "explain"]


@dataclass(frozen=True)
class Explanation:
    """A tree of outputs from the model that actually made the prediction.

    Attributes
    ----------
    output
        This model's result as a TimeSeries, matching ``model.run(data)``.
    kind
        Whether output is labels, scores or a transformed series.
    scores
        A detector's pre-threshold scores, or a scorer's own output. Other
        components leave this None; inspect their steps instead.
    contributions
        Squared per-column reconstruction residuals for PCA error models,
        propagated through AsScorer and Detector. They sum to the PCA score.
        Generic models leave this None rather than invent an attribution.
    steps
        Named child explanations: pipeline steps, graph nodes, detector score
        and threshold, or independently fitted input columns.

    Notes
    -----
    PCA contributions locate a broken relationship; they do not identify a
    causal culprit. Their units are squared input units. After feature
    transformations, contributions describe the transformed columns.
    """

    output: TimeSeries
    kind: OutputKind
    scores: TimeSeries | None = None
    contributions: TimeSeries | None = None
    steps: dict[str, Explanation] = field(default_factory=dict)

    def at(self, time: Any) -> dict[str, Any]:
        """Return this tree's named values at one timestamp.

        Accept the same timestamp forms as Stream.update. Raise KeyError if
        the timestamp is absent from the final output. Child values unavailable
        at that timestamp are None. Unknown numbers remain NaN.
        """
        stamp = _timestamp_to_ns(time)
        if _row(self.output, stamp) is None:
            raise KeyError(f"No prediction at {time!r}.")
        return self._snapshot(stamp)

    def _snapshot(self, stamp: int) -> dict[str, Any]:
        """Render a row without losing the hierarchy or claiming causality."""
        return {
            "kind": self.kind,
            "output": _row(self.output, stamp),
            "scores": _row(self.scores, stamp),
            "contributions": _row(self.contributions, stamp),
            "steps": {name: step._snapshot(stamp) for name, step in self.steps.items()},
        }


def explain(model: Component, data: Any) -> Explanation:
    """Trace a fitted detector, pipeline, graph or other component.

    Parameters
    ----------
    model
        Component to inspect, already fitted when training is required. No
        surrogate model is trained and no component is refitted.
    data
        Input accepted by :class:`~hazure.TimeSeries`.

    Returns
    -------
    Explanation
        Actual scores, verdicts and intermediate series, plus exact PCA error
        contributions when available. ``at(timestamp)`` inspects one decision.

    Notes
    -----
    This executes the component on the supplied data. Custom components retain
    their normal run-time behaviour, including any side effects they implement.

    Examples
    --------
    >>> from hazure.detectors import limits
    >>> ts = TimeSeries.from_arrays(np.arange(3), [1, 9, 1])
    >>> detail = explain(limits(high=5), ts)
    >>> detail.at(1)["output"]
    {'value': 1.0}
    >>> detail.at(1)["scores"]
    {'value': 9.0}
    """
    if not isinstance(model, Component):
        raise TypeError("explain needs a hazure component.")
    return _trace(model, TimeSeries.from_any(data))


def _trace(model: Component, ts: TimeSeries) -> Explanation:
    """Follow the same composition and column routing as component execution."""
    if not model.fitted:
        raise RuntimeError(f"{type(model).__name__} must be fitted before explanation.")
    ts = check_columns(model, ts)
    if not model.is_multivariate and not ts.is_univariate:
        parts = {
            name: _trace(
                model._column_models[name] if model._column_models else model,
                ts.select(name),
            )
            for name in ts.columns
        }
        return Explanation(
            _joined(parts, "output"),
            model.output_kind,
            _optional_join(parts, "scores"),
            _optional_join(parts, "contributions"),
            parts,
        )
    if isinstance(model, Graph):
        nodes: dict[str, Explanation] = {}

        def visit(node: Node, incoming: TimeSeries) -> TimeSeries:
            nodes[node.name] = _trace(node.model, incoming)
            return nodes[node.name].output

        produced = model._execute(ts, visit)
        output = produced[model._resolve().terminal]
        return Explanation(
            output,
            model.output_kind,
            scores=output if model.output_kind == "score" else None,
            steps=nodes,
        )
    if isinstance(model, Pipeline):
        steps: dict[str, Explanation] = {}
        current = ts
        for name, component in model.steps:
            steps[name] = _trace(component, current)
            current = steps[name].output
        return Explanation(
            current,
            model.output_kind,
            scores=current if model.output_kind == "score" else None,
            steps=steps,
        )
    if isinstance(model, Detector):
        steps = {}
        scores = ts
        contributions = None
        if model.scorer is not None:
            steps["score"] = _trace(model.scorer, ts)
            scores = steps["score"].output
            contributions = steps["score"].contributions
        steps["threshold"] = _trace(model.threshold, scores)
        output = steps["threshold"].output
        if model.is_multivariate and output.is_univariate:
            output = output.wrap(output.values, [_FRAME_LABEL])
        return Explanation(output, "labels", scores, contributions, steps)
    if isinstance(model, AsScorer) and not (model.is_trainable and not model.trained_):
        child = _trace(model.transformer, ts)
        return Explanation(
            child.output,
            "score",
            child.output,
            child.contributions,
            {"transformer": child},
        )
    output = model.run(ts)
    contributions = None
    if isinstance(model, PcaReconstructionError | PcaColumnError):
        contributions = ts.wrap(model._column_error(ts))
    return Explanation(
        output,
        model.output_kind,
        output if model.output_kind == "score" else None,
        contributions,
    )


def _joined(parts: dict[str, Explanation], attribute: str) -> TimeSeries:
    """Use component fan-out naming for every column's inspected result."""
    pieces: list[TimeSeries] = []
    for name, part in parts.items():
        series = getattr(part, attribute)
        if not isinstance(series, TimeSeries):
            raise TypeError(f"No {attribute} available for {name!r}.")
        if series.columns != (name,):
            series = series.wrap(series.values, [f"{name}_{c}" for c in series.columns])
        pieces.append(series)
    return join_all(pieces)


def _optional_join(parts: dict[str, Explanation], attribute: str) -> TimeSeries | None:
    """Only expose combined diagnostics when every column supplies them."""
    return (
        _joined(parts, attribute)
        if all(getattr(part, attribute) is not None for part in parts.values())
        else None
    )


def _row(series: TimeSeries | None, stamp: int) -> dict[str, float] | None:
    """Read a timestamp without silently selecting a neighbouring observation."""
    if series is None:
        return None
    position = int(np.searchsorted(series.time, stamp))
    if position == series.n_rows or series.time[position] != stamp:
        return None
    return {
        name: float(series.values[position, i]) for i, name in enumerate(series.columns)
    }
