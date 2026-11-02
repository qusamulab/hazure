"""Linear component chains and their graph representation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hazure.compose._base import (
    SOURCE,
    _check_is_component,
    _check_unique_names,
    _Composite,
)
from hazure.compose._graph import Graph, Node

if TYPE_CHECKING:
    from collections.abc import Sequence

    from hazure._core.component import Component
    from hazure._core.series import TimeSeries


class Pipeline(_Composite):
    """A straight chain of components, each fed by the previous one.

    Parameters
    ----------
    steps
        ``(name, component)`` pairs, executed in order. Names must be unique and
        exist so that :meth:`summary` and error messages can point at a step.

    Raises
    ------
    ValueError
        ``steps`` is empty, or a name is repeated.

    Notes
    -----
    Steps are reachable by name for reconfiguration and grid search:
    ``set_params(score__center="mean")`` reaches the step named ``score``.

    Examples
    --------
    Chain a feature, a score and a cut-off, fit it on a clean month and use it
    as one detector on the next:

    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> from hazure.scorers import DeviationScorer
    >>> from hazure.thresholds import IqrThreshold
    >>> from hazure.transformers import SeasonalDecomposition
    >>> rng = np.random.default_rng(0)
    >>> time = np.arange("2024-01-01", "2024-02-02", dtype="datetime64[D]")
    >>> values = np.tile([1.0, 5.0, 3.0, 2.0], 8) + rng.normal(scale=0.1, size=32)
    >>> model = Pipeline(
    ...     [
    ...         ("deseasonalise", SeasonalDecomposition(period=4)),
    ...         ("score", DeviationScorer()),
    ...         ("cut", IqrThreshold(factor=3.0)),
    ...     ]
    ... ).fit(TimeSeries.from_arrays(time, values))
    >>> values[13] = 12.0
    >>> labels = model.detect(TimeSeries.from_arrays(time, values))
    >>> np.flatnonzero(labels.values.ravel() == 1.0)
    array([13])
    """

    def __init__(self, steps: Sequence[tuple[str, Component]]) -> None:
        # Stored loosely so the runtime type check below is meaningful: callers
        # reach this constructor from untyped code more often than not.
        self.steps: list[tuple[str, Any]] = list(steps)
        self._validate()

    def _validate(self) -> None:
        """Check the chain is non-empty and its names are usable."""
        if not self.steps:
            msg = "A Pipeline needs at least one step."
            raise ValueError(msg)
        _check_unique_names([name for name, _ in self.steps], "Step")
        for name, component in self.steps:
            _check_is_component(component, "Step", name)

    @property
    def terminal(self) -> Component:
        """The last step's component."""
        end: Component = self.steps[-1][1]
        return end

    def named_steps(self) -> dict[str, Component]:
        """Return the steps as a mapping, for reaching a fitted sub-component."""
        return dict(self.steps)

    def _parts(self) -> dict[str, Component]:
        return dict(self.steps)

    def _learn(self, ts: TimeSeries) -> None:
        """Fit each step on the output of the one before it."""
        self._validate()
        current = ts
        for _, component in self.steps:
            current = component.fit(current).run(current)

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        current = ts
        for _, component in self.steps:
            current = component.run(current)
        return current

    def summary(self) -> str:
        """Return a plain-text description of the chain.

        Returns
        -------
        str
            One line per step, in execution order.
        """
        width = max(len(name) for name, _ in self.steps)
        lines = [f"Pipeline: {len(self.steps)} step(s) -> {self._verbs()[0]}()"]
        lines += [
            f"  {i}. {name:<{width}}  {component!r}"
            for i, (name, component) in enumerate(self.steps, start=1)
        ]
        return "\n".join(lines)

    def to_mermaid(self) -> str:
        """Return the chain as a Mermaid flowchart.

        Text rather than a drawing, so it renders in a README, a pull request or
        a notebook without pulling in a plotting dependency.

        Returns
        -------
        str
            Mermaid ``flowchart LR`` source.
        """
        return Graph(_chain_to_nodes(self.steps)).to_mermaid()

    def __repr__(self) -> str:
        inner = ", ".join(f"({name!r}, {c!r})" for name, c in self.steps)
        return f"Pipeline([{inner}])"


def _chain_to_nodes(steps: Sequence[tuple[str, Component]]) -> list[Node]:
    """Express a linear chain as graph nodes, for rendering."""
    nodes: list[Node] = []
    previous = SOURCE
    for name, component in steps:
        nodes.append(Node(name=name, model=component, inputs=(previous,)))
        previous = name
    return nodes
