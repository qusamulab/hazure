"""Directed graph nodes, wiring validation and execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from hazure._core.series import TimeSeries
from hazure.compose._base import (
    SOURCE,
    _check_is_component,
    _check_unique_names,
    _Composite,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

    from hazure._core.component import Component


@dataclass(frozen=True, slots=True)
class Node:
    """One vertex of a :class:`Graph`.

    Parameters
    ----------
    name
        Unique label for this node. ``"input"`` is reserved for the source data.
    model
        The component to run here.
    inputs
        Names of the nodes feeding this one, or ``"input"`` for the source data.
        Several inputs are joined on the time axis before the component sees
        them, as separate labelled columns.
    columns
        Optional per-input column selection, same length as ``inputs``. ``None``
        in a position means "every column from that input". Selecting a single
        column yields a univariate series, which is what a univariate component
        expects.
    """

    name: str
    model: Component
    inputs: tuple[str, ...] = (SOURCE,)
    columns: tuple[Sequence[str] | None, ...] | None = None

    def __post_init__(self) -> None:
        if self.columns is not None and len(self.columns) != len(self.inputs):
            msg = (
                f"Node {self.name!r} has {len(self.inputs)} input(s) but "
                f"{len(self.columns)} column selection(s); they must match."
            )
            raise ValueError(msg)


@dataclass
class _Plan:
    """A validated execution order, cached until the structure changes."""

    order: list[str] = field(default_factory=list)
    terminal: str = ""


class Graph(_Composite):
    """Components wired as a directed acyclic graph.

    Use this when the model branches or converges — scoring one series two ways
    and requiring both to agree, or feeding one transformed series to several
    detectors. For a straight chain, :class:`Pipeline` reads better.

    Parameters
    ----------
    nodes
        The vertices. Order is irrelevant; execution order is derived from the
        wiring. A mapping of ``name -> Node`` is also accepted, in which case the
        node's own ``name`` may be omitted by passing ``Node(name=key, ...)``.

    Raises
    ------
    ValueError
        Names repeat, a named input does not exist, the graph has a cycle, or
        more than one node is left unconsumed so the output would be ambiguous.

    Notes
    -----
    Nodes are reachable by name for reconfiguration and grid search:
    ``set_params(spike__threshold__side="positive")`` reaches the node named
    ``spike``.

    Examples
    --------
    Require two independent signals to agree:

    >>> import numpy as np
    >>> from hazure import TimeSeries, detectors
    >>> from hazure.ensemble import AndAggregator
    >>> values = np.concatenate([np.zeros(30), np.full(30, 10.0)])
    >>> time = np.arange(60) * np.timedelta64(1, "h") + np.datetime64("2024-01-01")
    >>> model = Graph(
    ...     [
    ...         Node("spike", detectors.spike(window=5)),
    ...         Node("shift", detectors.level_shift(window=5)),
    ...         Node("both", AndAggregator(), inputs=("spike", "shift")),
    ...     ]
    ... )
    >>> labels = model.fit_detect(TimeSeries.from_arrays(time, values))
    >>> np.flatnonzero(labels.values.ravel() == 1.0)
    array([30, 31, 32])
    """

    def __init__(self, nodes: Iterable[Node] | Mapping[str, Node]) -> None:
        self.nodes = list(nodes.values()) if hasattr(nodes, "values") else list(nodes)
        self._plan: _Plan | None = None
        self._resolve()

    # -- structure ----------------------------------------------------------

    def _by_name(self) -> dict[str, Node]:
        return {node.name: node for node in self.nodes}

    def _resolve(self) -> _Plan:
        """Validate the wiring and cache the execution order.

        Returns
        -------
        _Plan
            The execution order and the terminal node's name.
        """
        if self._plan is not None:
            return self._plan

        names = self._validate()
        order = self._topological_order()

        consumed = {i for node in self.nodes for i in node.inputs}
        unconsumed = [n for n in names if n not in consumed]
        if len(unconsumed) != 1:
            msg = (
                f"A Graph must have exactly one output, but {unconsumed} are "
                f"unconsumed. Feed the surplus into an aggregator, or drop it."
            )
            raise ValueError(msg)

        self._plan = _Plan(order=order, terminal=unconsumed[0])
        return self._plan

    def _validate(self) -> list[str]:
        """Check every node is nameable, wired to something, and a component.

        Returns
        -------
        list of str
            The node names, in declaration order, for the caller to plan with.

        Raises
        ------
        ValueError
            There are no nodes, a name is repeated or reserved, or a node has no
            inputs or names one that does not exist.
        TypeError
            A node holds something that is not a hazure component.
        """
        if not self.nodes:
            msg = "A Graph needs at least one node."
            raise ValueError(msg)

        names = [node.name for node in self.nodes]
        _check_unique_names(names, "Node")
        if SOURCE in names:
            msg = f"{SOURCE!r} is reserved for the source data; rename that node."
            raise ValueError(msg)

        known = set(names) | {SOURCE}
        for node in self.nodes:
            if not node.inputs:
                msg = f"Node {node.name!r} has no inputs."
                raise ValueError(msg)
            unknown = [i for i in node.inputs if i not in known]
            if unknown:
                msg = (
                    f"Node {node.name!r} names input(s) {unknown}, which do not "
                    f"exist. Available: {sorted(known)}."
                )
                raise ValueError(msg)
            _check_is_component(node.model, "Node", node.name)
        return names

    def _topological_order(self) -> list[str]:
        """Order nodes so every node follows all of its inputs."""
        done = {SOURCE}
        order: list[str] = []
        remaining = list(self.nodes)
        while remaining:
            ready = [n for n in remaining if all(i in done for i in n.inputs)]
            if not ready:
                stuck = sorted(n.name for n in remaining)
                msg = (
                    f"Node(s) {stuck} can never run: their inputs form a cycle, "
                    f"or depend on each other."
                )
                raise ValueError(msg)
            for node in ready:
                done.add(node.name)
                order.append(node.name)
                remaining.remove(node)
        return order

    @property
    def terminal(self) -> Component:
        """The component whose output is the graph's output."""
        plan = self._resolve()
        return self._by_name()[plan.terminal].model

    def _parts(self) -> dict[str, Component]:
        return {node.name: node.model for node in self.nodes}

    def named_nodes(self) -> dict[str, Component]:
        """Return the nodes as a mapping, for reaching a fitted sub-component."""
        return {node.name: node.model for node in self.nodes}

    def clone(self) -> Graph:
        """Return an unfitted copy whose nodes hold themselves fresh clones.

        Returns
        -------
        Graph
            A fresh, unfitted graph.
        """
        return Graph(
            [
                Node(
                    name=node.name,
                    model=node.model.clone(),
                    inputs=node.inputs,
                    columns=node.columns,
                )
                for node in self.nodes
            ]
        )

    # -- execution ----------------------------------------------------------

    def _gather(self, node: Node, produced: dict[str, TimeSeries]) -> TimeSeries:
        """Assemble one node's input from what its predecessors produced."""
        pieces: list[TimeSeries] = []
        for position, source in enumerate(node.inputs):
            piece = produced[source]
            wanted = None if node.columns is None else node.columns[position]
            if wanted is not None:
                piece = piece.select(wanted)
            pieces.append(piece)

        if len(pieces) == 1:
            return pieces[0]

        # Two detectors commonly emit identically named columns, so qualify each
        # contribution by the node it came from before joining.
        labelled = [
            piece.wrap(piece.values, [f"{source}__{c}" for c in piece.columns])
            for source, piece in zip(node.inputs, pieces, strict=True)
        ]
        return labelled[0].join(*labelled[1:])

    def _execute(
        self, ts: TimeSeries, step: Callable[[Node, TimeSeries], TimeSeries]
    ) -> dict[str, TimeSeries]:
        """Walk the graph in dependency order, returning what every node made.

        Fitting and running differ only in what happens at each node, so they
        share the walk and pass their own ``step`` — which keeps the source, the
        ordering and the input-gathering described in exactly one place.

        Parameters
        ----------
        ts
            The graph's input, which enters as the node named :data:`SOURCE`.
        step
            What to do at each node: :func:`_fit_and_run` or :func:`_run`.

        Returns
        -------
        dict
            Every node's output, keyed by name, including the source.
        """
        plan = self._resolve()
        lookup = self._by_name()
        produced: dict[str, TimeSeries] = {SOURCE: ts}
        for name in plan.order:
            node = lookup[name]
            produced[name] = step(node, self._gather(node, produced))
        return produced

    def _learn(self, ts: TimeSeries) -> None:
        """Fit every node in dependency order, on its own resolved input."""
        self._execute(ts, _fit_and_run)

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        return self._execute(ts, _run)[self._resolve().terminal]

    def trace(self, data: Any) -> dict[str, Any]:
        """Run the graph and return every node's output, not just the last.

        Useful for working out which branch flagged what, or for plotting the
        intermediate series while tuning.

        Parameters
        ----------
        data
            Any supported dataframe, series, or :class:`~hazure.TimeSeries`.

        Returns
        -------
        dict
            Node name mapped to that node's output in native form, plus
            ``"input"`` for the data as it entered.

        Raises
        ------
        RuntimeError
            The graph has not been fitted.
        """
        if not self.fitted:
            msg = "Fit the Graph before tracing it."
            raise RuntimeError(msg)

        produced = self._execute(TimeSeries.from_any(data), _run)
        return {name: series.to_native() for name, series in produced.items()}

    # -- description --------------------------------------------------------

    def summary(self) -> str:
        """Return a plain-text description of the wiring, in execution order.

        Returns
        -------
        str
            One line per node.
        """
        plan = self._resolve()
        lookup = self._by_name()
        width = max(len(n) for n in plan.order)
        lines = [
            f"Graph: {len(self.nodes)} node(s) -> "
            f"{plan.terminal} -> {self._verbs()[0]}()"
        ]
        for name in plan.order:
            node = lookup[name]
            wiring = ", ".join(node.inputs)
            lines.append(f"  {name:<{width}}  <- {wiring:<20}  {node.model!r}")
        return "\n".join(lines)

    def to_mermaid(self) -> str:
        """Return the wiring as a Mermaid flowchart.

        Text rather than a drawing, so it renders in a README, a pull request or
        a notebook without pulling in a plotting dependency.

        Returns
        -------
        str
            Mermaid ``flowchart LR`` source.
        """
        plan = self._resolve()
        lookup = self._by_name()
        lines = ["flowchart LR", f'  {SOURCE}(["{SOURCE}"])']
        for name in plan.order:
            node = lookup[name]
            lines.append(f'  {name}["{type(node.model).__name__}<br/>{name}"]')
        for name in plan.order:
            node = lookup[name]
            for position, source in enumerate(node.inputs):
                wanted = None if node.columns is None else node.columns[position]
                label = "" if wanted is None else f"|{', '.join(wanted)}|"
                lines.append(f"  {source} -->{label} {name}")
        lines.append(f'  {plan.terminal} --> output((["output"]))')
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Graph({len(self.nodes)} nodes, output={self._resolve().terminal!r})"

    def __iter__(self) -> Iterator[Node]:
        return iter(self.nodes)


def _fit_and_run(node: Node, incoming: TimeSeries) -> TimeSeries:
    """Fit a node's component, then run it."""
    return node.model.fit(incoming).run(incoming)


def _run(node: Node, incoming: TimeSeries) -> TimeSeries:
    """Run a node's component on its resolved input."""
    return node.model.run(incoming)
