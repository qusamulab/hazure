"""Combining components into models.

A single detector rarely captures a real-world notion of "anomalous". More often
the answer is a small graph: extract a feature, score it, threshold it, and
require two independent signals to agree before reporting anything.

Two shapes cover almost everything:

* :class:`Pipeline` for a straight chain — the common case, and the one worth
  keeping legible.
* :class:`Graph` for anything branching or converging, expressed as named nodes
  with named inputs.

Both are themselves components, so a pipeline can be a node of a graph, and
either can be handed to anything that accepts a detector.

Wiring is checked once, when the structure is first used, and the resolved
execution order is cached until the structure changes. Intermediate results stay
as :class:`~hazure.TimeSeries` throughout, so a chain of ten steps performs one
conversion in and one out.
"""

from __future__ import annotations

from hazure.compose._base import SOURCE as SOURCE
from hazure.compose._graph import Graph, Node
from hazure.compose._pipeline import Pipeline

__all__ = ["Graph", "Node", "Pipeline"]


# Keep saved component types and pickle references on their public paths.
Graph.__module__ = __name__
Node.__module__ = __name__
Pipeline.__module__ = __name__
