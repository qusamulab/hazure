"""A sorted, non-overlapping set of closed intervals in UTC nanoseconds."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any, cast

import narwhals.stable.v2 as nw
import numpy as np

# Private, but the same package: Events has to speak exactly the nanosecond
# dialect TimeSeries speaks, and a second copy of the datetime64 corner cases
# would be free to drift away from it.
from hazure._core.series import NO_BACKEND, Origin
from hazure._core.time import _as_epoch_ns, _from_epoch_ns

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray

__all__ = [
    "Events",
]


# Two intervals a nanosecond apart are treated as one. A nanosecond is the
# finest resolution hazure represents, so there is no instant between them: the
# gap is an artefact of storing inclusive end points, not real quiet time.
_ADJACENT: int = 1


@dataclass(frozen=True, slots=True, eq=False)
class Events:
    """A sorted, disjoint set of closed anomalous time intervals.

    Construct with :meth:`from_bounds`, :meth:`from_any` or :meth:`empty`.
    Calling ``Events(...)`` directly skips validation and is only for code that
    already holds sorted, merged bounds.

    Attributes
    ----------
    bounds
        Shape ``(n_events, 2)`` of UTC nanoseconds, ``[start, end]`` with both
        ends **inclusive**. Sorted by ``start``, and guaranteed not to overlap
        or touch. ``start == end`` marks an instantaneous event.
    origin
        Provenance borrowed from the series the events came from, so
        :meth:`to_frame` and :meth:`to_list` can hand back the caller's own
        timestamp flavour and time zone.

    Examples
    --------
    >>> events = Events.from_any([("2024-01-01T00:00", "2024-01-01T06:00")])
    >>> events
    Events([2024-01-01T00:00:00..2024-01-01T06:00:00])
    >>> events.n_events
    1
    """

    bounds: NDArray[np.int64]
    origin: Origin

    # -- construction -------------------------------------------------------

    @classmethod
    def from_bounds(
        cls,
        bounds: Any,
        *,
        origin: Origin | None = None,
    ) -> Events:
        """Validate, sort and merge ``bounds`` into an ``Events``.

        Parameters
        ----------
        bounds
            Anything array-like of shape ``(n, 2)``: integer UTC nanoseconds or
            ``datetime64`` of any unit. An empty input of any shape is accepted.
        origin
            Provenance for :meth:`to_frame` and :meth:`to_list`. Defaults to a
            pandas frame, matching ``TimeSeries.from_arrays``.

        Returns
        -------
        Events
            Sorted, disjoint events. Overlapping, nested and adjacent inputs are
            merged, so the result may be shorter than the input.

        Raises
        ------
        ValueError
            The array is not 2-D with two columns, or some ``end`` precedes its
            ``start``.
        TypeError
            The array's dtype is neither integer nor ``datetime64``.

        Examples
        --------
        Adjacent intervals collapse into one:

        >>> Events.from_bounds([[0, 9], [10, 19]]).bounds.tolist()
        [[0, 19]]
        """
        array = np.asarray(bounds)
        if array.size == 0:
            # ``[]`` arrives as shape (0,), which no reshape rule can tell apart
            # from a malformed input, so normalise it before validating.
            array = np.empty((0, 2), dtype=np.int64)
        if array.dtype.kind == "M":
            array = _as_epoch_ns(array.reshape(-1)).reshape(array.shape)
        elif array.dtype.kind not in "iu":
            msg = (
                f"Event bounds must be integer nanoseconds or datetime64, got "
                f"dtype {array.dtype}. Use Events.from_any for timestamps."
            )
            raise TypeError(msg)

        if array.ndim != 2 or array.shape[1] != 2:
            msg = (
                f"Event bounds must have shape (n, 2) of [start, end], got "
                f"shape {array.shape}."
            )
            raise ValueError(msg)

        starts = np.asarray(array[:, 0], dtype=np.int64)
        ends = np.asarray(array[:, 1], dtype=np.int64)
        backwards = np.flatnonzero(ends < starts)
        if backwards.size:
            first = int(backwards[0])
            msg = (
                f"Event {first} ends before it starts: "
                f"[{starts[first]}, {ends[first]}]. Swap the two, or pass "
                f"[t, t] for an instantaneous event."
            )
            raise ValueError(msg)

        return cls(
            bounds=_merge(starts, ends),
            origin=origin if origin is not None else Origin.default(),
        )

    @classmethod
    def from_any(cls, events: Any, *, origin: Origin | None = None) -> Events:
        """Build an ``Events`` from whatever the caller has to hand.

        Parameters
        ----------
        events
            One of:

            - an ``Events``, returned unchanged;
            - a sequence whose elements are either a timestamp (instantaneous)
              or a ``(start, end)`` pair. Timestamps may be ``datetime``,
              ``date``, ``numpy.datetime64``, a pandas ``Timestamp``, an ISO
              8601 string, or an ``int`` of UTC nanoseconds;
            - a dataframe with ``start`` and ``end`` columns;
            - an ``(n, 2)`` integer or ``datetime64`` array.
        origin
            Provenance override. Inferred from a dataframe input, otherwise
            defaults to a pandas frame.

        Returns
        -------
        Events
            Sorted, disjoint events.

        Raises
        ------
        TypeError
            The object is not one of the shapes above, or an element is not a
            recognisable timestamp.
        ValueError
            A dataframe input lacks a ``start`` or ``end`` column.

        Examples
        --------
        >>> Events.from_any(["2024-01-01T00:00"])
        Events([2024-01-01T00:00:00])
        """
        if isinstance(events, Events):
            return events
        if isinstance(events, np.ndarray) and events.dtype.kind in "iuM":
            return cls.from_bounds(events, origin=origin)

        frame = nw.from_native(events, eager_only=True, pass_through=True)
        if isinstance(frame, nw.DataFrame):
            return cls._from_frame(frame, origin=origin)

        if isinstance(events, str) or not hasattr(events, "__iter__"):
            msg = (
                f"Cannot read events from {type(events).__name__}. Pass an "
                f"Events, a list of timestamps or (start, end) pairs, or a "
                f"frame with 'start' and 'end' columns."
            )
            raise TypeError(msg)

        pairs = [_as_pair(item) for item in events]
        array = (
            np.asarray(pairs, dtype=np.int64)
            if pairs
            else np.empty((0, 2), dtype=np.int64)
        )
        return cls.from_bounds(array, origin=origin)

    @classmethod
    def _from_frame(cls, frame: nw.DataFrame[Any], *, origin: Origin | None) -> Events:
        """Read a ``start``/``end`` frame, keeping its backend and time zone."""
        missing = [c for c in ("start", "end") if c not in frame.columns]
        if missing:
            msg = (
                f"An events frame needs {missing} column(s); got {list(frame.columns)}."
            )
            raise ValueError(msg)
        if origin is None:
            zone = getattr(frame.schema["start"], "time_zone", None)
            origin = Origin(
                backend=frame.implementation.name.lower(),
                container="frame",
                time_on_index=False,
                time_name="start",
                time_unit="ns",
                time_zone=None if zone is None else str(zone),
            )
        stacked = np.column_stack(
            (
                _as_epoch_ns(frame["start"].to_numpy()),
                _as_epoch_ns(frame["end"].to_numpy()),
            )
        )
        return cls.from_bounds(stacked, origin=origin)

    @classmethod
    def empty(cls, *, origin: Origin | None = None) -> Events:
        """Return an ``Events`` holding nothing.

        Parameters
        ----------
        origin
            Provenance for :meth:`to_frame` and :meth:`to_list`.

        Returns
        -------
        Events
            An event set with ``n_events == 0``.
        """
        return cls(
            bounds=np.empty((0, 2), dtype=np.int64),
            origin=origin if origin is not None else Origin.default(),
        )

    # -- shape --------------------------------------------------------------

    @property
    def n_events(self) -> int:
        """Number of events."""
        return int(self.bounds.shape[0])

    @property
    def durations(self) -> NDArray[np.int64]:
        """Length of each event in nanoseconds, ``end - start + 1``.

        Inclusive integer bounds are half-open in effect: ``[start, end]`` in
        whole nanoseconds occupies ``[start, end + 1)`` of continuous time, so
        the ``+ 1`` counts the end nanosecond rather than being an off-by-one.

        Two things follow, and both are what makes the event-based metrics come
        out to round numbers. An instantaneous event lasts 1 ns, not 0. An event
        covering *k* consecutive samples of a series sampled every ``freq``
        lasts exactly ``k * freq``, so a duration ratio and a count of samples
        agree exactly.
        """
        return np.asarray(self.bounds[:, 1] - self.bounds[:, 0] + 1, dtype=np.int64)

    @property
    def total_duration(self) -> int:
        """Summed :attr:`durations`, in nanoseconds."""
        return int(self.durations.sum())

    def __len__(self) -> int:
        return self.n_events

    def __iter__(self) -> Iterator[tuple[int, int]]:
        """Yield ``(start, end)`` pairs of UTC nanoseconds.

        Yields
        ------
        tuple of int
            Inclusive bounds. Use :meth:`to_list` for native timestamps.
        """
        for start, end in self.bounds:
            yield int(start), int(end)

    def __repr__(self) -> str:
        if self.n_events == 0:
            return "Events([])"
        shown = [_format_interval(int(s), int(e)) for s, e in self.bounds[:3]]
        if self.n_events > 3:
            shown.append(f"... +{self.n_events - 3} more")
        return f"Events([{', '.join(shown)}])"

    def __eq__(self, other: object) -> bool:
        """Compare the event sets themselves, ignoring provenance.

        Two ``Events`` are equal when they cover the same instants. ``origin``
        is deliberately excluded: a round trip through labels and back may pick
        up a different backend without changing which instants are anomalous.
        """
        if not isinstance(other, Events):
            return NotImplemented
        return bool(np.array_equal(self.bounds, other.bounds))

    def __hash__(self) -> int:
        return hash(self.bounds.tobytes())

    # -- set operations -----------------------------------------------------

    def intersect(self, other: Events) -> Events:
        """Return the intervals covered by both this set and ``other``.

        Parameters
        ----------
        other
            Event set to intersect with.

        Returns
        -------
        Events
            Sorted, disjoint intersection, carrying this set's ``origin``.

        Raises
        ------
        TypeError
            ``other`` is not an ``Events``.

        Examples
        --------
        >>> a = Events.from_bounds([[0, 100]])
        >>> b = Events.from_bounds([[50, 200]])
        >>> a.intersect(b).bounds.tolist()
        [[50, 100]]
        """
        return self._sweep(other, min_cover=2)

    def union(self, other: Events) -> Events:
        """Return the intervals covered by either this set or ``other``.

        Parameters
        ----------
        other
            Event set to merge with.

        Returns
        -------
        Events
            Sorted, disjoint union, carrying this set's ``origin``.

        Raises
        ------
        TypeError
            ``other`` is not an ``Events``.

        Examples
        --------
        >>> a = Events.from_bounds([[0, 100]])
        >>> b = Events.from_bounds([[50, 200]])
        >>> a.union(b).bounds.tolist()
        [[0, 200]]
        """
        return self._sweep(other, min_cover=1)

    def _sweep(self, other: object, *, min_cover: int) -> Events:
        """Sweep both boundary lists once, keeping segments covered often enough.

        Each interval contributes ``+1`` at its start and ``-1`` one nanosecond
        past its end, so a cumulative sum over the sorted boundaries gives how
        many intervals cover the segment between each pair of boundaries. Both
        inputs are internally disjoint, so a coverage of 2 means "in both" and 1
        means "in either" — one routine serves intersection and union.
        """
        if not isinstance(other, Events):
            msg = (
                f"Expected an Events to combine with, got "
                f"{type(other).__name__}. Convert it with Events.from_any first."
            )
            raise TypeError(msg)

        starts = np.concatenate((self.bounds[:, 0], other.bounds[:, 0]))
        if starts.size == 0:
            return Events.empty(origin=self.origin)
        stops = np.concatenate((self.bounds[:, 1], other.bounds[:, 1])) + _ADJACENT

        points = np.concatenate((starts, stops))
        deltas = np.concatenate(
            (
                np.ones(starts.size, dtype=np.int64),
                np.full(stops.size, -1, dtype=np.int64),
            )
        )
        order = np.argsort(points, kind="stable")
        points = points[order]
        level = np.cumsum(deltas[order])

        # Coverage between two boundaries is the level *after* every delta at
        # the left boundary has been applied, so collapse ties to their last.
        last_of_run = np.empty(points.size, dtype=bool)
        last_of_run[:-1] = points[1:] != points[:-1]
        last_of_run[-1] = True
        coords = points[last_of_run]
        cover = level[last_of_run]

        # The level past the final boundary is always 0, so the segment opening
        # there is never hot and dropping it costs nothing.
        hot = cover[:-1] >= min_cover
        segments = np.column_stack((coords[:-1][hot], coords[1:][hot] - _ADJACENT))
        return Events.from_bounds(segments, origin=self.origin)

    # -- egress -------------------------------------------------------------

    def to_list(self, *, backend: str | None = None) -> list[Any]:
        """Return the events as native timestamps.

        An interval becomes a ``(start, end)`` 2-tuple; an instantaneous event
        becomes a bare timestamp, so a set of point anomalies reads as a plain
        list of the times they happened.

        Parameters
        ----------
        backend
            Emit timestamps of this backend's flavour instead of the original
            one. pandas yields ``Timestamp``; polars and pyarrow yield
            ``datetime.datetime``.

        Returns
        -------
        list
            One entry per event, in order.

        Examples
        --------
        >>> Events.from_any([0, (10, 20)]).to_list(backend="pandas")
        [Timestamp('1970-01-01 00:00:00'), (Timestamp('1970-01-01 00:00:00.000000010'), Timestamp('1970-01-01 00:00:00.000000020'))]
        """  # noqa: E501
        frame = self._to_narwhals(backend)
        starts = frame["start"].to_list()
        ends = frame["end"].to_list()
        return [
            start if start == end else (start, end)
            for start, end in zip(starts, ends, strict=True)
        ]

    def to_frame(self, *, backend: str | None = None) -> Any:
        """Return the events as a two-column ``start``/``end`` dataframe.

        Timestamps are emitted at nanosecond resolution regardless of the
        original series' unit: a merged period-based event ends one nanosecond
        short of the following sample, and a coarser unit would round that away.

        Parameters
        ----------
        backend
            Emit into this backend instead of the original one. Events built from
            raw arrays have no original backend, and default to pandas here:
            asking for a frame is asking for a dataframe library.

        Returns
        -------
        Any
            A native dataframe with columns ``start`` and ``end``.
        """
        return self._to_narwhals(backend).to_native()

    def _to_narwhals(self, backend: str | None) -> nw.DataFrame[Any]:
        """Build the ``start``/``end`` frame, time zone restored.

        Events built from raw arrays have no originating backend. Both callers of
        this method are explicit requests for native output, so pandas is the
        default there — asking for timestamps is asking for a dataframe library.
        """
        if backend is None and self.origin.backend == NO_BACKEND:
            backend = "pandas"
        payload = {
            "start": _from_epoch_ns(np.ascontiguousarray(self.bounds[:, 0]), "ns"),
            "end": _from_epoch_ns(np.ascontiguousarray(self.bounds[:, 1]), "ns"),
        }
        # Narwhals types ``backend`` as a literal union of the names it knows; an
        # unknown name raises from narwhals with a clearer message than ours.
        frame = nw.from_dict(
            payload, backend=cast("Any", backend or self.origin.backend)
        )
        if self.origin.time_zone is not None:
            # The stored instants are UTC; declare that, then shift the display
            # zone back to whatever the caller was using.
            frame = frame.with_columns(
                nw.col("start", "end")
                .dt.replace_time_zone("UTC")
                .dt.convert_time_zone(self.origin.time_zone)
            )
        return frame


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _merge(starts: NDArray[np.int64], ends: NDArray[np.int64]) -> NDArray[np.int64]:
    """Sort by start, then fuse overlapping, nested and adjacent intervals."""
    if starts.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    order = np.lexsort((ends, starts))
    starts = starts[order]
    ends = ends[order]

    # A running maximum, not just the previous end: [0, 100], [10, 20], [30, 40]
    # is one interval, and only the high-water mark knows that.
    reach = np.maximum.accumulate(ends)
    opens = np.empty(starts.size, dtype=bool)
    opens[0] = True
    opens[1:] = starts[1:] > reach[:-1] + _ADJACENT

    heads = np.flatnonzero(opens)
    return np.column_stack((starts[heads], np.maximum.reduceat(ends, heads)))


def _as_pair(item: Any) -> tuple[int, int]:
    """Read one list element as an inclusive ``(start, end)`` nanosecond pair."""
    if isinstance(item, tuple | list):
        if len(item) != 2:
            msg = (
                f"An event pair must have exactly 2 entries [start, end], got "
                f"{len(item)}: {item!r}."
            )
            raise ValueError(msg)
        return _timestamp_to_ns(item[0]), _timestamp_to_ns(item[1])
    stamp = _timestamp_to_ns(item)
    return stamp, stamp


def _timestamp_to_ns(value: Any) -> int:
    """Convert one timestamp-ish object to UTC nanoseconds."""
    if isinstance(value, np.datetime64):
        return int(value.astype("datetime64[ns]").astype(np.int64))
    if isinstance(value, int | np.integer):
        return int(value)
    # A pandas Timestamp is a datetime subclass, so ask it for the UTC instant
    # before the datetime branch below truncates its sub-microsecond digits.
    to_datetime64 = getattr(value, "to_datetime64", None)
    if callable(to_datetime64):
        return _timestamp_to_ns(to_datetime64())
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone(UTC).replace(tzinfo=None)
        return _timestamp_to_ns(np.datetime64(value))
    if isinstance(value, date | str):
        return _timestamp_to_ns(np.datetime64(value))
    msg = (
        f"Cannot read {value!r} as a timestamp. Pass a datetime, a "
        f"numpy.datetime64, a pandas Timestamp, an ISO 8601 string, or an int "
        f"of UTC nanoseconds."
    )
    raise TypeError(msg)


def _format_interval(start: int, end: int) -> str:
    """Render one event for :meth:`Events.__repr__`."""
    if start == end:
        return _format_stamp(start)
    return f"{_format_stamp(start)}..{_format_stamp(end)}"


def _format_stamp(value: int) -> str:
    """Render UTC nanoseconds as the shortest exact ISO 8601 string."""
    # A nanosecond-resolution datetime64 always renders nine fractional digits,
    # so there is always a '.' to strip back to, and no risk of eating a digit
    # from the seconds field.
    return str(np.datetime64(value, "ns")).rstrip("0").rstrip(".")
