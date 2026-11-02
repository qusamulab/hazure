"""Canonical time series, array validation, selection and joins."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from hazure._core.series._native import (
    Origin,
    _describe,
    _emit_native,
    _ingest,
    _time_from_column,
    _time_from_index,
    _to_float_matrix,
)
from hazure._core.time import _as_epoch_ns, _format_ns, _infer_freq

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class TimeSeries:
    """A time-indexed block of floating point data, backend-independent.

    Instances are immutable; every operation returns a new object. Construct
    them with :meth:`from_any` rather than calling ``TimeSeries(...)`` directly,
    unless you already hold validated arrays.

    Attributes
    ----------
    time
        UTC nanoseconds since the epoch, strictly increasing.
    values
        Shape ``(len(time), len(columns))``, always ``float64``. Missing
        observations are ``NaN``; boolean labels are stored as 0.0 / 1.0 / NaN.
    columns
        Column names, unique and in the same order as ``values``.
    freq
        Nanoseconds between consecutive samples, or None when the series is
        irregular or too short to tell. Calendar-based frequencies such as
        "month start" are irregular in nanoseconds and so report None.
    origin
        Provenance used to rebuild the caller's native type.
    """

    time: NDArray[np.int64]
    values: NDArray[np.float64]
    columns: tuple[str, ...]
    freq: int | None
    origin: Origin

    # -- construction -------------------------------------------------------

    @classmethod
    def from_any(
        cls,
        data: Any,
        *,
        time: str | None = None,
        sort: bool = True,
        drop_duplicates: bool = True,
    ) -> TimeSeries:
        """Build a ``TimeSeries`` from any supported native object.

        Parameters
        ----------
        data
            A pandas Series/DataFrame with a ``DatetimeIndex``, or a polars /
            pyarrow / modin / cuDF frame containing a temporal column. Passing
            an existing ``TimeSeries`` returns it unchanged.
        time
            Name of the column holding timestamps. Required only when a frame
            has more than one temporal column; otherwise it is detected. For
            pandas input this selects a column *instead of* the index.
        sort
            Sort by time when the input is not already ordered.
        drop_duplicates
            Keep the first observation at each timestamp. When False, duplicate
            timestamps raise instead.

        Returns
        -------
        TimeSeries
            Validated series with a strictly increasing time axis.

        Raises
        ------
        TypeError
            The object is not a recognised dataframe, or its time axis is not
            temporal.
        ValueError
            Column names are duplicated, timestamps are missing, or duplicate
            timestamps were found with ``drop_duplicates=False``.
        """
        if isinstance(data, TimeSeries):
            return data

        frame, origin, index = _ingest(data, time_name=time)
        if index is not None:
            time_ns, unit, tz = _time_from_index(index)
            value_names = tuple(frame.columns)
        else:
            time_ns, unit, tz = _time_from_column(frame, origin.time_name)
            value_names = tuple(c for c in frame.columns if c != origin.time_name)
        origin = replace(origin, time_unit=unit, time_zone=tz)

        if not value_names:
            msg = (
                f"{_describe(data)} has no value columns besides {origin.time_name!r}."
            )
            raise ValueError(msg)

        values = _to_float_matrix(frame, value_names)
        return cls._assemble(
            time_ns,
            values,
            value_names,
            origin,
            sort=sort,
            drop_duplicates=drop_duplicates,
        )

    @classmethod
    def from_arrays(
        cls,
        time: NDArray[Any],
        values: NDArray[Any],
        columns: Sequence[str] | None = None,
        *,
        origin: Origin | None = None,
        sort: bool = True,
        drop_duplicates: bool = True,
    ) -> TimeSeries:
        """Build a ``TimeSeries`` from raw numpy arrays.

        Parameters
        ----------
        time
            ``datetime64`` of any unit, or ``int64`` already in UTC nanoseconds.
        values
            1-D (one column) or 2-D ``(n_rows, n_columns)``. Cast to float64.
        columns
            Column names. Defaults to ``value``/``value_0``, ``value_1``, ...
        origin
            Provenance for :meth:`to_native`. Defaults to an array-backed series
            with no native container.
        sort
            Sort by time when not already ordered.
        drop_duplicates
            Keep the first observation at each timestamp.

        Returns
        -------
        TimeSeries
            Validated series.
        """
        time_ns = _as_epoch_ns(np.asarray(time))
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix[:, None]
        elif matrix.ndim != 2:
            msg = f"values must be 1-D or 2-D, got {matrix.ndim}-D."
            raise ValueError(msg)

        if columns is None:
            names = (
                ("value",)
                if matrix.shape[1] == 1
                else tuple(f"value_{i}" for i in range(matrix.shape[1]))
            )
        else:
            names = tuple(columns)

        return cls._assemble(
            time_ns,
            matrix,
            names,
            origin or Origin.default(),
            sort=sort,
            drop_duplicates=drop_duplicates,
        )

    @classmethod
    def _assemble(
        cls,
        time_ns: NDArray[np.int64],
        values: NDArray[np.float64],
        columns: tuple[str, ...],
        origin: Origin,
        *,
        sort: bool,
        drop_duplicates: bool,
    ) -> TimeSeries:
        """Validate, order and de-duplicate, then freeze into a ``TimeSeries``."""
        _check_shape(time_ns, values, columns)
        time_ns, values = _order_and_deduplicate(
            time_ns, values, sort=sort, drop_duplicates=drop_duplicates
        )
        return cls(
            time=np.ascontiguousarray(time_ns, dtype=np.int64),
            values=np.ascontiguousarray(values, dtype=np.float64),
            columns=columns,
            freq=_infer_freq(time_ns),
            origin=origin,
        )

    # -- shape --------------------------------------------------------------

    @property
    def n_rows(self) -> int:
        """Number of observations."""
        return int(self.time.shape[0])

    @property
    def n_columns(self) -> int:
        """Number of value columns."""
        return len(self.columns)

    @property
    def is_univariate(self) -> bool:
        """True when the series carries exactly one column."""
        return len(self.columns) == 1

    def __len__(self) -> int:
        return self.n_rows

    def __repr__(self) -> str:
        freq = "irregular" if self.freq is None else _format_ns(self.freq)
        return (
            f"TimeSeries({self.n_rows} rows x {self.n_columns} cols, "
            f"freq={freq}, columns={list(self.columns)})"
        )

    # -- derivation ---------------------------------------------------------

    def wrap(
        self,
        values: NDArray[Any],
        columns: Sequence[str] | None = None,
    ) -> TimeSeries:
        """Return a new series on this time axis carrying ``values``.

        This is how scorers and transformers emit results: the time axis, freq
        and provenance are inherited, so only the numbers need supplying.

        Parameters
        ----------
        values
            1-D or 2-D array with ``n_rows`` rows.
        columns
            Names for the new columns. Defaults to this series' names when the
            width is unchanged, otherwise ``value_0``, ``value_1``, ...

        Returns
        -------
        TimeSeries
            Series sharing this time axis.
        """
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix[:, None]
        if matrix.shape[0] != self.n_rows:
            msg = (
                f"Cannot wrap {matrix.shape[0]} rows onto a time axis of {self.n_rows}."
            )
            raise ValueError(msg)

        if columns is not None:
            names = tuple(columns)
        elif matrix.shape[1] == self.n_columns:
            names = self.columns
        elif matrix.shape[1] == 1:
            names = ("value",)
        else:
            names = tuple(f"value_{i}" for i in range(matrix.shape[1]))

        if len(names) != matrix.shape[1]:
            msg = f"Got {len(names)} column names for {matrix.shape[1]} columns."
            raise ValueError(msg)
        if len(set(names)) != len(names):
            msg = f"Column names must be unique; got {list(names)}."
            raise ValueError(msg)

        # The time axis is already validated, so skip straight to the frozen
        # object rather than re-running the ordering checks.
        return TimeSeries(
            time=self.time,
            values=np.ascontiguousarray(matrix),
            columns=names,
            freq=self.freq,
            origin=self.origin,
        )

    def select(self, columns: Sequence[str] | str) -> TimeSeries:
        """Return a series restricted to ``columns``, in the order given.

        Provenance is carried through unchanged. Narrowing to one column does not
        turn a frame the caller passed into a series they did not: whether the
        result is emitted as a series depends on what arrived, not on how many
        columns happen to be left mid-computation.

        Parameters
        ----------
        columns
            One name, or a sequence of names.

        Returns
        -------
        TimeSeries
            Series carrying only the requested columns.

        Raises
        ------
        KeyError
            A requested name is not present.
        """
        names = (columns,) if isinstance(columns, str) else tuple(columns)
        missing = [c for c in names if c not in self.columns]
        if missing:
            msg = f"Unknown column(s) {missing}; available: {list(self.columns)}."
            raise KeyError(msg)
        index = [self.columns.index(c) for c in names]
        return TimeSeries(
            time=self.time,
            values=np.ascontiguousarray(self.values[:, index]),
            columns=names,
            freq=self.freq,
            origin=self.origin,
        )

    def iter_columns(self) -> Iterator[TimeSeries]:
        """Yield each column as its own univariate ``TimeSeries``.

        Yields
        ------
        TimeSeries
            One single-column series per column, in order.
        """
        for name in self.columns:
            yield self.select(name)

    def join(self, *others: TimeSeries) -> TimeSeries:
        """Outer-join other series onto this one along the time axis.

        This replaces pandas' implicit index alignment with something explicit
        and backend-independent: the union of all time axes, with NaN where a
        series had no observation.

        Parameters
        ----------
        *others
            Series to merge in. Their columns are appended in order.

        Returns
        -------
        TimeSeries
            Combined series over the union of every time axis.

        Raises
        ------
        ValueError
            Two inputs contribute the same column name.
        """
        if not others:
            return self

        parts = (self, *others)
        names = _concatenated_names(parts)

        # Provenance is carried unchanged rather than forced to "frame": joining
        # is often a step on the way back down to one column, as when an
        # aggregator combines several label series, and the caller who passed a
        # series should get a series back.
        if all(_same_axis(part.time, self.time) for part in others):
            # Every axis is identical, so the columns line up positionally.
            return TimeSeries(
                time=self.time,
                values=np.hstack([p.values for p in parts]),
                columns=tuple(names),
                freq=self.freq,
                origin=self.origin,
            )

        axis = self.time
        for part in others:
            axis = np.union1d(axis, part.time)

        merged = np.full((axis.shape[0], len(names)), np.nan, dtype=np.float64)
        offset = 0
        for part in parts:
            rows = np.searchsorted(axis, part.time)
            merged[rows, offset : offset + part.n_columns] = part.values
            offset += part.n_columns

        return TimeSeries(
            time=axis,
            values=merged,
            columns=tuple(names),
            freq=_infer_freq(axis),
            origin=self.origin,
        )

    # -- egress -------------------------------------------------------------

    def to_native(self, *, backend: str | None = None) -> Any:
        """Rebuild the caller's native object.

        pandas input comes back with its ``DatetimeIndex`` restored, including
        time zone and resolution. polars and pyarrow input come back with the
        time column in its original position and name. A single-column series
        that arrived as a pandas Series leaves as a pandas Series.

        A series built by :meth:`from_arrays` has no native counterpart, so it
        returns *itself*. Building a dataframe would mean importing a library the
        caller never asked for — numpy is the only dependency such a series has,
        and emitting it should not add one. Pass ``backend`` to opt in.

        Parameters
        ----------
        backend
            Emit into this backend instead of the original one, e.g.
            ``"pandas"`` or ``"polars"``.

        Returns
        -------
        Any
            A native dataframe or series, or this ``TimeSeries`` when it was
            built from arrays and no ``backend`` was requested.
        """
        return _emit_native(self, backend=backend)

    def to_numpy(self) -> NDArray[np.float64]:
        """Return the values as a 2-D float64 array."""
        return self.values

    def __getitem__(self, name: str) -> NDArray[np.float64]:
        """Return one column as a 1-D float64 array.

        Parameters
        ----------
        name
            Column name.

        Returns
        -------
        numpy.ndarray
            The column's values.
        """
        return self.column_values(name)

    def column_values(self, name: str) -> NDArray[np.float64]:
        """Return one column as a 1-D float64 array.

        Parameters
        ----------
        name
            Column name.

        Returns
        -------
        numpy.ndarray
            The column's values.

        Raises
        ------
        KeyError
            The name is not present.
        """
        if name not in self.columns:
            msg = f"Unknown column {name!r}; available: {list(self.columns)}."
            raise KeyError(msg)
        return self.values[:, self.columns.index(name)]


def _concatenated_names(parts: Sequence[TimeSeries]) -> list[str]:
    """Return every part's columns in order, refusing a name used twice."""
    names: list[str] = []
    for part in parts:
        for name in part.columns:
            if name in names:
                msg = f"Cannot join: column {name!r} appears in more than one series."
                raise ValueError(msg)
            names.append(name)
    return names


def _same_axis(left: NDArray[np.int64], right: NDArray[np.int64]) -> bool:
    """Report whether two time axes hold the same instants in the same order."""
    return left.shape == right.shape and bool(np.array_equal(left, right))


def _check_shape(
    time_ns: NDArray[np.int64],
    values: NDArray[np.float64],
    columns: tuple[str, ...],
) -> None:
    """Fail unless the names, the values and the time axis agree on their shape."""
    if len(columns) != values.shape[1]:
        msg = f"Got {len(columns)} column names for {values.shape[1]} columns of data."
        raise ValueError(msg)
    if len(set(columns)) != len(columns):
        duplicated = sorted({c for c in columns if columns.count(c) > 1})
        msg = f"Column names must be unique; duplicated: {duplicated}."
        raise ValueError(msg)
    if time_ns.shape[0] != values.shape[0]:
        msg = (
            f"Time axis has {time_ns.shape[0]} entries but values have "
            f"{values.shape[0]} rows."
        )
        raise ValueError(msg)


def _order_and_deduplicate(
    time_ns: NDArray[np.int64],
    values: NDArray[np.float64],
    *,
    sort: bool,
    drop_duplicates: bool,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Return the rows sorted by time and carrying one observation per instant.

    Whether being out of order or repeating an instant is an error or something
    to repair is the caller's choice, which is what ``sort`` and
    ``drop_duplicates`` express.
    """
    # np.diff on an empty or single-element array is empty, so both checks below
    # degrade to no-ops naturally.
    steps = np.diff(time_ns)
    if steps.size and bool(np.any(steps < 0)):
        if not sort:
            msg = "Time axis is not sorted; pass sort=True to reorder it."
            raise ValueError(msg)
        order = np.argsort(time_ns, kind="stable")
        time_ns = time_ns[order]
        values = values[order]
        steps = np.diff(time_ns)

    if steps.size and bool(np.any(steps == 0)):
        if not drop_duplicates:
            n_dup = int(np.count_nonzero(steps == 0))
            msg = (
                f"Time axis has {n_dup} duplicated timestamp(s); pass "
                f"drop_duplicates=True to keep the first of each."
            )
            raise ValueError(msg)
        keep = np.ones(time_ns.shape[0], dtype=bool)
        keep[1:] = steps != 0
        time_ns = time_ns[keep]
        values = values[keep]

    return time_ns, values
