"""Native dataframe ingestion, provenance and result reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import narwhals.stable.v2 as nw
import numpy as np

from hazure._core.time import (
    _UNIT_TO_NS,
    _as_epoch_ns,
    _from_epoch_ns,
    _TimeUnit,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from hazure._core.series._model import TimeSeries


# Array-backed series have no native container and must not import a backend.
NO_BACKEND = "arrays"


@dataclass(frozen=True, slots=True)
class Origin:
    """How the caller's data was shaped, so we can hand back the same shape.

    Attributes
    ----------
    backend
        Narwhals backend name the data came from, e.g. ``"pandas"``.
    container
        Whether the caller passed a 1-D series or a 2-D frame.
    time_on_index
        True when the timestamps arrived on a pandas-style index rather than in
        a column. Determines whether :meth:`TimeSeries.to_native` restores an
        index or emits a time column.
    time_name
        Name of the index or column that carried the timestamps.
    time_unit
        Resolution of the original timestamps. pandas 3 and polars both default
        to microseconds, so this is rarely ``"ns"``.
    time_zone
        IANA zone name of the original timestamps, or None if they were naive.
    index_name
        Name the pandas index had, which may be None. Tracked separately from
        ``time_name`` because resetting an unnamed index invents a placeholder
        name, and we want to hand back the caller's unnamed index verbatim.
    """

    backend: str
    container: Literal["series", "frame"]
    time_on_index: bool
    time_name: str
    time_unit: _TimeUnit
    time_zone: str | None
    index_name: str | None = None

    @classmethod
    def default(cls) -> Origin:
        """Return the origin used for series built directly from numpy arrays.

        Its backend is :data:`NO_BACKEND`, because there was no native object to
        come back to. Emitting such a series returns the ``TimeSeries`` itself
        rather than reaching for a dataframe library the caller may not have
        installed — see :meth:`TimeSeries.to_native`.
        """
        return cls(
            backend=NO_BACKEND,
            container="frame",
            time_on_index=True,
            time_name="time",
            time_unit="ns",
            time_zone=None,
            index_name="time",
        )

    @classmethod
    def on_index(
        cls, backend: str, container: Literal["series", "frame"], index: Any
    ) -> Origin:
        """Return the origin for data whose timestamps arrived on a pandas index.

        ``time_unit`` and ``time_zone`` are left at their defaults: reading the
        index fills them in, and only reading it can.
        """
        return cls(
            backend=backend,
            container=container,
            time_on_index=True,
            time_name=index.name if index.name else "time",
            time_unit="ns",
            time_zone=None,
            index_name=index.name,
        )


def _ingest(
    data: Any, *, time_name: str | None
) -> tuple[nw.DataFrame[Any], Origin, Any]:
    """Normalise a native object into a value frame, provenance, and time index.

    The third element is the pandas index carrying the timestamps, or None when
    the timestamps live in a column of the frame instead. Narwhals deliberately
    has no index concept — ``maybe_reset_index`` discards an index rather than
    promoting it to a column — so an index-borne time axis is read straight off
    the pandas object and never enters the frame.
    """
    series = nw.from_native(data, series_only=True, pass_through=True)
    if isinstance(series, nw.Series):
        return _ingest_series(series, source=data)

    frame = nw.from_native(data, eager_only=True, pass_through=True)
    if not isinstance(frame, nw.DataFrame):
        msg = (
            f"{_describe(data)} is not a supported dataframe. Pass a pandas, "
            f"polars or pyarrow object, or use TimeSeries.from_arrays."
        )
        raise TypeError(msg)
    return _ingest_frame(frame, source=data, time_name=time_name)


def _ingest_series(
    series: nw.Series[Any], *, source: Any
) -> tuple[nw.DataFrame[Any], Origin, Any]:
    """Normalise a 1-D series, whose time axis can only be its index."""
    index = nw.maybe_get_index(series)
    if index is None or not _is_temporal_index(index):
        msg = (
            f"{_describe(source)} carries no time axis. Give it a "
            f"DatetimeIndex, or pass a dataframe with a temporal column."
        )
        raise TypeError(msg)
    name = series.name if series.name else "value"
    origin = Origin.on_index(series.implementation.name.lower(), "series", index)
    return series.rename(name).to_frame(), origin, index


def _ingest_frame(
    frame: nw.DataFrame[Any], *, source: Any, time_name: str | None
) -> tuple[nw.DataFrame[Any], Origin, Any]:
    """Normalise a 2-D frame, whose time axis may be an index or a column."""
    backend = frame.implementation.name.lower()

    # A pandas frame keeps its time on the index unless the caller pointed at a
    # column explicitly; polars and pyarrow always carry it in a column.
    index = nw.maybe_get_index(frame)
    if (
        index is not None
        and _is_temporal_index(index)
        and (time_name is None or time_name not in frame.columns)
    ):
        return frame, Origin.on_index(backend, "frame", index), index

    resolved = _resolve_time_column(
        frame, requested=time_name, source=_describe(source)
    )
    origin = Origin(
        backend=backend,
        container="frame",
        time_on_index=False,
        time_name=resolved,
        time_unit="ns",
        time_zone=None,
        index_name=resolved,
    )
    return frame, origin, None


def _is_temporal_index(index: Any) -> bool:
    """Report whether a pandas index holds datetimes rather than labels."""
    return bool(getattr(index.dtype, "kind", None) == "M")


def _resolve_time_column(
    frame: nw.DataFrame[Any], *, requested: str | None, source: str
) -> str:
    """Pick the column holding timestamps, or explain why we cannot."""
    schema = frame.schema
    if requested is not None:
        if requested not in schema:
            msg = f"{source} has no column {requested!r}; available: {list(schema)}."
            raise ValueError(msg)
        if not schema[requested].is_temporal():
            msg = (
                f"Column {requested!r} of {source} is {schema[requested]}, "
                f"not a temporal type."
            )
            raise TypeError(msg)
        return requested

    temporal = [name for name, dtype in schema.items() if dtype.is_temporal()]
    if not temporal:
        msg = (
            f"{source} has no temporal column to use as a time axis; "
            f"columns are {list(schema)}."
        )
        raise TypeError(msg)
    if len(temporal) > 1:
        msg = (
            f"{source} has several temporal columns {temporal}; pass "
            f"time=... to choose one."
        )
        raise ValueError(msg)
    return temporal[0]


def _time_from_index(index: Any) -> tuple[NDArray[np.int64], _TimeUnit, str | None]:
    """Read a pandas DatetimeIndex as UTC nanoseconds, with its unit and zone.

    ``asi8`` is used rather than ``np.asarray`` because a time-zone-aware index
    converts to an *object* array of Timestamps, whereas ``asi8`` is always
    UTC integers at the index's own resolution.
    """
    dtype = index.dtype
    zone = getattr(dtype, "tz", None)
    unit = _unit_of(dtype)
    raw = np.asarray(index.asi8, dtype=np.int64)
    _reject_missing(raw, f"index {index.name!r}" if index.name else "the index")
    return raw * _UNIT_TO_NS[unit], unit, None if zone is None else str(zone)


def _time_from_column(
    frame: nw.DataFrame[Any], name: str
) -> tuple[NDArray[np.int64], _TimeUnit, str | None]:
    """Read a temporal column as UTC nanoseconds, with its unit and zone."""
    dtype = frame.schema[name]
    unit: str = getattr(dtype, "time_unit", None) or "ns"
    zone = getattr(dtype, "time_zone", None)
    if unit not in _UNIT_TO_NS:  # pragma: no cover - guards future units
        msg = f"Unsupported time unit {unit!r} on column {name!r}."
        raise TypeError(msg)

    time_ns = _as_epoch_ns(frame[name].to_numpy())
    _reject_missing(time_ns, f"column {name!r}")
    return time_ns, unit, None if zone is None else str(zone)  # type: ignore[return-value]


def _unit_of(dtype: Any) -> _TimeUnit:
    """Extract the resolution of a pandas datetime dtype.

    A time-zone-aware dtype exposes ``.unit`` directly; a naive one is a plain
    numpy dtype whose resolution is only visible in its repr.
    """
    unit = getattr(dtype, "unit", None)
    if unit is None:
        text = str(dtype)
        unit = text[text.index("[") + 1 : text.index("]")] if "[" in text else "ns"
    if unit not in _UNIT_TO_NS:  # pragma: no cover - guards future units
        msg = f"Unsupported time resolution {unit!r}."
        raise TypeError(msg)
    return unit  # type: ignore[return-value]


def _reject_missing(raw: NDArray[np.int64], where: str) -> None:
    """Fail loudly on NaT rather than letting int64 sentinels leak downstream."""
    if bool(np.any(raw == np.iinfo(np.int64).min)):
        msg = (
            f"Time axis in {where} contains missing timestamps; drop or fill "
            f"them before detection."
        )
        raise ValueError(msg)


def _to_float_matrix(
    frame: nw.DataFrame[Any], names: Sequence[str]
) -> NDArray[np.float64]:
    """Cast the named columns to float64 and stack them column-wise."""
    schema = frame.schema
    bad = [n for n in names if schema[n].is_temporal() or schema[n] == nw.String]
    if bad:
        msg = (
            f"Column(s) {bad} are not numeric. Encode them before detection "
            f"(e.g. one-hot for categoricals)."
        )
        raise TypeError(msg)
    casted = frame.select(nw.col(n).cast(nw.Float64) for n in names)
    return np.asarray(casted.to_numpy(), dtype=np.float64)


def _describe(data: Any) -> str:
    """Name an object well enough to make an error message actionable."""
    return f"{type(data).__module__}.{type(data).__qualname__}"


def _emit_native(ts: TimeSeries, *, backend: str | None) -> Any:
    """Reconstruct the native container recorded by a series' origin."""
    origin = ts.origin
    target = backend or origin.backend
    if target == NO_BACKEND:
        return ts

    stamps = _from_epoch_ns(ts.time, origin.time_unit)
    payload: dict[str, NDArray[Any]] = {origin.time_name: stamps}
    for i, name in enumerate(ts.columns):
        payload[name] = ts.values[:, i]

    # Narwhals types `backend` as a literal union of the names it knows.
    # An unknown name raises from narwhals with a clearer message than any
    # check we could add here, so pass it straight through.
    frame = nw.from_dict(payload, backend=cast("Any", target))
    if origin.time_zone is not None:
        # The stored instants are UTC; declare that, then shift the display
        # zone back to whatever the caller was using.
        frame = frame.with_columns(
            nw.col(origin.time_name)
            .dt.replace_time_zone("UTC")
            .dt.convert_time_zone(origin.time_zone)
        )

    if origin.time_on_index:
        frame = nw.maybe_set_index(frame, column_names=[origin.time_name])

    native = frame.to_native()
    if origin.time_on_index and origin.index_name != origin.time_name:
        # Restoring an index that was originally unnamed: narwhals has no
        # index concept to express this through, and the object is one we
        # just built, so naming it directly is safe.
        native.index.name = origin.index_name
    if origin.container == "series" and ts.n_columns == 1:
        return native[ts.columns[0]]
    return native
