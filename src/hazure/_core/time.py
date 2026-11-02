"""Time-axis conversion and frequency inference on numpy arrays."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


# Normalise every backend's datetime resolution onto UTC nanoseconds.
_UNIT_TO_NS: dict[str, int] = {
    "s": 1_000_000_000,
    "ms": 1_000_000,
    "us": 1_000,
    "ns": 1,
}


_TimeUnit = Literal["s", "ms", "us", "ns"]


def _as_epoch_ns(values: NDArray[Any]) -> NDArray[np.int64]:
    """Convert a datetime64 or integer array to UTC epoch nanoseconds."""
    if values.dtype.kind == "M":
        return np.asarray(values.astype("datetime64[ns]").view(np.int64))
    if values.dtype.kind in "iu":
        return np.asarray(values, dtype=np.int64)
    if values.dtype.kind == "O":
        # pyarrow hands back objects for tz-aware columns on some versions.
        return np.asarray(
            np.array(values.tolist(), dtype="datetime64[ns]").view(np.int64)
        )
    msg = (
        f"Cannot read a time axis from dtype {values.dtype}; expected "
        f"datetime64 or integer nanoseconds."
    )
    raise TypeError(msg)


def _from_epoch_ns(time_ns: NDArray[np.int64], unit: _TimeUnit) -> NDArray[Any]:
    """Convert UTC epoch nanoseconds back to datetime64 at ``unit``."""
    return time_ns.view("datetime64[ns]").astype(f"datetime64[{unit}]")


def _infer_freq(time_ns: NDArray[np.int64]) -> int | None:
    """Return the constant step in nanoseconds, or None if there isn't one.

    Three points are the minimum needed to distinguish a real frequency from a
    single arbitrary gap, matching what pandas' ``inferred_freq`` requires.
    """
    if time_ns.shape[0] < 3:
        return None
    steps = np.diff(time_ns)
    first = int(steps[0])
    if first <= 0:
        return None
    return first if bool(np.all(steps == first)) else None


def _format_ns(nanoseconds: int) -> str:
    """Render a nanosecond step as a short human-readable duration."""
    for unit, size in (("d", 86_400), ("h", 3_600), ("m", 60), ("s", 1)):
        scale = size * 1_000_000_000
        if nanoseconds >= scale and nanoseconds % scale == 0:
            return f"{nanoseconds // scale}{unit}"
    return f"{nanoseconds}ns"
