"""Native dataframe factories shared across test modules."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import polars as pl

if TYPE_CHECKING:
    from collections.abc import Sequence

BACKENDS = ("pandas", "polars", "pyarrow")


def make_native(
    backend: str,
    values: Sequence[float] | np.ndarray,
    *,
    start: str = "2024-01-01",
    freq: str = "h",
    name: str = "x",
    tz: str | None = None,
) -> Any:
    """Build a single-column time series in the requested backend."""
    index = pd.date_range(start, periods=len(values), freq=freq, tz=tz, name="time")
    frame = pd.DataFrame({name: np.asarray(values, dtype=float)}, index=index)
    if backend == "pandas":
        return frame
    flat = frame.reset_index()
    if backend == "polars":
        return pl.from_pandas(flat)
    if backend == "pyarrow":
        return pl.from_pandas(flat).to_arrow()
    msg = f"Unknown backend {backend!r}."
    raise ValueError(msg)
