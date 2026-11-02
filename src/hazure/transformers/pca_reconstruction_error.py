"""How much of each point the retained components cannot express."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from hazure.transformers.pca_base import _PcaBase

if TYPE_CHECKING:
    from hazure._core import TimeSeries


__all__ = [
    "PcaReconstructionError",
]


class PcaReconstructionError(_PcaBase):
    """Measure how far each point lies from the principal subspace.

    The output, named ``error``, is the **squared** Euclidean distance between a
    point and its reconstruction — the sum over columns of the squared residual,
    not its square root. Squaring keeps it a sum of per-column contributions and
    avoids a square root that no threshold needs.

    A point can be unremarkable in every column and still have a large error,
    because the error measures whether the columns agree with each other.

    Parameters
    ----------
    k
        Number of principal components to keep.

    Examples
    --------
    Data lying exactly on a line is reconstructed exactly:

    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> base = np.array([0.0, 1.0, 2.0, 3.0])
    >>> ts = TimeSeries.from_arrays(
    ...     np.arange("2024-01-01", "2024-01-05", dtype="datetime64[D]"),
    ...     np.column_stack([base, 2.0 * base + 1.0]),
    ...     ["a", "b"],
    ... )
    >>> PcaReconstructionError(k=1).fit_transform(ts).values.ravel().round(12)
    array([0., 0., 0., 0.])
    """

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        squared, complete = self._squared_residuals(ts)
        error = np.full(ts.n_rows, np.nan, dtype=np.float64)
        error[complete] = squared.sum(axis=1)
        return ts.wrap(error, ["error"])
