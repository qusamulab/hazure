"""Which columns a point's distance from the subspace is made of."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hazure._core import TimeSeries

__all__ = [
    "PcaColumnError",
]


from hazure.transformers.pca_base import _PcaBase


class PcaColumnError(_PcaBase):
    """Split the reconstruction error across the columns that make it up.

    The output keeps the input's column names, one column per input column, each
    holding that column's own squared residual ``(x[t, j] - x_hat[t, j]) ** 2``.
    A squared Euclidean distance is a sum over coordinates,

    ``||x - x_hat||**2 == sum over j of (x[t, j] - x_hat[t, j])**2``

    so the row sums are :class:`PcaReconstructionError` exactly, to floating
    point. That identity is the reason to have this: it is not an attribution
    heuristic bolted onto the score, it is the score's own terms written out, so
    the shares are exhaustive and a flag raised by :func:`hazure.detectors.pca`
    can be taken apart without a second model that might disagree with the first.

    What a large share does **not** establish is which column is at fault. The
    residual is the part of the point the fitted subspace had no room for, and
    the subspace places a point using every column at once. When two columns that
    normally move together disagree, the disagreement lands on **both**, divided
    by the geometry of the subspace rather than by blame; which of the two
    actually moved is not in the data. Deciding that needs a third measurement
    the pair can be checked against, and PCA was not given one.

    Nor are the columns on a common footing. The SVD is of the covariance and not
    the correlation — the training rows are centred but never scaled — so a column
    that varies in the thousands carries more absolute squared error than one that
    varies in the ones, whatever either is doing. Where the columns carry
    different units, comparing their shares compares the units as much as the
    behaviour: put :class:`StandardScale` upstream first, and the shares become
    comparable.

    Rows with any missing value are excluded from fitting and are NaN across
    every output column, as for the other PCA views. A point is either placed in
    the subspace or it is not; there is no partial residual to divide up.

    Parameters
    ----------
    k
        Number of principal components to keep.

    Examples
    --------
    Two columns that normally agree, one of which breaks for two rows, plus a
    third column unrelated to either. Three columns on a plane, so ``k=2``:

    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> from hazure.transformers import PcaReconstructionError
    >>> a = np.arange(12.0)
    >>> b = a.copy()
    >>> b[6:8] += 6.0
    >>> c = np.tile([0.0, 9.0, 3.0, 12.0], 3)
    >>> ts = TimeSeries.from_arrays(
    ...     np.arange("2024-01-01", "2024-01-13", dtype="datetime64[D]"),
    ...     np.column_stack([a, b, c]),
    ...     ["a", "b", "c"],
    ... )
    >>> parts = PcaColumnError(k=2).fit_transform(ts)
    >>> parts.columns
    ('a', 'b', 'c')
    >>> parts.values[6:8].round(2)
    array([[5.62, 3.33, 0.  ],
           [4.6 , 2.73, 0.  ]])

    The unrelated column is clean and the broken pair holds everything — but note
    where inside the pair it went: the larger share is on ``a``, which never
    moved. The decomposition localises the *relationship* that failed, not the
    column that failed it.

    And the rows add back up to the score they came from:

    >>> total = PcaReconstructionError(k=2).fit_transform(ts)
    >>> bool(np.allclose(parts.values.sum(axis=1), total.values.ravel()))
    True
    """

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        return ts.wrap(self._column_error(ts), ts.columns)
