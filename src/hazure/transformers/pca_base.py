"""The fitted subspace the three PCA transformers share."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np

from hazure._core import Transformer
from hazure._core.missing import complete_rows

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from hazure._core import TimeSeries


# ---------------------------------------------------------------------------
# principal component analysis
# ---------------------------------------------------------------------------


class _PcaBase(Transformer):
    """Shared fitting for the three PCA views of a frame.

    Every time point is a point in as many dimensions as there are columns. PCA
    finds the directions those points actually vary along, which for correlated
    sensors is far fewer directions than columns.

    Parameters
    ----------
    k
        Number of principal components to keep.

    Attributes
    ----------
    mean_ : numpy.ndarray
        Column means of the training data, one per column.
    components_ : numpy.ndarray
        The ``k`` leading right singular vectors, shape ``(k, n_columns)``,
        ordered by decreasing variance explained.
    """

    multivariate: ClassVar[bool] = True

    def __init__(self, k: int = 1) -> None:
        self.k = k

    def _learn(self, ts: TimeSeries) -> None:
        k = int(self.k)
        if k < 1:
            msg = f"k must be at least 1, got {k}."
            raise ValueError(msg)
        if k > ts.n_columns:
            msg = (
                f"k={k} exceeds the {ts.n_columns} column(s) available, so "
                f"there are not that many components to find."
            )
            raise ValueError(msg)

        complete = complete_rows(ts.values)
        matrix = ts.values[complete]
        if matrix.shape[0] < k:
            msg = (
                f"Finding {k} component(s) needs at least {k} rows with no "
                f"missing values, but the training data has "
                f"{matrix.shape[0]}."
            )
            raise ValueError(msg)

        self.mean_ = matrix.mean(axis=0)
        directions = np.linalg.svd(matrix - self.mean_, full_matrices=False)[2]
        # Singular vector signs are arbitrary; fixing the largest entry of each
        # to be positive makes the projection reproducible across platforms.
        leading = np.argmax(np.abs(directions), axis=1)
        signs = np.sign(directions[np.arange(directions.shape[0]), leading])
        oriented = directions * np.where(signs == 0.0, 1.0, signs)[:, None]
        self.components_ = np.ascontiguousarray(oriented[:k], dtype=np.float64)

    def _project(self, ts: TimeSeries) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """Return each complete row's component scores, and which rows those are."""
        if int(self.k) != self.components_.shape[0]:
            msg = (
                f"k is now {self.k} but {self.components_.shape[0]} "
                f"component(s) were fitted. Call fit() again."
            )
            raise ValueError(msg)
        complete = complete_rows(ts.values)
        scores = np.full((ts.n_rows, self.components_.shape[0]), np.nan)
        scores[complete] = (ts.values[complete] - self.mean_) @ self.components_.T
        return scores, complete

    def _reconstruct(self, scores: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the points those component scores stand for, back in column space."""
        rebuilt: NDArray[np.float64] = self.mean_ + scores @ self.components_
        return rebuilt

    def _column_error(self, ts: TimeSeries) -> NDArray[np.float64]:
        """Return the fitted subspace's squared residual for each input column."""
        squared, complete = self._squared_residuals(ts)
        parts = np.full_like(ts.values, np.nan)
        parts[complete] = squared
        return parts

    def _squared_residuals(
        self, ts: TimeSeries
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """Keep complete rows compact when only the total error is required."""
        scores, complete = self._project(ts)
        rebuilt = self._reconstruct(scores[complete])
        return (ts.values[complete] - rebuilt) ** 2, complete
