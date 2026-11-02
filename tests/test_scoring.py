"""Tests for the scorers.

A score is only useful if the largest one lands on the anomaly, so most of these
plant a known anomaly at a known position and assert that the score peaks exactly
there. Where a window makes the peak a plateau, the plateau is asserted in full
rather than glossed over.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from hazure import Component, Scorer, TimeSeries
from hazure.scorers import (
    AsScorer,
    AutoregressionResidualScorer,
    DeviationScorer,
    MinClusterScorer,
    OutlierScorer,
)
from hazure.transformers import (
    DoubleRollingAggregate,
    PcaReconstructionError,
    RegressionResidual,
    RollingAggregate,
    SeasonalDecomposition,
)
from tests.helpers import BACKENDS, make_native

# -- helpers ----------------------------------------------------------------


def series(values: Any) -> TimeSeries:
    """Build a daily univariate series."""
    numbers = np.asarray(values, dtype=float)
    time = np.arange(
        np.datetime64("2024-01-01"),
        np.datetime64("2024-01-01") + len(numbers),
        dtype="datetime64[D]",
    )
    return TimeSeries.from_arrays(time, numbers)


def frame(**columns: Any) -> TimeSeries:
    """Build a daily multivariate series from named columns."""
    names = list(columns)
    stacked = np.column_stack([np.asarray(columns[n], dtype=float) for n in names])
    time = np.arange(
        np.datetime64("2024-01-01"),
        np.datetime64("2024-01-01") + stacked.shape[0],
        dtype="datetime64[D]",
    )
    return TimeSeries.from_arrays(time, stacked, names)


def scored(scorer: Scorer, ts: TimeSeries) -> np.ndarray:
    """Fit if needed, score, and return a flat array."""
    if scorer.is_trainable:
        scorer.fit(ts)
    return np.asarray(scorer.run(ts).values).ravel()


UNIVARIATE_SCORERS = (
    AsScorer(RollingAggregate(window=3)),
    AsScorer(DoubleRollingAggregate(window=3, agg="median")),
    DeviationScorer(),
    AutoregressionResidualScorer(n_steps=2),
)

MULTIVARIATE_SCORERS = (
    AsScorer(RegressionResidual(target="b")),
    AsScorer(PcaReconstructionError(k=1)),
)


class NearestOfTwo:
    """A minimal generalising clusterer: split on the training mean."""

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.split_ = float(X.mean())
        return self.predict(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (X.mean(axis=1) > self.split_).astype(int)


class Transductive:
    """A clusterer that can only label the batch it was given."""

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[0], dtype=int)


class FarFromCentre:
    """A minimal outlier model with fit and predict, marking outliers with -1."""

    def fit(self, X: np.ndarray) -> FarFromCentre:
        self.centre_ = np.median(X, axis=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(np.abs(X - self.centre_).sum(axis=1) > 5.0, -1, 1)


class BatchOnly:
    """An outlier model that only offers fit_predict, judging a batch on itself."""

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        distance = np.abs(X - np.median(X, axis=0)).sum(axis=1)
        return np.where(distance > 3.0 * np.median(distance), -1, 1)


# -- RollingAggregateScorer -------------------------------------------------


def test_a_rolling_aggregate_scorer_summarises_the_window_ending_at_each_point() -> (
    None
):
    ts = series([1.0, 1.0, 1.0, 9.0, 1.0, 1.0])
    assert list(
        AsScorer(RollingAggregate(window=2, agg="max")).score(ts).values.ravel()
    )[1:] == [1.0, 1.0, 9.0, 9.0, 1.0]


def test_a_rolling_aggregate_scorer_needs_no_fitting() -> None:
    assert AsScorer(RollingAggregate(window=2)).fitted


def test_a_rolling_aggregate_scorer_reports_nothing_for_a_short_window() -> None:
    scores = AsScorer(RollingAggregate(window=3, agg="mean")).score(
        series(np.arange(5.0))
    )
    assert np.isnan(scores.values.ravel()[:2]).all()


def test_a_rolling_aggregate_scorer_can_be_centred() -> None:
    ts = series([0.0, 0.0, 3.0, 0.0, 0.0])
    trailing = AsScorer(RollingAggregate(window=3, agg="max")).score(ts).values.ravel()
    centred = (
        AsScorer(RollingAggregate(window=3, agg="max", center=True))
        .score(ts)
        .values.ravel()
    )
    assert int(np.nanargmax(trailing)) == 2
    assert list(centred[1:4]) == [3.0, 3.0, 3.0]


def test_a_rolling_aggregate_scorer_passes_a_quantile_through() -> None:
    ts = series([1.0, 2.0, 3.0, 4.0])
    scores = AsScorer(
        RollingAggregate(window=4, agg="quantile", agg_params={"q": 0.5})
    ).score(ts)
    assert scores.values.ravel()[-1] == pytest.approx(2.5)


def test_a_rolling_aggregate_scorer_accepts_a_duration_window() -> None:
    ts = series(np.arange(6.0))
    scores = AsScorer(RollingAggregate(window="3d", agg="count")).score(ts)
    # A right-closed window spanning (t - 3d, t] holds three daily observations.
    assert list(scores.values.ravel()) == [1.0, 2.0, 3.0, 3.0, 3.0, 3.0]


# -- DoubleRollingScorer ----------------------------------------------------


def test_a_double_rolling_scorer_peaks_at_a_step_change() -> None:
    ts = series([0.0] * 6 + [5.0] * 6)
    scores = (
        AsScorer(DoubleRollingAggregate(window=3, diff="diff", agg="median"))
        .score(ts)
        .values.ravel()
    )
    # Windows are [i - 3, i) and [i, i + 3), so the step is fully visible at
    # i = 6 and partly visible either side of it.
    assert list(scores[5:8]) == [5.0, 5.0, 5.0]
    assert float(np.nanmax(scores)) == 5.0


def test_a_double_rolling_scorer_signs_the_direction_of_the_change() -> None:
    up = AsScorer(DoubleRollingAggregate(window=2, diff="diff", agg="median")).score(
        series([0.0] * 4 + [5.0] * 4)
    )
    down = AsScorer(DoubleRollingAggregate(window=2, diff="diff", agg="median")).score(
        series([5.0] * 4 + [0.0] * 4)
    )
    assert np.nanmax(up.values) > 0
    assert np.nanmin(down.values) < 0


def test_a_double_rolling_scorer_takes_an_asymmetric_window_for_spikes() -> None:
    """A right window of one measures a single point against its recent past."""
    ts = series([1.0] * 6 + [9.0] + [1.0] * 6)
    scores = (
        AsScorer(DoubleRollingAggregate(window=(4, 1), diff="diff", agg="median"))
        .score(ts)
        .values.ravel()
    )
    assert int(np.nanargmax(scores)) == 6
    assert scores[6] == 8.0
    # The median of the four preceding points absorbs the spike, so the point
    # after it reads as ordinary rather than as a second, mirrored anomaly.
    assert scores[7] == 0.0


def test_a_double_rolling_scorer_measures_volatility_relatively() -> None:
    rng = np.random.default_rng(0)
    quiet = rng.normal(scale=0.1, size=30)
    loud = rng.normal(scale=5.0, size=30)
    ts = series(np.concatenate([quiet, loud]))
    scores = (
        AsScorer(DoubleRollingAggregate(window=10, agg="std", diff="rel_diff"))
        .score(ts)
        .values.ravel()
    )
    assert int(np.nanargmax(scores)) == 30


def test_a_double_rolling_scorer_reports_a_plateau_around_a_shift() -> None:
    """Both windows straddle the change for as long as it takes them to clear it."""
    ts = series([0.0] * 6 + [10.0] * 6)
    scores = (
        AsScorer(DoubleRollingAggregate(window=3, diff="l1", agg="median"))
        .score(ts)
        .values.ravel()
    )
    # Windows are [i - 3, i) and [i, i + 3). One or both straddles the shift for
    # i = 4 to i = 8, and the medians differ by the full step for i = 5, 6, 7.
    assert list(scores[3:10]) == [0.0, 0.0, 10.0, 10.0, 10.0, 0.0, 0.0]
    assert np.isnan(scores[:3]).all()
    assert np.isnan(scores[10:]).all()


# -- DeviationScorer --------------------------------------------------------


def test_a_deviation_scorer_measures_distance_in_units_of_spread() -> None:
    ts = series([10.0, 12.0, 11.0, 13.0, 12.0, 10.0, 40.0])
    scorer = DeviationScorer().fit(ts)
    # median 12, Q1 10.5, Q3 12.5, IQR 2.
    assert (scorer.center_, scorer.scale_) == (12.0, 2.0)
    assert list(scorer.run(ts).values.ravel()) == [
        -1.0,
        0.0,
        -0.5,
        0.5,
        0.0,
        -1.0,
        14.0,
    ]


def test_a_deviation_scorer_keeps_the_sign_of_the_excursion() -> None:
    ts = series([0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 20.0, -20.0])
    scores = DeviationScorer().fit(ts).run(ts).values.ravel()
    assert scores[6] > 0
    assert scores[7] < 0


@pytest.mark.parametrize("scale", ["iqr", "idr", "mad", "std"])
def test_a_deviation_scorer_supports_every_scale(scale: str) -> None:
    ts = series([*np.random.default_rng(0).normal(size=40), 50.0])
    scores = DeviationScorer(scale=scale).fit(ts).run(ts).values.ravel()
    assert int(np.argmax(np.abs(scores))) == 40


def test_a_deviation_scorer_can_centre_on_the_mean_instead() -> None:
    ts = series([1.0, 2.0, 3.0, 94.0])
    assert DeviationScorer(center="mean").fit(ts).center_ == 25.0
    assert DeviationScorer(center="median").fit(ts).center_ == 2.5


def test_a_robust_deviation_scorer_is_unmoved_by_the_outlier_it_looks_for() -> None:
    clean = [10.0, 11.0, 12.0, 11.0, 10.0, 12.0]
    robust = DeviationScorer(center="median", scale="iqr")
    fragile = DeviationScorer(center="mean", scale="std")
    robust_shift = abs(
        robust.fit(series(clean)).scale_ - robust.fit(series([*clean, 5000.0])).scale_
    )
    fragile_shift = abs(
        fragile.fit(series(clean)).scale_ - fragile.fit(series([*clean, 5000.0])).scale_
    )
    assert robust_shift < fragile_shift


def test_a_deviation_scorer_calls_everything_unprecedented_after_a_constant_fit() -> (
    None
):
    scorer = DeviationScorer().fit(series([4.0] * 10))
    scores = scorer.run(series([4.0, 5.0, 3.0])).values.ravel()
    assert list(scores) == [0.0, np.inf, -np.inf]


def test_a_deviation_scorer_rejects_an_unknown_centre() -> None:
    with pytest.raises(ValueError, match=r"center='mode' is not one of"):
        DeviationScorer(center="mode")  # type: ignore[arg-type]


def test_a_deviation_scorer_rejects_an_unknown_scale() -> None:
    with pytest.raises(ValueError, match=r"scale='range' is not one of"):
        DeviationScorer(scale="range")  # type: ignore[arg-type]


# -- AsScorer(SeasonalDecomposition) -------------------------------------------------


def test_a_seasonal_residual_scorer_finds_the_break_in_the_pattern() -> None:

    values = np.tile([1.0, 5.0, 3.0, 2.0], 8)
    values[13] = 12.0
    ts = series(values)
    scorer = AsScorer(SeasonalDecomposition(period=4)).fit(ts)
    assert int(np.nanargmax(np.abs(scorer.run(ts).values.ravel()))) == 13


def test_a_seasonal_residual_scorer_learns_the_profile_of_one_cycle() -> None:

    ts = series(np.tile([0.0, 1.0, 0.0, -1.0], 5))
    scorer = AsScorer(SeasonalDecomposition(period=4)).fit(ts)
    assert scorer.transformer.period_ == 4
    np.testing.assert_allclose(scorer.transformer.seasonal_, [0.0, 1.0, 0.0, -1.0])
    np.testing.assert_allclose(scorer.run(ts).values.ravel(), 0.0)


def test_a_seasonal_residual_scorer_detects_the_period_from_autocorrelation() -> None:

    ts = series(np.tile([2.0, 8.0, 5.0, 1.0, 4.0, 7.0], 12))
    assert AsScorer(SeasonalDecomposition()).fit(ts).transformer.period_ == 6


def test_a_seasonal_residual_scorer_holds_its_profile_for_a_later_series() -> None:
    """A pattern learned once is what a later series is judged against."""

    profile = [1.0, 5.0, 3.0, 2.0]
    scorer = AsScorer(SeasonalDecomposition(period=4)).fit(series(np.tile(profile, 8)))
    later = series(np.tile(profile, 4))
    np.testing.assert_allclose(scorer.run(later).values.ravel(), 0.0, atol=1e-12)


# -- AutoregressionResidualScorer -------------------------------------------


def test_an_autoregression_residual_scorer_finds_a_break_in_the_dynamics() -> None:
    values = np.tile([1.0, 2.0, 3.0], 6)
    values[10] = 9.0
    ts = series(values)
    scores = AutoregressionResidualScorer(n_steps=3).fit(ts).run(ts).values.ravel()
    assert int(np.nanargmax(np.abs(scores))) == 10


def test_an_autoregression_residual_scorer_predicts_a_linear_series_exactly() -> None:
    ts = series(np.arange(20.0))
    scores = AutoregressionResidualScorer(n_steps=1).fit(ts).run(ts).values.ravel()
    np.testing.assert_allclose(scores[1:], 0.0, atol=1e-9)
    assert np.isnan(scores[0])


def test_an_autoregression_residual_scorer_leaves_an_incomplete_history_unknown() -> (
    None
):
    ts = series(np.arange(20.0))
    scores = (
        AutoregressionResidualScorer(n_steps=2, step_size=3)
        .fit(ts)
        .run(ts)
        .values.ravel()
    )
    assert np.isnan(scores[:6]).all()
    assert not np.isnan(scores[6:]).any()


def test_an_autoregression_residual_scorer_accepts_a_supplied_regressor() -> None:
    class AlwaysZero:
        def fit(self, X: np.ndarray, y: np.ndarray) -> AlwaysZero:
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            return np.zeros(X.shape[0])

    ts = series([5.0] * 8)
    scores = (
        AutoregressionResidualScorer(regressor=AlwaysZero())
        .fit(ts)
        .run(ts)
        .values.ravel()
    )
    assert list(scores[1:]) == [5.0] * 7


def test_an_autoregression_residual_scorer_leaves_the_given_regressor_unfitted() -> (
    None
):
    class Counter:
        fits = 0

        def fit(self, X: np.ndarray, y: np.ndarray) -> Counter:
            Counter.fits += 1
            self.seen_ = X.shape[0]
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            return np.zeros(X.shape[0])

    supplied = Counter()
    AutoregressionResidualScorer(regressor=supplied).fit(series(np.arange(10.0)))
    assert not hasattr(supplied, "seen_")


def test_an_autoregression_residual_scorer_gives_each_column_its_own_regressor() -> (
    None
):
    """Fanning out must not have every column overwrite one shared model."""

    class Recording:
        def fit(self, X: np.ndarray, y: np.ndarray) -> Recording:
            self.mean_ = float(y.mean())
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            return np.full(X.shape[0], self.mean_)

    index = pd.date_range("2024-01-01", periods=10, freq="h", name="time")
    data = pd.DataFrame(
        {"small": np.zeros(10), "large": np.full(10, 100.0)}, index=index
    )
    scorer = AutoregressionResidualScorer(regressor=Recording()).fit(data)

    assert scorer._column_models is not None
    means = {
        name: model.transformer_.regressor_.mean_
        for name, model in scorer._column_models.items()
    }
    assert means == {"small": 0.0, "large": 100.0}


@pytest.mark.parametrize(("n_steps", "step_size"), [(0, 1), (1, 0)])
def test_an_autoregression_residual_scorer_rejects_a_non_positive_lag(
    n_steps: int, step_size: int
) -> None:
    with pytest.raises(ValueError, match="must be at least 1"):
        AutoregressionResidualScorer(n_steps=n_steps, step_size=step_size)


# -- RegressionResidualScorer ----------------------------------------------


def test_a_regression_residual_scorer_finds_a_broken_relationship() -> None:
    drive = np.tile([1.0, 2.0, 3.0, 4.0], 5)
    follow = 3.0 * drive - 2.0
    follow[11] += 20.0
    ts = frame(drive=drive, follow=follow)
    scores = AsScorer(RegressionResidual(target="follow")).fit(ts).run(ts)
    assert int(np.argmax(np.abs(scores.values))) == 11
    assert scores.columns == ("residual",)


def test_a_regression_residual_scorer_sees_a_point_normal_in_every_column() -> None:
    """Both readings stay in range; only their combination is impossible."""
    drive = np.tile([1.0, 2.0, 3.0, 4.0], 6)
    follow = drive.copy()
    follow[10], follow[11] = follow[11], follow[10]
    ts = frame(drive=drive, follow=follow)
    scores = AsScorer(RegressionResidual(target="follow")).fit(ts).run(ts)
    flagged = set(np.flatnonzero(np.abs(scores.values.ravel()) > 0.5))
    assert flagged == {10, 11}
    assert drive.min() <= follow[10] <= drive.max()


def test_a_regression_residual_scorer_needs_every_column_at_once() -> None:
    assert AsScorer(RegressionResidual(target="b")).is_multivariate


def test_a_regression_residual_scorer_leaves_the_given_regressor_unfitted() -> None:
    class Recording:
        def fit(self, X: np.ndarray, y: np.ndarray) -> Recording:
            self.seen_ = X.shape[0]
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            return np.zeros(X.shape[0])

    supplied = Recording()
    ts = frame(a=np.arange(6.0), b=np.arange(6.0))
    AsScorer(RegressionResidual(target="b", regressor=supplied)).fit(ts)
    assert not hasattr(supplied, "seen_")


# -- PcaReconstructionErrorScorer ------------------------------------------


def test_a_pca_scorer_flags_the_point_off_the_subspace() -> None:
    base = np.arange(20.0)
    partner = 2.0 * base + 1.0
    partner[6] += 15.0
    ts = frame(a=base, b=partner)
    scores = AsScorer(PcaReconstructionError(k=1)).fit(ts).run(ts)
    assert int(np.argmax(scores.values)) == 6


def test_a_pca_scorer_reconstructs_data_on_a_line_exactly() -> None:
    base = np.arange(6.0)
    ts = frame(a=base, b=2.0 * base + 1.0)
    scores = AsScorer(PcaReconstructionError(k=1)).fit(ts).run(ts)
    np.testing.assert_allclose(scores.values.ravel(), 0.0, atol=1e-18)


def test_a_pca_scorer_exposes_the_basis_it_learned() -> None:
    base = np.arange(10.0)
    ts = frame(a=base, b=np.zeros(10))
    scorer = AsScorer(PcaReconstructionError(k=1)).fit(ts)
    pca = scorer.transformer
    assert pca.components_.shape == (1, 2)
    np.testing.assert_allclose(np.abs(pca.components_[0]), [1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(pca.mean_, [4.5, 0.0])


def test_a_pca_scorer_leaves_an_incomplete_row_unknown() -> None:
    base = np.arange(8.0)
    partner = 2.0 * base
    partner[3] = np.nan
    ts = frame(a=base, b=partner)
    scores = AsScorer(PcaReconstructionError(k=1)).fit(ts).run(ts)
    assert np.isnan(scores.values.ravel()[3])


def test_a_pca_scorer_rejects_more_components_than_columns() -> None:
    ts = frame(a=np.arange(6.0), b=np.arange(6.0))
    with pytest.raises(ValueError, match="exceeds the 2 column"):
        AsScorer(PcaReconstructionError(k=3)).fit(ts)


# -- MinClusterScorer -------------------------------------------------------


def test_a_min_cluster_scorer_flags_membership_of_the_rarest_cluster() -> None:
    ts = frame(a=[1.0, 1, 1, 1, 1, 9], b=[2.0, 2, 2, 2, 2, 9])
    scorer = MinClusterScorer(NearestOfTwo()).fit(ts)
    assert list(scorer.run(ts).values.ravel()) == [0.0] * 5 + [1.0]
    assert scorer.smallest_cluster_ == 1


def test_a_min_cluster_scorer_names_its_output() -> None:
    ts = frame(a=[1.0, 1, 9], b=[1.0, 1, 9])
    assert MinClusterScorer(NearestOfTwo()).fit(ts).run(ts).columns == ("min_cluster",)


def test_a_min_cluster_scorer_leaves_an_incomplete_row_unknown() -> None:
    ts = frame(a=[1.0, 1, np.nan, 9], b=[1.0, 1, 1, 9])
    scores = MinClusterScorer(NearestOfTwo()).fit(ts).run(ts).values.ravel()
    assert np.isnan(scores[2])


def test_a_min_cluster_scorer_rejects_a_model_that_cannot_place_a_new_point() -> None:
    ts = frame(a=[1.0, 1, 9], b=[1.0, 1, 9])
    with pytest.raises(ValueError, match=r"no predict\(\) method"):
        MinClusterScorer(Transductive()).fit(ts)


def test_a_min_cluster_scorer_leaves_the_given_model_unfitted() -> None:
    supplied = NearestOfTwo()
    ts = frame(a=[1.0, 1, 9], b=[1.0, 1, 9])
    MinClusterScorer(supplied).fit(ts)
    assert not hasattr(supplied, "split_")


# -- OutlierScorer ----------------------------------------------------------


def test_an_outlier_scorer_flags_what_the_model_rejects() -> None:
    ts = frame(a=[0.0, 1, 0, 1, 20], b=[0.0, 1, 1, 0, 20])
    scores = OutlierScorer(FarFromCentre()).fit(ts).run(ts)
    assert list(scores.values.ravel()) == [0.0, 0.0, 0.0, 0.0, 1.0]
    assert scores.columns == ("outlier",)


def test_an_outlier_scorer_handles_a_model_that_only_judges_a_whole_batch() -> None:
    ts = frame(a=[0.0, 0, 0, 0, 0, 30], b=[0.0, 0, 0, 0, 0, 30])
    scores = OutlierScorer(BatchOnly()).fit(ts).run(ts).values.ravel()
    assert list(scores) == [0.0] * 5 + [1.0]


def test_an_outlier_scorer_leaves_an_incomplete_row_unknown() -> None:
    ts = frame(a=[0.0, 1, np.nan, 20], b=[0.0, 1, 1, 20])
    scores = OutlierScorer(FarFromCentre()).fit(ts).run(ts).values.ravel()
    assert np.isnan(scores[2])


# -- shared behaviour -------------------------------------------------------


@pytest.mark.parametrize("scorer", UNIVARIATE_SCORERS, ids=lambda s: type(s).__name__)
def test_every_univariate_scorer_returns_all_unknown_for_an_all_missing_series(
    scorer: Scorer,
) -> None:
    ts = series([np.nan] * 12)
    assert np.isnan(scored(scorer.clone(), ts)).all()


@pytest.mark.parametrize("scorer", UNIVARIATE_SCORERS, ids=lambda s: type(s).__name__)
def test_every_univariate_scorer_is_quiet_on_a_constant_series(
    scorer: Scorer,
) -> None:
    """No warnings, and nothing that reads as a large deviation."""
    scores = scored(scorer.clone(), series([3.0] * 20))
    finite = scores[np.isfinite(scores)]
    assert np.all(np.abs(finite - finite.mean()) < 1e-9)


@pytest.mark.parametrize("scorer", UNIVARIATE_SCORERS, ids=lambda s: type(s).__name__)
def test_every_univariate_scorer_keeps_the_input_column_name(
    scorer: Scorer,
) -> None:
    ts = TimeSeries.from_arrays(
        np.arange("2024-01-01", "2024-01-15", dtype="datetime64[D]"),
        np.arange(14.0),
        ["sensor"],
    )
    copy = scorer.clone()
    if copy.is_trainable:
        copy.fit(ts)
    assert copy.run(ts).columns == ("sensor",)


@pytest.mark.parametrize(
    "scorer",
    UNIVARIATE_SCORERS + MULTIVARIATE_SCORERS,
    ids=lambda s: type(s).__name__,
)
def test_every_scorer_round_trips_its_parameters_through_clone(
    scorer: Scorer,
) -> None:
    copy = scorer.clone()
    assert _plain(copy.get_params()) == _plain(scorer.get_params())
    assert repr(copy) == repr(scorer)
    if isinstance(scorer, AsScorer):
        assert copy.transformer is not scorer.transformer


def _plain(params: dict[str, Any]) -> dict[str, Any]:
    """Drop nested components, which compare by identity, from a parameter dict."""
    return {k: v for k, v in params.items() if not isinstance(v, Component)}


def test_clone_carries_every_parameter_of_a_rolling_scorer() -> None:
    original = AsScorer(
        RollingAggregate(
            window="2h",
            agg="quantile",
            center=True,
            min_periods=2,
            closed="both",
            agg_params={"q": 0.9},
        )
    )
    assert original.clone().transformer.get_params() == {
        "window": "2h",
        "agg": "quantile",
        "agg_params": {"q": 0.9},
        "center": True,
        "min_periods": 2,
        "closed": "both",
    }


def test_clone_carries_every_parameter_of_a_double_rolling_scorer() -> None:
    original = AsScorer(
        DoubleRollingAggregate(
            window=(5, 1), agg=("median", "mean"), diff="rel_diff", min_periods=(3, 1)
        )
    )
    assert original.clone().transformer.get_params() == {
        "window": (5, 1),
        "agg": ("median", "mean"),
        "agg_params": None,
        "center": True,
        "min_periods": (3, 1),
        "diff": "rel_diff",
    }


def test_a_univariate_scorer_fits_each_column_of_a_frame_independently() -> None:
    index = pd.date_range("2024-01-01", periods=8, freq="h", name="time")
    data = pd.DataFrame(
        {"small": np.arange(8.0), "large": np.arange(8.0) * 1000.0}, index=index
    )
    scorer = DeviationScorer().fit(data)

    assert scorer._column_models is not None
    centres = {name: model.center_ for name, model in scorer._column_models.items()}
    assert centres == {"small": 3.5, "large": 3500.0}


def test_a_multivariate_scorer_does_not_fan_out() -> None:
    ts = frame(a=np.arange(8.0), b=np.arange(8.0) * 2.0)
    scorer = AsScorer(PcaReconstructionError(k=1)).fit(ts)
    assert scorer._column_models is None
    assert scorer.run(ts).n_columns == 1


# -- backends ---------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_scorer_returns_the_backend_it_was_given(backend: str) -> None:
    native = make_native(backend, np.arange(12.0))
    assert type(DeviationScorer().fit_score(native)) is type(native)


def test_every_backend_produces_identical_scores() -> None:
    values = np.concatenate([np.zeros(10), np.full(10, 5.0)])
    for scorer in (
        DeviationScorer(),
        AsScorer(DoubleRollingAggregate(window=3, diff="diff", agg="median")),
        AutoregressionResidualScorer(n_steps=2),
    ):
        results = [
            TimeSeries.from_any(scorer.clone().fit_score(make_native(b, values)))
            for b in BACKENDS
        ]
        for other in results[1:]:
            np.testing.assert_allclose(other.values, results[0].values, equal_nan=True)


# -- AsScorer ---------------------------------------------------------------


def test_as_scorer_refuses_something_that_is_not_a_transformer() -> None:
    with pytest.raises(TypeError, match="does not produce a series"):
        AsScorer(DeviationScorer())


def test_as_scorer_follows_the_transformer_it_wraps() -> None:
    assert not AsScorer(RollingAggregate(window=3)).is_trainable
    assert AsScorer(SeasonalDecomposition(period=4)).is_trainable
    assert AsScorer(PcaReconstructionError()).is_multivariate
    assert not AsScorer(RollingAggregate(window=3)).is_multivariate


def test_as_scorer_answers_unknown_when_training_held_no_observation() -> None:
    """The transformer alone would refuse to fit; the scorer says "no idea"."""
    scorer = AsScorer(SeasonalDecomposition(period=4)).fit(series(np.full(12, np.nan)))
    assert np.isnan(scorer.run(series(np.arange(12.0))).values).all()


def test_as_scorer_reaches_its_transformer_by_nested_parameter() -> None:
    scorer = AsScorer(RollingAggregate(window=3))
    scorer.set_params(transformer__agg="max")
    ts = series([1.0, 1.0, 9.0, 1.0, 1.0])
    np.testing.assert_array_equal(scorer.score(ts).values.ravel()[2:], [9.0] * 3)
