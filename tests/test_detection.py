"""Tests for the detectors.

Every detector here is checked against a synthetic series with an anomaly planted
at a known position, and the assertion is on that exact position rather than on
"something was found". Where a window makes the answer a plateau of several
consecutive points, the whole plateau is asserted.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest

from hazure import Component, Detector, TimeSeries, detectors
from hazure.scorers import AsScorer, DeviationScorer
from hazure.thresholds import (
    FixedThreshold,
    IqrThreshold,
    MadThreshold,
    SignedThreshold,
)
from hazure.transformers import DoubleRollingAggregate
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
    return TimeSeries.from_arrays(time, numbers, ["sensor"])


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


def labels_of(detector: Detector, ts: TimeSeries) -> np.ndarray:
    """Fit if needed, detect, and return a flat array of labels."""
    if detector.is_trainable:
        detector.fit(ts)
    return np.asarray(detector.run(ts).values).ravel()


def flagged(detector: Detector, ts: TimeSeries) -> list[int]:
    """Positions the detector calls anomalous."""
    return list(np.flatnonzero(labels_of(detector, ts) == 1.0))


class NearestOfTwo:
    """A minimal generalising clusterer: split on the training mean."""

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.split_ = float(X.mean())
        return self.predict(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (X.mean(axis=1) > self.split_).astype(int)


class FarFromCentre:
    """A minimal outlier model marking outliers with -1."""

    def fit(self, X: np.ndarray) -> FarFromCentre:
        self.centre_ = np.median(X, axis=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(np.abs(X - self.centre_).sum(axis=1) > 5.0, -1, 1)


UNIVARIATE_DETECTORS: tuple[Detector, ...] = (
    detectors.limits(low=-100.0, high=100.0),
    detectors.quantile(low=0.01, high=0.99),
    detectors.iqr(),
    detectors.esd(),
    detectors.spike(),
    detectors.level_shift(window=3),
    detectors.volatility_shift(window=5),
    detectors.seasonal(period=4),
    detectors.autoregression(n_steps=2),
)

MULTIVARIATE_DETECTORS: tuple[Detector, ...] = (
    detectors.regression(target="b"),
    detectors.pca(k=1),
    detectors.min_cluster(NearestOfTwo()),
    detectors.outlier(FarFromCentre()),
)


# -- the generic pairing ----------------------------------------------------


def test_a_score_detector_pairs_any_scorer_with_any_threshold() -> None:
    ts = series([5.0, 6.0, 5.0, 6.0, 5.0, 6.0, 40.0])
    detector = Detector(DeviationScorer(), MadThreshold())
    assert flagged(detector, ts) == [6]


def test_a_score_detector_exposes_both_halves() -> None:
    """The point of the split: the parts stay available."""
    scorer, threshold = DeviationScorer(), MadThreshold()
    detector = Detector(scorer, threshold)
    assert detector.scorer is scorer
    assert detector.threshold is threshold


def test_a_score_detector_fits_its_threshold_on_the_fitted_scorer_output() -> None:
    """The threshold must see the scale the scorer actually works on."""
    ts = series([10.0, 12.0, 11.0, 13.0, 12.0, 10.0, 40.0])
    detector = Detector(DeviationScorer(), IqrThreshold()).fit(ts)
    scores = detector.scorer.run(ts).values.ravel()
    assert detector.threshold.high_ == pytest.approx(
        np.quantile(scores, 0.75)
        + 3.0 * (np.quantile(scores, 0.75) - np.quantile(scores, 0.25))
    )


def test_a_score_detector_with_no_scorer_thresholds_the_values_themselves() -> None:
    ts = series([1.0, 2.0, 99.0])
    detector = Detector(None, FixedThreshold(high=50.0))
    assert flagged(detector, ts) == [2]


def test_a_score_detector_swaps_thresholds_without_touching_the_scorer() -> None:
    values = np.random.default_rng(11).normal(size=60)
    values[41] = 9.0
    ts = series(values)
    lenient = Detector(DeviationScorer(scale="std"), IqrThreshold(factor=50.0))
    usual = Detector(DeviationScorer(scale="std"), IqrThreshold(factor=3.0))
    assert flagged(lenient, ts) == []
    assert flagged(usual, ts) == [41]


def test_a_signed_score_detector_thresholds_magnitude_and_filters_direction() -> None:
    ts = series([1.0] * 8 + [9.0] + [1.0] * 8)
    build = lambda side: Detector(  # noqa: E731
        AsScorer(DoubleRollingAggregate(window=(3, 1), diff="diff", agg="median")),
        SignedThreshold(IqrThreshold(factor=(None, 3.0)), side=side),
    )
    assert flagged(build("both"), ts) == [8]
    assert flagged(build("positive"), ts) == [8]
    assert flagged(build("negative"), ts) == []


def test_a_multivariate_score_detector_reports_one_column_named_anomaly() -> None:
    ts = frame(a=np.arange(10.0), b=np.arange(10.0) * 2.0)
    detector = Detector(detectors.pca(k=1).scorer, IqrThreshold(factor=(None, 5.0)))
    detector.fit(ts)
    assert detector.run(ts).columns == ("anomaly",)


@pytest.mark.parametrize("side", ["up", "", None, "POSITIVE"])
def test_a_sided_detector_rejects_an_unknown_side(side: object) -> None:
    with pytest.raises(ValueError, match="is not one of"):
        detectors.spike(side=side)  # type: ignore[arg-type]


# -- values judged on their own ---------------------------------------------


def test_a_threshold_detector_flags_values_outside_the_given_range() -> None:
    ts = series([20.0, 21.0, 45.0, 19.0, -5.0])
    assert flagged(detectors.limits(low=0.0, high=40.0), ts) == [2, 4]


def test_a_threshold_detector_needs_no_fitting() -> None:
    assert detectors.limits(high=1.0).fitted


def test_a_quantile_detector_flags_the_planted_extreme() -> None:
    values = np.arange(1.0, 21.0)
    values[7] = 500.0
    assert flagged(detectors.quantile(high=0.95), series(values)) == [7]


def test_a_quantile_detector_flags_the_lower_tail_when_asked() -> None:
    values = np.arange(1.0, 21.0)
    values[7] = -500.0
    assert flagged(detectors.quantile(low=0.05), series(values)) == [7]


def test_an_iqr_detector_flags_the_planted_extreme() -> None:
    values = np.array([10.0, 11, 12, 11, 10, 12, 11, 10, 11, 60])
    assert flagged(detectors.iqr(), series(values)) == [9]


def test_a_smaller_iqr_factor_flags_more() -> None:
    values = np.array([*np.tile([10.0, 11.0], 10), 14.0])
    assert flagged(detectors.iqr(factor=3.0), series(values)) == []
    assert flagged(detectors.iqr(factor=0.5), series(values)) == [20]


def test_an_esd_detector_flags_the_planted_extreme() -> None:
    values = np.random.default_rng(1).normal(loc=20.0, size=60)
    values[42] = 30.0
    assert flagged(detectors.esd(), series(values)) == [42]


def test_a_stricter_esd_alpha_flags_less() -> None:
    values = np.random.default_rng(3).normal(size=100)
    values[[3, 50]] = [3.6, 8.0]
    ts = series(values)
    lenient = set(flagged(detectors.esd(alpha=0.2), ts))
    strict = set(flagged(detectors.esd(alpha=1e-9), ts))
    assert 50 in lenient
    assert strict < lenient


# -- spikes -----------------------------------------------------------------


def test_a_spike_detector_flags_the_spike_and_its_return() -> None:
    """With a one-point window the score moves twice: up, then back down."""
    values = np.ones(20)
    values[12] = 9.0
    assert flagged(detectors.spike(), series(values)) == [12, 13]


def test_a_spike_detector_isolates_the_spike_with_a_positive_side() -> None:
    values = np.ones(20)
    values[12] = 9.0
    assert flagged(detectors.spike(side="positive"), series(values)) == [12]
    assert flagged(detectors.spike(side="negative"), series(values)) == [13]


def test_a_spike_detector_reverses_which_end_it_flags_for_a_dip() -> None:
    values = np.ones(20)
    values[12] = -9.0
    assert flagged(detectors.spike(side="negative"), series(values)) == [12]
    assert flagged(detectors.spike(side="positive"), series(values)) == [13]


def test_a_wider_spike_window_flags_the_spike_alone() -> None:
    """A median over four points is unmoved by the spike inside it."""
    values = np.ones(20)
    values[12] = 9.0
    assert flagged(detectors.spike(window=4), series(values)) == [12]


def test_a_spike_detector_cannot_judge_its_first_observation() -> None:
    values = np.ones(20)
    values[12] = 9.0
    assert np.isnan(labels_of(detectors.spike(), series(values))[0])


def test_a_spike_detector_accepts_a_mean_window() -> None:
    values = np.ones(20)
    values[12] = 9.0
    assert flagged(detectors.spike(agg="mean", side="positive"), series(values)) == [12]


def test_a_spike_detector_rejects_an_aggregation_that_is_not_a_centre() -> None:
    with pytest.raises(ValueError, match=r"agg='std' must be one of"):
        detectors.spike(agg="std")


# -- level shifts -----------------------------------------------------------


def test_a_level_shift_detector_flags_the_plateau_around_the_shift() -> None:
    """Both windows straddle the change until they clear it, so a run is flagged."""
    values = np.concatenate([np.zeros(20), np.full(20, 10.0)])
    assert flagged(detectors.level_shift(window=3), series(values)) == [19, 20, 21]


def test_a_level_shift_detector_cannot_judge_the_ends() -> None:
    values = np.concatenate([np.zeros(20), np.full(20, 10.0)])
    unknown = np.flatnonzero(
        np.isnan(labels_of(detectors.level_shift(window=3), series(values)))
    )
    assert list(unknown) == [0, 1, 2, 38, 39]


def test_a_level_shift_detector_respects_the_direction_asked_for() -> None:
    up = np.concatenate([np.zeros(20), np.full(20, 10.0)])
    down = np.concatenate([np.full(20, 10.0), np.zeros(20)])
    assert flagged(detectors.level_shift(window=3, side="positive"), series(up)) == [
        19,
        20,
        21,
    ]
    assert flagged(detectors.level_shift(window=3, side="negative"), series(up)) == []
    assert flagged(detectors.level_shift(window=3, side="negative"), series(down)) == [
        19,
        20,
        21,
    ]


def test_a_level_shift_detector_ignores_a_lone_spike() -> None:
    """The distinction from spike detection: one odd point is not a new level."""
    values = np.zeros(40)
    values[20] = 10.0
    assert flagged(detectors.level_shift(window=3), series(values)) == []


def test_a_level_shift_detector_takes_an_asymmetric_window() -> None:
    values = np.concatenate([np.zeros(20), np.full(20, 10.0)])
    assert flagged(detectors.level_shift(window=(5, 2)), series(values)) != []


# -- volatility shifts ------------------------------------------------------


def test_a_volatility_shift_detector_flags_the_change_in_noise() -> None:
    rng = np.random.default_rng(0)
    values = np.concatenate(
        [rng.normal(scale=0.1, size=40), rng.normal(scale=5.0, size=40)]
    )
    positions = flagged(detectors.volatility_shift(window=10), series(values))
    # The right window reaches the loud stretch ten observations early, so the
    # run of flags ends exactly at the change point.
    assert positions[-1] == 40
    assert positions == list(range(31, 41))


def test_a_volatility_shift_detector_also_reacts_to_a_level_shift() -> None:
    """A step inside a window inflates that window's spread, so a step registers.

    Documented rather than fixed: it is what measuring spread over a window
    means. Telling the two apart is what pairing with a level-shift detector is
    for.
    """
    rng = np.random.default_rng(0)
    noise = rng.normal(scale=1.0, size=80)
    values = noise + np.concatenate([np.zeros(40), np.full(40, 50.0)])
    # The right window holds part of the step for i = 31 to 39; at i = 40 each
    # window sits wholly on one side of it and the spreads agree again.
    assert flagged(detectors.volatility_shift(window=10), series(values)) == list(
        range(31, 40)
    )


def test_a_volatility_shift_detector_respects_the_direction_asked_for() -> None:
    """A drop needs a smaller factor than a rise: a relative fall cannot pass -1."""
    rng = np.random.default_rng(0)
    loud_then_quiet = np.concatenate(
        [rng.normal(scale=5.0, size=60), rng.normal(scale=0.1, size=60)]
    )
    ts = series(loud_then_quiet)
    quieter = flagged(
        detectors.volatility_shift(window=10, factor=1.5, side="negative"), ts
    )
    louder = flagged(
        detectors.volatility_shift(window=10, factor=1.5, side="positive"), ts
    )
    assert 60 in quieter
    assert 60 not in louder


def test_a_volatility_shift_detector_accepts_a_robust_spread() -> None:
    rng = np.random.default_rng(0)
    values = np.concatenate(
        [rng.normal(scale=0.1, size=40), rng.normal(scale=5.0, size=40)]
    )
    assert 40 in flagged(
        detectors.volatility_shift(window=10, agg="iqr"), series(values)
    )


def test_a_volatility_shift_detector_rejects_an_aggregation_that_is_not_a_spread() -> (
    None
):
    with pytest.raises(ValueError, match=r"agg='median' must be one of"):
        detectors.volatility_shift(window=5, agg="median")


# -- seasonality ------------------------------------------------------------


def test_a_seasonal_detector_flags_the_break_in_the_pattern() -> None:
    values = np.tile([1.0, 5.0, 3.0, 2.0], 8)
    values[13] = 12.0
    assert flagged(detectors.seasonal(period=4), series(values)) == [13]


def test_a_seasonal_detector_finds_a_value_ordinary_elsewhere_in_the_cycle() -> None:
    """5.0 is normal at phase 1 and anomalous at phase 0."""
    values = np.tile([1.0, 5.0, 3.0, 2.0], 8)
    values[16] = 5.0
    assert flagged(detectors.seasonal(period=4), series(values)) == [16]
    assert values[16] in np.tile([1.0, 5.0, 3.0, 2.0], 8)


def test_a_seasonal_detector_respects_the_direction_asked_for() -> None:
    values = np.tile([1.0, 5.0, 3.0, 2.0], 8)
    values[13] = 12.0
    assert flagged(detectors.seasonal(period=4, side="positive"), series(values)) == [
        13
    ]
    assert flagged(detectors.seasonal(period=4, side="negative"), series(values)) == []


def test_a_seasonal_detector_detects_the_period_when_not_told() -> None:
    rng = np.random.default_rng(7)
    profile = [2.0, 8.0, 5.0, 1.0, 4.0, 7.0]
    history = np.tile(profile, 12) + rng.normal(scale=0.3, size=72)
    detector = detectors.seasonal().fit(series(history))
    assert detector.scorer.transformer.period_ == 6

    later = np.tile(profile, 12) + rng.normal(scale=0.3, size=72)
    later[40] += 15.0
    labels = detector.run(series(later)).values.ravel()
    assert list(np.flatnonzero(labels == 1.0)) == [40]


def test_a_seasonal_detector_can_remove_a_trend_as_well() -> None:
    values = np.tile([1.0, 5.0, 3.0, 2.0], 8) + np.arange(32) * 2.0
    values[13] += 12.0
    assert flagged(detectors.seasonal(period=4, trend=True), series(values)) == [13]


def test_a_seasonal_detector_without_a_trend_is_fooled_by_one() -> None:
    """Why ``trend=True`` exists: a drift otherwise dominates the residual."""
    values = np.tile([1.0, 5.0, 3.0, 2.0], 8) + np.arange(32) * 2.0
    values[13] += 12.0
    assert flagged(detectors.seasonal(period=4, trend=False), series(values)) != [13]


def test_a_seasonal_detector_needs_a_regular_time_axis() -> None:
    irregular = TimeSeries.from_arrays(
        np.array(
            ["2024-01-01", "2024-01-02", "2024-01-05", "2024-01-09"],
            dtype="datetime64[D]",
        ),
        np.array([1.0, 2.0, 3.0, 4.0]),
    )
    with pytest.raises(ValueError, match=r"regular|irregular|freq"):
        detectors.seasonal(period=2).fit(irregular)


# -- autoregression ---------------------------------------------------------


def test_an_autoregression_detector_flags_the_break_in_the_dynamics() -> None:
    values = np.tile([1.0, 3.0, 5.0, 3.0], 8)
    values[17] = 11.0
    assert flagged(detectors.autoregression(n_steps=3), series(values)) == [17]


def test_an_autoregression_detector_respects_the_direction_asked_for() -> None:
    values = np.tile([1.0, 3.0, 5.0, 3.0], 8)
    values[17] = 11.0
    ts = series(values)
    assert flagged(detectors.autoregression(n_steps=3, side="positive"), ts) == [17]
    assert flagged(detectors.autoregression(n_steps=3, side="negative"), ts) == []


def test_an_autoregression_detector_cannot_judge_an_incomplete_history() -> None:
    values = np.tile([1.0, 3.0, 5.0, 3.0], 8)
    values[17] = 11.0
    labels = labels_of(detectors.autoregression(n_steps=2, step_size=2), series(values))
    assert list(np.flatnonzero(np.isnan(labels))) == [0, 1, 2, 3]


def test_an_autoregression_detector_accepts_a_supplied_regressor() -> None:
    class LastValue:
        def fit(self, X: np.ndarray, y: np.ndarray) -> LastValue:
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            return X[:, 0]

    values = np.full(20, 5.0)
    values[11] = 15.0
    detector = detectors.autoregression(regressor=LastValue(), side="positive")
    assert flagged(detector, series(values)) == [11]


# -- multivariate -----------------------------------------------------------


def test_a_regression_detector_flags_the_broken_relationship() -> None:
    drive = np.tile([1.0, 2.0, 3.0, 4.0], 5)
    follow = 3.0 * drive - 2.0
    follow[11] += 20.0
    assert flagged(
        detectors.regression(target="follow"), frame(drive=drive, follow=follow)
    ) == [11]


def test_a_regression_detector_respects_the_direction_asked_for() -> None:
    drive = np.tile([1.0, 2.0, 3.0, 4.0], 5)
    follow = 3.0 * drive - 2.0
    follow[11] += 20.0
    ts = frame(drive=drive, follow=follow)
    assert flagged(detectors.regression(target="follow", side="positive"), ts) == [11]
    assert flagged(detectors.regression(target="follow", side="negative"), ts) == []


def test_a_regression_detector_reports_a_single_column_named_anomaly() -> None:
    drive = np.tile([1.0, 2.0, 3.0, 4.0], 5)
    ts = frame(drive=drive, follow=3.0 * drive)
    detector = detectors.regression(target="follow").fit(ts)
    assert detector.run(ts).columns == ("anomaly",)


def test_a_pca_detector_flags_the_point_off_the_subspace() -> None:
    base = np.arange(20.0)
    partner = 2.0 * base + 1.0
    partner[6] += 15.0
    assert flagged(detectors.pca(k=1), frame(a=base, b=partner)) == [6]


def test_a_pca_detector_never_flags_a_perfectly_reconstructed_point() -> None:
    """A squared error has no meaningful lower tail, so none is tested."""
    base = np.arange(30.0)
    partner = 2.0 * base + 1.0
    partner[[7, 20]] += [12.0, 9.0]
    labels = labels_of(detectors.pca(k=1), frame(a=base, b=partner))
    assert set(np.flatnonzero(labels == 1.0)) == {7, 20}


def test_a_min_cluster_detector_flags_the_rarest_group() -> None:
    ts = frame(a=[1.0, 1, 1, 1, 1, 9], b=[2.0, 2, 2, 2, 2, 9])
    assert flagged(detectors.min_cluster(NearestOfTwo()), ts) == [5]


def test_a_min_cluster_detector_flags_nothing_when_the_data_forms_one_group() -> None:
    """A single cluster has no rare minority, so nothing is anomalous."""
    ts = frame(a=np.full(20, 3.0), b=np.full(20, 7.0))
    assert flagged(detectors.min_cluster(NearestOfTwo()), ts) == []


def test_an_outlier_detector_flags_what_the_model_rejects() -> None:
    ts = frame(a=[0.0, 1, 0, 1, 20], b=[0.0, 1, 1, 0, 20])
    assert flagged(detectors.outlier(FarFromCentre()), ts) == [4]


@pytest.mark.parametrize(
    "detector", MULTIVARIATE_DETECTORS, ids=lambda d: type(d).__name__
)
def test_every_multivariate_detector_needs_every_column_at_once(
    detector: Detector,
) -> None:
    assert detector.is_multivariate
    ts = frame(a=np.arange(12.0), b=np.arange(12.0) * 2.0 + 1.0)
    fitted = detector.clone().fit(ts)
    assert fitted._column_models is None
    assert fitted.run(ts).columns == ("anomaly",)


# -- shared behaviour -------------------------------------------------------


@pytest.mark.parametrize(
    "detector", UNIVARIATE_DETECTORS, ids=lambda d: type(d).__name__
)
def test_every_univariate_detector_flags_nothing_in_a_constant_series(
    detector: Detector,
) -> None:
    """A series with no variation has no anomalies, and warns about nothing."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        labels = labels_of(detector.clone(), series(np.full(40, 3.0)))
    assert not np.any(labels == 1.0)


@pytest.mark.parametrize(
    "detector",
    UNIVARIATE_DETECTORS + MULTIVARIATE_DETECTORS,
    ids=lambda d: type(d).__name__,
)
def test_every_detector_returns_all_unknown_for_an_all_missing_series(
    detector: Detector,
) -> None:
    ts = (
        frame(a=np.full(24, np.nan), b=np.full(24, np.nan))
        if detector.is_multivariate
        else series(np.full(24, np.nan))
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        labels = labels_of(detector.clone(), ts)
    assert np.isnan(labels).all()


@pytest.mark.parametrize(
    "detector",
    UNIVARIATE_DETECTORS + MULTIVARIATE_DETECTORS,
    ids=lambda d: type(d).__name__,
)
def test_every_detector_emits_only_the_three_label_states(
    detector: Detector,
) -> None:
    values = np.tile([1.0, 5.0, 3.0, 2.0], 10)
    values[21] = 40.0
    ts = (
        frame(a=values, b=2.0 * values + 1.0)
        if detector.is_multivariate
        else series(values)
    )
    labels = labels_of(detector.clone(), ts)
    assert set(np.unique(labels[~np.isnan(labels)])) <= {0.0, 1.0}


@pytest.mark.parametrize(
    "detector",
    UNIVARIATE_DETECTORS + MULTIVARIATE_DETECTORS,
    ids=lambda d: type(d).__name__,
)
def test_every_detector_exposes_its_threshold(detector: Detector) -> None:
    assert detector.threshold is not None  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "detector",
    UNIVARIATE_DETECTORS + MULTIVARIATE_DETECTORS,
    ids=lambda d: type(d).__name__,
)
def test_every_detector_round_trips_its_parameters_through_clone(
    detector: Detector,
) -> None:
    copy = detector.clone()
    assert _plain(copy.get_params()) == _plain(detector.get_params())
    assert repr(copy) == repr(detector)
    assert copy is not detector


def _plain(params: dict[str, Any]) -> dict[str, Any]:
    """Drop nested components, which compare by identity, from a parameter dict."""
    return {k: v for k, v in params.items() if not isinstance(v, Component)}


def test_clone_carries_every_parameter_of_a_spike_detector() -> None:
    original = detectors.spike(
        window="2h", factor=4.5, side="negative", min_periods=2, agg="mean"
    )
    params = original.clone().get_params()
    assert params["scorer__transformer__window"] == ("2h", 1)
    assert params["scorer__transformer__min_periods"] == (2, 1)
    assert params["scorer__transformer__agg"] == "mean"
    assert params["threshold__threshold__factor"] == (None, 4.5)
    assert params["threshold__side"] == "negative"


def test_clone_carries_every_parameter_of_a_seasonal_detector() -> None:
    original = detectors.seasonal(period=7, factor=2.5, side="positive", trend=True)
    params = original.clone().get_params()
    assert params["scorer__transformer__period"] == 7
    assert params["scorer__transformer__trend"] is True
    assert params["threshold__threshold__factor"] == (None, 2.5)
    assert params["threshold__side"] == "positive"


def test_clone_carries_every_parameter_of_an_autoregression_detector() -> None:
    original = detectors.autoregression(
        n_steps=3, step_size=2, regressor=None, factor=2.0, side="negative"
    )
    params = original.clone().get_params()
    assert params["scorer__n_steps"] == 3
    assert params["scorer__step_size"] == 2
    assert params["scorer__regressor"] is None
    assert params["threshold__threshold__factor"] == (None, 2.0)
    assert params["threshold__side"] == "negative"


def test_clone_carries_every_parameter_of_a_volatility_shift_detector() -> None:
    original = detectors.volatility_shift(
        window=(9, 4), factor=7.0, side="positive", min_periods=(3, 2), agg="idr"
    )
    params = original.clone().get_params()
    assert params["scorer__transformer__window"] == (9, 4)
    assert params["scorer__transformer__min_periods"] == (3, 2)
    assert params["scorer__transformer__agg"] == "idr"
    assert params["scorer__transformer__diff"] == "rel_diff"
    assert params["threshold__threshold__factor"] == (None, 7.0)
    assert params["threshold__side"] == "positive"


def test_a_clone_rebuilds_its_composed_parts_rather_than_sharing_them() -> None:
    original = detectors.level_shift(window=3)
    copy = original.clone()
    assert copy.scorer is not original.scorer
    assert copy.threshold is not original.threshold


def test_a_parameter_change_takes_effect_at_the_next_fit() -> None:
    ts = series([*np.tile([10.0, 11.0], 12), 14.0])
    detector = detectors.iqr(factor=3.0)
    assert flagged(detector, ts) == []
    detector.set_params(threshold__factor=0.5)
    assert flagged(detector, ts) == [24]


def test_a_trainable_detector_must_be_fitted_first() -> None:
    with pytest.raises(RuntimeError, match="fit_detect"):
        detectors.iqr().detect(series([1.0, 2.0, 3.0]))


# -- fan-out and backends ---------------------------------------------------


def test_a_univariate_detector_fits_each_column_of_a_frame_independently() -> None:
    index = pd.date_range("2024-01-01", periods=11, freq="h", name="time")
    small = np.array([10.0, 11, 12, 11, 10, 12, 11, 10, 11, 10, 60])
    data = pd.DataFrame({"small": small, "large": small * 100.0}, index=index)

    detector = detectors.iqr(factor=1.5).fit(data)
    assert detector._column_models is not None
    cutoffs = {
        name: model.threshold.high_  # type: ignore[attr-defined]
        for name, model in detector._column_models.items()
    }
    assert cutoffs["large"] == pytest.approx(cutoffs["small"] * 100.0)

    labels = detector.detect(data)
    assert list(labels.columns) == ["small", "large"]
    assert list(np.flatnonzero(labels["small"].to_numpy() == 1.0)) == [10]
    assert list(np.flatnonzero(labels["large"].to_numpy() == 1.0)) == [10]


def test_a_univariate_detector_keeps_the_input_column_name() -> None:
    ts = series(np.arange(20.0))
    assert detectors.iqr().fit(ts).run(ts).columns == ("sensor",)


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_detector_returns_the_backend_it_was_given(backend: str) -> None:
    values = np.ones(20)
    values[12] = 9.0
    native = make_native(backend, values)
    assert type(detectors.spike().fit_detect(native)) is type(native)


@pytest.mark.parametrize(
    "detector",
    (
        detectors.iqr(),
        detectors.esd(),
        detectors.spike(side="positive"),
        detectors.level_shift(window=3),
        detectors.seasonal(period=4),
        detectors.autoregression(n_steps=2),
    ),
    ids=lambda d: type(d).__name__,
)
def test_every_backend_produces_identical_labels(detector: Detector) -> None:
    values = np.tile([1.0, 5.0, 3.0, 2.0], 10)
    values[21] = 40.0
    results = [
        TimeSeries.from_any(detector.clone().fit_detect(make_native(b, values)))
        for b in BACKENDS
    ]
    for other in results[1:]:
        np.testing.assert_array_equal(other.values, results[0].values)
