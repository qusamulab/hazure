"""Tests for the threshold rules.

A threshold is a decision, so the assertions are about exact cut-offs and exact
labels rather than about shapes: every rule here has a closed form, and the tests
pin it down.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from hazure import Threshold, TimeSeries
from hazure.thresholds import (
    MAD_SCALE,
    EsdThreshold,
    FixedThreshold,
    IqrThreshold,
    MadThreshold,
    QuantileThreshold,
    SignedThreshold,
)
from tests.helpers import BACKENDS, make_native

# -- helpers ----------------------------------------------------------------


def scores(*values: float) -> TimeSeries:
    """Build a daily score series from raw numbers."""
    time = np.arange(
        np.datetime64("2024-01-01"),
        np.datetime64("2024-01-01") + len(values),
        dtype="datetime64[D]",
    )
    return TimeSeries.from_arrays(time, np.asarray(values, dtype=float))


def labels_of(threshold: Threshold, ts: TimeSeries) -> np.ndarray:
    """Fit if needed, apply, and return a flat array of labels."""
    if threshold.trainable:
        threshold.fit(ts)
    return np.asarray(threshold.run(ts).values).ravel()


ALL_RULES = (
    FixedThreshold(low=-1.0, high=1.0),
    QuantileThreshold(low=0.05, high=0.95),
    IqrThreshold(),
    MadThreshold(),
    EsdThreshold(),
)


# -- FixedThreshold ---------------------------------------------------------


def test_a_fixed_threshold_flags_both_tails() -> None:
    ts = scores(-9.0, 0.0, 0.5, 9.0)
    assert list(FixedThreshold(low=-5.0, high=5.0).run(ts).values.ravel()) == [
        1.0,
        0.0,
        0.0,
        1.0,
    ]


def test_a_fixed_threshold_needs_no_fitting() -> None:
    assert FixedThreshold(high=1.0).fitted


def test_a_fixed_threshold_leaves_an_unset_side_unbounded() -> None:
    ts = scores(-1e18, 0.0, 2.0)
    assert list(FixedThreshold(high=1.0).run(ts).values.ravel()) == [0.0, 0.0, 1.0]


def test_a_fixed_threshold_treats_its_bounds_as_inclusive() -> None:
    """A score exactly on the bound is inside the normal range."""
    ts = scores(-5.0, 5.0)
    assert list(FixedThreshold(low=-5.0, high=5.0).run(ts).values.ravel()) == [0.0, 0.0]


def test_a_fixed_threshold_rejects_having_no_bound_at_all() -> None:
    with pytest.raises(ValueError, match="at least one of low"):
        FixedThreshold()


def test_a_fixed_threshold_rejects_losing_its_last_bound_later() -> None:
    threshold = FixedThreshold(high=1.0).set_params(high=None)
    with pytest.raises(ValueError, match="at least one of low"):
        threshold.run(scores(1.0))


# -- QuantileThreshold ------------------------------------------------------


def test_a_quantile_threshold_flags_the_upper_tail() -> None:
    ts = scores(1.0, 2.0, 3.0, 4.0, 100.0)
    threshold = QuantileThreshold(high=0.9).fit(ts)
    # Linear interpolation between the 4th and 5th of five sorted scores.
    assert threshold.high_ == pytest.approx(61.6)
    assert list(threshold.run(ts).values.ravel()) == [0.0, 0.0, 0.0, 0.0, 1.0]


def test_a_quantile_threshold_flags_the_lower_tail() -> None:
    ts = scores(-100.0, 1.0, 2.0, 3.0, 4.0)
    threshold = QuantileThreshold(low=0.1).fit(ts)
    assert threshold.low_ == pytest.approx(-59.6)
    assert list(threshold.run(ts).values.ravel()) == [1.0, 0.0, 0.0, 0.0, 0.0]


def test_a_quantile_threshold_leaves_an_unset_side_unbounded() -> None:
    threshold = QuantileThreshold(high=0.5).fit(scores(1.0, 2.0, 3.0))
    assert threshold.low_ == -math.inf
    assert threshold.high_ == 2.0


def test_a_quantile_threshold_holds_the_training_cutoff_for_later_data() -> None:
    """The line is drawn by history, not re-drawn for each new series."""
    threshold = QuantileThreshold(high=0.99).fit(scores(*range(100)))
    learned = threshold.high_
    threshold.run(scores(1000.0, 2000.0, 3000.0))
    assert threshold.high_ == learned


def test_a_quantile_threshold_rejects_a_quantile_out_of_range() -> None:
    with pytest.raises(ValueError, match=r"high=1.5 is not a quantile"):
        QuantileThreshold(high=1.5)


def test_a_quantile_threshold_rejects_having_no_bound_at_all() -> None:
    with pytest.raises(ValueError, match="at least one of low"):
        QuantileThreshold()


# -- IqrThreshold -----------------------------------------------------------


def test_an_iqr_threshold_learns_the_box_plot_whiskers() -> None:
    ts = scores(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 99.0)
    threshold = IqrThreshold(factor=1.5).fit(ts)
    # Q1 = 3, Q3 = 7, IQR = 4.
    assert (threshold.low_, threshold.high_) == (-3.0, 13.0)
    assert list(threshold.run(ts).values.ravel())[-1] == 1.0


def test_an_iqr_threshold_takes_a_different_factor_per_tail() -> None:
    ts = scores(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    threshold = IqrThreshold(factor=(1.0, 2.0)).fit(ts)
    # Q1 = 3, Q3 = 7, IQR = 4.
    assert (threshold.low_, threshold.high_) == (-1.0, 15.0)


def test_an_iqr_threshold_leaves_a_none_tail_unbounded() -> None:
    ts = scores(1.0, 2.0, 3.0, 4.0, 5.0)
    threshold = IqrThreshold(factor=(None, 1.0)).fit(ts)
    assert threshold.low_ == -math.inf
    assert threshold.high_ == 6.0


def test_a_one_sided_iqr_threshold_ignores_the_lower_tail() -> None:
    ts = scores(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0)
    labels = IqrThreshold(factor=(None, 1.0)).fit(ts).run(ts)
    assert list(labels.values.ravel()) == [0.0] * 7 + [1.0]


def test_an_iqr_threshold_resists_an_outlier_in_its_training_data() -> None:
    """The point of quartiles: the outlier does not widen the range hiding it."""
    quiet = [10.0] * 20 + [11.0] * 20
    with_outlier = [*quiet, 5000.0]
    without = IqrThreshold().fit(scores(*quiet)).high_
    with_it = IqrThreshold().fit(scores(*with_outlier)).high_
    assert with_it == pytest.approx(without, abs=0.5)


def test_an_iqr_threshold_rejects_a_negative_factor() -> None:
    with pytest.raises(ValueError, match="is negative"):
        IqrThreshold(factor=-1.0)


def test_an_iqr_threshold_rejects_a_factor_pair_of_the_wrong_length() -> None:
    with pytest.raises(ValueError, match="got 3 items"):
        IqrThreshold(factor=(1.0, 2.0, 3.0))  # type: ignore[arg-type]


# -- MadThreshold -----------------------------------------------------------


def test_a_mad_threshold_scales_the_deviation_to_a_standard_deviation() -> None:
    ts = scores(4.0, 5.0, 6.0, 5.0, 4.0, 6.0, 40.0)
    threshold = MadThreshold(factor=3.0).fit(ts)
    # median 5, absolute deviations [1, 0, 1, 0, 1, 1, 35] -> median 1.
    assert threshold.high_ == pytest.approx(5.0 + 3.0 * MAD_SCALE)
    assert threshold.low_ == pytest.approx(5.0 - 3.0 * MAD_SCALE)
    assert list(threshold.run(ts).values.ravel()) == [0.0] * 6 + [1.0]


def test_the_mad_scale_recovers_the_standard_deviation_of_a_normal_sample() -> None:
    """The reason for the 1.4826: it puts the MAD on the sigma scale."""
    sample = np.random.default_rng(0).normal(scale=3.0, size=200_000)
    estimate = MAD_SCALE * np.median(np.abs(sample - np.median(sample)))
    assert estimate == pytest.approx(3.0, rel=0.02)


def test_a_mad_threshold_takes_a_different_factor_per_tail() -> None:
    # median 0, absolute deviations [2, 1, 0, 1, 2] -> median 1.
    ts = scores(-2.0, -1.0, 0.0, 1.0, 2.0)
    threshold = MadThreshold(factor=(None, 2.0)).fit(ts)
    assert threshold.low_ == -math.inf
    assert threshold.high_ == pytest.approx(2.0 * MAD_SCALE)


def test_a_mad_threshold_rejects_a_negative_factor() -> None:
    with pytest.raises(ValueError, match="is negative"):
        MadThreshold(factor=(1.0, -2.0))


# -- EsdThreshold -----------------------------------------------------------


def test_an_esd_threshold_finds_the_planted_outlier() -> None:
    values = np.random.default_rng(0).normal(size=60)
    values[17] = 12.0
    ts = scores(*values)
    labels = EsdThreshold().fit(ts).run(ts)
    assert list(np.flatnonzero(labels.values.ravel() == 1.0)) == [17]


def test_an_esd_threshold_keeps_only_sufficient_statistics() -> None:
    """No training data is retained, so prediction is arithmetic on three numbers."""
    values = np.random.default_rng(1).normal(size=40)
    values[5] = 20.0
    threshold = EsdThreshold().fit(scores(*values))
    normal = np.delete(values, 5)
    assert threshold.count_ == 39
    assert threshold.sum_ == pytest.approx(normal.sum())
    assert threshold.sum_squares_ == pytest.approx((normal**2).sum())


def test_an_esd_threshold_excludes_every_planted_outlier_from_the_normal_set() -> None:
    values = np.random.default_rng(2).normal(size=80)
    values[[10, 40, 70]] = [15.0, -15.0, 18.0]
    threshold = EsdThreshold().fit(scores(*values))
    assert threshold.count_ == 77


def test_a_stricter_esd_alpha_flags_fewer_points() -> None:
    values = np.random.default_rng(3).normal(size=100)
    values[[3, 50]] = [3.6, 8.0]
    ts = scores(*values)
    lenient = EsdThreshold(alpha=0.2).fit(ts).run(ts).values.ravel()
    strict = EsdThreshold(alpha=1e-6).fit(ts).run(ts).values.ravel()
    assert np.nansum(strict) < np.nansum(lenient)


def test_an_esd_threshold_judges_later_points_against_the_training_set() -> None:
    threshold = EsdThreshold().fit(scores(*np.random.default_rng(4).normal(size=50)))
    labels = threshold.run(scores(0.1, 20.0, -0.2))
    assert list(labels.values.ravel()) == [0.0, 1.0, 0.0]


def test_an_esd_threshold_reports_unknown_when_trained_on_too_little() -> None:
    threshold = EsdThreshold().fit(scores(1.0))
    assert threshold.count_ == 0
    assert np.isnan(threshold.run(scores(1.0, 500.0)).values).all()


def test_an_esd_threshold_flags_nothing_in_a_constant_series() -> None:
    ts = scores(*([7.0] * 30))
    labels = EsdThreshold().fit(ts).run(ts)
    assert not np.any(labels.values == 1.0)


def test_an_esd_threshold_rejects_an_alpha_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        EsdThreshold(alpha=0.0)


# -- shared behaviour across every rule -------------------------------------


@pytest.mark.parametrize("threshold", ALL_RULES, ids=lambda t: type(t).__name__)
def test_every_threshold_passes_a_missing_score_through_as_unknown(
    threshold: Threshold,
) -> None:
    """An unmeasured point cannot be declared normal."""
    ts = scores(0.0, 1.0, np.nan, 0.5, 0.0, 1.0, 0.0, 0.5)
    labels = labels_of(threshold.clone(), ts)
    assert np.isnan(labels[2])
    assert not np.isnan(np.delete(labels, 2)).any()


@pytest.mark.parametrize("threshold", ALL_RULES, ids=lambda t: type(t).__name__)
def test_every_threshold_returns_all_unknown_for_an_all_missing_series(
    threshold: Threshold,
) -> None:
    ts = scores(np.nan, np.nan, np.nan, np.nan)
    assert np.isnan(labels_of(threshold.clone(), ts)).all()


@pytest.mark.parametrize("threshold", ALL_RULES, ids=lambda t: type(t).__name__)
def test_every_threshold_flags_nothing_in_a_constant_series(
    threshold: Threshold,
) -> None:
    """A series with no variation has no anomalies, and warns about nothing."""
    ts = scores(*([0.0] * 20))
    labels = labels_of(threshold.clone(), ts)
    assert not np.any(labels == 1.0)


@pytest.mark.parametrize("threshold", ALL_RULES, ids=lambda t: type(t).__name__)
def test_every_threshold_survives_a_single_observation(
    threshold: Threshold,
) -> None:
    labels = labels_of(threshold.clone(), scores(0.0))
    assert labels.shape == (1,)


@pytest.mark.parametrize("threshold", ALL_RULES, ids=lambda t: type(t).__name__)
def test_every_threshold_emits_only_the_three_label_states(
    threshold: Threshold,
) -> None:
    values = np.random.default_rng(5).normal(size=50)
    values[10] = 40.0
    labels = labels_of(threshold.clone(), scores(*values))
    assert set(np.unique(labels[~np.isnan(labels)])) <= {0.0, 1.0}


@pytest.mark.parametrize("threshold", ALL_RULES, ids=lambda t: type(t).__name__)
def test_every_threshold_round_trips_its_parameters_through_clone(
    threshold: Threshold,
) -> None:
    copy = threshold.clone()
    assert copy.get_params() == threshold.get_params()
    assert repr(copy) == repr(threshold)
    assert copy is not threshold


def test_clone_carries_a_factor_pair_intact() -> None:
    """Every constructor parameter travels, including a compound one."""
    original = IqrThreshold(factor=(None, 4.5))
    assert original.clone().factor == (None, 4.5)


# -- fan-out and backends ---------------------------------------------------


def test_a_threshold_fits_each_column_of_a_frame_independently() -> None:
    index = pd.date_range("2024-01-01", periods=9, freq="h", name="time")
    frame = pd.DataFrame(
        {
            "small": [1.0, 2, 3, 4, 5, 6, 7, 8, 99],
            "large": [100.0, 200, 300, 400, 500, 600, 700, 800, 9900],
        },
        index=index,
    )
    threshold = IqrThreshold(factor=1.5).fit(frame)

    assert threshold._column_models is not None
    cutoffs = {
        name: model.high_  # type: ignore[attr-defined]
        for name, model in threshold._column_models.items()
    }
    assert cutoffs == {"small": 13.0, "large": 1300.0}
    labels = threshold.apply(frame)
    assert list(labels["small"]) == [0.0] * 8 + [1.0]
    assert list(labels["large"]) == [0.0] * 8 + [1.0]


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_threshold_returns_the_backend_it_was_given(backend: str) -> None:
    native = make_native(backend, [1.0, 2, 3, 4, 5, 6, 7, 8, 99])
    assert type(IqrThreshold().fit_apply(native)) is type(native)


def test_every_backend_produces_identical_labels() -> None:
    values = [1.0, 2, 3, 4, 5, 6, 7, 8, 99]
    results = [
        TimeSeries.from_any(IqrThreshold(factor=1.5).fit_apply(make_native(b, values)))
        for b in BACKENDS
    ]
    for other in results[1:]:
        np.testing.assert_array_equal(other.values, results[0].values)


def test_every_backend_produces_identical_esd_labels() -> None:
    values = np.random.default_rng(6).normal(size=40)
    values[9] = 15.0
    results = [
        TimeSeries.from_any(EsdThreshold().fit_apply(make_native(b, values)))
        for b in BACKENDS
    ]
    for other in results[1:]:
        np.testing.assert_array_equal(other.values, results[0].values)


def test_a_threshold_must_be_fitted_before_use() -> None:
    with pytest.raises(RuntimeError, match="must be fitted"):
        IqrThreshold().apply(scores(1.0, 2.0))


# -- SignedThreshold -------------------------------------------------------


def test_a_signed_threshold_judges_magnitude_and_reports_the_chosen_side() -> None:
    time = np.arange("2024-01-01", "2024-01-11", dtype="datetime64[D]")
    scores = [0.1, -0.2, 0.0, 0.3, -0.1, 8.0, 0.1, -0.3, 0.2, -9.0]
    ts = TimeSeries.from_arrays(time, scores)
    fence = IqrThreshold(factor=(None, 3.0))
    for side, expected in (("both", [5, 9]), ("positive", [5]), ("negative", [9])):
        labels = SignedThreshold(fence.clone(), side=side).fit_apply(ts)
        assert list(np.flatnonzero(labels.values.ravel() == 1.0)) == expected


def test_a_signed_threshold_keeps_unknown_scores_unknown() -> None:
    time = np.arange("2024-01-01", "2024-01-04", dtype="datetime64[D]")
    ts = TimeSeries.from_arrays(time, [np.nan, -5.0, 5.0])
    labels = SignedThreshold(FixedThreshold(high=1.0), side="positive").apply(ts)
    np.testing.assert_array_equal(labels.values.ravel(), [np.nan, 0.0, 1.0])


def test_a_signed_threshold_rejects_a_bad_side_or_a_non_threshold() -> None:
    with pytest.raises(ValueError, match="is not one of"):
        SignedThreshold(FixedThreshold(high=1.0), side="up")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="does not produce labels"):
        SignedThreshold(object())  # type: ignore[arg-type]


def test_a_signed_threshold_is_as_trainable_as_what_it_wraps() -> None:
    assert not SignedThreshold(FixedThreshold(high=1.0)).is_trainable
    assert SignedThreshold(IqrThreshold()).is_trainable
