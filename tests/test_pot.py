"""Tests for the peaks-over-threshold fence.

Unlike the other threshold rules, this one has no closed form to pin down: the
cut-off is where a fitted generalised Pareto tail puts a requested probability.
So the assertions come in three kinds. The fit itself is checked against
``scipy.stats.genpareto`` by log-likelihood, which is the thing being maximised
and the only thing two different optimisers have to agree on. The fence is
checked against the analytic quantile of a distribution we sampled from, and
against the sample maximum it is supposed to extrapolate past. Everything else —
validation, missing data, serialisation, the streaming update — is exact.
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest
from scipy.stats import genpareto

from hazure import Component, Detector, TimeSeries
from hazure.scorers import DeviationScorer
from hazure.thresholds import PotThreshold
from hazure.thresholds.pot import (
    _fence,
    _fit_gpd,
    _log_likelihood,
    _stationary_points,
)
from tests.helpers import BACKENDS, make_native

if TYPE_CHECKING:
    from collections.abc import Callable

# -- helpers ----------------------------------------------------------------


def scores(values: np.ndarray | list[float]) -> TimeSeries:
    """Build a minutely score series from raw numbers."""
    column = np.asarray(values, dtype=float)
    time = np.arange(column.size, dtype=np.int64) * 60_000_000_000
    return TimeSeries.from_arrays(time, column)


def pareto_sample(
    rng: np.random.Generator, shape: float, size: int, scale: float = 1.0
) -> np.ndarray:
    """Draw from a generalised Pareto by inverting its own quantile function.

    Sampling this way rather than through scipy keeps the data a fixed function
    of the seed, so the comparison against scipy's fit is not also a comparison
    against its random variates.
    """
    uniform = rng.uniform(size=size)
    if shape == 0.0:
        return -scale * np.log(uniform)
    return scale * (uniform**-shape - 1.0) / shape


def exponential(size: int = 4000, seed: int = 0) -> np.ndarray:
    """Standard exponential scores, whose tail quantiles are known in closed form."""
    return np.random.default_rng(seed).exponential(size=size)


def laplace(size: int = 12_000, seed: int = 0) -> np.ndarray:
    """Symmetric scores with exactly exponential tails on both sides."""
    return np.random.default_rng(seed).laplace(size=size)


#: The one-in-ten-thousand point of a standard Laplace, whose tails are exactly
#: exponential in both directions: P(X > x) = exp(-x) / 2, so the point where
#: exceedance has probability q is -log(2q). Both tails are therefore a
#: generalised Pareto with a shape of zero — the one case where the model being
#: fitted is not an approximation, so the fence has an exact target to hit.
LAPLACE_TAIL = -math.log(2e-4)


# -- the Grimshaw fit against scipy ------------------------------------------


@pytest.mark.parametrize(
    ("name", "shape"),
    [
        ("exponential", 0.0),
        ("heavy", 0.5),
        ("light", -0.3),
        ("lighter", -0.6),
    ],
)
def test_the_tail_fit_matches_scipys_likelihood(name: str, shape: float) -> None:
    """Two optimisers need not agree on parameters, only on what they maximise."""
    peaks = pareto_sample(np.random.default_rng(0), shape, 500)
    ours = _log_likelihood(peaks, *_fit_gpd(peaks))
    theirs = _log_likelihood(peaks, *_scipy_fit(peaks))
    assert ours >= theirs - 1e-6, name


def _scipy_fit(peaks: np.ndarray) -> tuple[float, float]:
    """Maximum-likelihood shape and scale from scipy, with the location pinned."""
    shape, _, scale = genpareto.fit(peaks, floc=0.0)
    return float(shape), float(scale)


@pytest.mark.parametrize("shape", [0.0, 0.25, 0.75, -0.2, -0.5])
def test_the_tail_fit_is_no_worse_than_scipy_across_sample_sizes(shape: float) -> None:
    rng = np.random.default_rng(1)
    for size in (20, 60, 250, 1000):
        peaks = pareto_sample(rng, shape, size)
        ours = _log_likelihood(peaks, *_fit_gpd(peaks))
        theirs = _log_likelihood(peaks, *_scipy_fit(peaks))
        assert ours >= theirs - 1e-6, (shape, size)


def test_the_tail_fit_recovers_the_shape_it_was_sampled_from() -> None:
    peaks = pareto_sample(np.random.default_rng(2), 0.5, 20_000)
    shape, scale = _fit_gpd(peaks)
    assert shape == pytest.approx(0.5, abs=0.05)
    assert scale == pytest.approx(1.0, abs=0.05)


def test_the_tail_fit_always_returns_a_positive_scale() -> None:
    rng = np.random.default_rng(3)
    for shape in (-0.9, -0.4, 0.0, 0.6, 1.5):
        _, scale = _fit_gpd(pareto_sample(rng, shape, 200))
        assert scale > 0.0


def test_the_tail_fit_stays_above_the_regularity_boundary() -> None:
    """Below a shape of -1 the likelihood is unbounded, so no estimate is taken."""
    rng = np.random.default_rng(4)
    for shape in (-1.0, -1.5, -3.0):
        peaks = pareto_sample(rng, shape, 400)
        fit_shape, fit_scale = _fit_gpd(peaks)
        assert fit_shape > -1.0
        assert math.isfinite(_log_likelihood(peaks, fit_shape, fit_scale))


def test_every_root_of_the_likelihood_equation_solves_it() -> None:
    peaks = pareto_sample(np.random.default_rng(5), 0.4, 300)
    largest, mean = float(peaks.max()), float(peaks.mean())
    roots = list(_stationary_points(peaks, largest, mean))
    assert roots
    for theta in roots:
        terms = 1.0 + theta * peaks
        residual = np.mean(1.0 / terms) * (1.0 + np.mean(np.log(terms))) - 1.0
        assert abs(float(residual)) < 1e-6


def test_the_log_likelihood_rejects_parameters_that_exclude_an_excess() -> None:
    peaks = np.array([0.5, 1.0, 4.0])
    # A shape of -0.1 with scale 1 puts the tail's endpoint at 10, inside which
    # every excess must lie; scale 0.2 moves it to 2 and leaves 4.0 outside.
    assert math.isfinite(_log_likelihood(peaks, -0.1, 1.0))
    assert _log_likelihood(peaks, -0.1, 0.2) == -math.inf
    assert _log_likelihood(peaks, 0.3, 0.0) == -math.inf


def test_the_fence_uses_the_exponential_form_at_a_shape_of_zero() -> None:
    assert _fence(2.0, 0.0, 3.0, 0.01) == pytest.approx(2.0 - 3.0 * math.log(0.01))
    # And approaches it as the shape is taken to zero from either side.
    for shape in (1e-7, -1e-7):
        assert _fence(2.0, shape, 3.0, 0.01) == pytest.approx(
            _fence(2.0, 0.0, 3.0, 0.01), rel=1e-4
        )


# -- extrapolation ----------------------------------------------------------


def test_a_pot_threshold_places_the_fence_beyond_the_largest_training_score() -> None:
    """The whole point: a quantile of the sample could never reach here."""
    sample = exponential()
    threshold = PotThreshold(high=1e-4).fit(scores(sample))
    assert threshold.high_ > sample.max()
    # A standard exponential exceeds -log(q) with probability q.
    assert threshold.high_ == pytest.approx(-math.log(1e-4), rel=0.1)


def test_a_pot_threshold_records_the_tail_it_fitted() -> None:
    threshold = PotThreshold(high=1e-4, level=0.98).fit(scores(exponential()))
    tail = threshold.tail_["high"]
    assert set(tail) == {"start", "shape", "scale", "peaks"}
    assert tail["peaks"] == 80.0
    assert tail["start"] == pytest.approx(np.quantile(exponential(), 0.98))
    # An exponential is the boundary case of the generalised Pareto family.
    assert tail["shape"] == pytest.approx(0.0, abs=0.15)


def test_a_stricter_probability_puts_the_fence_further_out() -> None:
    sample = scores(exponential())
    lenient = PotThreshold(high=1e-2).fit(sample).high_
    strict = PotThreshold(high=1e-6).fit(sample).high_
    assert strict > lenient


def test_a_pot_threshold_flags_nothing_in_the_scores_it_was_fitted_on() -> None:
    """One in ten thousand, asked of four thousand samples, should mean silence."""
    sample = scores(exponential())
    labels = PotThreshold(high=1e-4).fit(sample).run(sample)
    assert float(labels.values.sum()) == 0.0


def test_a_pot_threshold_flags_a_score_beyond_the_fence() -> None:
    threshold = PotThreshold(high=1e-4).fit(scores(exponential()))
    labels = threshold.run(scores([0.5, threshold.high_, threshold.high_ + 1.0]))
    assert list(labels.values.ravel()) == [0.0, 0.0, 1.0]


# -- the lower tail ---------------------------------------------------------


def test_a_pot_threshold_flags_the_lower_tail() -> None:
    sample = laplace()
    threshold = PotThreshold(low=1e-4, high=None).fit(scores(sample))
    assert threshold.high_ == math.inf
    assert set(threshold.tail_) == {"low"}
    assert threshold.low_ == pytest.approx(-LAPLACE_TAIL, rel=0.15)
    # The lower tail is fitted on negated scores; reported in the original sign
    # it must come out below the body, not above it.
    assert threshold.low_ < float(np.quantile(sample, 1e-3))
    assert threshold.tail_["low"]["start"] < float(np.median(sample))
    labels = threshold.run(scores([threshold.low_ - 1.0, 0.0, 100.0]))
    assert list(labels.values.ravel()) == [1.0, 0.0, 0.0]


def test_a_pot_threshold_fits_both_tails_at_once() -> None:
    sample = laplace()
    threshold = PotThreshold(low=1e-4, high=1e-4).fit(scores(sample))
    assert set(threshold.tail_) == {"low", "high"}
    assert threshold.high_ == pytest.approx(LAPLACE_TAIL, rel=0.15)
    assert threshold.low_ == pytest.approx(-LAPLACE_TAIL, rel=0.15)
    # A symmetric distribution should get a roughly symmetric pair of fences.
    assert threshold.high_ == pytest.approx(-threshold.low_, rel=0.2)
    labels = threshold.run(scores([-99.0, 0.0, 99.0]))
    assert list(labels.values.ravel()) == [1.0, 0.0, 1.0]


def test_a_pot_threshold_leaves_an_unset_side_unbounded() -> None:
    threshold = PotThreshold(high=1e-3).fit(scores(exponential()))
    assert threshold.low_ == -math.inf
    assert set(threshold.tail_) == {"high"}


# -- validation -------------------------------------------------------------


def test_a_pot_threshold_rejects_having_no_bound_at_all() -> None:
    with pytest.raises(ValueError, match="at least one of low"):
        PotThreshold(low=None, high=None)


def test_a_pot_threshold_rejects_losing_its_last_bound_later() -> None:
    threshold = PotThreshold(high=1e-3).set_params(high=None)
    with pytest.raises(ValueError, match="at least one of low"):
        threshold.fit(scores(exponential()))


@pytest.mark.parametrize("value", [0.0, 1.0, -0.1, 1.5, 42.0])
def test_a_pot_threshold_rejects_a_probability_outside_the_unit_interval(
    value: float,
) -> None:
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        PotThreshold(high=value)
    with pytest.raises(ValueError, match=f"low={value}"):
        PotThreshold(low=value, high=None)


@pytest.mark.parametrize("level", [0.0, 1.0, -0.5, 2.0])
def test_a_pot_threshold_rejects_a_level_outside_the_unit_interval(
    level: float,
) -> None:
    with pytest.raises(ValueError, match="level"):
        PotThreshold(high=1e-3, level=level)


def test_a_pot_threshold_refuses_a_probability_that_is_not_in_the_tail() -> None:
    """Asked about the body of the distribution, it says so rather than answering."""
    with pytest.raises(ValueError, match=r"Lower high below 0.02, or lower level"):
        PotThreshold(high=0.1, level=0.98).fit(scores(exponential()))


def test_the_refusal_names_the_side_that_could_not_be_placed() -> None:
    with pytest.raises(ValueError, match=r"the low fence at low=0.5"):
        PotThreshold(low=0.5, high=None, level=0.9).fit(scores(exponential()))


def test_a_lower_level_lets_a_looser_probability_be_asked_for() -> None:
    """The fix the error message suggests actually works."""
    threshold = PotThreshold(high=0.1, level=0.5).fit(scores(exponential()))
    assert threshold.high_ == pytest.approx(-math.log(0.1), rel=0.15)


# -- too little data, and missing data --------------------------------------


def test_a_pot_threshold_leaves_a_side_unknown_with_too_few_excesses(
    hourly: pd.Series,
) -> None:
    """Twenty-four points leave no tail to fit, so no label can be justified."""
    threshold = PotThreshold(high=1e-3).fit(hourly)
    assert math.isnan(threshold.high_)
    assert threshold.tail_ == {}
    assert np.isnan(threshold.apply(hourly).to_numpy()).all()


def test_a_side_is_unknown_when_the_excesses_fall_one_short() -> None:
    # 500 scores at level 0.98 leave 10 excesses, which is the fewest allowed;
    # 450 leave 9, which is not.
    sample = exponential()
    assert not math.isnan(PotThreshold(high=1e-3).fit(scores(sample[:500])).high_)
    assert math.isnan(PotThreshold(high=1e-3).fit(scores(sample[:450])).high_)


def test_a_pot_threshold_reports_unknown_for_an_all_missing_series() -> None:
    threshold = PotThreshold(high=1e-3).fit(scores([np.nan] * 600))
    assert math.isnan(threshold.high_)
    assert np.isnan(threshold.run(scores([np.nan, 1.0])).values).all()


def test_a_pot_threshold_flags_nothing_in_a_constant_series() -> None:
    ts = scores([7.0] * 600)
    labels = PotThreshold(high=1e-3).fit(ts).run(ts)
    assert not np.any(labels.values == 1.0)


def test_a_pot_threshold_survives_a_single_observation() -> None:
    labels = PotThreshold(high=1e-3).fit_apply(scores([0.0]))
    assert labels.values.shape == (1, 1)


def test_a_pot_threshold_passes_a_missing_score_through_as_unknown() -> None:
    threshold = PotThreshold(high=1e-3).fit(scores(exponential()))
    labels = np.asarray(threshold.run(scores([0.0, np.nan, 1e9])).values).ravel()
    assert np.isnan(labels[1])
    assert list(np.delete(labels, 1)) == [0.0, 1.0]


def test_a_pot_threshold_ignores_missing_scores_when_fitting() -> None:
    """A gap in the training scores is absent, not zero."""
    sample = exponential()
    with_gaps = np.concatenate([sample, np.full(500, np.nan)])
    assert PotThreshold(high=1e-4).fit(scores(with_gaps)).high_ == pytest.approx(
        PotThreshold(high=1e-4).fit(scores(sample)).high_
    )


def test_a_pot_threshold_emits_only_the_three_label_states() -> None:
    sample = exponential()
    sample[123] = 40.0
    labels = PotThreshold(high=1e-4).fit_apply(scores(sample)).values.ravel()
    assert set(np.unique(labels[~np.isnan(labels)])) <= {0.0, 1.0}


# -- streaming --------------------------------------------------------------


def test_update_before_fitting_is_refused() -> None:
    with pytest.raises(RuntimeError, match="must be fitted before update"):
        PotThreshold(high=1e-3).update(1.0)


def test_update_many_before_fitting_is_refused() -> None:
    with pytest.raises(RuntimeError, match="must be fitted before update"):
        PotThreshold(high=1e-3).update_many([1.0, 2.0])


def test_update_passes_a_missing_score_through_and_changes_nothing() -> None:
    threshold = PotThreshold(high=1e-3).fit(scores(exponential()))
    state = (threshold.high_, threshold._seen, threshold.tail_["high"]["peaks"])
    assert math.isnan(threshold.update(math.nan))
    assert (threshold.high_, threshold._seen, threshold.tail_["high"]["peaks"]) == state


def test_update_reports_unknown_while_the_fence_is_unknown(hourly: pd.Series) -> None:
    threshold = PotThreshold(high=1e-3).fit(hourly)
    assert math.isnan(threshold.update(1e9))


def test_update_flags_a_score_beyond_the_fence_without_widening_the_tail() -> None:
    """An anomaly is not evidence about how the normal tail behaves."""
    threshold = PotThreshold(high=1e-3).fit(scores(exponential()))
    before, peaks = threshold.high_, threshold.tail_["high"]["peaks"]
    assert threshold.update(before + 50.0) == 1.0
    assert threshold.high_ == before
    assert threshold.tail_["high"]["peaks"] == peaks


def test_update_takes_an_accepted_excess_into_the_tail() -> None:
    threshold = PotThreshold(high=1e-3).fit(scores(exponential()))
    before, peaks = threshold.high_, threshold.tail_["high"]["peaks"]
    inside = 0.5 * (threshold.tail_["high"]["start"] + before)
    assert threshold.update(inside) == 0.0
    assert threshold.tail_["high"]["peaks"] == peaks + 1.0
    assert threshold.high_ != before


def test_update_leaves_the_tail_alone_for_a_score_below_it() -> None:
    """An ordinary score is one more observation and nothing else."""
    threshold = PotThreshold(high=1e-3).fit(scores(exponential()))
    before, peaks = threshold.high_, threshold.tail_["high"]["peaks"]
    assert threshold.update(0.1) == 0.0
    assert threshold.tail_["high"]["peaks"] == peaks
    # The same excesses now stand for a rarer event, so the fence eases down.
    assert threshold.high_ < before


def test_update_moves_the_lower_fence_down_as_its_tail_fills_in() -> None:
    threshold = PotThreshold(low=1e-3, high=None).fit(
        scores(np.random.default_rng(8).normal(size=4000))
    )
    before = threshold.low_
    for _ in range(20):
        threshold.update(0.5 * (threshold.tail_["low"]["start"] + threshold.low_))
    assert threshold.low_ < before
    assert threshold.tail_["low"]["peaks"] == 100.0
    assert threshold.update(threshold.low_ - 50.0) == 1.0


def test_update_many_returns_one_label_per_score() -> None:
    threshold = PotThreshold(high=1e-3).fit(scores(exponential()))
    arriving = np.random.default_rng(9).exponential(size=200)
    labels = threshold.update_many(arriving)
    assert labels.shape == (200,)
    assert labels.dtype == np.float64
    assert set(np.unique(labels)) <= {0.0, 1.0}


def test_update_many_roughly_honours_the_requested_alert_rate() -> None:
    """Streaming data from the fitted distribution should alarm about q of the time."""
    rng = np.random.default_rng(10)
    threshold = PotThreshold(high=1e-3).fit(scores(rng.exponential(size=4000)))
    alerts = float(threshold.update_many(rng.exponential(size=5000)).sum())
    # Five expected. The bound is loose because the count is a small integer
    # whose variance is its mean, and because the fence keeps moving.
    assert 0.0 < alerts < 25.0


def test_update_many_judges_each_score_against_the_fence_of_its_moment() -> None:
    """Streaming a batch matches feeding it one at a time."""
    arriving = np.random.default_rng(11).exponential(size=100)
    history = scores(exponential())
    batched = PotThreshold(high=1e-3).fit(history).update_many(arriving)
    single = PotThreshold(high=1e-3).fit(history)
    one_by_one = [single.update(float(value)) for value in arriving]
    np.testing.assert_array_equal(batched, np.asarray(one_by_one))


def test_a_burst_of_anomalies_does_not_drag_the_fence_up_after_it() -> None:
    """The failure mode the discard exists to prevent: accepting the next one."""
    threshold = PotThreshold(high=1e-3).fit(scores(exponential()))
    before, peaks = threshold.high_, threshold.tail_["high"]["peaks"]
    assert threshold.update_many([1e6] * 20).sum() == 20.0
    assert threshold.high_ <= before
    assert threshold.tail_["high"]["peaks"] == peaks
    assert threshold.update(1e6) == 1.0


# -- serialisation ----------------------------------------------------------


def test_a_fitted_pot_threshold_survives_a_round_trip_through_json() -> None:
    ts = scores(np.random.default_rng(12).normal(size=4000))
    fitted = PotThreshold(low=1e-3, high=1e-4, level=0.97).fit(ts)
    expected = fitted.run(ts)

    restored = PotThreshold.from_dict(json.loads(json.dumps(fitted.to_dict())))

    assert restored.get_params() == fitted.get_params()
    assert (restored.low_, restored.high_) == (fitted.low_, fitted.high_)
    assert restored.tail_ == fitted.tail_
    np.testing.assert_array_equal(restored.run(ts).values, expected.values)


def test_a_restored_pot_threshold_can_still_be_updated() -> None:
    fitted = PotThreshold(high=1e-3).fit(scores(exponential()))
    restored = PotThreshold.from_dict(json.loads(json.dumps(fitted.to_dict())))
    # The excesses travel too, so streaming picks up where it left off rather
    # than refitting from nothing.
    np.testing.assert_array_equal(restored._peaks["high"], fitted._peaks["high"])
    inside = 0.5 * (restored.tail_["high"]["start"] + restored.high_)
    assert restored.update(inside) == fitted.update(inside)
    assert restored.high_ == fitted.high_


def test_a_fitted_pot_threshold_round_trips_through_the_generic_loader() -> None:
    ts = scores(exponential())
    fitted = PotThreshold(high=1e-4).fit(ts)
    restored = Component.from_dict(fitted.to_dict())
    assert isinstance(restored, PotThreshold)
    assert restored.high_ == fitted.high_


def test_an_unfitted_pot_threshold_carries_only_its_parameters() -> None:
    stored = PotThreshold(low=0.01, high=None, level=0.9).to_dict()
    assert stored["state"] == {"low": 0.01, "high": None, "level": 0.9}
    assert not PotThreshold.from_dict(stored).fitted


# -- parameters -------------------------------------------------------------


def test_get_params_reports_the_probabilities_and_the_level() -> None:
    threshold = PotThreshold(low=0.02, high=1e-5, level=0.95)
    assert threshold.get_params() == {"low": 0.02, "high": 1e-5, "level": 0.95}


def test_clone_gives_an_unfitted_copy_with_the_same_parameters() -> None:
    fitted = PotThreshold(low=1e-3, high=1e-4, level=0.95).fit(scores(exponential()))
    copy = fitted.clone()
    assert copy.get_params() == fitted.get_params()
    assert repr(copy) == repr(fitted)
    assert not copy.fitted
    assert not hasattr(copy, "high_")
    with pytest.raises(RuntimeError, match="must be fitted"):
        copy.apply(scores(exponential()))


def test_fitting_twice_on_the_same_scores_gives_the_same_fence() -> None:
    """No randomness anywhere in the fit: the scan and the bisection are fixed."""
    ts = scores(exponential())
    first = PotThreshold(low=1e-3, high=1e-4).fit(ts)
    second = PotThreshold(low=1e-3, high=1e-4).fit(ts)
    assert (first.low_, first.high_) == (second.low_, second.high_)
    assert first.tail_ == second.tail_


# -- composition, fan-out and backends --------------------------------------


def test_a_pot_threshold_works_inside_a_score_detector() -> None:
    values = np.random.default_rng(13).normal(size=4000)
    values[1234] += 30.0
    frame = pd.DataFrame(
        {"x": values},
        index=pd.date_range("2024-01-01", periods=4000, freq="h", name="time"),
    )
    detector = Detector(DeviationScorer(), PotThreshold(high=1e-4))
    labels = detector.fit_detect(frame)
    assert list(np.flatnonzero(labels["x"].to_numpy() == 1.0)) == [1234]


def test_a_pot_threshold_fits_each_column_of_a_frame_independently() -> None:
    values = np.random.default_rng(14).exponential(size=4000)
    frame = pd.DataFrame(
        {"small": values, "large": values * 100.0},
        index=pd.date_range("2024-01-01", periods=4000, freq="h", name="time"),
    )
    threshold = PotThreshold(high=1e-4).fit(frame)

    assert threshold._column_models is not None
    cutoffs = {
        name: model.high_  # type: ignore[attr-defined]
        for name, model in threshold._column_models.items()
    }
    # Each column learns its own scale, so the fences differ by the same factor.
    assert cutoffs["large"] == pytest.approx(cutoffs["small"] * 100.0)
    labels = threshold.apply(frame)
    assert float(np.nansum(labels.to_numpy())) == 0.0


def test_a_pot_threshold_returns_the_backend_it_was_given(
    native_factory: Callable[..., Any],
) -> None:
    native = native_factory(exponential(size=2000))
    assert type(PotThreshold(high=1e-3).fit_apply(native)) is type(native)


def test_a_fitted_pot_threshold_round_trips_in_every_backend(
    native_factory: Callable[..., Any],
) -> None:
    sample = exponential(size=2000)
    native = native_factory(sample)
    fitted = PotThreshold(high=1e-4).fit(native)
    restored = PotThreshold.from_dict(json.loads(json.dumps(fitted.to_dict())))
    expected = TimeSeries.from_any(fitted.apply(native))
    actual = TimeSeries.from_any(restored.apply(native))
    assert actual.columns == expected.columns
    np.testing.assert_array_equal(actual.values, expected.values)


def test_every_backend_produces_identical_pot_labels() -> None:
    sample = exponential(size=2000)
    sample[42] = 60.0
    results = [
        TimeSeries.from_any(PotThreshold(high=1e-4).fit_apply(make_native(b, sample)))
        for b in BACKENDS
    ]
    for other in results[1:]:
        np.testing.assert_array_equal(other.values, results[0].values)


def test_a_pot_threshold_must_be_fitted_before_use() -> None:
    with pytest.raises(RuntimeError, match="must be fitted"):
        PotThreshold(high=1e-3).apply(scores(exponential()))


# ---------------------------------------------------------------------------
# what an alarm must not do to the tail
# ---------------------------------------------------------------------------


def test_a_flagged_score_moves_neither_the_excesses_nor_the_count() -> None:
    """A burst of anomalies must not tighten the fence.

    The count of observations is the denominator the target probability is
    rescaled by, so letting a flagged score raise it would *lower* the fence — and
    a run of anomalies would then talk the threshold into accepting the next one.
    The excesses and the count both describe the sample the tail was estimated
    from, and a flagged score has just been declared not to belong to it.
    """
    rng = np.random.default_rng(3)
    time = np.arange(4000, dtype=np.int64) * 60_000_000_000
    history = TimeSeries.from_arrays(time, rng.exponential(size=4000))
    threshold = PotThreshold(high=1e-3).fit(history)

    fence, seen, peaks = (
        threshold.high_,
        threshold._seen,
        threshold.tail_["high"]["peaks"],
    )
    for _ in range(500):
        assert threshold.update(1e6) == 1.0
    assert threshold.high_ == fence
    assert threshold._seen == seen
    assert threshold.tail_["high"]["peaks"] == peaks


def test_streaming_a_threshold_fitted_on_a_frame_is_refused() -> None:
    """One fence per column means no single tail for ``update`` to extend."""
    rng = np.random.default_rng(4)
    time = np.arange(3000, dtype=np.int64) * 60_000_000_000
    frame = TimeSeries.from_arrays(
        time, np.column_stack([rng.exponential(size=3000)] * 2), ["a", "b"]
    )
    threshold = PotThreshold(high=1e-3).fit(frame)
    with pytest.raises(RuntimeError, match="one independent fence per column"):
        threshold.update(1.0)
