"""Tests for the methods in ``hazure.methods``.

Each component gets two tests: one that plants an anomaly at a known position and
asserts the component finds it, and one that checks the plumbing — output length,
output type, no warnings. Anything with a closed-form answer is asserted exactly.

The three adapters are guarded with ``importorskip``, so this file passes whether
or not the optional extras are installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

from hazure import Component, TimeSeries, detectors
from hazure.scorers import (
    DampScorer,
    HampelScorer,
    MatrixProfileScorer,
    MstlResidualScorer,
    PeltScorer,
    RollingQuantileScorer,
    RupturesScorer,
    SpectralResidualScorer,
    StlResidualScorer,
)
from tests.helpers import BACKENDS, make_native

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def series(
    values: Sequence[float] | NDArray[np.float64], freq: str = "h"
) -> TimeSeries:
    """A regular univariate TimeSeries carrying ``values``."""
    array = np.asarray(values, dtype=float)
    index = pd.date_range("2024-01-01", periods=array.shape[0], freq=freq, name="time")
    return TimeSeries.from_any(pd.DataFrame({"x": array}, index=index))


def irregular(values: Sequence[float]) -> TimeSeries:
    """A series whose timestamps are unevenly spaced."""
    stamps = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-04", "2024-01-08"])
    return TimeSeries.from_any(
        pd.DataFrame({"x": np.asarray(values, dtype=float)}, index=stamps)
    )


def sine(n: int, period: int = 24, scale: float = 0.05, seed: int = 0) -> NDArray[Any]:
    """A noisy sine wave."""
    rng = np.random.default_rng(seed)
    return np.sin(np.arange(n) * 2 * np.pi / period) + rng.normal(scale=scale, size=n)


def spiky(n: int = 200, at: int = 137) -> NDArray[Any]:
    """A smooth series with one spike, at ``at``."""
    values = 10.0 + np.sin(np.arange(n) * np.pi / 12)
    values[at] = 30.0
    return values


def step(n: int = 120, at: int = 60, size: float = 10.0, seed: int = 0) -> NDArray[Any]:
    """A noisy series that steps up by ``size`` at ``at``."""
    rng = np.random.default_rng(seed)
    values = rng.normal(size=n)
    values[at:] += size
    return values


def discord(n: int = 400, window: int = 20, at: int = 200) -> NDArray[Any]:
    """A repeating shape with one stretch that does not repeat."""
    values = sine(n, period=window)
    values[at : at + window] = np.linspace(3.0, -3.0, window)
    return values


def numbers(result: Any) -> NDArray[np.float64]:
    """The values of a result, whatever flavour it came back as.

    The public verbs hand back the caller's native type, so a test that passed a
    TimeSeries gets a pandas frame back; ``run`` is the TimeSeries-in,
    TimeSeries-out entry point. This flattens either.
    """
    return np.asarray(TimeSeries.from_any(result).values.ravel(), dtype=float)


def flat_argmax(result: Any) -> int:
    """Position of the largest value of a univariate result."""
    return int(np.nanargmax(numbers(result)))


def flagged(result: Any) -> NDArray[np.int64]:
    """Positions labelled anomalous."""
    return np.flatnonzero(numbers(result) == 1.0)


# ---------------------------------------------------------------------------
# spectral residual
# ---------------------------------------------------------------------------


def test_the_spectral_residual_scorer_makes_a_planted_spike_most_salient() -> None:
    scores = SpectralResidualScorer().run(series(spiky(at=137)))
    assert flat_argmax(scores) == 137


def test_the_spectral_residual_scorer_returns_one_score_per_observation() -> None:
    ts = series(sine(120))
    scores = SpectralResidualScorer().run(ts)
    assert isinstance(scores, TimeSeries)
    assert scores.n_rows == ts.n_rows
    assert scores.columns == ("x",)
    assert np.isfinite(scores.values).all()


def test_the_spectral_residual_scorer_can_judge_the_most_recent_observation() -> None:
    # The extrapolated tail is what makes this possible: the edge artefact of the
    # transform lands on the invented points instead of on the last real one, so a
    # spike on the most recent observation is still among the most salient points
    # of the series.
    values = 10.0 + np.sin(np.arange(200) * np.pi / 12)
    values[-1] = 30.0
    scores = numbers(SpectralResidualScorer().run(series(values)))
    assert 199 in np.argsort(scores)[-2:]
    assert scores[-1] > 5.0 * np.median(scores)


def test_the_spectral_residual_scorer_leaves_missing_observations_unscored() -> None:
    values = spiky()
    values[40] = np.nan
    scores = SpectralResidualScorer().run(series(values)).values.ravel()
    assert np.isnan(scores[40])
    assert not np.isnan(np.delete(scores, 40)).any()


def test_the_spectral_residual_scorer_rejects_an_irregular_time_axis() -> None:
    with pytest.raises(ValueError, match="regular time axis"):
        SpectralResidualScorer().run(irregular([1.0, 2.0, 3.0, 4.0]))


def test_the_spectral_residual_scorer_rejects_a_series_window_of_one() -> None:
    with pytest.raises(ValueError, match="series_window"):
        SpectralResidualScorer(series_window=1)


def test_the_spectral_residual_detector_flags_the_planted_spike_alone() -> None:
    labels = detectors.spectral_residual(factor=12.0).fit_detect(series(spiky(at=137)))
    assert list(flagged(labels)) == [137]


def test_the_spectral_residual_detector_returns_binary_labels() -> None:
    ts = series(sine(120))
    labels = detectors.spectral_residual().fit(ts).run(ts)
    assert labels.n_rows == ts.n_rows
    assert set(np.unique(labels.values)) <= {0.0, 1.0}


# ---------------------------------------------------------------------------
# local order statistics
# ---------------------------------------------------------------------------


def test_the_hampel_scorer_ranks_a_planted_outlier_highest() -> None:
    values = np.tile([10.0, 11.0, 12.0, 11.0], 8)
    values[17] = 40.0
    assert flat_argmax(HampelScorer().run(series(values, freq="D"))) == 17


def test_the_hampel_scorer_blanks_half_a_window_at_each_end() -> None:
    ts = series(10.0 + np.random.default_rng(0).normal(size=40), freq="D")
    scores = numbers(HampelScorer(window=7).run(ts))
    assert scores.shape == (40,)
    assert np.isnan(scores[:3]).all()
    assert np.isnan(scores[-3:]).all()
    assert not np.isnan(scores[3:-3]).any()


def test_the_hampel_scorer_reports_an_undefined_score_where_there_is_no_spread() -> (
    None
):
    # A locally constant window has no scale, so a deviation cannot be expressed
    # in units of it.
    scores = HampelScorer().run(series(np.full(20, 5.0), freq="D"))
    assert np.isnan(scores.values).all()


def test_the_hampel_scorer_measures_a_deviation_in_local_standard_deviations() -> None:
    # Alternating 0 and 1 with a window of 3: every window is either (0, 1, 0) or
    # (1, 0, 1), so every local median is one of the two values and every
    # deviation from it is exactly 1. The local MAD is therefore 1, the scaled
    # spread is 1.4826, and every interior point scores 1 / 1.4826 = 0.6745.
    ts = series(np.tile([0.0, 1.0], 12), freq="D")
    scores = HampelScorer(window=3).run(ts).values.ravel()
    assert np.allclose(scores[1:-1], 1.0 / 1.482602218505602)
    assert np.isnan(scores[[0, -1]]).all()


def test_the_hampel_detector_flags_the_planted_outlier_without_fitting() -> None:
    values = np.tile([10.0, 11.0, 12.0, 11.0], 8)
    values[17] = 40.0
    labels = detectors.hampel().detect(series(values, freq="D"))
    assert list(flagged(labels)) == [17]


def test_the_hampel_detector_returns_binary_labels_with_an_unknown_margin() -> None:
    ts = series(10.0 + np.random.default_rng(0).normal(size=40), freq="D")
    labels = detectors.hampel(window=7).run(ts)
    assert labels.n_rows == ts.n_rows
    assert np.isnan(labels.values.ravel()[:3]).all()
    assert set(np.unique(labels.values[3:-3])) == {0.0}


def test_the_rolling_quantile_scorer_ranks_a_break_in_a_drift_highest() -> None:
    values = np.arange(40.0) + np.tile([0.0, 0.5, -0.5, 0.2], 10)
    values[26] += 12.0
    scores = RollingQuantileScorer(window=10).run(series(values, freq="D"))
    ranked = scores.values.ravel()
    assert flat_argmax(scores) == 26
    # The drift itself is worth far less than the break in it.
    assert ranked[26] > 5.0 * np.nanmax(np.delete(ranked, 26))


def test_the_rolling_quantile_scorer_scores_zero_inside_the_band() -> None:
    ts = series(np.tile([1.0, 2.0, 3.0, 2.0], 8), freq="D")
    scores = RollingQuantileScorer(window=8, low=0.0, high=1.0).run(ts).values.ravel()
    assert scores.shape == (32,)
    assert np.isnan(scores[:7]).all()
    # With the band at the window's own extremes, nothing can fall outside it.
    assert (scores[7:] == 0.0).all()


def test_the_rolling_quantile_scorer_rejects_a_bound_that_is_not_a_quantile() -> None:
    with pytest.raises(ValueError, match="not a quantile"):
        RollingQuantileScorer(window=5, high=1.5)


def test_the_rolling_quantile_detector_flags_the_break_in_a_drift() -> None:
    values = np.arange(40.0) + np.tile([0.0, 0.5, -0.5, 0.2], 10)
    values[26] += 12.0
    labels = detectors.rolling_quantile(window=10).fit_detect(series(values))
    assert list(flagged(labels)) == [26]


def test_the_rolling_quantile_detector_does_not_flag_an_ordinary_drift() -> None:
    # The window trails and includes the point being scored, so on a rising
    # series the newest observation is routinely its own window's maximum and
    # leaves the band by a little. Reporting every such excursion would flag
    # nearly half of this series; the fitted fence is what stops it.
    values = np.arange(40.0) + np.tile([0.0, 0.5, -0.5, 0.2], 10)
    ts = series(values)
    excursions = numbers(RollingQuantileScorer(window=10).score(ts))
    assert (np.nan_to_num(excursions) > 0.0).sum() > 10
    assert list(flagged(detectors.rolling_quantile(window=10).fit_detect(ts))) == []


def test_the_rolling_quantile_detector_returns_binary_labels() -> None:
    ts = series(sine(200))
    labels = detectors.rolling_quantile(window=24).fit(ts).run(ts)
    assert labels.n_rows == 200
    assert labels.columns == ("x",)
    assert set(np.unique(numbers(labels)[23:])) <= {0.0, 1.0}


# ---------------------------------------------------------------------------
# change-point segmentation
# ---------------------------------------------------------------------------


def _optimal_segmentation(
    values: NDArray[np.float64], penalty: float, min_size: int = 2
) -> list[int]:
    """Segment by unpruned dynamic programming, for comparison with PELT.

    Deliberately naive: every candidate start is reconsidered at every end, and
    each segment's cost is computed from scratch. PELT must agree with it exactly,
    since pruning is supposed to discard only candidates that could not have won.
    """
    n = values.shape[0]
    best = np.full(n + 1, np.inf)
    best[0] = -penalty
    previous = np.zeros(n + 1, dtype=int)
    for end in range(min_size, n + 1):
        for start in range(0, end - min_size + 1):
            segment = values[start:end]
            cost = float(((segment - segment.mean()) ** 2).sum())
            total = best[start] + cost + penalty
            if total < best[end]:
                best[end] = total
                previous[end] = start
    found: list[int] = []
    position = n
    while position > 0:
        start = int(previous[position])
        if start > 0:
            found.append(start)
        position = start
    return sorted(found)


def test_the_pelt_scorer_finds_a_planted_level_shift() -> None:
    scorer = PeltScorer().fit(series(step(at=60)))
    assert list(scorer.breakpoints_) == [60]


def test_the_pelt_scorer_reports_the_exact_size_of_a_noiseless_change() -> None:
    values = np.concatenate([np.zeros(50), np.full(50, 10.0)])
    scores = PeltScorer(penalty=1.0).fit_score(series(values)).values.ravel()
    assert scores[50] == pytest.approx(10.0)
    assert (np.delete(scores, 50) == 0.0).all()


def test_the_pelt_scorer_agrees_with_an_unpruned_dynamic_programme() -> None:
    values = step(n=60, at=30, size=4.0, seed=3)
    scorer = PeltScorer(penalty=5.0, min_size=2).fit(series(values))
    assert list(scorer.breakpoints_) == _optimal_segmentation(values, 5.0)


def test_the_pelt_scorer_finds_nothing_in_a_series_that_never_changes() -> None:
    scorer = PeltScorer().fit(series(np.random.default_rng(1).normal(size=200)))
    assert scorer.breakpoints_.size == 0
    assert (
        scorer.run(series(np.random.default_rng(1).normal(size=200))).values == 0.0
    ).all()


def test_the_pelt_scorer_honours_min_size_and_the_jump_grid() -> None:
    values = step(n=200, at=100, size=8.0, seed=2)
    values[150:] -= 8.0
    scorer = PeltScorer(min_size=10, jump=10).fit(series(values))
    assert list(scorer.breakpoints_) == [100, 150]
    assert (scorer.breakpoints_ % 10 == 0).all()


def test_the_pelt_scorer_derives_a_penalty_from_the_roughness_of_the_series() -> None:
    scorer = PeltScorer().fit(series(step(at=60)))
    # sigma is about 1 here and log(120) about 4.79, so 2 * sigma^2 * log(n) is
    # around 9.6.
    assert 5.0 < scorer.penalty_ < 15.0


def test_the_pelt_scorer_with_an_absolute_cost_ignores_an_outlier_in_a_segment() -> (
    None
):
    values = np.concatenate([np.zeros(60), np.full(60, 10.0)])
    values[20] = 500.0
    scorer = PeltScorer(cost="l1", penalty=20.0).fit(series(values))
    assert list(scorer.breakpoints_) == [60]


def test_the_pelt_scorer_returns_one_score_per_observation() -> None:
    ts = series(step())
    scores = PeltScorer().fit(ts).run(ts)
    assert isinstance(scores, TimeSeries)
    assert scores.n_rows == ts.n_rows
    assert scores.columns == ("x",)
    assert (scores.values >= 0.0).all()


def test_the_pelt_scorer_rejects_an_unknown_cost() -> None:
    with pytest.raises(ValueError, match="cost='l3'"):
        PeltScorer(cost="l3")  # type: ignore[arg-type]


def test_the_pelt_detector_flags_the_change_and_nothing_else() -> None:
    labels = detectors.pelt().fit_detect(series(step(at=60)))
    assert list(flagged(labels)) == [60]


def test_the_pelt_detector_returns_binary_labels() -> None:
    ts = series(step())
    labels = detectors.pelt().fit(ts).run(ts)
    assert labels.n_rows == 120
    assert set(np.unique(labels.values)) <= {0.0, 1.0}


def test_the_ruptures_scorer_finds_a_planted_level_shift() -> None:
    pytest.importorskip("ruptures")
    scorer = RupturesScorer(model="dynp", n_bkps=1).fit(series(step(at=60)))
    assert list(scorer.breakpoints_) == [60]


def test_the_ruptures_scorer_scores_the_change_it_found() -> None:
    pytest.importorskip("ruptures")
    ts = series(step(at=60, size=10.0))
    scores = RupturesScorer(model="binseg").fit(ts).run(ts)
    assert scores.n_rows == ts.n_rows
    assert scores.values.ravel()[60] == pytest.approx(10.0, abs=1.0)
    assert (np.delete(scores.values.ravel(), 60) == 0.0).all()


def test_the_ruptures_scorer_rejects_an_unknown_search_strategy() -> None:
    with pytest.raises(ValueError, match="model='wavelet'"):
        RupturesScorer(model="wavelet")


def test_the_ruptures_detector_flags_the_change_it_was_asked_for() -> None:
    pytest.importorskip("ruptures")
    labels = detectors.ruptures(model="dynp", n_bkps=1).fit_detect(series(step(at=60)))
    assert list(flagged(labels)) == [60]


def test_the_ruptures_detector_rejects_an_unknown_strategy_without_ruptures() -> None:
    # Validation precedes the lazy import, so this raises whether or not the
    # extra is installed.
    with pytest.raises(ValueError, match="model='wavelet'"):
        detectors.ruptures(model="wavelet")


# ---------------------------------------------------------------------------
# matrix profile
# ---------------------------------------------------------------------------


def test_the_matrix_profile_scorer_peaks_on_the_planted_discord() -> None:
    pytest.importorskip("stumpy")
    scores = MatrixProfileScorer(window=20).run(series(discord()))
    # A subsequence's distance is broadcast over the window it covers, so the
    # peak lands within one window of the discord rather than exactly on it.
    assert abs(flat_argmax(scores) - 200) < 20


def test_the_matrix_profile_scorer_returns_one_score_per_observation() -> None:
    pytest.importorskip("stumpy")
    ts = series(sine(200, period=20))
    scores = MatrixProfileScorer(window=20).run(ts)
    assert isinstance(scores, TimeSeries)
    assert scores.n_rows == ts.n_rows
    assert scores.columns == ("x",)
    assert (scores.values >= 0.0).all()


def test_the_matrix_profile_scorer_needs_two_windows_of_data() -> None:
    pytest.importorskip("stumpy")
    with pytest.raises(ValueError, match="at least 40 observations"):
        MatrixProfileScorer(window=20).run(series(sine(30, period=20)))


def test_the_matrix_profile_scorer_rejects_a_window_with_no_shape() -> None:
    with pytest.raises(ValueError, match="at least 3 observations"):
        MatrixProfileScorer(window=2)


def test_the_matrix_profile_detector_flags_the_discord() -> None:
    pytest.importorskip("stumpy")
    labels = detectors.matrix_profile(window=20).fit_detect(series(discord()))
    positions = flagged(labels)
    assert 200 in positions
    # Every flagged point shares a subsequence with the discord.
    assert positions.min() >= 200 - 19
    assert positions.max() <= 219 + 19


def test_the_damp_scorer_withholds_judgement_over_its_warm_up() -> None:
    pytest.importorskip("stumpy")
    scores = DampScorer(window=20).run(series(discord())).values.ravel()
    assert np.isnan(scores[:40]).all()
    assert not np.isnan(scores[60:]).any()


def test_the_damp_scorer_peaks_on_the_first_sight_of_the_discord() -> None:
    pytest.importorskip("stumpy")
    scores = DampScorer(window=20).run(series(discord()))
    assert abs(flat_argmax(scores) - 200) < 20


def test_the_damp_detector_flags_the_first_occurrence_of_a_repeated_shape() -> None:
    pytest.importorskip("stumpy")
    # The same odd stretch twice: a full matrix profile explains each away with
    # the other, while scoring against the past alone leaves the first unmatched.
    values = discord()
    values[300:320] = values[200:220]
    positions = flagged(detectors.damp(window=20).fit_detect(series(values)))
    assert positions.size > 0
    assert positions.min() >= 200 - 19
    assert positions.max() <= 219 + 19


def test_the_damp_detector_withholds_judgement_over_the_warm_up() -> None:
    pytest.importorskip("stumpy")
    labels = numbers(detectors.damp(window=20).fit_detect(series(discord())))
    assert np.isnan(labels[:40]).all()
    assert not np.isnan(labels[60:]).any()


def test_the_matrix_profile_detector_passes_normalize_to_its_scorer() -> None:
    # No stumpy needed: the pairing is built at construction, the backend is
    # imported only when a series is scored.
    detector = detectors.matrix_profile(window=20, normalize=False)
    assert detector.scorer.normalize is False
    assert detectors.matrix_profile(window=20).scorer.normalize is True


# ---------------------------------------------------------------------------
# seasonal-trend decomposition
# ---------------------------------------------------------------------------


def _seasonal(n: int = 240, at: int = 100, seed: int = 0) -> NDArray[Any]:
    """An hourly series with a daily rhythm and one anomaly."""
    rng = np.random.default_rng(seed)
    values = 10.0 + 3.0 * np.sin(np.arange(n) * 2 * np.pi / 24) + rng.normal(size=n)
    values[at] += 12.0
    return values


def _two_rhythms(n: int = 24 * 28, at: int = 300) -> NDArray[Any]:
    """An hourly series with both a daily and a weekly rhythm."""
    hours = np.arange(n)
    values = (
        10.0
        + 3.0 * np.sin(hours * 2 * np.pi / 24)
        + 5.0 * np.sin(hours * 2 * np.pi / 168)
    )
    values[at] += 20.0
    return values


def test_the_stl_residual_scorer_ranks_the_planted_anomaly_highest() -> None:
    pytest.importorskip("statsmodels")
    scores = StlResidualScorer().run(series(_seasonal(at=100)))
    assert flat_argmax(scores) == 100


def test_the_stl_residual_scorer_infers_a_daily_period_from_an_hourly_axis() -> None:
    pytest.importorskip("statsmodels")
    ts = series(_seasonal())
    stated = StlResidualScorer(period=24).run(ts)
    inferred = StlResidualScorer().run(ts)
    assert np.allclose(stated.values, inferred.values)


def test_the_stl_residual_scorer_returns_one_score_per_observation() -> None:
    pytest.importorskip("statsmodels")
    ts = series(_seasonal())
    scores = StlResidualScorer(robust=False, seasonal=25).run(ts)
    assert isinstance(scores, TimeSeries)
    assert scores.n_rows == ts.n_rows
    assert scores.columns == ("x",)
    assert (scores.values >= 0.0).all()


def test_the_stl_residual_scorer_rejects_an_irregular_time_axis() -> None:
    pytest.importorskip("statsmodels")
    with pytest.raises(ValueError, match="regular time axis"):
        StlResidualScorer(period=2).run(irregular([1.0, 2.0, 3.0, 4.0]))


def test_the_stl_residual_scorer_declines_to_guess_an_unguessable_period() -> None:
    pytest.importorskip("statsmodels")
    with pytest.raises(ValueError, match="could not infer a period"):
        StlResidualScorer().run(series(_seasonal(n=48, at=10), freq="7h"))


def test_the_stl_detector_flags_the_planted_anomaly_alone() -> None:
    pytest.importorskip("statsmodels")
    labels = detectors.stl(factor=6.0).fit_detect(series(_seasonal(at=100)))
    assert list(flagged(labels)) == [100]


def test_the_mstl_residual_scorer_ranks_an_anomaly_under_two_rhythms_highest() -> None:
    pytest.importorskip("statsmodels")
    scores = MstlResidualScorer(periods=(24, 168)).run(series(_two_rhythms(at=300)))
    assert flat_argmax(scores) == 300


def test_the_mstl_residual_scorer_accepts_a_single_period() -> None:
    pytest.importorskip("statsmodels")
    ts = series(_seasonal())
    scores = MstlResidualScorer(periods=24).run(ts)
    assert scores.n_rows == ts.n_rows
    assert scores.columns == ("x",)
    assert flat_argmax(scores) == 100


def test_the_mstl_residual_scorer_rejects_an_empty_set_of_periods() -> None:
    with pytest.raises(ValueError, match="periods is empty"):
        MstlResidualScorer(periods=())


def test_the_mstl_detector_flags_the_planted_anomaly_alone() -> None:
    pytest.importorskip("statsmodels")
    labels = detectors.mstl(periods=(24, 168), factor=25.0).fit_detect(
        series(_two_rhythms(at=300))
    )
    assert list(flagged(labels)) == [300]


# ---------------------------------------------------------------------------
# cross-cutting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_detector_gives_the_same_answer_on_every_backend(backend: str) -> None:
    values = np.tile([10.0, 11.0, 12.0, 11.0], 8)
    values[17] = 40.0
    native = make_native(backend, values)
    labels = detectors.hampel().detect(native)
    assert type(labels) is type(native)
    assert list(flagged(TimeSeries.from_any(labels))) == [17]


def test_every_component_carries_its_parameters_through_a_clone() -> None:
    components: list[Any] = [
        SpectralResidualScorer(window=5, series_window=11, score_window=13),
        detectors.spectral_residual(factor=4.0),
        HampelScorer(window=9, center=False),
        detectors.hampel(window=9, factor=4.0, center=False),
        RollingQuantileScorer(window=8, low=0.1, high=0.8),
        PeltScorer(penalty=2.0, cost="l1", min_size=3, jump=2),
        detectors.pelt(penalty=2.0, cost="l1", min_size=3, jump=2),
        RupturesScorer(model="window", cost="l1", n_bkps=2),
        MatrixProfileScorer(window=8, normalize=False),
        detectors.matrix_profile(window=8, factor=5.0),
        DampScorer(window=8, normalize=False),
        StlResidualScorer(period=12, robust=False, seasonal=9),
        detectors.stl(period=12, robust=False, seasonal=9, factor=5.0),
        MstlResidualScorer(periods=(12, 24), robust=False),
        detectors.mstl(periods=(12, 24), robust=False, factor=5.0),
    ]
    for component in components:
        clone = component.clone()
        assert repr(clone) == repr(component)
        assert _plain(clone.get_params()) == _plain(component.get_params())
        assert type(component).__name__ in repr(component)


def _plain(params: dict[str, Any]) -> dict[str, Any]:
    """Drop nested components, which compare by identity, from a parameter dict."""
    return {k: v for k, v in params.items() if not isinstance(v, Component)}


def test_the_scorer_and_detector_packages_export_what_they_advertise() -> None:
    import hazure.detectors as detector_presets
    import hazure.scorers as scorers

    for module in (scorers, detector_presets):
        for name in module.__all__:
            assert hasattr(module, name)
