"""Tests for the transformers.

Every transformer here has a closed form on the small synthetic series used
below, so the assertions are exact numbers rather than shapes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

from hazure import TimeSeries
from hazure.transformers import (
    CustomizedTransformer,
    DoubleRollingAggregate,
    OrdinaryLeastSquares,
    PcaColumnError,
    PcaProjection,
    PcaReconstruction,
    PcaReconstructionError,
    RegressionResidual,
    Retrospect,
    RollingAggregate,
    SeasonalDecomposition,
    StandardScale,
    SumAll,
)
from tests.helpers import BACKENDS, make_native

if TYPE_CHECKING:
    from hazure import Transformer

#: A repeating profile that sums to zero, so it survives centring untouched.
PROFILE = np.array([0.0, 1.0, 0.0, -1.0])


def daily(values: Any, start: str = "2024-01-01") -> TimeSeries:
    """Build a univariate daily series from a sequence of numbers."""
    matrix = np.asarray(values, dtype=float)
    time = np.datetime64(start) + np.arange(matrix.shape[0]) * np.timedelta64(1, "D")
    return TimeSeries.from_arrays(time, matrix)


def daily_frame(columns: dict[str, Any], start: str = "2024-01-01") -> TimeSeries:
    """Build a multi-column daily series from named sequences."""
    matrix = np.column_stack([np.asarray(v, dtype=float) for v in columns.values()])
    time = np.datetime64(start) + np.arange(matrix.shape[0]) * np.timedelta64(1, "D")
    return TimeSeries.from_arrays(time, matrix, list(columns))


def flat(ts: TimeSeries) -> np.ndarray:
    """Return a single-column result as a 1-D array."""
    return np.asarray(ts.values[:, 0])


def applied(transformer: Transformer, ts: TimeSeries) -> TimeSeries:
    """Fit on a series and transform it, staying inside the TimeSeries world."""
    return transformer.fit(ts).run(ts)


# ---------------------------------------------------------------------------
# RollingAggregate
# ---------------------------------------------------------------------------


def test_a_rolling_mean_averages_the_trailing_window() -> None:
    result = RollingAggregate(window=2).run(daily([1.0, 2.0, 3.0, 4.0, 5.0]))
    np.testing.assert_array_equal(flat(result), [np.nan, 1.5, 2.5, 3.5, 4.5])


def test_a_centred_rolling_window_looks_both_ways() -> None:
    result = RollingAggregate(window=3, center=True).run(daily([1.0, 2.0, 3.0, 4.0]))
    np.testing.assert_array_equal(flat(result), [np.nan, 2.0, 3.0, np.nan])


def test_min_periods_decides_how_much_data_a_result_needs() -> None:
    series = daily([1.0, 2.0, 3.0])
    lenient = RollingAggregate(window=3, min_periods=1).run(series)
    np.testing.assert_array_equal(flat(lenient), [1.0, 1.5, 2.0])
    strict = RollingAggregate(window=3).run(series)
    np.testing.assert_array_equal(flat(strict), [np.nan, np.nan, 2.0])


def test_a_duration_window_measures_the_window_in_time() -> None:
    result = RollingAggregate(window="2d", agg="sum").run(daily([1.0, 2.0, 3.0, 4.0]))
    np.testing.assert_array_equal(flat(result), [1.0, 3.0, 5.0, 7.0])


def test_closed_moves_the_window_boundary() -> None:
    series = daily([1.0, 2.0, 3.0, 4.0])
    trailing = RollingAggregate(window=2, agg="max").run(series)
    np.testing.assert_array_equal(flat(trailing), [np.nan, 2.0, 3.0, 4.0])
    exclusive = RollingAggregate(window=2, agg="max", closed="left").run(series)
    np.testing.assert_array_equal(flat(exclusive), [np.nan, np.nan, 2.0, 3.0])


def test_a_rolling_aggregate_keeps_the_column_name() -> None:
    frame = daily_frame({"sensor": [1.0, 2.0, 3.0]})
    assert RollingAggregate(window=2).run(frame).columns == ("sensor",)


def test_a_rolling_aggregate_needs_no_fitting() -> None:
    assert RollingAggregate(window=2).fitted


def test_an_all_missing_series_rolls_to_all_missing() -> None:
    result = RollingAggregate(window=2, agg="median").run(daily([np.nan] * 4))
    assert np.isnan(flat(result)).all()


def test_an_empty_series_rolls_to_an_empty_series() -> None:
    empty = TimeSeries.from_arrays(
        np.array([], dtype="datetime64[D]"), np.zeros((0, 1))
    )
    assert RollingAggregate(window=2).run(empty).n_rows == 0


# -- quantiles --------------------------------------------------------------


def test_a_single_quantile_gives_one_column() -> None:
    result = RollingAggregate(window=3, agg="quantile", agg_params={"q": 0.5}).run(
        daily([1.0, 2.0, 9.0, 4.0])
    )
    np.testing.assert_array_equal(flat(result), [np.nan, np.nan, 2.0, 4.0])


def test_a_list_of_quantiles_gives_one_column_per_quantile() -> None:
    result = RollingAggregate(
        window=3, agg="quantile", agg_params={"q": [0.0, 0.5, 1.0]}
    ).run(daily([1.0, 2.0, 9.0, 4.0]))
    assert result.columns == ("q0.0", "q0.5", "q1.0")
    np.testing.assert_array_equal(
        result.values[2:],
        [[1.0, 2.0, 9.0], [2.0, 4.0, 9.0]],
    )


def test_a_quantile_aggregation_without_a_quantile_is_reported() -> None:
    with pytest.raises(ValueError, match="needs a quantile"):
        RollingAggregate(window=2, agg="quantile").run(daily([1.0, 2.0]))


def test_an_empty_quantile_list_is_reported() -> None:
    with pytest.raises(ValueError, match="asks for no columns"):
        RollingAggregate(window=2, agg="quantile", agg_params={"q": []}).run(
            daily([1.0, 2.0])
        )


# -- histograms -------------------------------------------------------------


def test_a_histogram_counts_each_bin_of_each_window() -> None:
    result = RollingAggregate(
        window=2, agg="hist", agg_params={"bins": [0, 1, 2, 3]}
    ).run(daily([0.5, 1.5, 2.5, 0.5]))
    assert result.columns == ("[0, 1)", "[1, 2)", "[2, 3]")
    np.testing.assert_array_equal(
        result.values,
        [
            [np.nan, np.nan, np.nan],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
        ],
    )


def test_a_histogram_closes_its_last_bin_on_the_right() -> None:
    """An observation exactly on the top edge belongs to the last bin."""
    result = RollingAggregate(window=1, agg="hist", agg_params={"bins": [0, 1, 2]}).run(
        daily([2.0, 1.0, 0.0])
    )
    np.testing.assert_array_equal(result.values, [[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])


def test_a_histogram_ignores_observations_outside_the_bins() -> None:
    result = RollingAggregate(window=1, agg="hist", agg_params={"bins": [0, 1]}).run(
        daily([-5.0, 0.5, 99.0])
    )
    np.testing.assert_array_equal(flat(result), [0.0, 1.0, 0.0])


def test_a_bin_count_spans_the_observed_range() -> None:
    result = RollingAggregate(window=1, agg="hist", agg_params={"bins": 2}).run(
        daily([0.0, 1.0, 2.0])
    )
    assert result.columns == ("[0, 1)", "[1, 2]")
    np.testing.assert_array_equal(result.values, [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])


def test_a_histogram_blanks_a_window_that_fails_min_periods() -> None:
    result = RollingAggregate(
        window=3, agg="hist", agg_params={"bins": [0, 10]}, min_periods=2
    ).run(daily([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(flat(result), [np.nan, 2.0, 3.0])


def test_a_histogram_without_bins_is_reported() -> None:
    with pytest.raises(ValueError, match="needs bins"):
        RollingAggregate(window=2, agg="hist").run(daily([1.0, 2.0]))


def test_histogram_bin_edges_must_increase() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        RollingAggregate(window=2, agg="hist", agg_params={"bins": [0, 2, 1]}).run(
            daily([1.0, 2.0])
        )


def test_a_histogram_needs_at_least_two_edges() -> None:
    with pytest.raises(ValueError, match="at least 2 edges"):
        RollingAggregate(window=2, agg="hist", agg_params={"bins": [0]}).run(
            daily([1.0, 2.0])
        )


def test_a_histogram_needs_a_positive_bin_count() -> None:
    with pytest.raises(ValueError, match="bins must be at least 1"):
        RollingAggregate(window=2, agg="hist", agg_params={"bins": 0}).run(
            daily([1.0, 2.0])
        )


def test_bins_cannot_be_chosen_for_a_series_with_no_observations() -> None:
    with pytest.raises(ValueError, match="Cannot choose bins"):
        RollingAggregate(window=2, agg="hist", agg_params={"bins": 3}).run(
            daily([np.nan, np.nan, np.nan])
        )


# -- fan-out ----------------------------------------------------------------


def test_a_widening_rolling_aggregate_qualifies_its_column_names() -> None:
    frame = daily_frame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    result = RollingAggregate(
        window=2, agg="quantile", agg_params={"q": [0.5, 1.0]}
    ).run(frame)
    assert result.columns == ("a_q0.5", "a_q1.0", "b_q0.5", "b_q1.0")


def test_a_rolling_aggregate_fans_out_over_a_frame() -> None:
    frame = daily_frame({"a": [1.0, 3.0, 5.0], "b": [10.0, 30.0, 50.0]})
    result = RollingAggregate(window=2).run(frame)
    assert result.columns == ("a", "b")
    np.testing.assert_array_equal(
        result.values, [[np.nan, np.nan], [2.0, 20.0], [4.0, 40.0]]
    )


# ---------------------------------------------------------------------------
# DoubleRollingAggregate
# ---------------------------------------------------------------------------


def test_double_rolling_peaks_at_a_level_shift() -> None:
    result = DoubleRollingAggregate(window=2, diff="diff").run(
        daily([0.0, 0.0, 0.0, 5.0, 5.0, 5.0])
    )
    np.testing.assert_array_equal(flat(result), [np.nan, np.nan, 2.5, 5.0, 2.5, np.nan])
    assert int(np.nanargmax(flat(result))) == 3


def test_double_rolling_reports_at_the_window_end_when_not_centred() -> None:
    """The same numbers, moved to the last observation of the right window."""
    result = DoubleRollingAggregate(window=2, diff="diff", center=False).run(
        daily([0.0, 0.0, 0.0, 5.0, 5.0, 5.0])
    )
    np.testing.assert_array_equal(flat(result), [np.nan, np.nan, np.nan, 2.5, 5.0, 2.5])


def test_double_rolling_shifts_a_duration_window_to_its_end() -> None:
    result = DoubleRollingAggregate(window="2d", diff="diff", center=False).run(
        daily([0.0, 0.0, 0.0, 5.0, 5.0, 5.0])
    )
    np.testing.assert_array_equal(flat(result), [np.nan, np.nan, 0.0, 2.5, 5.0, 0.0])


def test_double_rolling_accepts_a_different_window_per_side() -> None:
    """A long left window describes normal; a short right one catches a blip."""
    values = np.zeros(8)
    values[6] = 10.0
    result = DoubleRollingAggregate(window=(4, 1), agg="mean", diff="diff").run(
        daily(values)
    )
    assert int(np.nanargmax(flat(result))) == 6
    assert flat(result)[6] == 10.0


def test_double_rolling_accepts_a_different_aggregation_per_side() -> None:
    result = DoubleRollingAggregate(window=2, agg=("min", "max"), diff="diff").run(
        daily([1.0, 2.0, 3.0, 4.0])
    )
    np.testing.assert_array_equal(flat(result), [np.nan, np.nan, 3.0, np.nan])


def test_double_rolling_accepts_min_periods_per_side() -> None:
    result = DoubleRollingAggregate(
        window=3, agg="mean", min_periods=(1, 3), diff="diff"
    ).run(daily([0.0, 0.0, 3.0, 3.0, 3.0]))
    np.testing.assert_array_equal(flat(result), [np.nan, 2.0, 3.0, np.nan, np.nan])


def test_double_rolling_shares_one_quantile_between_the_sides() -> None:
    result = DoubleRollingAggregate(
        window=2, agg="quantile", agg_params={"q": 1.0}, diff="diff"
    ).run(daily([1.0, 2.0, 3.0, 4.0]))
    np.testing.assert_array_equal(flat(result), [np.nan, np.nan, 2.0, np.nan])


def test_double_rolling_rejects_a_vector_aggregation() -> None:
    with pytest.raises(ValueError, match="returns a vector per window"):
        DoubleRollingAggregate(window=2, agg="hist", agg_params={"bins": [0, 1]}).run(
            daily([1.0, 2.0])
        )


def test_double_rolling_rejects_several_quantiles() -> None:
    with pytest.raises(ValueError, match="several quantiles"):
        DoubleRollingAggregate(
            window=2, agg="quantile", agg_params={"q": [0.1, 0.9]}
        ).run(daily([1.0, 2.0]))


def test_double_rolling_rejects_disagreeing_quantiles() -> None:
    with pytest.raises(ValueError, match="different quantiles"):
        DoubleRollingAggregate(
            window=2,
            agg="quantile",
            agg_params=({"q": 0.1}, {"q": 0.9}),
        ).run(daily([1.0, 2.0]))


def test_double_rolling_needs_a_quantile_for_a_quantile_aggregation() -> None:
    with pytest.raises(ValueError, match="on the left window needs a quantile"):
        DoubleRollingAggregate(window=2, agg="quantile").run(daily([1.0, 2.0]))


def test_double_rolling_rejects_a_tuple_that_is_not_a_pair() -> None:
    with pytest.raises(ValueError, match="left, right"):
        DoubleRollingAggregate(window=2, agg=("min", "max", "mean")).run(  # type: ignore[arg-type]
            daily([1.0, 2.0])
        )


def test_clone_preserves_the_diff_parameter() -> None:
    """``diff`` is a constructor parameter, so get_params and clone carry it."""
    original = DoubleRollingAggregate(window=2, diff="rel_diff")
    assert original.get_params()["diff"] == "rel_diff"

    copy = original.clone()
    assert copy.diff == "rel_diff"

    series = daily([2.0, 2.0, 8.0, 8.0])
    np.testing.assert_array_equal(flat(copy.run(series)), flat(original.run(series)))
    # And the parameter actually changes the answer, so carrying it matters.
    magnitude = DoubleRollingAggregate(window=2, diff="l1").run(series)
    assert not np.allclose(
        flat(copy.run(series))[2:3], flat(magnitude)[2:3], equal_nan=True
    )


# ---------------------------------------------------------------------------
# Retrospect
# ---------------------------------------------------------------------------


def test_retrospect_produces_the_exact_lag_matrix() -> None:
    result = Retrospect(n_steps=3, step_size=2, till=1).run(
        daily([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    )
    assert result.columns == ("t-1", "t-3", "t-5")
    np.testing.assert_array_equal(
        result.values,
        [
            [np.nan, np.nan, np.nan],
            [0.0, np.nan, np.nan],
            [1.0, np.nan, np.nan],
            [2.0, 0.0, np.nan],
            [3.0, 1.0, np.nan],
            [4.0, 2.0, 0.0],
        ],
    )


def test_retrospect_defaults_to_the_current_observation() -> None:
    result = Retrospect().run(daily([1.0, 2.0, 3.0]))
    assert result.columns == ("t-0",)
    np.testing.assert_array_equal(flat(result), [1.0, 2.0, 3.0])


def test_retrospect_can_look_ahead() -> None:
    """A negative lag is a forward shift, and says so in the column name."""
    result = Retrospect(n_steps=2, till=-1).run(daily([1.0, 2.0, 3.0]))
    assert result.columns == ("t+1", "t-0")
    np.testing.assert_array_equal(
        result.values, [[2.0, 1.0], [3.0, 2.0], [np.nan, 3.0]]
    )


def test_a_lag_longer_than_the_series_is_all_missing() -> None:
    result = Retrospect(n_steps=1, till=9).run(daily([1.0, 2.0, 3.0]))
    assert np.isnan(flat(result)).all()


def test_retrospect_needs_a_regular_time_axis() -> None:
    irregular = TimeSeries.from_arrays(
        np.array(["2024-01-01", "2024-01-02", "2024-01-05"], dtype="datetime64[D]"),
        np.array([1.0, 2.0, 3.0]),
    )
    with pytest.raises(ValueError, match="needs a regular time axis"):
        Retrospect().run(irregular)


def test_retrospect_rejects_a_non_positive_step() -> None:
    with pytest.raises(ValueError, match="step_size must be at least 1"):
        Retrospect(n_steps=2, step_size=0).run(daily([1.0, 2.0, 3.0]))


def test_retrospect_rejects_a_non_positive_number_of_steps() -> None:
    with pytest.raises(ValueError, match="n_steps must be at least 1"):
        Retrospect(n_steps=0).run(daily([1.0, 2.0, 3.0]))


def test_retrospect_qualifies_its_column_names_under_fan_out() -> None:
    frame = daily_frame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    result = Retrospect(n_steps=2).run(frame)
    assert result.columns == ("a_t-0", "a_t-1", "b_t-0", "b_t-1")


# ---------------------------------------------------------------------------
# StandardScale
# ---------------------------------------------------------------------------


def test_standard_scale_centres_and_scales_by_the_sample_deviation() -> None:
    result = StandardScale().run(daily([1.0, 2.0, 3.0, 4.0, 5.0]))
    expected = (np.arange(1.0, 6.0) - 3.0) / np.sqrt(2.5)
    np.testing.assert_allclose(flat(result), expected)


def test_standard_scale_leaves_a_constant_series_at_zero() -> None:
    """A series with no spread is centred but not divided by zero."""
    result = StandardScale().run(daily([7.0, 7.0, 7.0]))
    np.testing.assert_array_equal(flat(result), [0.0, 0.0, 0.0])


def test_standard_scale_handles_a_single_observation() -> None:
    result = StandardScale().run(daily([7.0]))
    np.testing.assert_array_equal(flat(result), [0.0])


def test_standard_scale_ignores_missing_observations() -> None:
    result = StandardScale().run(daily([1.0, np.nan, 5.0]))
    np.testing.assert_allclose(
        flat(result), [-1.0 / np.sqrt(2), np.nan, 1 / np.sqrt(2)]
    )


def test_standard_scale_passes_an_all_missing_series_through() -> None:
    result = StandardScale().run(daily([np.nan, np.nan]))
    assert np.isnan(flat(result)).all()


def test_standard_scale_has_no_parameters_and_needs_no_fitting() -> None:
    assert StandardScale().get_params() == {}
    assert StandardScale().fitted
    assert repr(StandardScale()) == "StandardScale()"


def test_standard_scale_scales_each_column_separately() -> None:
    frame = daily_frame({"small": [1.0, 2.0, 3.0], "large": [100.0, 200.0, 300.0]})
    result = StandardScale().run(frame)
    np.testing.assert_allclose(result.values[:, 0], result.values[:, 1])


# ---------------------------------------------------------------------------
# SeasonalDecomposition
# ---------------------------------------------------------------------------


def test_seasonal_decomposition_recovers_a_planted_profile_exactly() -> None:
    model = SeasonalDecomposition(period=4).fit(daily(np.tile(PROFILE, 6)))
    assert model.period_ == 4
    np.testing.assert_array_equal(model.seasonal_, PROFILE)


def test_a_pure_cycle_leaves_no_residual() -> None:
    series = daily(np.tile(PROFILE, 6))
    residual = applied(SeasonalDecomposition(period=4), series)
    np.testing.assert_allclose(flat(residual), np.zeros(24))


def test_the_seasonal_profile_carries_the_level_without_a_trend() -> None:
    """With trend=False there is nowhere else for the level to live."""
    model = SeasonalDecomposition(period=4).fit(daily(np.tile(PROFILE + 10.0, 6)))
    np.testing.assert_array_equal(model.seasonal_, PROFILE + 10.0)


def test_seasonal_decomposition_flags_a_planted_anomaly() -> None:
    values = np.tile(PROFILE, 6)
    values[13] += 5.0
    residual = applied(SeasonalDecomposition(period=4), daily(values))
    assert int(np.nanargmax(np.abs(flat(residual)))) == 13


def test_seasonal_decomposition_returns_the_seasonal_component() -> None:
    series = daily(np.tile(PROFILE, 6))
    seasonal = applied(SeasonalDecomposition(period=4, component="seasonal"), series)
    np.testing.assert_array_equal(flat(seasonal), np.tile(PROFILE, 6))


def test_a_trend_is_recovered_by_the_centred_moving_average() -> None:
    values = 0.5 * np.arange(24) + np.tile(PROFILE, 6)
    series = daily(values)
    trend = applied(
        SeasonalDecomposition(period=4, trend=True, component="trend"), series
    )
    # A cycle-wide average cancels the cycle, leaving the straight line, and
    # cannot be centred on the two observations at either end.
    np.testing.assert_allclose(flat(trend)[2:22], 0.5 * np.arange(2, 22))
    assert np.isnan(flat(trend)[[0, 1, 22, 23]]).all()


def test_removing_a_trend_leaves_a_zero_centred_profile() -> None:
    values = 0.5 * np.arange(24) + np.tile(PROFILE, 6)
    model = SeasonalDecomposition(period=4, trend=True).fit(daily(values))
    np.testing.assert_allclose(model.seasonal_, PROFILE)
    residual = flat(model.run(daily(values)))
    np.testing.assert_allclose(residual[2:22], np.zeros(20))
    assert np.isnan(residual[[0, 1, 22, 23]]).all()


def test_seasonal_decomposition_detects_a_known_period() -> None:
    cycle = np.sin(2 * np.pi * np.arange(120) / 12)
    assert SeasonalDecomposition().fit(daily(cycle)).period_ == 12


def test_seasonal_decomposition_detects_a_short_period() -> None:
    assert SeasonalDecomposition().fit(daily(np.tile(PROFILE, 30))).period_ == 4


def test_a_window_far_after_training_is_still_in_phase() -> None:
    """The phase comes from arithmetic on the timestamps, not from a search."""
    model = SeasonalDecomposition(period=4).fit(daily(np.tile(PROFILE, 6)))
    offset = int(
        (np.datetime64("2030-01-01") - np.datetime64("2024-01-01")).astype(int)
    )
    future = daily(PROFILE[(offset + np.arange(8)) % 4], start="2030-01-01")
    np.testing.assert_allclose(flat(model.run(future)), np.zeros(8))


def test_a_window_before_training_is_still_in_phase() -> None:
    model = SeasonalDecomposition(period=4).fit(daily(np.tile(PROFILE, 6)))
    offset = int(
        (np.datetime64("2020-01-01") - np.datetime64("2024-01-01")).astype(int)
    )
    past = daily(PROFILE[(offset + np.arange(8)) % 4], start="2020-01-01")
    np.testing.assert_allclose(flat(model.run(past)), np.zeros(8))


def test_an_out_of_phase_window_is_rejected() -> None:
    model = SeasonalDecomposition(period=4).fit(daily(np.tile(PROFILE, 6)))
    shifted = TimeSeries.from_arrays(
        np.datetime64("2024-02-01T12") + np.arange(8) * np.timedelta64(1, "D"),
        np.tile(PROFILE, 2),
    )
    with pytest.raises(ValueError, match="out of phase with training"):
        model.run(shifted)


def test_a_differently_sampled_window_is_rejected() -> None:
    model = SeasonalDecomposition(period=4).fit(daily(np.tile(PROFILE, 6)))
    hourly = TimeSeries.from_arrays(
        np.datetime64("2024-01-01") + np.arange(8) * np.timedelta64(1, "h"),
        np.tile(PROFILE, 2),
    )
    with pytest.raises(ValueError, match="sampled every"):
        model.run(hourly)


def test_seasonal_decomposition_needs_a_regular_time_axis() -> None:
    irregular = TimeSeries.from_arrays(
        np.array(
            ["2024-01-01", "2024-01-02", "2024-01-05", "2024-01-06"],
            dtype="datetime64[D]",
        ),
        np.tile(PROFILE, 1),
    )
    with pytest.raises(ValueError, match="needs a regular time axis"):
        SeasonalDecomposition(period=2).fit(irregular)


def test_seasonal_decomposition_rejects_an_unknown_component() -> None:
    with pytest.raises(ValueError, match="Unknown component"):
        SeasonalDecomposition(period=4, component="noise").fit(  # type: ignore[arg-type]
            daily(np.tile(PROFILE, 6))
        )


def test_a_trend_cannot_be_returned_if_it_was_never_estimated() -> None:
    with pytest.raises(ValueError, match="nothing to return while trend=False"):
        SeasonalDecomposition(period=4, component="trend").fit(
            daily(np.tile(PROFILE, 6))
        )


def test_seasonal_decomposition_needs_two_full_cycles() -> None:
    with pytest.raises(ValueError, match="at least two full cycles"):
        SeasonalDecomposition(period=4).fit(daily([1.0, 2.0, 3.0, 4.0, 5.0]))


def test_a_period_below_two_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 2 observations"):
        SeasonalDecomposition(period=1).fit(daily(np.tile(PROFILE, 6)))


def test_a_constant_series_has_no_period_to_detect() -> None:
    with pytest.raises(ValueError, match="constant"):
        SeasonalDecomposition().fit(daily(np.zeros(20)))


def test_a_series_without_a_cycle_has_no_period_to_detect() -> None:
    """A ramp autocorrelates strongly but never peaks, so nothing is claimed."""
    with pytest.raises(ValueError, match="No autocorrelation peak"):
        SeasonalDecomposition().fit(daily(np.arange(40.0)))


def test_period_detection_needs_a_few_observations() -> None:
    with pytest.raises(ValueError, match="at least 4 observations"):
        SeasonalDecomposition().fit(daily([1.0, 2.0, 3.0]))


def test_a_phase_with_no_training_observation_is_reported() -> None:
    values = np.tile(PROFILE, 3).astype(float)
    values[1::4] = np.nan
    with pytest.raises(ValueError, match=r"Phases \[1\]"):
        SeasonalDecomposition(period=4).fit(daily(values))


# ---------------------------------------------------------------------------
# SumAll
# ---------------------------------------------------------------------------


def test_sum_all_adds_the_columns() -> None:
    frame = daily_frame({"a": [1.0, 2.0], "b": [10.0, 20.0], "c": [100.0, 200.0]})
    result = SumAll().run(frame)
    assert result.columns == ("sum",)
    np.testing.assert_array_equal(flat(result), [111.0, 222.0])


def test_sum_all_propagates_a_missing_observation() -> None:
    frame = daily_frame({"a": [1.0, 2.0], "b": [np.nan, 20.0]})
    np.testing.assert_array_equal(flat(SumAll().run(frame)), [np.nan, 22.0])


def test_sum_all_has_no_parameters_and_needs_no_fitting() -> None:
    assert SumAll().get_params() == {}
    assert SumAll().fitted


# ---------------------------------------------------------------------------
# RegressionResidual
# ---------------------------------------------------------------------------


def test_an_exact_linear_relationship_leaves_no_residual() -> None:
    drive = np.arange(6.0)
    frame = daily_frame({"drive": drive, "follow": 2.0 * drive + 1.0})
    residual = applied(RegressionResidual(target="follow"), frame)
    assert residual.columns == ("residual",)
    np.testing.assert_allclose(flat(residual), np.zeros(6), atol=1e-12)


def test_a_broken_relationship_shows_up_in_the_residual() -> None:
    drive = np.arange(6.0)
    follow = 2.0 * drive + 1.0
    follow[4] += 10.0
    frame = daily_frame({"drive": drive, "follow": follow})
    residual = flat(applied(RegressionResidual(target="follow"), frame))
    assert int(np.argmax(np.abs(residual))) == 4


def test_regression_residual_accepts_any_fit_predict_object() -> None:
    class AlwaysZero:
        def fit(self, X: Any, y: Any) -> Any:
            return self

        def predict(self, X: Any) -> Any:
            return np.zeros(X.shape[0])

    frame = daily_frame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    model = RegressionResidual(target="b", regressor=AlwaysZero())
    np.testing.assert_array_equal(flat(applied(model, frame)), [4.0, 5.0, 6.0])


def test_regression_residual_leaves_incomplete_rows_missing() -> None:
    frame = daily_frame({"a": [1.0, 2.0, np.nan, 4.0], "b": [2.0, 4.0, 6.0, 8.0]})
    residual = flat(applied(RegressionResidual(target="b"), frame))
    assert np.isnan(residual[2])
    np.testing.assert_allclose(residual[[0, 1, 3]], np.zeros(3), atol=1e-12)


def test_regression_residual_rejects_an_unknown_target() -> None:
    frame = daily_frame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    with pytest.raises(ValueError, match="not a column of the training data"):
        RegressionResidual(target="missing").fit(frame)


def test_regression_residual_needs_a_feature_column() -> None:
    with pytest.raises(ValueError, match="at least one column besides"):
        RegressionResidual(target="a").fit(daily_frame({"a": [1.0, 2.0]}))


def test_regression_residual_needs_a_complete_training_row() -> None:
    frame = daily_frame({"a": [np.nan, 2.0], "b": [3.0, np.nan]})
    with pytest.raises(ValueError, match="nothing to regress on"):
        RegressionResidual(target="b").fit(frame)


def test_regression_residual_rejects_an_input_missing_a_feature() -> None:
    frame = daily_frame({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0]})
    model = RegressionResidual(target="b").fit(frame)
    with pytest.raises(ValueError, match="the input is missing"):
        model.run(daily_frame({"b": [1.0, 2.0, 3.0], "c": [1.0, 1.0, 1.0]}))


def test_ordinary_least_squares_recovers_known_coefficients() -> None:
    X = np.column_stack([np.arange(5.0), np.ones(5)])
    model = OrdinaryLeastSquares().fit(X, 3.0 * np.arange(5.0) - 2.0)
    assert model.coefficients_[0] == pytest.approx(3.0)
    assert model.intercept_ + model.coefficients_[1] == pytest.approx(-2.0)


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------


def rank_one_frame() -> TimeSeries:
    """A frame whose points lie exactly on a line in two dimensions."""
    base = np.arange(5.0)
    return daily_frame({"a": base, "b": 2.0 * base + 1.0})


def test_reconstruction_of_a_rank_one_subspace_is_exact() -> None:
    error = applied(PcaReconstructionError(k=1), rank_one_frame())
    assert error.columns == ("error",)
    np.testing.assert_allclose(flat(error), np.zeros(5), atol=1e-18)


def test_reconstruction_recovers_rank_one_data_column_by_column() -> None:
    frame = rank_one_frame()
    rebuilt = applied(PcaReconstruction(k=1), frame)
    assert rebuilt.columns == ("a", "b")
    np.testing.assert_allclose(rebuilt.values, frame.values, atol=1e-12)


def test_reconstruction_error_finds_a_point_off_the_subspace() -> None:
    base = np.arange(5.0)
    other = 2.0 * base
    other[2] += 1.0
    frame = daily_frame({"a": base, "b": other})
    error = flat(applied(PcaReconstructionError(k=1), frame))
    assert int(np.argmax(error)) == 2
    assert error[2] > 10 * error[0]


def test_the_projection_names_one_column_per_component() -> None:
    frame = daily_frame(
        {"a": [1.0, 2.0, 3.0], "b": [3.0, 1.0, 2.0], "c": [2.0, 3.0, 1.0]}
    )
    assert applied(PcaProjection(k=2), frame).columns == ("pc0", "pc1")


def test_the_projection_of_a_rank_one_subspace_is_exact() -> None:
    """Coordinates along the single direction, scaled by its unit length."""
    scores = flat(applied(PcaProjection(k=1), rank_one_frame()))
    centred = np.arange(5.0) - 2.0
    np.testing.assert_allclose(scores, centred * np.sqrt(5.0))


def test_the_projection_sign_is_fixed_rather_than_arbitrary() -> None:
    model = PcaProjection(k=1).fit(rank_one_frame())
    assert model.components_[0, np.argmax(np.abs(model.components_[0]))] > 0


def test_pca_stores_the_training_mean_and_components() -> None:
    model = PcaProjection(k=1).fit(rank_one_frame())
    np.testing.assert_allclose(model.mean_, [2.0, 5.0])
    np.testing.assert_allclose(model.components_, [[1.0, 2.0] / np.sqrt(5.0)])


def test_pca_leaves_rows_with_a_missing_value_missing() -> None:
    base = np.arange(5.0)
    other = 2.0 * base
    other[3] = np.nan
    frame = daily_frame({"a": base, "b": other})
    for transformer in (
        PcaProjection(k=1),
        PcaReconstruction(k=1),
        PcaReconstructionError(k=1),
        PcaColumnError(k=1),
    ):
        result = applied(transformer, frame)
        assert np.isnan(result.values[3]).all()
        assert not np.isnan(result.values[[0, 1, 2, 4]]).any()


def test_pca_rejects_more_components_than_columns() -> None:
    with pytest.raises(ValueError, match="exceeds the 2 column"):
        PcaProjection(k=3).fit(rank_one_frame())


def test_pca_rejects_a_non_positive_component_count() -> None:
    with pytest.raises(ValueError, match="k must be at least 1"):
        PcaProjection(k=0).fit(rank_one_frame())


def test_pca_needs_enough_complete_rows() -> None:
    frame = daily_frame({"a": [1.0, np.nan, np.nan], "b": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="at least 2 rows with no missing"):
        PcaProjection(k=2).fit(frame)


def test_pca_rejects_a_component_count_raised_after_fitting() -> None:
    model = PcaProjection(k=1).fit(rank_one_frame())
    model.set_params(k=2)
    with pytest.raises(ValueError, match="component"):
        model.run(rank_one_frame())


def test_the_column_errors_sum_to_the_reconstruction_error() -> None:
    """The identity the decomposition exists for, at two subspace sizes."""
    rng = np.random.default_rng(11)
    frame = daily_frame({name: rng.normal(size=40) for name in "abcd"})
    for k in (1, 3):
        parts = applied(PcaColumnError(k=k), frame)
        total = flat(applied(PcaReconstructionError(k=k), frame))
        np.testing.assert_allclose(parts.values.sum(axis=1), total)


def test_the_column_error_lands_on_the_pair_that_disagrees() -> None:
    """A break in one of two matched columns charges the pair, not the third."""
    base = np.arange(20.0)
    partner = base.copy()
    partner[8:10] += 8.0
    frame = daily_frame(
        {"a": base, "b": partner, "c": np.tile([0.0, 9.0, 3.0, 12.0], 5)}
    )
    parts = applied(PcaColumnError(k=2), frame).values
    totals = parts.sum(axis=1)
    # The two broken rows dominate, and inside them the unrelated column is
    # untouched: what the residual localises is the failed relation.
    assert totals[8:10].min() > 10 * np.delete(totals, [8, 9]).max()
    assert (parts[8:10, :2].sum(axis=1) / totals[8:10] > 0.99).all()


def test_the_column_error_keeps_the_input_columns_in_order() -> None:
    frame = daily_frame(
        {"b": [1.0, 2.0, 3.0], "a": [3.0, 1.0, 2.0], "c": [2.0, 3.0, 1.0]}
    )
    assert applied(PcaColumnError(k=2), frame).columns == ("b", "a", "c")


def test_the_column_error_rejects_a_component_count_raised_after_fitting() -> None:
    model = PcaColumnError(k=1).fit(rank_one_frame())
    model.set_params(k=2)
    with pytest.raises(ValueError, match="component"):
        model.run(rank_one_frame())


# ---------------------------------------------------------------------------
# CustomizedTransformer
# ---------------------------------------------------------------------------


def test_a_customized_transformer_receives_the_value_matrix() -> None:
    seen: list[tuple[int, ...]] = []

    def spread(values: np.ndarray) -> np.ndarray:
        seen.append(values.shape)
        return values.max(axis=1) - values.min(axis=1)

    frame = daily_frame({"a": [1.0, 2.0], "b": [4.0, 8.0]})
    result = CustomizedTransformer(transform_func=spread).run(frame)
    assert seen == [(2, 2)]
    np.testing.assert_array_equal(flat(result), [3.0, 6.0])


def test_a_customized_transformer_needs_no_fitting_without_a_fit_func() -> None:
    assert CustomizedTransformer(transform_func=lambda values: values).fitted


def test_a_customized_transformer_learns_parameters_from_fit_func() -> None:
    frame = daily_frame({"a": [1.0, 3.0], "b": [5.0, 7.0]})
    model = CustomizedTransformer(
        transform_func=lambda values, centre: values.sum(axis=1) - centre,
        fit_func=lambda values: {"centre": float(values.mean())},
    )
    np.testing.assert_array_equal(flat(applied(model, frame)), [2.0, 6.0])
    assert model.learned_params_ == {"centre": 4.0}


def test_an_explicit_parameter_beats_a_learned_one() -> None:
    frame = daily_frame({"a": [1.0, 3.0], "b": [5.0, 7.0]})
    model = CustomizedTransformer(
        transform_func=lambda values, centre: values.sum(axis=1) - centre,
        transform_func_params={"centre": 0.0},
        fit_func=lambda values: {"centre": 4.0},
    )
    np.testing.assert_array_equal(flat(applied(model, frame)), [6.0, 10.0])


def test_a_customized_transformer_needs_fitting_when_it_has_a_fit_func() -> None:
    model = CustomizedTransformer(
        transform_func=lambda values, centre: values.sum(axis=1) - centre,
        fit_func=lambda values: {"centre": 0.0},
    )
    with pytest.raises(RuntimeError, match="must be fitted"):
        model.transform(daily_frame({"a": [1.0], "b": [2.0]}))


def test_a_customized_transformer_rejects_a_non_dict_from_fit_func() -> None:
    model = CustomizedTransformer(
        transform_func=lambda values: values, fit_func=lambda values: 1.0
    )
    with pytest.raises(TypeError, match="must return a dict"):
        model.fit(daily_frame({"a": [1.0], "b": [2.0]}))


def test_a_customized_transformer_rejects_a_changed_row_count() -> None:
    model = CustomizedTransformer(transform_func=lambda values: values[:1, 0])
    with pytest.raises(ValueError, match="must have shape"):
        model.transform(daily_frame({"a": [1.0, 2.0], "b": [3.0, 4.0]}))


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------


def configured_transformers() -> list[Transformer]:
    """One instance of every transformer, with non-default parameters."""
    return [
        RollingAggregate(
            window="7d",
            agg="quantile",
            agg_params={"q": [0.1, 0.9]},
            center=True,
            min_periods=2,
            closed="both",
        ),
        DoubleRollingAggregate(
            window=(3, 5),
            agg=("mean", "median"),
            agg_params=({"unused": 1}, None),
            center=False,
            min_periods=(1, 2),
            diff="abs_rel_diff",
        ),
        Retrospect(n_steps=3, step_size=2, till=1),
        StandardScale(),
        SeasonalDecomposition(period=7, trend=True, component="seasonal"),
        SumAll(),
        RegressionResidual(target="y"),
        PcaProjection(k=2),
        PcaReconstruction(k=3),
        PcaReconstructionError(k=4),
        PcaColumnError(k=5),
        CustomizedTransformer(
            transform_func=np.abs, transform_func_params={"out": None}
        ),
    ]


def test_clone_round_trips_every_transformer_parameter() -> None:
    for original in configured_transformers():
        copy = original.clone()
        assert copy.get_params() == original.get_params()
        assert repr(copy) == repr(original)


def test_get_params_names_every_constructor_parameter() -> None:
    assert set(RollingAggregate(window=2).get_params()) == {
        "window",
        "agg",
        "agg_params",
        "center",
        "min_periods",
        "closed",
    }
    assert set(DoubleRollingAggregate(window=2).get_params()) == {
        "window",
        "agg",
        "agg_params",
        "center",
        "min_periods",
        "diff",
    }


def test_clone_produces_an_unfitted_copy() -> None:
    original = SeasonalDecomposition(period=4).fit(daily(np.tile(PROFILE, 6)))
    assert original.fitted
    assert not original.clone().fitted


# ---------------------------------------------------------------------------
# backend independence
# ---------------------------------------------------------------------------


def test_a_widened_native_series_comes_back_as_a_native_frame() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="h", name="time")
    series = pd.Series([1.0, 2.0, 3.0, 4.0], index=index, name="x")
    lagged = Retrospect(n_steps=2).transform(series)
    assert list(lagged.columns) == ["t-0", "t-1"]
    assert lagged.index.equals(index)
    np.testing.assert_array_equal(lagged["t-1"], [np.nan, 1.0, 2.0, 3.0])


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_transformer_returns_the_backend_it_was_given(backend: str) -> None:
    native = make_native(backend, np.arange(10.0))
    assert type(RollingAggregate(window=3).transform(native)) is type(native)


def test_every_backend_produces_the_same_rolling_aggregate() -> None:
    values = np.concatenate([np.zeros(8), np.full(4, 5.0)])
    results = [
        TimeSeries.from_any(
            DoubleRollingAggregate(window=3, diff="diff").transform(
                make_native(backend, values)
            )
        )
        for backend in BACKENDS
    ]
    for other in results[1:]:
        np.testing.assert_array_equal(other.values, results[0].values)
    assert int(np.nanargmax(results[0].values[:, 0])) == 8


def test_every_backend_produces_the_same_seasonal_residual() -> None:
    values = np.tile(PROFILE, 8)
    values[9] += 3.0
    results = [
        TimeSeries.from_any(
            SeasonalDecomposition(period=4).fit_transform(
                make_native(backend, values, freq="D")
            )
        )
        for backend in BACKENDS
    ]
    for other in results[1:]:
        np.testing.assert_allclose(other.values, results[0].values)
    assert int(np.argmax(np.abs(results[0].values[:, 0]))) == 9


def test_every_backend_produces_the_same_reconstruction_error() -> None:
    base = np.arange(12.0)
    other = 2.0 * base
    other[5] += 4.0
    frame = daily_frame({"a": base, "b": other})
    results = [
        TimeSeries.from_any(
            PcaReconstructionError(k=1).fit_transform(frame.to_native(backend=backend))
        )
        for backend in BACKENDS
    ]
    for other_result in results[1:]:
        np.testing.assert_allclose(other_result.values, results[0].values)
    assert int(np.argmax(results[0].values[:, 0])) == 5


# ---------------------------------------------------------------------------
# degenerate inputs
# ---------------------------------------------------------------------------


def empty_series(columns: int = 1) -> TimeSeries:
    """A series with no observations at all."""
    return TimeSeries.from_arrays(
        np.array([], dtype="datetime64[D]"),
        np.zeros((0, columns)),
        [f"c{i}" for i in range(columns)],
    )


def test_an_empty_series_survives_the_window_end_shift() -> None:
    result = DoubleRollingAggregate(window=2, center=False).run(empty_series())
    assert result.n_rows == 0


def test_an_empty_frame_sums_to_an_empty_series() -> None:
    assert SumAll().run(empty_series(2)).n_rows == 0


def test_a_seasonal_model_rejects_an_irregular_series_at_transform_time() -> None:
    model = SeasonalDecomposition(period=4).fit(daily(np.tile(PROFILE, 6)))
    irregular = TimeSeries.from_arrays(
        np.array(
            ["2024-03-01", "2024-03-02", "2024-03-05", "2024-03-06"],
            dtype="datetime64[D]",
        ),
        np.tile(PROFILE, 1),
    )
    with pytest.raises(ValueError, match="needs a regular time axis"):
        model.run(irregular)


def test_a_regression_residual_is_missing_where_no_row_is_complete() -> None:
    frame = daily_frame({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0]})
    model = RegressionResidual(target="b").fit(frame)
    gappy = daily_frame({"a": [np.nan, np.nan], "b": [1.0, 2.0]})
    assert np.isnan(flat(model.run(gappy))).all()


def test_pca_is_missing_everywhere_when_no_row_is_complete() -> None:
    model = PcaReconstructionError(k=1).fit(rank_one_frame())
    gappy = daily_frame({"a": [np.nan, np.nan], "b": [1.0, 2.0]})
    assert np.isnan(flat(model.run(gappy))).all()
