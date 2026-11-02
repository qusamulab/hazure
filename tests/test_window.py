"""Tests for the rolling window engine.

Window semantics follow ``pandas.Series.rolling``, so pandas is the reference for
the boundary rules. Shape statistics use a brute-force per-window reference
because older pandas releases could leave rolling skewness permanently NaN
after a gap; see :func:`test_rolling_skew_recovers_after_gaps`.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as npst

from hazure._core import window as window_module
from hazure._core.window import (
    AGGREGATIONS,
    aggregate_windows,
    default_min_periods,
    double_rolling,
    parse_duration,
    rolling,
    window_bounds,
)

# Statistics where pandas' rolling implementation is trustworthy.
PANDAS_SAFE = ("sum", "mean", "median", "min", "max", "std", "var")
CLOSED = ("right", "left", "both", "neither")


def hourly_index(n: int) -> pd.DatetimeIndex:
    """Return a regular hourly index of length ``n``."""
    return pd.date_range("2024-01-01", periods=n, freq="h")


def epoch_ns(index: pd.DatetimeIndex) -> np.ndarray:
    """Return a naive pandas index as UTC nanoseconds."""
    return index.to_numpy().astype("datetime64[ns]").view(np.int64)


def brute_force(
    values: np.ndarray, window: int, how: str, min_periods: int
) -> np.ndarray:
    """Reference implementation: one independent pandas call per window."""
    out = np.full(len(values), np.nan)
    for i in range(len(values)):
        chunk = pd.Series(values[max(0, i - window + 1) : i + 1]).dropna()
        if len(chunk) >= min_periods:
            out[i] = getattr(chunk, how)()
    return out


@pytest.fixture
def gappy() -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """Random values with scattered gaps, plus a matching time axis."""
    rng = np.random.default_rng(42)
    values = rng.normal(size=40)
    values[[3, 17, 18, 31]] = np.nan
    index = hourly_index(40)
    return values, epoch_ns(index), pd.Series(values, index=index)


# -- parity with pandas -----------------------------------------------------


@pytest.mark.parametrize("agg", PANDAS_SAFE)
@pytest.mark.parametrize("size", [2, 3, 5, 8])
def test_integer_window_matches_pandas(
    gappy: tuple[np.ndarray, np.ndarray, pd.Series], agg: str, size: int
) -> None:
    values, _, series = gappy
    expected = getattr(series.rolling(size), agg)().to_numpy()
    np.testing.assert_allclose(
        rolling(values, size, agg), expected, equal_nan=True, rtol=1e-9
    )


@pytest.mark.parametrize("agg", PANDAS_SAFE)
@pytest.mark.parametrize("size", [3, 4, 5])
def test_centred_integer_window_matches_pandas(
    gappy: tuple[np.ndarray, np.ndarray, pd.Series], agg: str, size: int
) -> None:
    """Includes even widths, where pandas puts the extra observation left."""
    values, _, series = gappy
    expected = getattr(series.rolling(size, center=True), agg)().to_numpy()
    np.testing.assert_allclose(
        rolling(values, size, agg, center=True), expected, equal_nan=True, rtol=1e-9
    )


@pytest.mark.parametrize("agg", PANDAS_SAFE)
@pytest.mark.parametrize("closed", CLOSED)
def test_integer_window_closed_matches_pandas(
    gappy: tuple[np.ndarray, np.ndarray, pd.Series], agg: str, closed: str
) -> None:
    values, _, series = gappy
    expected = getattr(
        series.rolling(4, closed=closed, min_periods=1), agg
    )().to_numpy()
    np.testing.assert_allclose(
        rolling(values, 4, agg, closed=closed, min_periods=1),
        expected,
        equal_nan=True,
        rtol=1e-9,
    )


@pytest.mark.parametrize("agg", PANDAS_SAFE)
@pytest.mark.parametrize("span", ["3h", "5h", "12h"])
@pytest.mark.parametrize("closed", CLOSED)
def test_duration_window_matches_pandas(
    gappy: tuple[np.ndarray, np.ndarray, pd.Series],
    agg: str,
    span: str,
    closed: str,
) -> None:
    values, time, series = gappy
    expected = getattr(series.rolling(span, closed=closed), agg)().to_numpy()
    np.testing.assert_allclose(
        rolling(values, span, agg, time=time, closed=closed),
        expected,
        equal_nan=True,
        rtol=1e-9,
    )


@pytest.mark.parametrize("agg", PANDAS_SAFE)
@pytest.mark.parametrize("span", ["3h", "4h"])
def test_centred_duration_window_matches_pandas(
    gappy: tuple[np.ndarray, np.ndarray, pd.Series], agg: str, span: str
) -> None:
    values, time, series = gappy
    expected = getattr(series.rolling(span, center=True), agg)().to_numpy()
    np.testing.assert_allclose(
        rolling(values, span, agg, time=time, center=True),
        expected,
        equal_nan=True,
        rtol=1e-9,
    )


@pytest.mark.parametrize("agg", PANDAS_SAFE)
@pytest.mark.parametrize(("size", "min_periods"), [(5, 1), (5, 3), (8, 2), (8, 8)])
def test_min_periods_matches_pandas(
    gappy: tuple[np.ndarray, np.ndarray, pd.Series],
    agg: str,
    size: int,
    min_periods: int,
) -> None:
    values, _, series = gappy
    expected = getattr(series.rolling(size, min_periods=min_periods), agg)().to_numpy()
    np.testing.assert_allclose(
        rolling(values, size, agg, min_periods=min_periods),
        expected,
        equal_nan=True,
        rtol=1e-9,
    )


@pytest.mark.parametrize("q", [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
def test_quantile_matches_pandas(
    gappy: tuple[np.ndarray, np.ndarray, pd.Series], q: float
) -> None:
    values, _, series = gappy
    np.testing.assert_allclose(
        rolling(values, 5, "quantile", q=q),
        series.rolling(5).quantile(q).to_numpy(),
        equal_nan=True,
        rtol=1e-9,
    )


def test_duration_window_on_an_irregular_axis_matches_pandas() -> None:
    """The case integer windows cannot express: gaps shrink the window."""
    index = pd.DatetimeIndex(
        [
            "2024-01-01 00:00",
            "2024-01-01 00:10",
            "2024-01-01 00:15",
            "2024-01-01 03:00",
            "2024-01-01 03:05",
            "2024-01-01 09:00",
        ]
    )
    values = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 50.0])
    series = pd.Series(values, index=index)
    time = epoch_ns(index)
    for agg in PANDAS_SAFE:
        np.testing.assert_allclose(
            rolling(values, "1h", agg, time=time),
            getattr(series.rolling("1h"), agg)().to_numpy(),
            equal_nan=True,
            rtol=1e-9,
            err_msg=f"agg={agg}",
        )


@settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    values=npst.arrays(
        dtype=np.float64,
        shape=npst.array_shapes(min_dims=1, max_dims=1, min_side=1, max_side=40),
        elements=st.one_of(
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
            st.just(np.nan),
        ),
    ),
    size=st.integers(min_value=1, max_value=8),
    fraction=st.floats(min_value=0.0, max_value=1.0),
    center=st.booleans(),
    agg=st.sampled_from(PANDAS_SAFE),
)
def test_rolling_matches_pandas_on_arbitrary_input(
    values: np.ndarray,
    size: int,
    fraction: float,
    center: bool,
    agg: str,
) -> None:
    """Property test over shapes, gaps, widths and min_periods together.

    ``min_periods`` is derived from ``size`` because pandas refuses to build a
    window when it exceeds the width, whereas hazure treats that as "never
    enough observations" and returns all-missing.

    The absolute tolerance scales with the data because pandas accumulates
    running sums as the window slides, so a large value that has already left
    the window still perturbs the result. hazure recomputes from the window's
    own observations and does not drift — see
    :func:`test_variance_is_exact_where_pandas_drifts`.
    """
    min_periods = 1 + int(fraction * (size - 1))
    series = pd.Series(values, index=hourly_index(len(values)))
    expected = getattr(
        series.rolling(size, min_periods=min_periods, center=center), agg
    )().to_numpy()
    magnitude = float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 1.0
    np.testing.assert_allclose(
        rolling(values, size, agg, min_periods=min_periods, center=center),
        expected,
        equal_nan=True,
        rtol=1e-7,
        atol=1e-6 * max(magnitude, 1.0),
    )


def test_variance_is_exact_where_pandas_drifts() -> None:
    """A window of identical values has zero spread, and hazure reports zero.

    pandas' incremental variance still carries the earlier large observation in
    its running sums, so it reports a small non-zero standard deviation for a
    window that is entirely zeros.
    """
    values = np.array([31.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    series = pd.Series(values, index=hourly_index(len(values)))

    assert rolling(values, 6, "std")[7] == 0.0
    assert series.rolling(6).std().to_numpy()[7] > 1e-8


# -- skewness and kurtosis --------------------------------------------------


@pytest.mark.parametrize("agg", ["skew", "kurt"])
@pytest.mark.parametrize("size", [3, 4, 5, 8])
def test_shape_statistics_match_a_brute_force_reference(
    gappy: tuple[np.ndarray, np.ndarray, pd.Series], agg: str, size: int
) -> None:
    values, _, _ = gappy
    np.testing.assert_allclose(
        rolling(values, size, agg),
        brute_force(values, size, agg, size),
        equal_nan=True,
        rtol=1e-9,
    )


def test_rolling_skew_recovers_after_gaps(
    gappy: tuple[np.ndarray, np.ndarray, pd.Series],
) -> None:
    """Only windows lacking three observations should be missing.

    Older pandas releases reported 28 missing values on this input instead of
    12. Check hazure against independent per-window reductions so upstream
    rolling fixes do not affect this regression test.
    """
    values, _, _ = gappy
    reference = brute_force(values, 3, "skew", 3)
    result = rolling(values, 3, "skew")

    np.testing.assert_array_equal(
        np.flatnonzero(np.isnan(result)), [0, 1, 3, 4, 5, 17, 18, 19, 20, 31, 32, 33]
    )
    np.testing.assert_allclose(result, reference, equal_nan=True, rtol=1e-9)


def test_skewness_needs_three_observations() -> None:
    assert np.isnan(rolling(np.array([1.0, 2.0]), 2, "skew", min_periods=1)).all()


def test_kurtosis_needs_four_observations() -> None:
    result = rolling(np.array([1.0, 2.0, 4.0]), 3, "kurt", min_periods=1)
    assert np.isnan(result).all()


def test_shape_statistics_are_undefined_without_spread() -> None:
    """A flat window has no shape, so reporting a number would be a lie."""
    flat = np.full(6, 2.5)
    assert np.isnan(rolling(flat, 4, "skew")[3:]).all()
    assert np.isnan(rolling(flat, 4, "kurt")[3:]).all()


# -- statistics pandas has no direct equivalent for -------------------------


def test_count_returns_the_number_of_observations() -> None:
    values = np.array([1.0, np.nan, 3.0, 4.0])
    np.testing.assert_array_equal(
        rolling(values, 2, "count", min_periods=1), [1.0, 1.0, 1.0, 2.0]
    )


def test_count_respects_min_periods_unlike_pandas() -> None:
    """The one deliberate divergence, kept so count behaves like its siblings."""
    values = np.array([1.0, np.nan, np.nan, 4.0])
    series = pd.Series(values, index=hourly_index(4))

    ours = rolling(values, 3, "count", min_periods=3)
    theirs = series.rolling(3, min_periods=3).count().to_numpy()

    assert np.isnan(ours).all()
    np.testing.assert_array_equal(theirs[2:], [1.0, 1.0])


def test_nnz_counts_non_zero_observations() -> None:
    values = np.array([0.0, 1.0, 0.0, 2.0, np.nan])
    np.testing.assert_array_equal(
        rolling(values, 3, "nnz", min_periods=1), [0.0, 1.0, 1.0, 2.0, 1.0]
    )


def test_nunique_counts_distinct_observations() -> None:
    values = np.array([1.0, 1.0, 2.0, 2.0, 3.0])
    np.testing.assert_array_equal(
        rolling(values, 3, "nunique", min_periods=1), [1.0, 1.0, 2.0, 2.0, 2.0]
    )


def test_nunique_ignores_gaps() -> None:
    values = np.array([1.0, np.nan, 1.0, 2.0])
    np.testing.assert_array_equal(
        rolling(values, 3, "nunique", min_periods=1), [1.0, 1.0, 1.0, 2.0]
    )


def test_iqr_is_the_inter_quartile_range(
    gappy: tuple[np.ndarray, np.ndarray, pd.Series],
) -> None:
    values, _, series = gappy
    expected = (
        series.rolling(6).quantile(0.75) - series.rolling(6).quantile(0.25)
    ).to_numpy()
    np.testing.assert_allclose(
        rolling(values, 6, "iqr"), expected, equal_nan=True, rtol=1e-9
    )


def test_idr_is_the_inter_decile_range(
    gappy: tuple[np.ndarray, np.ndarray, pd.Series],
) -> None:
    values, _, series = gappy
    expected = (
        series.rolling(6).quantile(0.9) - series.rolling(6).quantile(0.1)
    ).to_numpy()
    np.testing.assert_allclose(
        rolling(values, 6, "idr"), expected, equal_nan=True, rtol=1e-9
    )


def test_every_advertised_aggregation_runs() -> None:
    """Guard against :data:`AGGREGATIONS` drifting from the dispatch table."""
    values = np.arange(12.0)
    for agg in sorted(AGGREGATIONS):
        result = rolling(values, 5, agg, min_periods=1, q=0.5)
        assert result.shape == values.shape, agg


# -- window bounds ----------------------------------------------------------


def test_bounds_are_well_formed() -> None:
    start, stop = window_bounds(None, 3, 6)
    assert (start >= 0).all()
    assert (stop <= 6).all()
    assert (start <= stop).all()


def test_trailing_integer_bounds() -> None:
    start, stop = window_bounds(None, 3, 5)
    assert start.tolist() == [0, 0, 0, 1, 2]
    assert stop.tolist() == [1, 2, 3, 4, 5]


def test_centred_odd_integer_bounds_are_symmetric() -> None:
    start, stop = window_bounds(None, 3, 5, center=True)
    assert start.tolist() == [0, 0, 1, 2, 3]
    assert stop.tolist() == [2, 3, 4, 5, 5]


def test_centred_even_integer_bounds_lean_left() -> None:
    """Matches pandas: the extra observation of an even window goes left."""
    start, stop = window_bounds(None, 4, 6, center=True)
    assert (stop - start).tolist() == [2, 3, 4, 4, 4, 3]


def test_duration_bounds_shrink_across_a_gap() -> None:
    index = pd.DatetimeIndex(
        ["2024-01-01 00:00", "2024-01-01 00:30", "2024-01-01 06:00"]
    )
    start, stop = window_bounds(epoch_ns(index), "1h", 3)
    assert (stop - start).tolist() == [1, 2, 1]


def test_a_duration_window_needs_a_time_axis() -> None:
    with pytest.raises(ValueError, match="needs a time axis"):
        window_bounds(None, "1h", 5)


def test_a_time_axis_of_the_wrong_length_is_rejected() -> None:
    with pytest.raises(ValueError, match="but n_rows is"):
        window_bounds(np.array([0, 1], dtype=np.int64), "1h", 5)


def test_a_non_positive_integer_window_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        window_bounds(None, 0, 5)


def test_an_unknown_closed_rule_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown closed"):
        window_bounds(None, 3, 5, closed="middle")  # type: ignore[arg-type]


def test_default_min_periods_follows_pandas() -> None:
    assert default_min_periods(5) == 5
    assert default_min_periods(5, "both") == 6
    assert default_min_periods(5, "neither") == 4
    assert default_min_periods("1h") == 1


# -- duration parsing -------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("1ns", 1),
        ("500ms", 500_000_000),
        ("2s", 2_000_000_000),
        ("30min", 1_800_000_000_000),
        ("1h", 3_600_000_000_000),
        ("7d", 604_800_000_000_000),
        ("2w", 1_209_600_000_000_000),
        ("h", 3_600_000_000_000),
        ("1.5h", 5_400_000_000_000),
        ("  3 h  ", 10_800_000_000_000),
        ("2HOURS", 7_200_000_000_000),
    ],
)
def test_parse_duration(spec: str, expected: int) -> None:
    assert parse_duration(spec) == expected


def test_parse_duration_accepts_timedelta_objects() -> None:
    assert parse_duration(timedelta(hours=2)) == 7_200_000_000_000
    assert parse_duration(np.timedelta64(2, "h")) == 7_200_000_000_000


def test_minutes_and_months_are_distinguished_by_case() -> None:
    """A lowercase 'm' is minutes; an uppercase 'M' would be a calendar month."""
    assert parse_duration("5m") == 300_000_000_000
    with pytest.raises(ValueError, match="calendar unit"):
        parse_duration("5M")


@pytest.mark.parametrize("spec", ["1Y", "2Q", "3months", "1mo"])
def test_calendar_units_are_rejected(spec: str) -> None:
    with pytest.raises(ValueError, match="calendar unit"):
        parse_duration(spec)


def test_an_unknown_unit_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown duration unit"):
        parse_duration("5furlongs")


@pytest.mark.parametrize("spec", ["3 weeks ago", "", "1h30min", "5%"])
def test_an_unparseable_duration_is_rejected(spec: str) -> None:
    with pytest.raises(ValueError, match="Cannot read"):
        parse_duration(spec)


def test_a_zero_duration_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        parse_duration("0s")


# -- double rolling ---------------------------------------------------------


def test_double_rolling_compares_adjacent_windows() -> None:
    step = np.array([0.0, 0.0, 0.0, 5.0, 5.0, 5.0])
    np.testing.assert_allclose(
        double_rolling(step, 2, "mean", diff="diff"),
        [np.nan, np.nan, 2.5, 5.0, 2.5, np.nan],
        equal_nan=True,
    )


def test_the_left_window_holds_exactly_w_observations_before_the_row() -> None:
    """Regression: an off-by-one here silently halves the left window."""
    ramp = np.arange(8.0)
    # Left mean over [i-3, i) minus nothing; check via a signed diff of means.
    left_only = double_rolling(ramp, 3, "mean", diff="diff")
    # right mean over [i, i+3) is left mean + 3 for a unit ramp.
    np.testing.assert_allclose(left_only[3:5], [3.0, 3.0])


def test_double_rolling_flags_a_spike_at_the_right_place() -> None:
    spike = np.array([1.0, 1.0, 1.0, 1.0, 9.0, 1.0, 1.0, 1.0, 1.0])
    scores = double_rolling(spike, (4, 1), "median", diff="l1")
    assert np.nanargmax(scores) == 4
    np.testing.assert_allclose(scores[5:], 0.0)


def test_double_rolling_flags_a_level_shift_at_the_right_place() -> None:
    """The first observation of the new level scores maximally.

    A width-3 median cannot resolve the shift more finely than three rows, so
    the score plateaus either side of the change point. What matters is that the
    change point itself is at the top of that plateau, and that the score
    recovers the size of the shift.
    """
    level = np.concatenate([np.ones(5), np.full(5, 6.0)])
    scores = double_rolling(level, 3, "median", diff="l1")
    assert scores[5] == np.nanmax(scores)
    np.testing.assert_allclose(np.nanmax(scores), 5.0)
    np.testing.assert_allclose(scores[[3, 7]], 0.0)


def test_double_rolling_flags_a_volatility_shift() -> None:
    calm = np.zeros(8)
    wild = np.tile([-3.0, 3.0], 4)
    scores = double_rolling(np.concatenate([calm, wild]), 4, "std", diff="l1")
    assert 6 <= int(np.nanargmax(scores)) <= 9


@pytest.mark.parametrize("diff", ["l1", "l2", "diff", "rel_diff", "abs_rel_diff"])
def test_every_diff_mode_runs(diff: str) -> None:
    values = np.arange(1.0, 9.0)
    assert double_rolling(values, 2, "mean", diff=diff).shape == values.shape  # type: ignore[arg-type]


def test_l1_and_l2_agree_for_scalars() -> None:
    values = np.array([1.0, 5.0, 2.0, 8.0, 3.0, 9.0])
    np.testing.assert_allclose(
        double_rolling(values, 2, "mean", diff="l1"),
        double_rolling(values, 2, "mean", diff="l2"),
        equal_nan=True,
    )


def test_relative_diff_is_scale_free() -> None:
    values = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    absolute = double_rolling(values * 100, 2, "mean", diff="abs_rel_diff")
    relative = double_rolling(values, 2, "mean", diff="abs_rel_diff")
    np.testing.assert_allclose(absolute, relative, equal_nan=True)


def test_asymmetric_windows_and_aggregations() -> None:
    values = np.arange(12.0)
    result = double_rolling(values, (4, 2), ("mean", "median"), diff="diff")
    assert result.shape == values.shape
    assert not np.isnan(result[4:-2]).any()


def test_double_rolling_on_a_duration_window() -> None:
    index = hourly_index(12)
    values = np.concatenate([np.ones(6), np.full(6, 4.0)])
    scores = double_rolling(values, "3h", "median", time=epoch_ns(index), diff="l1")
    assert scores[6] == np.nanmax(scores)
    np.testing.assert_allclose(np.nanmax(scores), 3.0)


def test_double_rolling_rejects_an_unknown_diff() -> None:
    with pytest.raises(ValueError, match="Unknown diff"):
        double_rolling(np.arange(5.0), 2, diff="cosine")  # type: ignore[arg-type]


def test_double_rolling_rejects_a_two_dimensional_input() -> None:
    with pytest.raises(ValueError, match="1-D array"):
        double_rolling(np.zeros((3, 2)), 2)


def test_a_malformed_window_pair_is_rejected() -> None:
    with pytest.raises(ValueError, match="one value or a 2-tuple"):
        double_rolling(np.arange(5.0), (1, 2, 3))  # type: ignore[arg-type]


# -- misc -------------------------------------------------------------------


def test_rolling_rejects_a_two_dimensional_input() -> None:
    with pytest.raises(ValueError, match="1-D array"):
        rolling(np.zeros((3, 2)), 2)


def test_an_unknown_aggregation_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown aggregation"):
        rolling(np.arange(5.0), 2, "mode")


def test_quantile_requires_q() -> None:
    with pytest.raises(ValueError, match="needs q="):
        rolling(np.arange(5.0), 2, "quantile")


@pytest.mark.parametrize("q", [-0.1, 1.5])
def test_quantile_rejects_an_out_of_range_q(q: float) -> None:
    with pytest.raises(ValueError, match=r"q must lie in \[0, 1\]"):
        rolling(np.arange(5.0), 2, "quantile", q=q)


def test_mismatched_bounds_are_rejected() -> None:
    with pytest.raises(ValueError, match="but stop has"):
        aggregate_windows(
            np.arange(5.0),
            np.zeros(5, dtype=np.int64),
            np.zeros(4, dtype=np.int64),
            "mean",
        )


def test_an_empty_series_produces_an_empty_result() -> None:
    assert rolling(np.array([]), 3, "mean").shape == (0,)


def test_an_all_missing_series_produces_all_missing() -> None:
    assert np.isnan(rolling(np.full(5, np.nan), 3, "mean", min_periods=1)).all()


def test_chunking_does_not_change_the_result(
    gappy: tuple[np.ndarray, np.ndarray, pd.Series],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Peak memory is bounded by chunking rows; results must not depend on it."""
    values, _, _ = gappy
    expected = rolling(values, 9, "median", min_periods=1)
    monkeypatch.setattr(window_module, "_MAX_BLOCK_ELEMENTS", 9)
    np.testing.assert_allclose(
        rolling(values, 9, "median", min_periods=1), expected, equal_nan=True
    )
