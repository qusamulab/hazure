"""Tests for the dataframe boundary.

The contract under test: whatever goes in comes back out unchanged, and the
internal representation is identical no matter which backend supplied it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

from hazure import TimeSeries
from tests.helpers import BACKENDS, make_native

VALUES = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


# -- round trips ------------------------------------------------------------


def test_pandas_series_round_trips_exactly() -> None:
    index = pd.date_range("2024-01-01", periods=6, freq="h", name="ts")
    original = pd.Series(VALUES, index=index, name="x")
    assert TimeSeries.from_any(original).to_native().equals(original)


def test_pandas_series_with_unnamed_index_keeps_index_unnamed() -> None:
    original = pd.Series(
        VALUES, index=pd.date_range("2024-01-01", periods=6, freq="h"), name="x"
    )
    restored = TimeSeries.from_any(original).to_native()
    assert restored.index.name is None
    assert restored.equals(original)


def test_unnamed_series_gets_a_placeholder_column_name() -> None:
    index = pd.date_range("2024-01-01", periods=6, freq="h", name="time")
    ts = TimeSeries.from_any(pd.Series(VALUES, index=index))
    assert ts.columns == ("value",)


def test_time_zone_survives_the_round_trip() -> None:
    index = pd.date_range("2024-01-01", periods=6, freq="h", tz="Asia/Tokyo", name="t")
    original = pd.Series(VALUES, index=index, name="x")
    restored = TimeSeries.from_any(original).to_native()
    assert restored.equals(original)
    assert str(restored.index.tz) == "Asia/Tokyo"


def test_pandas_frame_round_trips_exactly() -> None:
    index = pd.date_range("2024-01-01", periods=6, freq="h", name="time")
    original = pd.DataFrame({"a": VALUES, "b": VALUES[::-1]}, index=index)
    assert TimeSeries.from_any(original).to_native().equals(original)


@pytest.mark.parametrize("backend", BACKENDS)
def test_every_backend_round_trips_exactly(backend: str) -> None:
    original = make_native(backend, VALUES)
    restored = TimeSeries.from_any(original).to_native()
    assert type(restored) is type(original)
    assert restored.equals(original)


def test_backends_agree_on_the_internal_representation() -> None:
    """The whole point of the boundary: one canonical form, whatever came in."""
    parsed = [TimeSeries.from_any(make_native(b, VALUES)) for b in BACKENDS]
    reference = parsed[0]
    for other in parsed[1:]:
        assert np.array_equal(other.time, reference.time)
        assert np.array_equal(other.values, reference.values)
        assert other.columns == reference.columns
        assert other.freq == reference.freq


def test_microsecond_resolution_is_preserved() -> None:
    """pandas 3 and polars both default to microseconds, not nanoseconds."""
    index = pd.date_range("2024-01-01", periods=6, freq="h", name="time")
    assert index.dtype == "datetime64[us]"
    restored = TimeSeries.from_any(pd.Series(VALUES, index=index, name="x")).to_native()
    assert restored.index.dtype == "datetime64[us]"


def test_emitting_into_a_different_backend() -> None:
    ts = TimeSeries.from_any(make_native("pandas", VALUES))
    assert isinstance(ts.to_native(backend="polars"), pl.DataFrame)


# -- time axis --------------------------------------------------------------


def test_freq_is_inferred_for_a_regular_axis() -> None:
    ts = TimeSeries.from_any(make_native("pandas", VALUES, freq="30min"))
    assert ts.freq == 30 * 60 * 1_000_000_000


def test_freq_is_none_for_an_irregular_axis() -> None:
    time = np.array(
        ["2024-01-01T00", "2024-01-01T01", "2024-01-01T03"], dtype="datetime64[ns]"
    )
    assert TimeSeries.from_arrays(time, [1.0, 2.0, 3.0]).freq is None


def test_freq_is_none_when_too_short_to_tell() -> None:
    """Two points imply a step but do not establish a frequency."""
    time = np.array(["2024-01-01T00", "2024-01-01T01"], dtype="datetime64[ns]")
    assert TimeSeries.from_arrays(time, [1.0, 2.0]).freq is None


def test_unsorted_input_is_sorted() -> None:
    time = np.array(
        ["2024-01-01T02", "2024-01-01T00", "2024-01-01T01"], dtype="datetime64[ns]"
    )
    ts = TimeSeries.from_arrays(time, [3.0, 1.0, 2.0])
    assert np.array_equal(ts.values[:, 0], [1.0, 2.0, 3.0])


def test_unsorted_input_raises_when_sorting_is_declined() -> None:
    time = np.array(["2024-01-01T02", "2024-01-01T00"], dtype="datetime64[ns]")
    with pytest.raises(ValueError, match="not sorted"):
        TimeSeries.from_arrays(time, [1.0, 2.0], sort=False)


def test_duplicate_timestamps_keep_the_first_observation() -> None:
    time = np.array(
        ["2024-01-01T00", "2024-01-01T00", "2024-01-01T01"], dtype="datetime64[ns]"
    )
    ts = TimeSeries.from_arrays(time, [1.0, 99.0, 2.0])
    assert ts.n_rows == 2
    assert np.array_equal(ts.values[:, 0], [1.0, 2.0])


def test_duplicate_timestamps_raise_when_deduplication_is_declined() -> None:
    time = np.array(["2024-01-01T00", "2024-01-01T00"], dtype="datetime64[ns]")
    with pytest.raises(ValueError, match="duplicated timestamp"):
        TimeSeries.from_arrays(time, [1.0, 2.0], drop_duplicates=False)


def test_missing_timestamps_are_rejected() -> None:
    index = pd.DatetimeIndex(["2024-01-01", pd.NaT, "2024-01-03"])
    with pytest.raises(ValueError, match="missing timestamps"):
        TimeSeries.from_any(pd.Series([1.0, 2.0, 3.0], index=index, name="x"))


# -- rejections -------------------------------------------------------------


def test_a_non_temporal_index_is_rejected() -> None:
    with pytest.raises(TypeError, match="no time axis"):
        TimeSeries.from_any(pd.Series(VALUES, name="x"))


def test_a_frame_without_a_temporal_column_is_rejected() -> None:
    with pytest.raises(TypeError, match="no temporal column"):
        TimeSeries.from_any(pl.DataFrame({"a": VALUES}))


def test_ambiguous_temporal_columns_require_an_explicit_choice() -> None:
    stamps = pl.datetime_range(
        pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01 05:00"), "1h", eager=True
    )
    frame = pl.DataFrame({"t1": stamps, "t2": stamps, "a": VALUES})
    with pytest.raises(ValueError, match="several temporal columns"):
        TimeSeries.from_any(frame)


def test_naming_the_time_column_resolves_the_ambiguity() -> None:
    stamps = pl.datetime_range(
        pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01 05:00"), "1h", eager=True
    )
    frame = pl.DataFrame({"t": stamps, "a": VALUES, "b": VALUES[::-1]})
    assert TimeSeries.from_any(frame, time="t").columns == ("a", "b")


def test_a_leftover_temporal_column_is_still_rejected() -> None:
    """Naming one time column does not make a second one a valid value column."""
    stamps = pl.datetime_range(
        pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01 05:00"), "1h", eager=True
    )
    frame = pl.DataFrame({"t1": stamps, "t2": stamps, "a": VALUES})
    with pytest.raises(TypeError, match="not numeric"):
        TimeSeries.from_any(frame, time="t1")


def test_a_string_column_is_rejected_with_a_useful_message() -> None:
    stamps = pl.datetime_range(
        pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01 05:00"), "1h", eager=True
    )
    frame = pl.DataFrame({"t": stamps, "label": ["a"] * 6})
    with pytest.raises(TypeError, match="not numeric"):
        TimeSeries.from_any(frame)


def test_a_frame_with_only_a_time_column_is_rejected() -> None:
    stamps = pl.datetime_range(
        pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01 05:00"), "1h", eager=True
    )
    with pytest.raises(ValueError, match="no value columns"):
        TimeSeries.from_any(pl.DataFrame({"t": stamps}))


def test_an_unsupported_object_is_rejected() -> None:
    with pytest.raises(TypeError, match="not a supported dataframe"):
        TimeSeries.from_any([1, 2, 3])


def test_duplicate_column_names_are_rejected() -> None:
    time = np.array(["2024-01-01", "2024-01-02", "2024-01-03"], dtype="datetime64[ns]")
    with pytest.raises(ValueError, match="must be unique"):
        TimeSeries.from_arrays(time, np.zeros((3, 2)), ["a", "a"])


# -- derivation -------------------------------------------------------------


def test_from_any_passes_an_existing_timeseries_through() -> None:
    ts = TimeSeries.from_any(make_native("pandas", VALUES))
    assert TimeSeries.from_any(ts) is ts


def test_wrap_inherits_the_time_axis_and_names() -> None:
    ts = TimeSeries.from_any(make_native("pandas", VALUES))
    wrapped = ts.wrap(np.arange(6.0))
    assert wrapped.time is ts.time
    assert wrapped.columns == ts.columns
    assert wrapped.freq == ts.freq


def test_wrap_rejects_a_length_mismatch() -> None:
    ts = TimeSeries.from_any(make_native("pandas", VALUES))
    with pytest.raises(ValueError, match="Cannot wrap"):
        ts.wrap(np.arange(3.0))


def test_wrap_names_a_widened_result() -> None:
    ts = TimeSeries.from_any(make_native("pandas", VALUES))
    assert ts.wrap(np.zeros((6, 3))).columns == ("value_0", "value_1", "value_2")


def test_select_narrows_and_reorders() -> None:
    index = pd.date_range("2024-01-01", periods=6, freq="h", name="time")
    ts = TimeSeries.from_any(
        pd.DataFrame({"a": VALUES, "b": VALUES[::-1]}, index=index)
    )
    assert ts.select(["b", "a"]).columns == ("b", "a")
    assert ts.select("a").is_univariate


def test_select_rejects_an_unknown_column() -> None:
    ts = TimeSeries.from_any(make_native("pandas", VALUES))
    with pytest.raises(KeyError, match="Unknown column"):
        ts.select("nope")


def test_iter_columns_yields_one_series_per_column() -> None:
    index = pd.date_range("2024-01-01", periods=6, freq="h", name="time")
    ts = TimeSeries.from_any(
        pd.DataFrame({"a": VALUES, "b": VALUES[::-1]}, index=index)
    )
    parts = list(ts.iter_columns())
    assert [p.columns for p in parts] == [("a",), ("b",)]


def test_column_values_returns_a_flat_array() -> None:
    ts = TimeSeries.from_any(make_native("pandas", VALUES))
    assert np.array_equal(ts.column_values("x"), VALUES)


# -- join -------------------------------------------------------------------


def test_join_on_an_identical_axis_concatenates_columns() -> None:
    left = TimeSeries.from_any(make_native("pandas", VALUES, name="a"))
    right = TimeSeries.from_any(make_native("pandas", VALUES[::-1], name="b"))
    joined = left.join(right)
    assert joined.columns == ("a", "b")
    assert joined.n_rows == 6
    assert not np.isnan(joined.values).any()


def test_join_on_disjoint_axes_outer_joins_with_nan() -> None:
    """This is what replaces pandas' implicit index alignment."""
    left = TimeSeries.from_arrays(
        np.array(["2024-01-01T00", "2024-01-01T01"], dtype="datetime64[ns]"),
        [1.0, 2.0],
        ["a"],
    )
    right = TimeSeries.from_arrays(
        np.array(["2024-01-01T01", "2024-01-01T02"], dtype="datetime64[ns]"),
        [3.0, 4.0],
        ["b"],
    )
    joined = left.join(right)
    assert joined.n_rows == 3
    assert np.array_equal(np.isnan(joined.values[:, 0]), [False, False, True])
    assert np.array_equal(np.isnan(joined.values[:, 1]), [True, False, False])
    assert joined.values[1].tolist() == [2.0, 3.0]


def test_join_rejects_a_column_name_collision() -> None:
    left = TimeSeries.from_any(make_native("pandas", VALUES, name="a"))
    with pytest.raises(ValueError, match="more than one series"):
        left.join(left)


def test_join_with_no_arguments_is_the_identity() -> None:
    ts = TimeSeries.from_any(make_native("pandas", VALUES))
    assert ts.join() is ts


# -- misc -------------------------------------------------------------------


def test_repr_reports_shape_and_frequency() -> None:
    ts = TimeSeries.from_any(make_native("pandas", VALUES))
    assert "6 rows x 1 cols" in repr(ts)
    assert "freq=1h" in repr(ts)


def test_len_is_the_row_count() -> None:
    assert len(TimeSeries.from_any(make_native("pandas", VALUES))) == 6


def test_integer_values_are_promoted_to_float() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="h", name="time")
    ts = TimeSeries.from_any(pd.Series([1, 2, 3], index=index, name="x"))
    assert ts.values.dtype == np.float64


def test_boolean_values_become_one_and_zero() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="h", name="time")
    ts = TimeSeries.from_any(pd.Series([True, False, True], index=index, name="x"))
    assert np.array_equal(ts.values[:, 0], [1.0, 0.0, 1.0])


def test_nulls_become_nan() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="h", name="time")
    ts = TimeSeries.from_any(pd.Series([1.0, None, 3.0], index=index, name="x"))
    assert np.isnan(ts.values[1, 0])


def test_from_arrays_accepts_epoch_nanoseconds() -> None:
    ts = TimeSeries.from_arrays(
        np.array([0, 1_000_000_000, 2_000_000_000]), [1.0, 2, 3]
    )
    assert ts.freq == 1_000_000_000


def test_from_arrays_rejects_a_bad_time_dtype() -> None:
    with pytest.raises(TypeError, match="Cannot read a time axis"):
        TimeSeries.from_arrays(np.array([0.5, 1.5]), [1.0, 2.0])


def test_from_arrays_rejects_three_dimensional_values() -> None:
    time = np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]")
    with pytest.raises(ValueError, match="1-D or 2-D"):
        TimeSeries.from_arrays(time, np.zeros((2, 1, 1)))


def test_from_arrays_rejects_mismatched_lengths() -> None:
    time = np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]")
    with pytest.raises(ValueError, match="entries but values have"):
        TimeSeries.from_arrays(time, [1.0, 2.0, 3.0])


def test_timeseries_is_immutable() -> None:
    ts = TimeSeries.from_any(make_native("pandas", VALUES))
    with pytest.raises((AttributeError, TypeError)):
        ts.freq = 1  # type: ignore[misc]


def test_naming_a_time_column_overrides_the_pandas_index() -> None:
    """An explicit ``time=`` should win over whatever the index happens to be."""
    frame = pd.DataFrame(
        {
            "when": pd.date_range("2024-01-01", periods=3, freq="h"),
            "a": [1.0, 2.0, 3.0],
        }
    )
    ts: Any = TimeSeries.from_any(frame, time="when")
    assert ts.columns == ("a",)
    assert ts.origin.time_on_index is False
