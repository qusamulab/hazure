"""Tests for hazure.events: the Events value object, the converters and the splits."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from hazure import TimeSeries
from hazure.evaluation import split_train_test
from hazure.events import Events, expand_events, to_events, to_labels, validate_series
from tests.helpers import BACKENDS, make_native

HOUR = 3_600_000_000_000
DAY = 24 * HOUR


def hourly_labels(flags: list[float], **kwargs: Any) -> pd.Series:
    """Build an hourly pandas label series starting 2024-01-01."""
    index = pd.date_range("2024-01-01", periods=len(flags), freq="h", name="time")
    return pd.Series(flags, index=index, name="anomaly", **kwargs)


# ---------------------------------------------------------------------------
# Events: construction
# ---------------------------------------------------------------------------


def test_from_bounds_sorts_and_merges_overlapping_intervals() -> None:
    events = Events.from_bounds([[200, 300], [0, 100], [50, 120]])
    assert events.bounds.tolist() == [[0, 120], [200, 300]]


def test_from_bounds_merges_a_nested_interval_into_its_container() -> None:
    events = Events.from_bounds([[0, 100], [10, 20], [30, 40]])
    assert events.bounds.tolist() == [[0, 100]]


def test_from_bounds_merges_intervals_one_nanosecond_apart() -> None:
    # A nanosecond is the finest resolution represented, so nothing can sit
    # between 9 and 10; this is what makes the label round trip exact.
    assert Events.from_bounds([[0, 9], [10, 19]]).bounds.tolist() == [[0, 19]]


def test_from_bounds_keeps_intervals_two_nanoseconds_apart_separate() -> None:
    assert Events.from_bounds([[0, 9], [11, 19]]).bounds.tolist() == [[0, 9], [11, 19]]


def test_from_bounds_accepts_datetime64_of_any_unit() -> None:
    stamps = np.array([["2024-01-01", "2024-01-02"]], dtype="datetime64[s]")
    events = Events.from_bounds(stamps)
    assert events.bounds.tolist() == [[1704067200000000000, 1704153600000000000]]


def test_from_bounds_accepts_an_empty_list() -> None:
    events = Events.from_bounds([])
    assert events.n_events == 0
    assert events.bounds.shape == (0, 2)


def test_from_bounds_rejects_an_array_that_is_not_two_columns() -> None:
    with pytest.raises(ValueError, match=r"shape \(n, 2\)"):
        Events.from_bounds([[0, 1, 2]])


def test_from_bounds_rejects_a_float_array() -> None:
    with pytest.raises(TypeError, match="integer nanoseconds or datetime64"):
        Events.from_bounds([[0.5, 1.5]])


def test_from_bounds_rejects_an_interval_that_ends_before_it_starts() -> None:
    with pytest.raises(ValueError, match="Event 1 ends before it starts"):
        Events.from_bounds([[0, 10], [50, 20]])


def test_from_any_passes_an_events_through_unchanged() -> None:
    events = Events.from_bounds([[0, 10]])
    assert Events.from_any(events) is events


def test_from_any_reads_pairs_and_bare_timestamps_from_one_list() -> None:
    events = Events.from_any(
        [("2024-01-02T00:00", "2024-01-02T06:00"), "2024-01-01T00:00"]
    )
    assert events.n_events == 2
    assert events.durations.tolist() == [1, 6 * HOUR + 1]


def test_from_any_reads_every_flavour_of_timestamp_to_the_same_instant() -> None:
    expected = 1704067200000000000  # 2024-01-01T00:00:00Z
    variants: list[Any] = [
        "2024-01-01T00:00:00",
        np.datetime64("2024-01-01", "s"),
        datetime(2024, 1, 1),
        datetime(2024, 1, 1, 9, tzinfo=timezone(timedelta(hours=9))),
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-01", tz="UTC"),
        expected,
    ]
    for value in variants:
        assert Events.from_any([value]).bounds[0, 0] == expected


def test_from_any_keeps_the_nanoseconds_of_a_pandas_timestamp() -> None:
    stamp = pd.Timestamp("2024-01-01 00:00:00.000000007")
    assert Events.from_any([stamp]).bounds[0, 0] % 1000 == 7


def test_from_any_reads_a_start_end_frame_and_remembers_its_backend() -> None:
    frame = pl.DataFrame(
        {
            "start": [datetime(2024, 1, 1)],
            "end": [datetime(2024, 1, 2)],
        }
    )
    events = Events.from_any(frame)
    assert events.durations.tolist() == [DAY + 1]
    assert events.origin.backend == "polars"


def test_from_any_remembers_the_time_zone_of_a_frame() -> None:
    frame = pd.DataFrame(
        {
            "start": pd.to_datetime(["2024-01-01"]).tz_localize("Asia/Tokyo"),
            "end": pd.to_datetime(["2024-01-02"]).tz_localize("Asia/Tokyo"),
        }
    )
    events = Events.from_any(frame)
    assert events.origin.time_zone == "Asia/Tokyo"
    assert events.to_frame()["start"].dt.tz is not None


def test_from_any_lets_an_explicit_origin_override_the_frame_s_own() -> None:
    frame = pl.DataFrame(
        {"start": [datetime(2024, 1, 1)], "end": [datetime(2024, 1, 2)]}
    )
    origin = TimeSeries.from_any(hourly_labels([1.0])).origin
    assert Events.from_any(frame, origin=origin).origin.backend == "pandas"


def test_from_any_rejects_a_frame_without_start_and_end_columns() -> None:
    with pytest.raises(ValueError, match=r"needs \['end'\] column"):
        Events.from_any(pl.DataFrame({"start": [datetime(2024, 1, 1)]}))


def test_from_any_rejects_a_scalar() -> None:
    with pytest.raises(TypeError, match="Cannot read events from int"):
        Events.from_any(7)


def test_from_any_rejects_an_unreadable_element() -> None:
    with pytest.raises(TypeError, match=r"Cannot read .* as a timestamp"):
        Events.from_any([None])


def test_from_any_rejects_a_pair_of_the_wrong_length() -> None:
    with pytest.raises(ValueError, match="exactly 2 entries"):
        Events.from_any([(0, 1, 2)])


def test_from_any_reads_an_n_by_2_integer_array() -> None:
    assert Events.from_any(np.array([[0, 5]])).bounds.tolist() == [[0, 5]]


# ---------------------------------------------------------------------------
# Events: properties and protocol
# ---------------------------------------------------------------------------


def test_an_instantaneous_event_lasts_one_nanosecond() -> None:
    # Inclusive bounds measure the half-open span they occupy, so the single
    # nanosecond [10, 11) is one nanosecond long, not zero.
    events = Events.from_bounds([[10, 10]])
    assert events.durations.tolist() == [1]
    assert events.total_duration == 1


def test_total_duration_sums_every_event() -> None:
    events = Events.from_bounds([[0, 10], [100, 130]])
    assert events.total_duration == 42


def test_events_supports_len_and_iteration_over_nanosecond_pairs() -> None:
    events = Events.from_bounds([[0, 10], [100, 130]])
    assert len(events) == 2
    assert list(events) == [(0, 10), (100, 130)]


def test_the_repr_shows_the_first_three_events_and_counts_the_rest() -> None:
    assert repr(Events.empty()) == "Events([])"
    assert repr(Events.from_bounds([[0, 0]])) == "Events([1970-01-01T00:00:00])"
    assert repr(Events.from_bounds([[0, DAY]])) == (
        "Events([1970-01-01T00:00:00..1970-01-02T00:00:00])"
    )
    many = Events.from_bounds([[i * DAY, i * DAY] for i in range(5)])
    assert repr(many).endswith("... +2 more])")


def test_the_repr_keeps_sub_second_digits_that_matter() -> None:
    assert repr(Events.from_bounds([[1, 1]])) == (
        "Events([1970-01-01T00:00:00.000000001])"
    )


def test_equality_ignores_provenance() -> None:
    left = Events.from_bounds([[0, 10]])
    right = Events.from_bounds(
        [[0, 10]], origin=TimeSeries.from_any(hourly_labels([1.0])).origin
    )
    assert left == right
    assert hash(left) == hash(right)
    assert left != Events.from_bounds([[0, 11]])
    assert left != "not events"


# ---------------------------------------------------------------------------
# Events: set operations
# ---------------------------------------------------------------------------


def test_intersect_keeps_only_the_shared_span() -> None:
    left = Events.from_bounds([[0, 100], [200, 300]])
    right = Events.from_bounds([[50, 250]])
    assert left.intersect(right).bounds.tolist() == [[50, 100], [200, 250]]


def test_union_fuses_intervals_that_now_touch() -> None:
    left = Events.from_bounds([[0, 100], [200, 300]])
    right = Events.from_bounds([[101, 199]])
    assert left.union(right).bounds.tolist() == [[0, 300]]


def test_intersecting_with_an_empty_set_gives_nothing() -> None:
    assert Events.from_bounds([[0, 10]]).intersect(Events.empty()).n_events == 0


def test_uniting_with_an_empty_set_gives_the_other_set() -> None:
    events = Events.from_bounds([[0, 10]])
    assert events.union(Events.empty()) == events
    assert Events.empty().union(events) == events


def test_combining_two_empty_sets_gives_an_empty_set() -> None:
    assert Events.empty().union(Events.empty()).n_events == 0


def test_touching_intervals_intersect_at_a_single_instant() -> None:
    left = Events.from_bounds([[0, 10]])
    right = Events.from_bounds([[10, 20]])
    assert left.intersect(right).bounds.tolist() == [[10, 10]]


def test_set_operations_reject_a_plain_list() -> None:
    with pytest.raises(TypeError, match="Expected an Events"):
        Events.from_bounds([[0, 1]]).intersect([(0, 1)])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Events: egress
# ---------------------------------------------------------------------------


def test_to_list_gives_bare_timestamps_for_instantaneous_events() -> None:
    listed = Events.from_any([0, (10, 20)]).to_list(backend="pandas")
    assert listed[0] == pd.Timestamp(0)
    assert listed[1] == (pd.Timestamp(10), pd.Timestamp(20))


@pytest.mark.parametrize("backend", BACKENDS)
def test_to_frame_emits_start_and_end_on_every_backend(backend: str) -> None:
    frame = Events.from_bounds([[0, 10]]).to_frame(backend=backend)
    if backend == "pandas":
        assert list(frame.columns) == ["start", "end"]
    elif backend == "polars":
        assert frame.columns == ["start", "end"]
    else:
        assert frame.column_names == ["start", "end"]


def test_an_empty_events_still_produces_an_empty_frame_and_list() -> None:
    assert Events.empty().to_list() == []
    assert len(Events.empty().to_frame()) == 0


# ---------------------------------------------------------------------------
# to_events
# ---------------------------------------------------------------------------


def test_to_events_merges_consecutive_labels_into_one_period_event() -> None:
    events = to_events(hourly_labels([0.0, 1.0, 1.0, 0.0]))
    assert isinstance(events, Events)
    # The two labelled samples own 01:00-02:00 and 02:00-03:00, so together
    # they own everything up to one nanosecond before 03:00.
    assert events.bounds.tolist() == [
        [
            pd.Timestamp("2024-01-01 01:00").value,
            pd.Timestamp("2024-01-01 03:00").value - 1,
        ]
    ]


@pytest.mark.parametrize("n_samples", [1, 2, 3, 7])
def test_a_period_event_lasts_exactly_one_step_per_sample(n_samples: int) -> None:
    # The property the metrics rest on: a duration ratio and a count of samples
    # measure the same thing, with no nanosecond left over.
    events = to_events(hourly_labels([1.0] * n_samples + [0.0, 0.0]), as_periods=True)
    assert isinstance(events, Events)
    assert events.durations.tolist() == [n_samples * HOUR]


def test_to_events_treats_each_label_as_an_instant_when_asked() -> None:
    events = to_events(hourly_labels([0.0, 1.0, 1.0, 0.0]), as_periods=False)
    assert isinstance(events, Events)
    assert events.n_events == 2
    # Two instants, one nanosecond each.
    assert events.total_duration == 2


def test_to_events_falls_back_to_instants_on_an_irregular_axis() -> None:
    index = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-05"])
    events = to_events(pd.Series([1.0, 1.0, 0.0], index=index))
    assert isinstance(events, Events)
    assert events.n_events == 2
    assert events.total_duration == 2


def test_as_periods_true_is_the_default_on_a_regular_axis() -> None:
    labels = hourly_labels([0.0, 1.0, 1.0, 0.0])
    assert to_events(labels, as_periods=True) == to_events(labels)


def test_to_events_rejects_as_periods_on_an_irregular_axis() -> None:
    index = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-05"])
    with pytest.raises(ValueError, match="as_periods=True needs a sampling frequency"):
        to_events(pd.Series([1.0, 1.0, 0.0], index=index), as_periods=True)


def test_to_events_reads_nan_as_not_anomalous() -> None:
    events = to_events(hourly_labels([np.nan, 1.0, np.nan, np.nan]), as_periods=False)
    assert isinstance(events, Events)
    assert events.n_events == 1


def test_to_events_accepts_boolean_labels() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="h")
    events = to_events(pd.Series([False, True, True, False], index=index))
    assert isinstance(events, Events)
    assert events.durations.tolist() == [2 * HOUR]


def test_to_events_ignores_a_fractional_score() -> None:
    # Labels are 1.0 / 0.0; a value in between is not a label, so it is not
    # rounded into one.
    events = to_events(hourly_labels([0.0, 0.5, 0.0, 0.0]))
    assert isinstance(events, Events)
    assert events.n_events == 0


def test_to_events_clips_a_value_above_one_to_anomalous() -> None:
    events = to_events(hourly_labels([0.0, 3.0, 0.0, 0.0]), as_periods=False)
    assert isinstance(events, Events)
    assert events.n_events == 1


def test_to_events_returns_a_dict_for_a_multi_column_input() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="h")
    frame = pd.DataFrame({"a": [1.0, 0, 0, 0], "b": [0, 0, 1.0, 1.0]}, index=index)
    events = to_events(frame)
    assert isinstance(events, dict)
    assert set(events) == {"a", "b"}
    assert events["a"].n_events == 1
    assert events["b"].durations.tolist() == [2 * HOUR]


def test_to_events_returns_one_events_for_a_single_column_frame() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="h")
    assert isinstance(
        to_events(pd.DataFrame({"a": [1.0, 0, 0, 0]}, index=index)), Events
    )


def test_to_events_on_an_all_normal_series_gives_no_events() -> None:
    events = to_events(hourly_labels([0.0, 0.0, 0.0]))
    assert isinstance(events, Events)
    assert events.n_events == 0


def test_a_single_row_series_converts_without_error() -> None:
    events = to_events(hourly_labels([1.0]))
    assert isinstance(events, Events)
    # One sample cannot establish a frequency, so the label is an instant.
    assert events.bounds.tolist() == [
        [pd.Timestamp("2024-01-01").value, pd.Timestamp("2024-01-01").value]
    ]


def test_a_single_row_series_round_trips_back_to_labels() -> None:
    labels = hourly_labels([1.0])
    assert to_labels(to_events(labels), labels).tolist() == [1.0]


@pytest.mark.parametrize("backend", BACKENDS)
def test_to_events_gives_identical_bounds_on_every_backend(backend: str) -> None:
    native = make_native(backend, [0.0, 1.0, 1.0, 0.0])
    events = to_events(native)
    assert isinstance(events, Events)
    assert events.durations.tolist() == [2 * HOUR]


# ---------------------------------------------------------------------------
# to_labels
# ---------------------------------------------------------------------------


def test_events_survive_a_round_trip_through_period_labels() -> None:
    labels = hourly_labels([0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    events = to_events(labels)
    assert to_events(to_labels(events, labels)) == events


def test_events_survive_a_round_trip_through_instant_labels() -> None:
    labels = hourly_labels([0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    events = to_events(labels, as_periods=False)
    relabelled = to_labels(events, labels, as_periods=False)
    assert to_events(relabelled, as_periods=False) == events


def test_labels_survive_a_round_trip_through_events() -> None:
    flags = [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0]
    labels = hourly_labels(flags)
    assert to_labels(to_events(labels), labels).tolist() == flags


def test_to_labels_marks_the_expected_positions() -> None:
    labels = hourly_labels([0.0] * 5)
    events = Events.from_any([("2024-01-01T01:00", "2024-01-01T02:00")])
    assert to_labels(events, labels).tolist() == [0.0, 1.0, 1.0, 0.0, 0.0]


def test_to_labels_gives_one_column_per_dict_key() -> None:
    labels = hourly_labels([0.0] * 4)
    frame = to_labels(
        {"spike": ["2024-01-01T00:00"], "shift": ["2024-01-01T03:00"]}, labels
    )
    assert list(frame.columns) == ["spike", "shift"]
    assert frame["spike"].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert frame["shift"].tolist() == [0.0, 0.0, 0.0, 1.0]


def test_a_dict_of_events_round_trips_through_labels() -> None:
    index = pd.date_range("2024-01-01", periods=6, freq="h")
    frame = pd.DataFrame(
        {"a": [0.0, 1.0, 1.0, 0.0, 0.0, 0.0], "b": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]},
        index=index,
    )
    events = to_events(frame)
    assert isinstance(events, dict)
    assert to_events(to_labels(events, frame)) == events


def test_to_labels_accepts_a_nanosecond_array_as_its_time_axis() -> None:
    axis = np.arange(4, dtype=np.int64) * HOUR
    labelled = to_labels(Events.from_bounds([[HOUR, HOUR]]), axis, backend="polars")
    assert labelled["anomaly"].to_list() == [0.0, 1.0, 0.0, 0.0]


def test_to_labels_accepts_a_datetime64_array_as_its_time_axis() -> None:
    axis = np.array(["2024-01-01", "2024-01-02", "2024-01-03"], dtype="datetime64[s]")
    labelled = to_labels(Events.from_any(["2024-01-02"]), axis)
    assert labelled["anomaly"].tolist() == [0.0, 1.0, 0.0]


def test_to_labels_accepts_a_timeseries_as_its_time_axis() -> None:
    ts = TimeSeries.from_any(hourly_labels([0.0, 0.0, 0.0]))
    instant = Events.from_bounds([[ts.time[1], ts.time[1]]])
    assert to_labels(instant, ts).tolist() == [0.0, 1.0, 0.0]


def test_an_instant_event_does_not_survive_a_period_round_trip() -> None:
    # Under period semantics a labelled sample owns the whole period it opens,
    # so an instantaneous event is not expressible on the axis and comes back
    # widened. The round trip is the identity for events to_events produced.
    labels = hourly_labels([0.0, 0.0, 0.0])
    instant = Events.from_any(["2024-01-01T01:00"])
    widened = to_events(to_labels(instant, labels))
    assert isinstance(widened, Events)
    assert widened.durations.tolist() == [HOUR]
    assert to_events(
        to_labels(instant, labels, as_periods=False), as_periods=False
    ) == (instant)


def test_to_labels_rejects_a_float_time_axis() -> None:
    with pytest.raises(TypeError, match="int64 UTC nanoseconds or datetime64"):
        to_labels(Events.empty(), np.array([1.0, 2.0]))


def test_to_labels_rejects_an_empty_dict() -> None:
    with pytest.raises(ValueError, match="at least one key"):
        to_labels({}, hourly_labels([0.0, 0.0]))


def test_to_labels_rejects_as_periods_on_an_irregular_axis() -> None:
    index = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-05"])
    labels = pd.Series([0.0, 0.0, 0.0], index=index)
    with pytest.raises(ValueError, match="as_periods=True needs a sampling frequency"):
        to_labels(Events.empty(), labels, as_periods=True)


def test_to_labels_of_no_events_is_all_zeros() -> None:
    assert to_labels(Events.empty(), hourly_labels([0.0, 0.0])).tolist() == [0.0, 0.0]


@pytest.mark.parametrize("backend", BACKENDS)
def test_to_labels_returns_the_flavour_it_was_given(backend: str) -> None:
    native = make_native(backend, [0.0, 1.0, 1.0, 0.0])
    events = to_events(native)
    relabelled = to_labels(events, native)
    assert type(relabelled) is type(native)
    assert to_events(relabelled) == events


# ---------------------------------------------------------------------------
# expand_events
# ---------------------------------------------------------------------------


def test_expand_events_widens_by_a_duration_string() -> None:
    events = Events.from_any([("2024-01-01T06:00", "2024-01-01T07:00")])
    widened = expand_events(events, before="1h", after="30min")
    assert isinstance(widened, Events)
    assert widened.durations.tolist() == [int(2.5 * HOUR) + 1]


def test_a_bare_int_margin_is_nanoseconds() -> None:
    widened = expand_events(Events.from_bounds([[100, 200]]), before=10, after=10)
    assert isinstance(widened, Events)
    assert widened.bounds.tolist() == [[90, 210]]


def test_expand_events_accepts_a_timedelta() -> None:
    widened = expand_events(Events.from_bounds([[0, 0]]), after=timedelta(hours=1))
    assert isinstance(widened, Events)
    assert widened.bounds.tolist() == [[0, HOUR]]


def test_expand_events_accepts_a_numpy_timedelta() -> None:
    widened = expand_events(Events.from_bounds([[0, 0]]), after=np.timedelta64(5, "s"))
    assert isinstance(widened, Events)
    assert widened.bounds.tolist() == [[0, 5_000_000_000]]


def test_expanding_re_merges_events_that_now_overlap() -> None:
    widened = expand_events(Events.from_bounds([[0, 10], [30, 40]]), after=20)
    assert isinstance(widened, Events)
    assert widened.bounds.tolist() == [[0, 60]]


def test_expand_events_applies_the_same_margins_to_every_dict_key() -> None:
    widened = expand_events(
        {"a": Events.from_bounds([[100, 100]]), "b": Events.from_bounds([[500, 500]])},
        before=5,
        after=7,
    )
    assert isinstance(widened, dict)
    assert widened["a"].bounds.tolist() == [[95, 107]]
    assert widened["b"].bounds.tolist() == [[495, 507]]


def test_expand_events_accepts_a_plain_list() -> None:
    widened = expand_events([(0, 10)], after=5)
    assert isinstance(widened, Events)
    assert widened.bounds.tolist() == [[0, 15]]


def test_expanding_nothing_gives_nothing() -> None:
    widened = expand_events(Events.empty(), before="1h")
    assert isinstance(widened, Events)
    assert widened.n_events == 0


def test_expand_events_rejects_a_negative_margin() -> None:
    with pytest.raises(ValueError, match="is negative"):
        expand_events(Events.from_bounds([[0, 10]]), before=-5)


def test_expand_events_rejects_a_float_margin() -> None:
    # Silently truncating 1.5ns to 1ns would be worse than refusing.
    with pytest.raises(TypeError, match="must be a duration string"):
        expand_events(Events.from_bounds([[0, 10]]), after=1.5)  # type: ignore[arg-type]


def test_expand_events_rejects_a_bool_margin() -> None:
    with pytest.raises(TypeError, match="must be a duration, not a bool"):
        expand_events(Events.from_bounds([[0, 10]]), before=True)


def test_expand_events_rejects_a_calendar_unit() -> None:
    with pytest.raises(ValueError, match="calendar unit"):
        expand_events(Events.from_bounds([[0, 10]]), before="1M")


# ---------------------------------------------------------------------------
# validate_series
# ---------------------------------------------------------------------------


def test_validate_series_sorts_and_deduplicates() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-01", "2024-01-01"])
    out = validate_series(pd.Series([3.0, 1.0, 9.0], index=index))
    assert out.tolist() == [1.0, 3.0]
    assert out.index.tolist() == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
    ]


def test_validate_series_can_refuse_to_deduplicate() -> None:
    index = pd.to_datetime(["2024-01-01", "2024-01-01"])
    with pytest.raises(ValueError, match="duplicated timestamp"):
        validate_series(pd.Series([1.0, 2.0], index=index), drop_duplicates=False)


def test_validate_series_can_refuse_to_sort() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-01"])
    with pytest.raises(ValueError, match="not sorted"):
        validate_series(pd.Series([1.0, 2.0], index=index), sort=False)


@pytest.mark.parametrize("backend", BACKENDS)
def test_validate_series_returns_the_flavour_it_was_given(backend: str) -> None:
    native = make_native(backend, [1.0, 2.0, 3.0])
    assert type(validate_series(native)) is type(native)


def test_validate_series_rejects_a_frame_without_a_time_axis() -> None:
    with pytest.raises(TypeError, match="no temporal column"):
        validate_series(pl.DataFrame({"x": [1.0, 2.0]}))


# ---------------------------------------------------------------------------
# split_train_test
# ---------------------------------------------------------------------------


def daily(n_rows: int) -> pd.DataFrame:
    """A daily frame whose values are their own positions."""
    index = pd.date_range("2024-01-01", periods=n_rows, freq="D", name="time")
    return pd.DataFrame({"x": np.arange(float(n_rows))}, index=index)


def layout(data: pd.DataFrame, folds: list[tuple[Any, Any]]) -> list[str]:
    """Render each fold as a train/test/unused position string."""
    rendered = []
    for train, test in folds:
        row = ["0"] * len(data)
        for value in train["x"].to_numpy():
            row[int(value)] = "1"
        for value in test["x"].to_numpy():
            row[int(value)] = "2"
        rendered.append("".join(row))
    return rendered


def test_mode_1_cuts_equal_disjoint_folds() -> None:
    data = daily(40)
    assert layout(data, split_train_test(data, mode=1, n_splits=4)) == [
        "1111111222000000000000000000000000000000",
        "0000000000111111122200000000000000000000",
        "0000000000000000000011111112220000000000",
        "0000000000000000000000000000001111111222",
    ]


def test_mode_2_nests_folds_that_all_start_at_the_first_observation() -> None:
    data = daily(40)
    assert layout(data, split_train_test(data, mode=2, n_splits=4)) == [
        "1111111222000000000000000000000000000000",
        "1111111111111122222200000000000000000000",
        "1111111111111111111112222222220000000000",
        "1111111111111111111111111111222222222222",
    ]


def test_mode_3_appends_a_fixed_size_test_block_to_a_growing_train_window() -> None:
    data = daily(40)
    assert layout(data, split_train_test(data, mode=3, n_splits=4)) == [
        "1111111122222222000000000000000000000000",
        "1111111111111111222222220000000000000000",
        "1111111111111111111111112222222200000000",
        "1111111111111111111111111111111122222222",
    ]


def test_mode_4_tests_a_growing_train_window_against_the_whole_remainder() -> None:
    data = daily(40)
    assert layout(data, split_train_test(data, mode=4, n_splits=4)) == [
        "1111111122222222222222222222222222222222",
        "1111111111111111222222222222222222222222",
        "1111111111111111111111112222222222222222",
        "1111111111111111111111111111111122222222",
    ]


def test_the_default_split_is_one_fold_cut_at_the_train_ratio() -> None:
    folds = split_train_test(daily(10))
    assert len(folds) == 1
    assert (len(folds[0][0]), len(folds[0][1])) == (7, 3)


def test_train_ratio_moves_the_cut() -> None:
    folds = split_train_test(daily(10), train_ratio=0.9)
    assert (len(folds[0][0]), len(folds[0][1])) == (9, 1)


def test_modes_3_and_4_ignore_train_ratio() -> None:
    data = daily(20)
    for mode in (3, 4):
        lax = split_train_test(data, mode=mode, n_splits=2, train_ratio=0.2)
        strict = split_train_test(data, mode=mode, n_splits=2, train_ratio=0.9)
        assert layout(data, lax) == layout(data, strict)


def test_every_mode_1_fold_uses_each_observation_exactly_once() -> None:
    data = daily(23)
    seen = [
        value
        for train, test in split_train_test(data, mode=1, n_splits=3)
        for value in [*train["x"].to_numpy(), *test["x"].to_numpy()]
    ]
    assert sorted(seen) == list(np.arange(23.0))


def test_a_split_keeps_the_frequency_of_the_parent_series() -> None:
    train, _ = split_train_test(daily(10), mode=1)[0]
    assert TimeSeries.from_any(train).freq == DAY


@pytest.mark.parametrize("backend", BACKENDS)
def test_split_train_test_returns_the_flavour_it_was_given(backend: str) -> None:
    native = make_native(backend, np.arange(10.0))
    train, test = split_train_test(native, mode=1)[0]
    assert type(train) is type(native)
    assert type(test) is type(native)
    assert len(train) == 7
    assert len(test) == 3


def test_split_train_test_rejects_a_plain_list() -> None:
    with pytest.raises(TypeError, match="not a supported dataframe"):
        split_train_test([1.0, 2.0, 3.0])


def test_split_train_test_rejects_a_non_integer_mode() -> None:
    with pytest.raises(TypeError, match="mode must be an int"):
        split_train_test(daily(10), mode=1.0)  # type: ignore[arg-type]


def test_split_train_test_rejects_a_non_integer_n_splits() -> None:
    with pytest.raises(TypeError, match="n_splits must be an int"):
        split_train_test(daily(10), n_splits="2")  # type: ignore[arg-type]


def test_split_train_test_rejects_a_non_numeric_train_ratio() -> None:
    with pytest.raises(TypeError, match="train_ratio must be a number"):
        split_train_test(daily(10), train_ratio="0.7")  # type: ignore[arg-type]


def test_split_train_test_rejects_an_unknown_mode() -> None:
    with pytest.raises(ValueError, match=r"mode must be one of \[1, 2, 3, 4\]"):
        split_train_test(daily(10), mode=5)


def test_split_train_test_rejects_a_non_positive_n_splits() -> None:
    with pytest.raises(ValueError, match="n_splits must be at least 1"):
        split_train_test(daily(10), n_splits=0)


@pytest.mark.parametrize("ratio", [0.0, 1.0, -0.5, 1.5])
def test_split_train_test_rejects_a_train_ratio_outside_the_open_unit_interval(
    ratio: float,
) -> None:
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        split_train_test(daily(10), train_ratio=ratio)


def test_split_train_test_refuses_to_return_an_empty_side() -> None:
    with pytest.raises(ValueError, match="would be empty on one side"):
        split_train_test(daily(4), mode=1, n_splits=4)


def test_a_pyarrow_table_splits_into_tables() -> None:
    table = make_native("pyarrow", np.arange(12.0))
    folds = split_train_test(table, mode=3, n_splits=2)
    assert all(isinstance(part, pa.Table) for fold in folds for part in fold)
    assert [(len(a), len(b)) for a, b in folds] == [(4, 4), (8, 4)]
