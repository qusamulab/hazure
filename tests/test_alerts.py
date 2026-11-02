"""Notification rules are causal and agree across batch, chunks and resume."""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

import numpy as np
import pytest

from hazure import AlertPolicy, Stream, TimeSeries, detectors
from tests.helpers import BACKENDS, make_native


def series(values: list[float], times: list[int] | None = None) -> TimeSeries:
    return TimeSeries.from_arrays(
        np.asarray(times) if times is not None else np.arange(len(values)), values
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_batch_suppresses_short_runs_and_keeps_the_backend(backend: str) -> None:
    source = make_native(backend, [1, 0, 1, 1, 1, 0, 1, 1])
    output = AlertPolicy(2).apply(source)
    assert type(output) is type(source)
    np.testing.assert_array_equal(
        TimeSeries.from_any(output).values[:, 0], [0, 0, 0, 1, 0, 0, 0, 1]
    )


def test_a_sustained_anomaly_never_renotifies_after_cooldown() -> None:
    output = AlertPolicy(cooldown=2).apply(series([1] * 20))
    np.testing.assert_array_equal(output.values[:, 0], [1] + [0] * 19)


def test_cooldown_waits_for_actual_elapsed_time_and_includes_the_boundary() -> None:
    output = AlertPolicy(2, cooldown=10).apply(
        series([1, 1, 0, 1, 1, 1, 1], [0, 1, 2, 3, 4, 10, 11])
    )
    np.testing.assert_array_equal(output.values[:, 0], [0, 1, 0, 0, 0, 0, 1])


def test_a_short_run_inside_cooldown_does_not_notify_later() -> None:
    output = AlertPolicy(cooldown=10).apply(series([1, 0, 1, 0, 0], [0, 1, 2, 3, 20]))
    np.testing.assert_array_equal(output.values[:, 0], [1, 0, 0, 0, 0])


def test_unknown_labels_stay_unknown_and_break_continuity() -> None:
    output = AlertPolicy(2).apply(series([1, np.nan, 1, 1, np.nan, 1, 1]))
    np.testing.assert_array_equal(output.values[:, 0], [0, np.nan, 0, 1, np.nan, 0, 1])


def test_columns_have_independent_runs_and_cooldowns() -> None:
    ts = TimeSeries.from_arrays(
        np.arange(6), [[1, 0], [1, 1], [0, 1], [1, 0], [1, 1], [1, 1]], ["a", "b"]
    )
    result = AlertPolicy(2, cooldown=4).apply(ts)
    np.testing.assert_array_equal(
        result.values, [[0, 0], [1, 0], [0, 1], [0, 0], [0, 0], [1, 0]]
    )


@pytest.mark.parametrize("cut", range(1, 9))
def test_chunked_processing_and_json_resume_match_one_batch(cut: int) -> None:
    values = [1, 1, 0, 1, np.nan, 1, 1, 1, 0, 1, 1]
    ts = series(values)
    policy = AlertPolicy(2, cooldown=5)
    expected = policy.apply(ts)
    first = policy.update_many(series(values[:cut]))
    resumed = AlertPolicy.from_dict(json.loads(json.dumps(policy.to_dict())))
    second = resumed.update_many(series(values[cut:], list(range(cut, len(values)))))
    np.testing.assert_array_equal(
        np.vstack([first.values, second.values]), expected.values
    )
    assert policy.clone().to_dict()["state"]["_last_time"] is None


def test_single_updates_match_batch_and_accept_reordered_mappings() -> None:
    policy = AlertPolicy(2)
    assert policy.update(0, {"a": 1, "b": 0}) == {"a": 0, "b": 0}
    assert policy.update(1, {"b": 1, "a": 1}) == {"a": 1, "b": 0}
    assert policy.update(2, {"a": 1, "b": 1}) == {"a": 0, "b": 1}


def test_apply_does_not_change_running_state_and_reset_clears_it() -> None:
    policy = AlertPolicy(2)
    assert policy.update(0, 1) == 0
    stored = policy.to_dict()
    policy.apply(series([0, 1, 1]))
    assert policy.to_dict() == stored
    assert policy.update(1, 1) == 1
    assert policy.reset().update(0, 1) == 0


@pytest.mark.parametrize(
    "cooldown", ["1s", timedelta(seconds=1), np.timedelta64(1, "s"), 1_000_000_000]
)
def test_duration_forms_survive_json(cooldown: Any) -> None:
    policy = AlertPolicy(cooldown=cooldown)
    assert policy.update(0, 1) == 1
    assert policy.update(1, 0) == 0
    policy = AlertPolicy.from_dict(json.loads(json.dumps(policy.to_dict())))
    assert policy.update(999_999_999, 1) == 0
    assert policy.update(1_000_000_000, 1) == 1


@pytest.mark.parametrize("value", [0, -1])
def test_nonpositive_runs_are_rejected(value: int) -> None:
    with pytest.raises(ValueError):
        AlertPolicy(value)


@pytest.mark.parametrize("value", [True, 1.2, "2"])
def test_noninteger_runs_are_rejected(value: Any) -> None:
    with pytest.raises(TypeError):
        AlertPolicy(value)


@pytest.mark.parametrize("value", [-1, timedelta(seconds=-1), np.timedelta64("NaT")])
def test_negative_or_missing_cooldown_is_rejected(value: Any) -> None:
    with pytest.raises(ValueError):
        AlertPolicy(cooldown=value)


@pytest.mark.parametrize("value", [True, 1.5, None])
def test_invalid_cooldown_type_is_rejected(value: Any) -> None:
    with pytest.raises(TypeError):
        AlertPolicy(cooldown=value)


def test_invalid_timestamp_and_schema_do_not_advance_state() -> None:
    policy = AlertPolicy(2)
    policy.update(1, {"x": 1})
    stored = policy.to_dict()
    for stamp, row in [
        (1, {"x": 1}),
        (2, {"y": 1}),
        (2, {}),
        (np.datetime64("NaT"), {"x": 1}),
    ]:
        with pytest.raises(ValueError):
            policy.update(stamp, row)
        assert policy.to_dict() == stored
    assert policy.update(2, {"x": 1}) == {"x": 1}


def test_batch_validation_is_atomic_and_does_not_reorder_arrivals() -> None:
    policy = AlertPolicy()
    policy.update(2, 1)
    stored = policy.to_dict()
    with pytest.raises(ValueError):
        policy.update_many(series([1, 1], [1, 3]))
    assert policy.to_dict() == stored
    unsorted = make_native("pandas", [1, 0, 1]).iloc[[1, 0, 2]]
    with pytest.raises(ValueError, match="not sorted"):
        AlertPolicy().update_many(unsorted)


def test_empty_batch_preserves_state_and_columns() -> None:
    policy = AlertPolicy()
    result = policy.update_many(series([]))
    assert result.values.shape == (0, 1)
    assert policy.to_dict()["state"]["_columns"] is None


@pytest.mark.parametrize("cut", [2, 5, 8])
def test_stream_and_policy_resume_preserve_notification_history(cut: int) -> None:
    training = series([0] * 4)
    data = series([0] * 4 + [9, 9, 9, 0, 9, 9, 0, 9, 9, 9])
    detector = detectors.limits(high=5).fit(training)
    expected = AlertPolicy(2, cooldown=5).apply(detector.detect(data)).values[4:, 0]
    stream = Stream(detector, history=2).prime(training)
    policy = AlertPolicy(2, cooldown=5)
    decisions = []
    for i in range(4, data.n_rows):
        if i == cut + 4:
            stream = Stream.from_dict(json.loads(json.dumps(stream.to_dict())))
            policy = AlertPolicy.from_dict(json.loads(json.dumps(policy.to_dict())))
        label = stream.update(i, float(data.values[i, 0]))
        decisions.append(policy.update(i, label))
    np.testing.assert_array_equal(decisions, expected)
