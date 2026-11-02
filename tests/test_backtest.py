"""Held-out comparison must keep fitting, truth and notification state isolated."""

from __future__ import annotations

import json
import math
from typing import Any, ClassVar

import numpy as np
import pytest

from hazure import AlertPolicy, Threshold, TimeSeries, detectors
from hazure.evaluation import backtest
from hazure.events import Events
from hazure.scorers import DeviationScorer
from tests.helpers import BACKENDS, make_native


def series(values: list[float]) -> TimeSeries:
    return TimeSeries.from_arrays(np.arange(len(values)), values)


@pytest.mark.parametrize("backend", BACKENDS)
def test_compares_two_detectors_with_exact_held_out_predictions(backend: str) -> None:
    data = make_native(backend, [0] * 6 + [9, 0, 9, 0, 0, 0])
    truth = make_native(backend, [0] * 6 + [1, 0, 1, 0, 0, 0])
    result = backtest(
        {"finds": detectors.limits(high=5), "misses": detectors.limits(high=10)},
        data,
        truth,
        n_splits=1,
        events=False,
    )
    records = result.to_records()
    assert len(records) == 2
    assert records[0]["n_train"] == 6
    assert records[0]["n_test"] == 6
    assert records[0]["precision"] == records[0]["recall"] == records[0]["f1"] == 1
    assert records[1]["recall"] == 0
    assert records[0]["n_alerts"] == 2
    np.testing.assert_array_equal(
        result.folds[0].labels.values[:, 0], [1, 0, 1, 0, 0, 0]
    )


def test_fitted_templates_are_unchanged_and_each_fold_has_its_own_training() -> None:
    data = series([0, 1, 0, 1, 10, 0, 10, 0, 100, 0, 100, 0])
    template = detectors.iqr().fit(series([1000, 1100, 1200]))
    before = json.dumps(template.to_dict())
    result = backtest(template, data, n_splits=2)
    assert [fold.n_train for fold in result.folds] == [4, 8]
    assert json.dumps(template.to_dict()) == before
    assert result.folds[0].labels.time[0] == 4
    assert result.folds[1].labels.time[0] == 8


class RecordTraining(Threshold):
    seen: ClassVar[list[list[int]]] = []

    def _learn(self, ts: TimeSeries) -> None:
        self.seen.append(ts.time.tolist())

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        return ts.wrap(np.zeros(ts.n_rows))


def test_no_fold_trains_on_its_test_or_future_rows() -> None:
    RecordTraining.seen = []
    result = backtest(RecordTraining(), series(list(range(16))), n_splits=3)
    assert RecordTraining.seen == [list(range(4)), list(range(8)), list(range(12))]
    assert all(
        max(train) < fold.labels.time[0]
        for train, fold in zip(RecordTraining.seen, result.folds, strict=True)
    )


def test_interval_truth_is_clipped_to_each_test_block() -> None:
    result = backtest(
        detectors.limits(high=5),
        series([9] * 12),
        Events.from_any([(0, 11)]),
        n_splits=2,
    )
    assert [row["recall"] for row in result.to_records()] == [1, 1]
    assert [row["precision"] for row in result.to_records()] == [1, 1]


def test_label_truth_can_be_broadcast_across_columns() -> None:
    data = TimeSeries.from_arrays(
        np.arange(8),
        np.column_stack([[0] * 4 + [9, 0, 0, 0], [0] * 4 + [9, 0, 0, 0]]),
        ["a", "b"],
    )
    truth = series([0] * 4 + [1, 0, 0, 0])
    result = backtest(detectors.limits(high=5), data, truth, n_splits=1)
    assert {row["column"] for row in result.to_records()} == {"a", "b"}
    assert all(row["recall"] == 1 for row in result.to_records())


def test_truth_mapping_uses_each_columns_own_incidents() -> None:
    data = TimeSeries.from_arrays(
        np.arange(8),
        np.column_stack([[0] * 4 + [9, 0, 0, 0], [0] * 4 + [0, 9, 0, 0]]),
        ["a", "b"],
    )
    result = backtest(
        detectors.limits(high=5), data, {"a": [(4, 4)], "b": [(5, 5)]}, n_splits=1
    )
    assert all(row["f1"] == 1 for row in result.to_records())


def test_notification_policy_does_not_change_detection_metrics() -> None:
    data = series([0] * 6 + [9, 9, 0, 9, 0, 9])
    truth = series([0] * 6 + [1, 1, 0, 1, 0, 1])
    policy = AlertPolicy(min_consecutive=2)
    policy.update(-1, 1)
    stored = policy.to_dict()
    result = backtest(
        detectors.limits(high=5), data, truth, n_splits=1, alert_policy=policy
    )
    record = result.to_records()[0]
    assert record["n_alerts"] == 3
    assert record["n_notifications"] == 1
    assert record["f1"] == 1
    assert policy.to_dict() == stored
    assert result.folds[0].notifications is not None


def test_summary_means_defined_fold_metrics_and_sums_counts() -> None:
    result = backtest(
        detectors.limits(high=5), series([0] * 7 + [9, 0, 9, 0, 0]), n_splits=2
    )
    assert result.summary() == [
        {"model": "Detector", "column": "value", "n_folds": 2, "n_alerts": 2.0}
    ]
    empty_truth = series([0] * 12)
    empty = backtest(
        detectors.limits(high=100), series([0] * 12), empty_truth, n_splits=2
    )
    assert math.isnan(empty.summary()[0]["recall"])


def test_online_mode_primes_training_before_evaluating_new_observations() -> None:
    data = series([0] * 7 + [9, 0, 9, 0, 0])
    result = backtest(detectors.limits(high=5), data, n_splits=2, history=3)
    assert [row["n_alerts"] for row in result.to_records()] == [1, 1]


def test_online_mode_rejects_a_detector_that_reads_the_future() -> None:
    data = series([float(i % 7) for i in range(80)])
    with pytest.raises(ValueError, match=r"streamed|too short"):
        backtest(detectors.level_shift(window=3), data, n_splits=1, history=20)


@pytest.mark.parametrize(
    "models", [{}, {"score": DeviationScorer()}, {"bad": object()}, [], None]
)
def test_invalid_models_are_rejected(models: Any) -> None:
    with pytest.raises(TypeError):
        backtest(models, series([0] * 8))


def test_mismatched_truth_axis_is_rejected_before_fitting() -> None:
    RecordTraining.seen = []
    with pytest.raises(ValueError, match="time axis"):
        backtest(RecordTraining(), series([0] * 8), series([0] * 7), n_splits=1)
    assert RecordTraining.seen == []


def test_truth_column_mismatches_are_not_silently_aggregated() -> None:
    data = TimeSeries.from_arrays(np.arange(8), np.zeros((8, 2)), ["a", "b"])
    with pytest.raises(ValueError, match="columns"):
        backtest(detectors.limits(high=5), data, {"a": []}, n_splits=1)


def test_batch_predictions_keep_unknown_warmup() -> None:
    result = backtest(detectors.hampel(window=5), series(list(range(20))), n_splits=1)
    assert np.isnan(result.folds[0].labels.values[:4]).all()
