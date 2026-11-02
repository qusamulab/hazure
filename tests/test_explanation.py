"""Explanations inspect the fitted model's computations, not a surrogate."""

from __future__ import annotations

import json
from typing import Any, ClassVar

import numpy as np
import pytest

from hazure import (
    Detector,
    Graph,
    Node,
    Pipeline,
    Threshold,
    TimeSeries,
    detectors,
    explain,
)
from hazure.ensemble import OrAggregator
from hazure.scorers import AsScorer, DeviationScorer
from hazure.thresholds import FixedThreshold
from hazure.transformers import PcaReconstructionError, Retrospect
from tests.helpers import BACKENDS, make_native


def series(values: list[float]) -> TimeSeries:
    return TimeSeries.from_arrays(np.arange(len(values)), values)


@pytest.mark.parametrize("backend", BACKENDS)
def test_simple_detector_exposes_raw_values_and_the_exact_verdict(backend: str) -> None:
    source = make_native(backend, [0, 9, 0])
    model = detectors.limits(high=5)
    result = explain(model, source)
    expected = TimeSeries.from_any(model.detect(source))
    np.testing.assert_array_equal(result.output.values, expected.values)
    assert result.scores is not None
    assert result.at(int(expected.time[1]))["output"] == {"x": 1}
    assert result.at(int(expected.time[1]))["scores"] == {"x": 9}
    assert result.contributions is None


def test_fanout_uses_the_fitted_model_for_each_column() -> None:
    train = TimeSeries.from_arrays(
        np.arange(6),
        [[1, 100], [2, 200], [1, 100], [2, 200], [1, 100], [2, 200]],
        ["a", "b"],
    )
    model = Detector(DeviationScorer(), FixedThreshold(high=3)).fit(train)
    actual = train.wrap(
        np.array([[1, 100], [9, 100], [1, 900], [2, 200], [1, 100], [2, 200]])
    )
    detail = explain(model, actual)
    np.testing.assert_array_equal(detail.output.values, model.run(actual).values)
    assert set(detail.steps) == {"a", "b"}
    assert detail.scores is not None
    assert detail.scores.columns == ("a", "b")
    assert detail.at(1)["scores"]["a"] > detail.at(1)["scores"]["b"]


def test_graph_records_each_branch_and_the_aggregated_verdict() -> None:
    model = Graph(
        [
            Node("high", detectors.limits(high=5)),
            Node("low", detectors.limits(low=-5)),
            Node("either", OrAggregator(), inputs=("high", "low")),
        ]
    )
    data = series([0, 9, -9])
    result = explain(model, data)
    np.testing.assert_array_equal(result.output.values, model.run(data).values)
    row = result.at(1)
    assert row["steps"]["high"]["output"] == {"value": 1}
    assert row["steps"]["low"]["output"] == {"value": 0}
    assert row["output"] == {"anomaly": 1}


def test_pipeline_records_the_series_fed_to_each_successive_step() -> None:
    model = Pipeline([("lag", Retrospect(n_steps=2)), ("cut", FixedThreshold(high=5))])
    data = series([0, 9, 0, 0])
    result = explain(model, data)
    np.testing.assert_array_equal(result.output.values, model.run(data).values)
    assert set(result.steps) == {"lag", "cut"}
    assert result.steps["lag"].output.n_columns == 2


def test_pca_contributions_sum_to_the_score_of_the_same_fitted_model() -> None:
    base = np.arange(8.0)
    train = TimeSeries.from_arrays(
        np.arange(8), np.column_stack([base, base * 2, base * 3]), ["a", "b", "c"]
    )
    model = Detector(AsScorer(PcaReconstructionError(k=1)), FixedThreshold(high=1)).fit(
        train
    )
    values = train.values.copy()
    values[4, 1] += 20
    values[5, 0] = np.nan
    actual = train.wrap(values)
    stored = json.dumps(model.to_dict())
    result = explain(model, actual)
    assert result.contributions is not None
    assert result.scores is not None
    np.testing.assert_allclose(
        result.contributions.values.sum(axis=1),
        result.scores.values[:, 0],
        equal_nan=True,
    )
    np.testing.assert_array_equal(result.output.values, model.run(actual).values)
    assert result.output.columns == ("anomaly",)
    assert np.isnan(result.contributions.values[5]).all()
    assert result.at(4)["contributions"]["b"] > 0
    assert json.dumps(model.to_dict()) == stored


def test_pca_contributions_use_the_training_column_order() -> None:
    data = TimeSeries.from_arrays(
        np.arange(6), np.column_stack([np.arange(6.0), np.arange(6.0) * 2]), ["a", "b"]
    )
    model = detectors.pca(k=1).fit(data)
    result = explain(model, data.select(["b", "a"]))
    assert result.contributions is not None
    assert result.contributions.columns == ("a", "b")


def test_an_all_missing_as_scorer_model_does_not_trace_an_unfitted_transformer() -> (
    None
):
    data = series([np.nan] * 8)
    model = AsScorer(PcaReconstructionError()).fit(data)
    result = explain(model, data)
    assert np.isnan(result.output.values).all()
    assert result.contributions is None


class CountRuns(Threshold):
    trainable: ClassVar[bool] = False
    calls: int = 0

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        self.calls += 1
        return ts.wrap(np.zeros(ts.n_rows))


def test_tracing_does_not_execute_a_leaf_component_twice() -> None:
    leaf = CountRuns()
    explain(Pipeline([("leaf", leaf)]), series([0, 1]))
    assert leaf.calls == 1


def test_unknown_timestamps_are_not_replaced_by_a_neighbour() -> None:
    result = explain(detectors.limits(high=5), series([0, 9]))
    with pytest.raises(KeyError):
        result.at(2)


@pytest.mark.parametrize("model", [DeviationScorer(), object()])
def test_invalid_or_unfitted_models_fail_clearly(model: Any) -> None:
    with pytest.raises((RuntimeError, TypeError)):
        explain(model, series([0, 1, 0]))
