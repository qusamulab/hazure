"""Tests for combining components into pipelines and graphs.

Exercised with small stand-in components so a failure points at the wiring rather
than at an algorithm.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from hazure import Aggregator, Detector, Scorer, Threshold, TimeSeries, Transformer
from hazure.compose import SOURCE, Graph, Node, Pipeline
from hazure.transformers import double_rolling, rolling
from tests.helpers import BACKENDS, make_native

# -- stand-ins --------------------------------------------------------------


class Smooth(Transformer):
    """Rolling median, to stand in for a feature step."""

    trainable: ClassVar[bool] = False

    def __init__(self, window: int = 3) -> None:
        self.window = window

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        return ts.wrap(rolling(ts.values[:, 0], self.window, "median", min_periods=1))


class ShiftScore(Scorer):
    """Magnitude of the change between adjacent windows."""

    trainable: ClassVar[bool] = False

    def __init__(self, window: int = 3) -> None:
        self.window = window

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        return ts.wrap(
            double_rolling(ts.values[:, 0], self.window, "median", diff="l1")
        )


class Iqr(Threshold):
    """Upper inter-quartile fence."""

    def __init__(self, factor: float = 3.0) -> None:
        self.factor = factor

    def _learn(self, ts: TimeSeries) -> None:
        low, high = np.nanquantile(ts.values[:, 0], [0.25, 0.75])
        self.high_ = float(high + self.factor * (high - low))

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        return ts.wrap((ts.values[:, 0] > self.high_).astype(float))


class Flagger(Threshold):
    """A whole detection rule in one class, used through ``Detector(None, ...)``."""

    def __init__(self, window: int = 3, factor: float = 3.0) -> None:
        self.window = window
        self.factor = factor

    def _learn(self, ts: TimeSeries) -> None:
        self.inner_ = Iqr(self.factor).fit(ShiftScore(self.window).run(ts))

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        return self.inner_.run(ShiftScore(self.window).run(ts))


class AndAgg(Aggregator):
    """Anomalous only where every input agrees."""

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        return ts.wrap(np.nanmin(ts.values, axis=1), ["anomaly"])


class OrAgg(Aggregator):
    """Anomalous where any input says so."""

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        return ts.wrap(np.nanmax(ts.values, axis=1), ["anomaly"])


@pytest.fixture
def stepped() -> pd.Series:
    """A flat series that steps up part-way through, with one earlier blip."""
    index = pd.date_range("2024-01-01", periods=30, freq="h", name="time")
    values = np.zeros(30)
    values[15:] = 7.0
    values[5] = 4.0
    return pd.Series(values, index=index, name="sensor")


def detector_pipeline() -> Pipeline:
    """Return a three-step chain ending in binary labels."""
    return Pipeline(
        [("smooth", Smooth(3)), ("shift", ShiftScore(4)), ("cut", Iqr(3.0))]
    )


def agreeing_graph(aggregator: Aggregator) -> Graph:
    """Return a graph that scores the series twice and combines the verdicts."""
    return Graph(
        [
            Node("raw_score", ShiftScore(4)),
            Node("raw_flag", Iqr(3.0), inputs=("raw_score",)),
            Node("smooth", Smooth(5)),
            Node("smooth_score", ShiftScore(4), inputs=("smooth",)),
            Node("smooth_flag", Iqr(3.0), inputs=("smooth_score",)),
            Node("verdict", aggregator, inputs=("raw_flag", "smooth_flag")),
        ]
    )


# -- Pipeline ---------------------------------------------------------------


def test_a_pipeline_chains_its_steps(stepped: pd.Series) -> None:
    flags = detector_pipeline().fit_detect(stepped)
    assert set(flags.unique()) <= {0.0, 1.0}
    assert flags.loc[stepped.index[15]] == 1.0


def test_a_pipeline_returns_the_input_flavour(stepped: pd.Series) -> None:
    assert isinstance(detector_pipeline().fit_detect(stepped), pd.Series)


def test_a_pipeline_ending_in_a_threshold_is_a_detector() -> None:
    """Binary labels out means the caller should be able to say ``detect``."""
    assert detector_pipeline().output_kind == "labels"


def test_a_pipeline_ending_in_a_threshold_also_answers_to_apply(
    stepped: pd.Series,
) -> None:
    pipeline = detector_pipeline()
    np.testing.assert_array_equal(
        pipeline.fit_apply(stepped).to_numpy(),
        pipeline.detect(stepped).to_numpy(),
    )


def test_a_pipeline_ending_in_a_scorer_answers_to_score(stepped: pd.Series) -> None:
    pipeline = Pipeline([("smooth", Smooth(3)), ("shift", ShiftScore(4))])
    assert pipeline.output_kind == "score"
    assert pipeline.fit_score(stepped).nunique() > 2


def test_a_pipeline_ending_in_a_transformer_answers_to_transform(
    stepped: pd.Series,
) -> None:
    pipeline = Pipeline([("smooth", Smooth(3))])
    assert pipeline.output_kind == "series"
    assert len(pipeline.fit_transform(stepped)) == len(stepped)


def test_using_the_wrong_verb_says_which_one_to_use(stepped: pd.Series) -> None:
    with pytest.raises(TypeError, match=r"binary labels, so call detect\(\)"):
        detector_pipeline().fit_transform(stepped)


def test_the_wrong_verb_is_refused_for_a_scorer_terminal(stepped: pd.Series) -> None:
    pipeline = Pipeline([("shift", ShiftScore(4))])
    with pytest.raises(TypeError, match=r"continuous scores, so call score\(\)"):
        pipeline.fit_detect(stepped)


def test_a_pipeline_exposes_its_fitted_steps(stepped: pd.Series) -> None:
    pipeline = detector_pipeline().fit(stepped)
    cut = pipeline.named_steps()["cut"]
    assert isinstance(cut, Iqr)
    assert hasattr(cut, "high_")


def test_a_pipeline_summary_lists_steps_in_order() -> None:
    summary = detector_pipeline().summary()
    assert "1. smooth" in summary
    assert summary.index("smooth") < summary.index("shift") < summary.index("cut")


def test_an_empty_pipeline_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one step"):
        Pipeline([])


def test_repeated_step_names_are_rejected() -> None:
    with pytest.raises(ValueError, match=r"unique; repeated: \['x'\]"):
        Pipeline([("x", Smooth()), ("x", Smooth())])


def test_a_non_component_step_is_rejected() -> None:
    with pytest.raises(TypeError, match="not a hazure component"):
        Pipeline([("bad", "not a component")])  # type: ignore[list-item]


def test_an_aggregator_combines_the_columns_a_pipeline_step_produced(
    stepped: pd.Series,
) -> None:
    """A univariate step fans out over a frame, and an aggregator joins it back."""
    frame = pd.DataFrame({"a": stepped, "b": stepped[::-1].to_numpy()})
    per_column = Iqr().fit(frame).apply(frame)
    pipeline = Pipeline([("cut", Iqr()), ("combine", OrAgg())])
    combined = pipeline.fit_detect(frame)
    assert list(combined.columns) == ["anomaly"]
    np.testing.assert_array_equal(
        combined["anomaly"].to_numpy(), per_column.max(axis=1).to_numpy()
    )


def test_a_pipeline_must_be_fitted_first(stepped: pd.Series) -> None:
    with pytest.raises(RuntimeError, match="must be fitted"):
        detector_pipeline().detect(stepped)


# -- Graph ------------------------------------------------------------------


def test_a_graph_runs_branches_and_converges(stepped: pd.Series) -> None:
    flags = agreeing_graph(AndAgg()).fit_detect(stepped)
    assert flags.loc[stepped.index[15]] == 1.0


def test_requiring_agreement_is_stricter_than_accepting_either(
    stepped: pd.Series,
) -> None:
    """The point of a graph: two signals combined behave unlike either alone."""
    both = agreeing_graph(AndAgg()).fit_detect(stepped)
    either = agreeing_graph(OrAgg()).fit_detect(stepped)
    assert both.sum() < either.sum()
    assert set(both[both == 1].index) < set(either[either == 1].index)


def test_a_graph_orders_nodes_by_dependency() -> None:
    graph = Graph(
        [
            Node("last", Iqr(), inputs=("middle",)),
            Node("middle", ShiftScore(), inputs=("first",)),
            Node("first", Smooth()),
        ]
    )
    order = graph.summary().splitlines()[1:]
    assert [line.split()[0] for line in order] == ["first", "middle", "last"]


def test_a_graph_selects_columns_per_input() -> None:
    index = pd.date_range("2024-01-01", periods=12, freq="h", name="time")
    frame = pd.DataFrame(
        {"a": np.concatenate([np.zeros(6), np.full(6, 5.0)]), "b": np.arange(12.0)},
        index=index,
    )
    graph = Graph([Node("score", ShiftScore(3), inputs=(SOURCE,), columns=(["a"],))])
    result = graph.fit_score(frame)
    assert list(result.columns) == ["a"]


def test_a_graph_qualifies_column_names_when_joining_inputs(
    stepped: pd.Series,
) -> None:
    """Two detectors emitting the same column name must not collide."""
    graph = agreeing_graph(OrAgg())
    traced = graph.fit(stepped).trace(stepped)
    assert traced["verdict"].name == "anomaly"


def test_trace_returns_every_intermediate(stepped: pd.Series) -> None:
    traced = agreeing_graph(AndAgg()).fit(stepped).trace(stepped)
    assert set(traced) == {
        SOURCE,
        "raw_score",
        "raw_flag",
        "smooth",
        "smooth_score",
        "smooth_flag",
        "verdict",
    }
    assert len(traced["raw_score"]) == len(stepped)


def test_trace_requires_fitting(stepped: pd.Series) -> None:
    with pytest.raises(RuntimeError, match="Fit the Graph"):
        agreeing_graph(AndAgg()).trace(stepped)


def test_a_graph_renders_as_mermaid() -> None:
    mermaid = agreeing_graph(AndAgg()).to_mermaid()
    assert mermaid.startswith("flowchart LR")
    assert "raw_flag --> verdict" in mermaid
    assert "smooth_flag --> verdict" in mermaid
    assert "verdict --> output" in mermaid


def test_a_pipeline_renders_as_mermaid() -> None:
    mermaid = detector_pipeline().to_mermaid()
    assert "input --> smooth" in mermaid
    assert "smooth --> shift" in mermaid
    assert "shift --> cut" in mermaid


def test_a_graph_accepts_a_mapping_of_nodes() -> None:
    graph = Graph({"only": Node("only", Smooth())})
    assert graph.output_kind == "series"


def test_a_detector_node_makes_the_graph_a_detector(stepped: pd.Series) -> None:
    graph = Graph([Node("flag", Detector(None, Flagger(window=4)))])
    assert graph.output_kind == "labels"
    assert graph.fit_detect(stepped).loc[stepped.index[15]] == 1.0


def test_an_empty_graph_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one node"):
        Graph([])


def test_repeated_node_names_are_rejected() -> None:
    with pytest.raises(ValueError, match=r"unique; repeated: \['a'\]"):
        Graph([Node("a", Smooth()), Node("a", Smooth())])


def test_the_source_name_is_reserved() -> None:
    with pytest.raises(ValueError, match="reserved for the source data"):
        Graph([Node(SOURCE, Smooth())])


def test_an_unknown_input_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"input\(s\) \['nope'\]"):
        Graph([Node("a", Smooth(), inputs=("nope",))])


def test_a_cycle_is_reported_as_such() -> None:
    with pytest.raises(ValueError, match="form a cycle"):
        Graph(
            [
                Node("a", Smooth(), inputs=("b",)),
                Node("b", Smooth(), inputs=("a",)),
            ]
        )


def test_an_ambiguous_output_is_rejected() -> None:
    """Two unconsumed nodes give no single answer to return."""
    with pytest.raises(ValueError, match="exactly one output"):
        Graph([Node("a", Smooth()), Node("b", Smooth())])


def test_a_node_with_no_inputs_is_rejected() -> None:
    with pytest.raises(ValueError, match="no inputs"):
        Graph([Node("a", Smooth(), inputs=())])


def test_a_non_component_node_is_rejected() -> None:
    with pytest.raises(TypeError, match="not a hazure component"):
        Graph([Node("a", "nope")])  # type: ignore[arg-type]


def test_a_column_selection_must_match_the_input_count() -> None:
    with pytest.raises(ValueError, match="column selection"):
        Node("a", Smooth(), inputs=(SOURCE,), columns=(None, None))


# -- cloning ----------------------------------------------------------------


def test_cloning_a_pipeline_clones_its_steps(stepped: pd.Series) -> None:
    """Sharing a step between copies would leak fitted state between columns."""
    original = detector_pipeline().fit(stepped)
    copy = original.clone()

    assert copy.named_steps()["cut"] is not original.named_steps()["cut"]
    assert not copy.fitted
    assert original.fitted
    assert copy.named_steps()["shift"].window == 4  # type: ignore[union-attr]


def test_cloning_a_graph_clones_its_nodes(stepped: pd.Series) -> None:
    original = agreeing_graph(AndAgg()).fit(stepped)
    copy = original.clone()

    assert copy.named_nodes()["raw_flag"] is not original.named_nodes()["raw_flag"]
    assert not copy.fitted


def test_a_component_without_a_constructor_reports_no_parameters() -> None:
    """A component with nothing to configure need not define ``__init__``."""
    assert AndAgg().get_params() == {}
    assert repr(AndAgg()) == "AndAgg()"
    assert isinstance(AndAgg().clone(), AndAgg)


# -- composition and backends -----------------------------------------------


def test_a_pipeline_can_be_a_node_of_a_graph(stepped: pd.Series) -> None:
    inner = Pipeline([("smooth", Smooth(5)), ("shift", ShiftScore(4))])
    graph = Graph(
        [
            Node("branch", inner),
            Node("flag", Iqr(3.0), inputs=("branch",)),
        ]
    )
    assert graph.fit_detect(stepped).loc[stepped.index[15]] == 1.0


def test_a_multi_column_frame_passes_through_and_inner_steps_fan_out() -> None:
    """Each column gets its own fence, so wildly different scales both work.

    The series is long enough that the raised scores are a small tail. An
    inter-quartile fence widens with the fraction of the sample that is unusual,
    so on a short series where a third of the scores are raised it legitimately
    declines to flag any of them.
    """
    index = pd.date_range("2024-01-01", periods=40, freq="h", name="time")
    step = np.concatenate([np.zeros(20), np.full(20, 6.0)])
    frame = pd.DataFrame({"a": step, "b": step * 100}, index=index)

    flags = Pipeline([("shift", ShiftScore(4)), ("cut", Iqr(3.0))]).fit_detect(frame)

    assert list(flags.columns) == ["a", "b"]
    assert flags["a"].loc[index[20]] == 1.0
    assert flags["b"].loc[index[20]] == 1.0
    assert flags["a"].sum() < 10


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_pipeline_returns_the_backend_it_was_given(backend: str) -> None:
    step = np.concatenate([np.zeros(20), np.full(20, 6.0)])
    native = make_native(backend, step)
    assert type(detector_pipeline().fit_detect(native)) is type(native)


def test_every_backend_gives_the_same_verdict() -> None:
    step = np.concatenate([np.zeros(20), np.full(20, 6.0)])
    results = [
        TimeSeries.from_any(detector_pipeline().fit_detect(make_native(b, step)))
        for b in BACKENDS
    ]
    for other in results[1:]:
        np.testing.assert_array_equal(other.values, results[0].values)
