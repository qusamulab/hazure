"""Tests for the component hierarchy and parameter introspection.

Exercised through small concrete components rather than the real algorithms, so
a failure here points at the base machinery instead of at an algorithm.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest

from hazure import (
    Aggregator,
    Component,
    Detector,
    Graph,
    Node,
    Pipeline,
    Scorer,
    Threshold,
    TimeSeries,
    Transformer,
    detectors,
)
from hazure._core import Configurable
from tests.helpers import BACKENDS, make_native

# -- test doubles -----------------------------------------------------------


class Doubler(Transformer):
    """Multiplies by a factor. Untrainable, so usable straight away."""

    trainable: ClassVar[bool] = False

    def __init__(self, factor: float = 2.0) -> None:
        self.factor = factor

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        return ts.wrap(ts.values * self.factor)


class DeviationScorer(Scorer):
    """Scores each point by its distance from the training mean."""

    def __init__(self, absolute: bool = True) -> None:
        self.absolute = absolute

    def _learn(self, ts: TimeSeries) -> None:
        self.centre_ = float(np.nanmean(ts.values))

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        deviation = ts.values[:, 0] - self.centre_
        return ts.wrap(np.abs(deviation) if self.absolute else deviation)


class OverMean(Threshold):
    """Flags scores above the training mean times a factor."""

    def __init__(self, factor: float = 1.0) -> None:
        self.factor = factor

    def _learn(self, ts: TimeSeries) -> None:
        self.cutoff_ = float(np.nanmean(ts.values)) * self.factor

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        return ts.wrap((ts.values[:, 0] > self.cutoff_).astype(float))


class Widener(Transformer):
    """Turns one column into two, to exercise output naming under fan-out."""

    trainable: ClassVar[bool] = False

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        stacked = np.column_stack([ts.values[:, 0], ts.values[:, 0] * -1.0])
        return ts.wrap(stacked, ["up", "down"])


class Total(Transformer):
    """Sums every column, so it genuinely needs them all at once."""

    multivariate: ClassVar[bool] = True
    trainable: ClassVar[bool] = False

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        return ts.wrap(np.nansum(ts.values, axis=1), ["total"])


class AnyOf(Aggregator):
    """Flags a point when any input flagged it."""

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        return ts.wrap(np.nanmax(ts.values, axis=1), ["anomaly"])


@pytest.fixture
def stepped() -> pd.Series:
    """A quiet series with a clear step change part-way through."""
    index = pd.date_range("2024-01-01", periods=20, freq="h", name="time")
    values = np.concatenate([np.zeros(15), np.full(5, 10.0)])
    return pd.Series(values, index=index, name="sensor")


# -- the layering -----------------------------------------------------------


def test_a_scorer_returns_continuous_values() -> None:
    """A scorer is usable on its own, with no threshold in sight."""
    index = pd.date_range("2024-01-01", periods=20, freq="h", name="time")
    values = np.arange(20.0)
    values[7] = 500.0
    noisy = pd.Series(values, index=index, name="sensor")

    scores = DeviationScorer().fit_score(noisy)

    assert scores.nunique() > 2
    assert scores.idxmax() == index[7]


def test_a_threshold_can_be_swapped_without_touching_the_scorer(
    stepped: pd.Series,
) -> None:
    scorer = DeviationScorer().fit(stepped)
    scores = scorer.run(TimeSeries.from_any(stepped)).to_native()

    lenient = OverMean(factor=3.0).fit_apply(scores)
    strict = OverMean(factor=0.5).fit_apply(scores)

    assert strict.sum() > lenient.sum()


def test_a_detector_pairs_a_scorer_with_a_threshold(stepped: pd.Series) -> None:
    flags = Detector(DeviationScorer(), OverMean()).fit_detect(stepped)
    assert set(flags.unique()) <= {0.0, 1.0}
    assert flags.loc[stepped.index[15:]].eq(1.0).all()
    assert flags.loc[stepped.index[:15]].eq(0.0).all()


def test_an_aggregator_reduces_several_label_series() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="h", name="time")
    left = pd.Series([1.0, 0.0, 0.0, 0.0], index=index, name="flag")
    right = pd.Series([0.0, 0.0, 1.0, 0.0], index=index, name="flag")

    combined = AnyOf().aggregate(left, right)

    assert list(combined) == [1.0, 0.0, 1.0, 0.0]
    assert combined.name == "anomaly"


def test_an_aggregator_accepts_a_single_frame() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="h", name="time")
    frame = pd.DataFrame({"a": [1.0, 0.0, 0.0], "b": [0.0, 0.0, 1.0]}, index=index)
    assert list(AnyOf().aggregate(frame)["anomaly"]) == [1.0, 0.0, 1.0]


def test_an_aggregator_aligns_disjoint_time_axes() -> None:
    """Alignment is an explicit join on the time axis, not an implicit one."""
    left = pd.Series(
        [1.0, 0.0],
        index=pd.DatetimeIndex(["2024-01-01 00:00", "2024-01-01 01:00"], name="time"),
        name="flag",
    )
    right = pd.Series(
        [0.0, 1.0],
        index=pd.DatetimeIndex(["2024-01-01 01:00", "2024-01-01 02:00"], name="time"),
        name="flag",
    )
    combined = AnyOf().aggregate(left, right)
    assert len(combined) == 3
    assert list(combined) == [1.0, 0.0, 1.0]


def test_an_aggregator_needs_more_than_one_series() -> None:
    single = pd.Series(
        [1.0], index=pd.date_range("2024-01-01", periods=1, name="time"), name="flag"
    )
    with pytest.raises(ValueError, match="needs several label series"):
        AnyOf().aggregate(single)


def test_an_aggregator_needs_at_least_one_argument() -> None:
    with pytest.raises(ValueError, match="at least one label series"):
        AnyOf().aggregate()


def test_an_aggregator_rejects_a_name_count_mismatch() -> None:
    index = pd.date_range("2024-01-01", periods=2, freq="h", name="time")
    part = pd.Series([1.0, 0.0], index=index, name="flag")
    with pytest.raises(ValueError, match="names for 2 label series"):
        AnyOf().aggregate(part, part, names=["only_one"])


# -- fan-out across columns -------------------------------------------------


def test_a_univariate_component_fans_out_over_a_frame() -> None:
    index = pd.date_range("2024-01-01", periods=10, freq="h", name="time")
    frame = pd.DataFrame(
        {"small": np.arange(10.0), "large": np.arange(10.0) * 1000},
        index=index,
    )
    scores = DeviationScorer().fit_score(frame)
    assert list(scores.columns) == ["small", "large"]
    # Each column learned its own centre, so both peak at their own extremes.
    assert scores["large"].max() > scores["small"].max() * 100


def test_fan_out_fits_each_column_independently() -> None:
    index = pd.date_range("2024-01-01", periods=6, freq="h", name="time")
    frame = pd.DataFrame({"a": np.zeros(6), "b": np.full(6, 100.0)}, index=index)
    scorer = DeviationScorer().fit(frame)

    assert scorer._column_models is not None
    centres = {n: m.centre_ for n, m in scorer._column_models.items()}  # type: ignore[attr-defined]
    assert centres == {"a": 0.0, "b": 100.0}


def test_a_widening_component_qualifies_its_output_names() -> None:
    """Without prefixing, two columns of ``up``/``down`` would collide."""
    index = pd.date_range("2024-01-01", periods=4, freq="h", name="time")
    frame = pd.DataFrame({"a": np.arange(4.0), "b": np.arange(4.0)}, index=index)
    result = Widener().transform(frame)
    assert list(result.columns) == ["a_up", "a_down", "b_up", "b_down"]


def test_a_widening_component_keeps_plain_names_for_one_column() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="h", name="time")
    result = Widener().transform(pd.Series(np.arange(4.0), index=index, name="a"))
    assert list(result.columns) == ["up", "down"]


def test_a_multivariate_component_sees_every_column_at_once() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="h", name="time")
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]}, index=index)
    assert list(Total().transform(frame)["total"]) == [11.0, 22.0, 33.0]


def test_a_multivariate_component_reorders_columns_to_the_training_layout() -> None:
    index = pd.date_range("2024-01-01", periods=2, freq="h", name="time")
    trained = Total().fit(pd.DataFrame({"a": [1.0, 1.0], "b": [2.0, 2.0]}, index=index))
    swapped = pd.DataFrame({"b": [2.0, 2.0], "a": [1.0, 1.0]}, index=index)
    assert trained.feature_names == ("a", "b")
    assert list(trained.transform(swapped)["total"]) == [3.0, 3.0]


def test_a_multivariate_component_rejects_a_missing_column() -> None:
    index = pd.date_range("2024-01-01", periods=2, freq="h", name="time")
    trained = Total().fit(pd.DataFrame({"a": [1.0, 1.0], "b": [2.0, 2.0]}, index=index))
    with pytest.raises(ValueError, match=r"missing \['b'\]"):
        trained.transform(pd.DataFrame({"a": [1.0, 1.0]}, index=index))


def test_a_fanned_out_component_rejects_an_unseen_column() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="h", name="time")
    trained = DeviationScorer().fit(
        pd.DataFrame({"a": np.arange(4.0), "b": np.arange(4.0)}, index=index)
    )
    with pytest.raises(ValueError, match=r"nothing trained for \['c'\]"):
        trained.score(pd.DataFrame({"c": np.arange(4.0)}, index=index))


# -- fitting contract -------------------------------------------------------


def test_a_trainable_component_must_be_fitted_first(stepped: pd.Series) -> None:
    with pytest.raises(RuntimeError, match="must be fitted"):
        DeviationScorer().score(stepped)


def test_the_error_names_the_shortcut_method(stepped: pd.Series) -> None:
    with pytest.raises(RuntimeError, match="fit_detect"):
        Detector(DeviationScorer(), OverMean()).detect(stepped)


def test_an_untrainable_component_works_without_fitting(stepped: pd.Series) -> None:
    assert Doubler().fitted
    assert list(Doubler(factor=3.0).transform(stepped)[:3]) == [0.0, 0.0, 0.0]


def test_fit_returns_self_for_chaining(stepped: pd.Series) -> None:
    scorer = DeviationScorer()
    assert scorer.fit(stepped) is scorer


def test_feature_names_records_what_training_saw() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="h", name="time")
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]}, index=index)
    assert DeviationScorer().fit(frame).feature_names == ("x", "y")


def test_feature_names_is_none_before_fitting() -> None:
    assert DeviationScorer().feature_names is None


def test_refitting_replaces_the_previous_state() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="h", name="time")
    scorer = DeviationScorer()
    scorer.fit(pd.DataFrame({"a": np.zeros(4)}, index=index))
    scorer.fit(pd.DataFrame({"b": np.zeros(4)}, index=index))
    assert scorer.feature_names == ("b",)
    assert scorer.score(pd.DataFrame({"b": np.zeros(4)}, index=index)) is not None


# -- backend independence ---------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_component_returns_the_backend_it_was_given(backend: str) -> None:
    native = make_native(backend, np.arange(10.0))
    result = DeviationScorer().fit_score(native)
    assert type(result) is type(native)


def test_every_backend_produces_the_same_scores() -> None:
    values = np.concatenate([np.zeros(8), np.full(4, 5.0)])
    results = [
        TimeSeries.from_any(DeviationScorer().fit_score(make_native(b, values)))
        for b in BACKENDS
    ]
    for other in results[1:]:
        np.testing.assert_allclose(other.values, results[0].values)


# -- parameter introspection ------------------------------------------------


def test_get_params_reads_the_constructor_signature() -> None:
    """No hand-maintained list, so it cannot drift from ``__init__``."""
    assert Doubler(factor=5.0).get_params() == {"factor": 5.0}


def test_set_params_updates_in_place_and_chains() -> None:
    component = Doubler()
    assert component.set_params(factor=7.0) is component
    assert component.factor == 7.0


def test_set_params_rejects_an_unknown_name() -> None:
    with pytest.raises(KeyError, match="not parameters of Doubler"):
        Doubler().set_params(multiplier=2.0)


def test_clone_preserves_every_parameter_including_nested_ones() -> None:
    """Every setting survives a clone, including those of nested components."""
    original = Detector(DeviationScorer(absolute=False), OverMean(factor=2.5))
    copy = original.clone()

    assert copy.scorer.absolute is False
    assert copy.threshold.factor == 2.5
    assert repr(copy) == repr(original)


def test_clone_produces_an_unfitted_copy(stepped: pd.Series) -> None:
    original = DeviationScorer().fit(stepped)
    copy = original.clone()
    assert original.fitted
    assert not copy.fitted
    assert copy is not original


def test_clone_does_not_share_nested_components() -> None:
    original = Detector(DeviationScorer(), OverMean())
    copy = original.clone()
    assert copy.scorer is not original.scorer
    assert copy.threshold is not original.threshold


def test_repr_omits_defaults() -> None:
    assert repr(Doubler()) == "Doubler()"
    assert repr(Doubler(factor=3.0)) == "Doubler(factor=3.0)"


def test_repr_shows_required_parameters() -> None:
    rendered = repr(Detector(DeviationScorer(), OverMean()))
    assert rendered.startswith("Detector(scorer=DeviationScorer()")


def test_a_star_args_constructor_is_rejected() -> None:
    """Positional varargs cannot be named, so they cannot be round-tripped."""

    class Sloppy(Configurable):
        def __init__(self, *args: Any) -> None:
            self.args = args

    with pytest.raises(TypeError, match=r"takes \*args"):
        Sloppy(1, 2).get_params()


def test_keyword_varargs_are_ignored_rather_than_rejected() -> None:
    class Flexible(Configurable):
        def __init__(self, size: int = 1, **extra: Any) -> None:
            self.size = size
            self.extra = extra

    assert Flexible(size=4, colour="red").get_params() == {"size": 4}


# -- serialisation ----------------------------------------------------------


def _fittable() -> list[tuple[str, Any]]:
    """Real components with fitted state worth round-tripping, one per shape.

    Built here rather than at import time so a missing optional extra cannot
    stop this module from being collected.
    """

    from hazure.scorers import DeviationScorer as RealDeviationScorer
    from hazure.thresholds import EsdThreshold, IqrThreshold, MadThreshold
    from hazure.transformers import SeasonalDecomposition, StandardScale

    return [
        ("iqr", detectors.iqr()),
        ("esd", detectors.esd()),
        ("spike", detectors.spike(window=12)),
        ("level shift", detectors.level_shift(window=12)),
        ("seasonal", detectors.seasonal(period=24)),
        ("autoregression", detectors.autoregression(n_steps=2)),
        ("deviation", RealDeviationScorer(center="mean", scale="mad")),
        ("decomposition", SeasonalDecomposition(period=24, trend=True)),
        ("standard scale", StandardScale()),
        ("iqr threshold", IqrThreshold(factor=(None, 2.0))),
        ("mad threshold", MadThreshold()),
        ("esd threshold", EsdThreshold(alpha=0.01)),
        ("hampel", detectors.hampel(window=11)),
        ("pelt", detectors.pelt()),
        ("spectral residual", detectors.spectral_residual()),
        ("pca", detectors.pca(k=1)),
        ("regression", detectors.regression(target="b")),
    ]


def _seasonal_frame(columns: int = 1) -> TimeSeries:
    """A regular hourly frame with a daily cycle and one planted spike."""
    rng = np.random.default_rng(0)
    index = pd.date_range("2024-01-01", periods=240, freq="h", name="time")
    cycle = 10.0 + 3.0 * np.sin(2 * np.pi * np.arange(240) / 24)
    data = {}
    for position in range(columns):
        values = cycle * (position + 1) + rng.normal(scale=0.2, size=240)
        values[150] += 12.0
        data[chr(ord("a") + position)] = values
    return TimeSeries.from_any(pd.DataFrame(data, index=index))


@pytest.mark.parametrize(("name", "component"), _fittable())
def test_a_fitted_component_survives_a_round_trip_through_json(
    name: str, component: Any
) -> None:
    import json

    # Multivariate components need two columns to have a relationship to model.
    ts = _seasonal_frame(2 if component.is_multivariate else 1)
    fitted = component.fit(ts)
    expected = fitted.run(ts)

    restored = type(component).from_dict(json.loads(json.dumps(fitted.to_dict())))
    actual = restored.run(ts)

    assert repr(restored) == repr(fitted), name
    assert actual.columns == expected.columns, name
    assert np.array_equal(actual.values, expected.values, equal_nan=True), name


def test_a_restored_component_needs_no_second_fit() -> None:
    import json

    ts = _seasonal_frame()
    fitted = detectors.seasonal(period=24).fit(ts)
    restored = Detector.from_dict(json.loads(json.dumps(fitted.to_dict())))
    # run() raises on an unfitted trainable component, so this passing is the
    # assertion: the fitted state came back, not just the parameters.
    assert restored.run(ts).n_rows == ts.n_rows


def test_an_unfitted_component_carries_only_its_parameters() -> None:
    stored = Doubler(factor=3.0).to_dict()
    assert stored["state"] == {"factor": 3.0}


def test_serialising_keeps_a_window_pair_a_pair() -> None:
    restored = Detector.from_dict(detectors.level_shift(window=(6, 12)).to_dict())
    assert restored.scorer.transformer.window == (6, 12)


def test_serialising_keeps_an_arrays_dtype() -> None:
    from hazure.scorers import PeltScorer

    stored = PeltScorer().fit(_seasonal_frame()).to_dict()
    # int64 nanosecond timestamps read back as float64 would lose their last
    # digits, so the dtype travels with the values.
    assert PeltScorer.from_dict(stored).breakpoints_.dtype == np.int64


def test_a_model_hazure_did_not_build_cannot_be_serialised() -> None:
    from hazure.scorers import OutlierScorer

    class Rejector:
        def fit_predict(self, X: Any) -> Any:
            return np.full(X.shape[0], -1)

    scorer = OutlierScorer(model=Rejector())
    with pytest.raises(TypeError, match="pickle"):
        scorer.to_dict()


def test_a_payload_may_not_name_a_class_outside_hazure() -> None:
    payload = {"type": "builtins.dict", "state": {}}
    with pytest.raises(ValueError, match="outside hazure"):
        Component.from_dict(payload)


def test_a_payload_may_not_name_something_that_is_not_a_component() -> None:
    payload = {"type": "hazure.TimeSeries", "state": {}}
    with pytest.raises(TypeError, match="not a hazure component"):
        Component.from_dict(payload)


def test_a_payload_may_not_name_a_class_that_no_longer_exists() -> None:
    payload = {"type": "hazure.detection.spike.RemovedDetector", "state": {}}
    with pytest.raises(ValueError, match="does not exist"):
        Component.from_dict(payload)


def test_from_dict_refuses_to_hand_back_a_different_component() -> None:
    from hazure.thresholds import IqrThreshold, MadThreshold

    with pytest.raises(TypeError, match="is not a MadThreshold"):
        MadThreshold.from_dict(IqrThreshold().to_dict())


def test_a_reserved_key_in_a_mapping_parameter_is_refused() -> None:
    from hazure.transformers import RollingAggregate

    component = RollingAggregate(
        window=3, agg="quantile", agg_params={"q": 0.5, "__tuple__": 1}
    )
    with pytest.raises(TypeError, match="reserved key"):
        component.to_dict()


def test_a_component_of_your_own_round_trips_when_you_name_the_class() -> None:
    # Deserialising has to import the class a payload names, and an import runs
    # code, so a bare Component.from_dict stays inside hazure. Naming the
    # class yourself is proof it is already imported, so this is allowed.
    stored = Doubler(factor=3.0).to_dict()
    assert Doubler.from_dict(stored).factor == 3.0
    with pytest.raises(ValueError, match="outside hazure"):
        Component.from_dict(stored)


# -- nesting ----------------------------------------------------------------


def test_nested_parameters_are_reported_by_name() -> None:
    detector = Detector(DeviationScorer(absolute=False), OverMean(factor=2.5))
    params = detector.get_params()
    assert params["scorer__absolute"] is False
    assert params["threshold__factor"] == 2.5
    assert set(detector.get_params(deep=False)) == {"scorer", "threshold"}


def test_a_nested_parameter_is_set_through_its_part(stepped: pd.Series) -> None:
    detector = Detector(DeviationScorer(), OverMean())
    assert detector.fit_detect(stepped).sum() > 0
    detector.set_params(threshold__factor=100.0)
    assert detector.threshold.factor == 100.0
    assert detector.fit_detect(stepped).sum() == 0


def test_replacing_a_part_and_configuring_it_in_one_call_configures_the_new_part() -> (
    None
):
    detector = Detector(DeviationScorer(), OverMean())
    replacement = OverMean()
    detector.set_params(threshold=replacement, threshold__factor=3.0)
    assert detector.threshold is replacement
    assert replacement.factor == 3.0


def test_an_unknown_nested_part_is_refused() -> None:
    with pytest.raises(KeyError, match="reaches into"):
        Detector(DeviationScorer(), OverMean()).set_params(scorre__absolute=True)


def test_pipeline_steps_are_parts_named_by_their_step() -> None:
    pipeline = Pipeline([("double", Doubler()), ("score", DeviationScorer())])
    assert pipeline.get_params()["double__factor"] == 2.0
    pipeline.set_params(double__factor=5.0)
    assert pipeline.named_steps()["double"].factor == 5.0


def test_graph_nodes_are_parts_named_by_their_node() -> None:
    graph = Graph([Node("double", Doubler())])
    graph.set_params(double__factor=5.0)
    assert graph.named_nodes()["double"].factor == 5.0


# -- the detector ------------------------------------------------------------


def test_a_detector_refuses_a_scorer_that_does_not_score() -> None:
    with pytest.raises(TypeError, match="AsScorer"):
        Detector(Doubler(), OverMean())  # type: ignore[arg-type]


def test_a_detector_refuses_a_threshold_that_does_not_label() -> None:
    with pytest.raises(TypeError, match="must produce labels"):
        Detector(DeviationScorer(), DeviationScorer())


def test_a_detector_of_untrainable_parts_needs_no_fit(stepped: pd.Series) -> None:
    from hazure.thresholds import FixedThreshold

    detector = Detector(None, FixedThreshold(high=5.0))
    assert not detector.is_trainable
    assert detector.detect(stepped).sum() == 5.0


def test_a_detector_is_as_multivariate_as_its_scorer() -> None:
    assert not Detector(DeviationScorer(), OverMean()).is_multivariate
    assert detectors.pca().is_multivariate


def test_a_composite_of_untrainable_parts_needs_no_fit(stepped: pd.Series) -> None:
    pipeline = Pipeline([("double", Doubler()), ("again", Doubler(factor=3.0))])
    assert not pipeline.is_trainable
    assert pipeline.transform(stepped).max() == 60.0


# -- storage conventions ----------------------------------------------------


class Cached(Transformer):
    """Holds a cache, which is not state, and a declared private anchor, which is."""

    trainable: ClassVar[bool] = True
    _persisted: ClassVar[tuple[str, ...]] = (*Transformer._persisted, "_anchor")

    def _learn(self, ts: TimeSeries) -> None:
        self.level_ = float(np.nanmean(ts.values))
        self._anchor = int(ts.time[0])
        self._cache = object()

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        return ts.wrap(ts.values - self.level_)


def test_only_parameters_fitted_attributes_and_declared_state_are_stored(
    stepped: pd.Series,
) -> None:
    stored = Cached().fit(stepped).to_dict()["state"]
    assert "level_" in stored
    assert "_anchor" in stored
    assert "_cache" not in stored
    restored = Cached.from_dict(Cached().fit(stepped).to_dict())
    assert restored._anchor == stored["_anchor"]
    assert restored.fitted


def test_a_pipeline_round_trips_with_its_steps(stepped: pd.Series) -> None:
    import json

    from hazure.scorers import DeviationScorer as RealDeviationScorer
    from hazure.thresholds import IqrThreshold

    pipeline = Pipeline([("score", RealDeviationScorer()), ("cut", IqrThreshold())])
    expected = pipeline.fit_detect(stepped)
    restored = Pipeline.from_dict(json.loads(json.dumps(pipeline.to_dict())))
    pd.testing.assert_series_equal(restored.detect(stepped), expected)


def test_sklearn_style_shallow_parameters_are_available() -> None:
    """``sklearn.base.clone`` asks for ``get_params(deep=False)``."""
    detector = detectors.iqr()
    shallow = detector.get_params(deep=False)
    rebuilt = type(detector)(**shallow)
    assert repr(rebuilt) == repr(detector)
