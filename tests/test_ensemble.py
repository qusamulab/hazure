"""Tests for the aggregators.

The three-valued logic is the substance here, so every cell of every truth table
is asserted rather than sampled.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

from hazure import TimeSeries
from hazure.compose import Graph, Node
from hazure.ensemble import (
    AndAggregator,
    CustomizedAggregator,
    OrAggregator,
    ScoreAggregator,
    VoteAggregator,
)
from hazure.scorers import AsScorer, DeviationScorer
from hazure.transformers import DoubleRollingAggregate
from tests.helpers import BACKENDS, make_native

if TYPE_CHECKING:
    from hazure import Aggregator

NAN = np.nan

#: Every combination of two labels, in a fixed order the expectations mirror.
PAIRS = [
    (1.0, 1.0),
    (1.0, 0.0),
    (1.0, NAN),
    (0.0, 1.0),
    (0.0, 0.0),
    (0.0, NAN),
    (NAN, 1.0),
    (NAN, 0.0),
    (NAN, NAN),
]


def combine(aggregator: Aggregator, columns: Any) -> np.ndarray:
    """Aggregate a matrix whose columns are label series, returning the labels."""
    values = np.asarray(columns, dtype=float)
    time = np.datetime64("2024-01-01") + np.arange(values.shape[0]) * np.timedelta64(
        1, "h"
    )
    ts = TimeSeries.from_arrays(
        time, values, [f"input_{i}" for i in range(values.shape[1])]
    )
    result = aggregator.aggregate(ts)
    return np.asarray(result["anomaly"], dtype=float)


# -- truth tables -----------------------------------------------------------


def test_or_is_anomalous_when_any_input_says_so() -> None:
    expected = [1.0, 1.0, 1.0, 1.0, 0.0, NAN, 1.0, NAN, NAN]
    np.testing.assert_array_equal(combine(OrAggregator(), PAIRS), expected)


def test_and_is_anomalous_only_when_every_input_agrees() -> None:
    expected = [1.0, 0.0, NAN, 0.0, 0.0, 0.0, NAN, 0.0, NAN]
    np.testing.assert_array_equal(combine(AndAggregator(), PAIRS), expected)


def test_a_majority_vote_ignores_unknown_inputs() -> None:
    """Unknown labels abstain, so a tie among the known ones still carries."""
    expected = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, NAN]
    np.testing.assert_array_equal(combine(VoteAggregator(), PAIRS), expected)


def test_a_unanimous_vote_needs_every_known_input() -> None:
    expected = [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, NAN]
    np.testing.assert_array_equal(
        combine(VoteAggregator(threshold=1.0), PAIRS), expected
    )


def test_a_vote_of_zero_flags_every_row_with_a_known_label() -> None:
    expected = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, NAN]
    np.testing.assert_array_equal(
        combine(VoteAggregator(threshold=0.0), PAIRS), expected
    )


def test_a_vote_counts_the_exact_fraction_of_three_inputs() -> None:
    """One of three is 1/3, two of three is 2/3, either side of a 0.5 threshold."""
    rows = [
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (1.0, 1.0, 1.0),
        (1.0, 0.0, NAN),
    ]
    np.testing.assert_array_equal(
        combine(VoteAggregator(threshold=0.5), rows), [0.0, 1.0, 1.0, 1.0]
    )
    np.testing.assert_array_equal(
        combine(VoteAggregator(threshold=2 / 3), rows), [0.0, 1.0, 1.0, 0.0]
    )


def test_an_all_unknown_row_is_unknown_for_every_aggregator() -> None:
    aggregators = (
        OrAggregator(),
        AndAggregator(),
        VoteAggregator(),
        ScoreAggregator(),
    )
    for aggregator in aggregators:
        assert np.isnan(combine(aggregator, [(NAN, NAN, NAN)])[0])


def test_a_non_binary_label_counts_as_anomalous() -> None:
    """A detector emitting counts or confidences still aggregates sensibly."""
    np.testing.assert_array_equal(
        combine(AndAggregator(), [(3.0, 0.5), (3.0, 0.0)]), [1.0, 0.0]
    )


# -- shape and naming -------------------------------------------------------


def test_every_aggregator_emits_a_single_column_named_anomaly() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="h", name="time")
    frame = pd.DataFrame({"a": [1.0, 0.0, 0.0], "b": [0.0, 0.0, 1.0]}, index=index)
    aggregators = (
        OrAggregator(),
        AndAggregator(),
        VoteAggregator(),
        ScoreAggregator(),
    )
    for aggregator in aggregators:
        result = aggregator.aggregate(frame)
        assert list(result.columns) == ["anomaly"]


def test_aggregating_separate_series_joins_them_first() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="h", name="time")
    left = pd.Series([1.0, 1.0, 0.0, 0.0], index=index, name="flag")
    right = pd.Series([1.0, 0.0, 1.0, 0.0], index=index, name="flag")
    combined = AndAggregator().aggregate(left, right)
    assert list(combined) == [1.0, 0.0, 0.0, 0.0]
    assert combined.name == "anomaly"


def test_a_single_row_is_aggregated_like_any_other() -> None:
    np.testing.assert_array_equal(combine(OrAggregator(), [(0.0, 1.0)]), [1.0])


# -- customized aggregation -------------------------------------------------


def test_a_customized_aggregator_receives_the_label_matrix() -> None:
    seen: list[tuple[int, ...]] = []

    def first_input(labels: np.ndarray) -> np.ndarray:
        seen.append(labels.shape)
        return labels[:, 0]

    result = combine(CustomizedAggregator(aggregate_func=first_input), PAIRS)
    assert seen == [(9, 2)]
    np.testing.assert_array_equal(result, [row[0] for row in PAIRS])


def test_a_customized_aggregator_passes_its_parameters() -> None:
    weighted = CustomizedAggregator(
        aggregate_func=lambda labels, w: (labels @ w >= 0.5).astype(float),
        aggregate_func_params={"w": np.array([0.67, 0.33])},
    )
    np.testing.assert_array_equal(
        combine(weighted, [(1.0, 0.0), (0.0, 1.0), (0.0, 0.0)]), [1.0, 0.0, 0.0]
    )


def test_a_customized_aggregator_accepts_a_single_column_result() -> None:
    reshaped = CustomizedAggregator(aggregate_func=lambda labels: labels[:, :1].copy())
    np.testing.assert_array_equal(
        combine(reshaped, [(1.0, 0.0), (0.0, 0.0)]), [1.0, 0.0]
    )


def test_a_customized_aggregator_rejects_a_wrong_row_count() -> None:
    shrinking = CustomizedAggregator(aggregate_func=lambda labels: labels[:1, 0])
    with pytest.raises(ValueError, match=r"must give shape \(2,\)"):
        combine(shrinking, [(1.0, 0.0), (0.0, 0.0)])


# -- score aggregation ------------------------------------------------------


#: Two scorers over five rows, the second in units a thousand times larger.
SMALL = [0.1, 0.4, 0.2, 0.9, 0.3]
LARGE = [200.0, 100.0, 500.0, 400.0, 300.0]
#: The same ranking as ``LARGE``, in units comparable with ``SMALL``.
COMPARABLE = [0.2, 0.1, 0.5, 0.4, 0.3]


def test_the_score_aggregator_ranks_the_difference_in_scale_away() -> None:
    """Ranking is scale-free, so only the order each input reports survives."""
    lopsided = combine(ScoreAggregator(), list(zip(SMALL, LARGE, strict=True)))
    even = combine(ScoreAggregator(), list(zip(SMALL, COMPARABLE, strict=True)))
    np.testing.assert_allclose(lopsided, even)

    # The other half of the claim: without normalisation the large column simply
    # is the answer, and the small one might as well not have been passed.
    raw = combine(
        ScoreAggregator(normalize="none"), list(zip(SMALL, LARGE, strict=True))
    )
    assert np.argsort(raw).tolist() == np.argsort(LARGE).tolist()
    assert np.argsort(lopsided).tolist() != np.argsort(LARGE).tolist()


#: One row where a single input screams, one where every input is mildly raised,
#: one where every input is loud. Each column is ranked on its own.
LONE_VOICE = [
    (0.0, 0.0, 0.0),
    (10.0, 0.0, 0.0),
    (3.0, 3.0, 3.0),
    (9.0, 9.0, 9.0),
]


def test_the_score_aggregator_hears_a_lone_loud_input_only_with_max() -> None:
    averaged = combine(ScoreAggregator(how="mean"), LONE_VOICE)
    loudest = combine(ScoreAggregator(how="max"), LONE_VOICE)

    # Averaging dilutes the lone voice below the row every input agrees on.
    assert averaged[1] < averaged[2] < averaged[3]
    # Taking the maximum puts it above that row, and level with the loud one.
    assert loudest[2] < loudest[1]
    assert loudest[1] == loudest[3] == 1.0


def test_the_score_aggregator_median_ignores_one_input_entirely() -> None:
    """The middle ground: a single input cannot move the median either way."""
    middling = combine(ScoreAggregator(how="median"), LONE_VOICE)
    assert middling[1] < middling[2] < middling[3]


def test_the_score_aggregator_lets_unknown_inputs_abstain() -> None:
    """A row is combined from whatever is known, and is NaN only if nothing is."""
    rows = [(0.0, 0.0, 0.0), (1.0, 2.0, NAN), (4.0, NAN, NAN), (NAN, NAN, NAN)]
    combined = combine(ScoreAggregator(normalize="none"), rows)
    np.testing.assert_array_equal(combined, [0.0, 1.5, 4.0, NAN])


def test_the_score_aggregator_keeps_a_ranked_row_of_unknowns_unknown() -> None:
    rows = [(0.0, 0.0), (1.0, NAN), (NAN, NAN), (2.0, 5.0)]
    combined = combine(ScoreAggregator(), rows)
    assert np.isnan(combined[2])
    assert not np.isnan(combined[[0, 1, 3]]).any()


def test_the_score_aggregator_scales_by_the_mad_when_asked() -> None:
    """``robust`` keeps the spacing between scores that ranking flattens."""
    rows = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (20.0, 20.0)]
    robust = combine(ScoreAggregator(normalize="robust"), rows)
    ranked = combine(ScoreAggregator(), rows)
    # Both agree on the order; only the robust one says how far out the last
    # point is, which the ranks cap at 1.
    assert list(np.argsort(robust)) == list(np.argsort(ranked))
    assert robust[3] > 10.0
    assert ranked[3] == 1.0


def test_the_score_aggregator_treats_a_column_without_spread_as_centred() -> None:
    """A zero MAD leaves the column unscaled rather than dividing by zero."""
    rows = [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (5.0, 5.0)]
    combined = combine(ScoreAggregator(normalize="robust"), rows)
    np.testing.assert_array_equal(combined, [0.0, 0.0, 0.0, 5.0])


def test_the_score_aggregator_emits_one_float_column_named_anomaly() -> None:
    index = pd.date_range("2024-01-01", periods=5, freq="h", name="time")
    frame = pd.DataFrame({"a": SMALL, "b": LARGE}, index=index)
    result = ScoreAggregator().aggregate(frame)
    assert list(result.columns) == ["anomaly"]
    assert len(result) == 5
    assert result["anomaly"].dtype == np.float64


def test_the_score_aggregator_combines_two_branches_of_a_graph() -> None:
    """The wiring case: two scorers on the same series, one combined ranking."""
    index = pd.date_range("2024-01-01", periods=60, freq="h", name="time")
    values = np.tile([1.0, 2.0, 1.5, 2.5], 15)
    values[40] = 20.0
    series = pd.Series(values, index=index, name="x")

    graph = Graph(
        [
            Node("deviation", DeviationScorer()),
            Node("shift", AsScorer(DoubleRollingAggregate(window=4, agg="median"))),
            Node(
                "combined",
                ScoreAggregator(),
                inputs=("deviation", "shift"),
            ),
        ]
    )
    # The combination is still a score, so the graph answers to score().
    combined = graph.fit_score(series)
    assert len(combined) == 60
    assert combined.name == "anomaly"
    assert combined.idxmax() == index[40]


def test_a_score_aggregator_rejects_an_unknown_reduction() -> None:
    with pytest.raises(ValueError, match=r"how='total' is not one of"):
        ScoreAggregator(how="total")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"how='total' is not one of"):
        combine(ScoreAggregator().set_params(how="total"), LONE_VOICE)


def test_a_score_aggregator_rejects_an_unknown_normalisation() -> None:
    with pytest.raises(ValueError, match=r"normalize='zscore' is not one of"):
        ScoreAggregator(normalize="zscore")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"normalize='zscore' is not one of"):
        combine(ScoreAggregator().set_params(normalize="zscore"), LONE_VOICE)


# -- parameters -------------------------------------------------------------


def test_the_quantifying_aggregators_have_no_parameters() -> None:
    assert OrAggregator().get_params() == {}
    assert AndAggregator().get_params() == {}
    assert repr(AndAggregator()) == "AndAggregator()"


def test_a_vote_exposes_its_threshold() -> None:
    assert VoteAggregator(threshold=0.75).get_params() == {"threshold": 0.75}
    assert repr(VoteAggregator(threshold=0.75)) == "VoteAggregator(threshold=0.75)"


def test_a_score_aggregator_exposes_its_choices_and_learns_nothing() -> None:
    aggregator = ScoreAggregator(how="max", normalize="robust")
    assert aggregator.get_params() == {"how": "max", "normalize": "robust"}
    assert repr(aggregator) == "ScoreAggregator(how='max', normalize='robust')"
    assert ScoreAggregator.trainable is False


def test_clone_round_trips_every_aggregator_parameter() -> None:
    def half(labels: np.ndarray) -> np.ndarray:
        return labels[:, 0]

    originals: list[Aggregator] = [
        OrAggregator(),
        AndAggregator(),
        VoteAggregator(threshold=0.8),
        ScoreAggregator(how="median", normalize="none"),
        CustomizedAggregator(aggregate_func=half, aggregate_func_params={"unused": 1}),
    ]
    for original in originals:
        copy = original.clone()
        assert copy.get_params() == original.get_params()
        assert repr(copy) == repr(original)


@pytest.mark.parametrize("threshold", [-0.1, 1.5])
def test_a_vote_threshold_must_be_a_fraction(threshold: float) -> None:
    with pytest.raises(ValueError, match=r"threshold must be a fraction"):
        combine(VoteAggregator(threshold=threshold), PAIRS)


# -- backend independence ---------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_an_aggregator_returns_the_backend_it_was_given(backend: str) -> None:
    left = make_native(backend, [1.0, 0.0, 1.0], name="left")
    right = make_native(backend, [1.0, 1.0, 0.0], name="right")
    result = OrAggregator().aggregate(left, right)
    assert type(result) is type(left)


def test_every_backend_produces_the_same_labels() -> None:
    left_values = [1.0, 0.0, np.nan, 1.0]
    right_values = [1.0, 1.0, 0.0, np.nan]
    results = [
        TimeSeries.from_any(
            VoteAggregator(threshold=0.5).aggregate(
                make_native(backend, left_values, name="left"),
                make_native(backend, right_values, name="right"),
            )
        )
        for backend in BACKENDS
    ]
    for other in results[1:]:
        np.testing.assert_array_equal(other.values, results[0].values)
    np.testing.assert_array_equal(results[0].values.ravel(), [1.0, 1.0, 0.0, 1.0])
