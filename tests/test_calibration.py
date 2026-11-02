"""Tests for hazure.calibration: choosing a cut-off from labels or from a budget."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest

from hazure.calibration import Calibration, budget_threshold, tune_threshold
from hazure.events import Events, to_events
from hazure.thresholds import FixedThreshold
from tests.helpers import make_native

HOUR = 3_600_000_000_000


def series(
    values: Any, *, start: str = "2024-01-01", freq: str = "h", name: str = "x"
) -> pd.Series:
    """An hourly pandas series on a named time axis."""
    index = pd.date_range(start, periods=len(values), freq=freq, name="time")
    return pd.Series(np.asarray(values, dtype=float), index=index, name=name)


def planted() -> tuple[pd.Series, pd.Series]:
    """Two cleanly separated outages: background below 1, anomalies at exactly 5.

    Nothing sits between the two levels, so a fence anywhere in between flags the
    planted samples and nothing else.
    """
    rng = np.random.default_rng(0)
    scores = rng.uniform(0.0, 1.0, 200)
    truth = np.zeros(200)
    for start, stop in ((50, 55), (120, 124)):
        scores[start:stop] = 5.0
        truth[start:stop] = 1.0
    return series(truth), series(scores)


def graded() -> tuple[pd.Series, pd.Series]:
    """One ten-sample outage whose first six samples score far above its last four.

    Six of ten samples is 0.6 of the outage's duration, so the event-based metrics
    call it caught while the sample-based ones still want the other four.
    """
    rng = np.random.default_rng(1)
    scores = rng.uniform(0.0, 1.0, 100)
    scores[40:46] = 8.0
    scores[46:50] = 3.0
    truth = np.zeros(100)
    truth[40:50] = 1.0
    return series(truth), series(scores)


def excursions() -> pd.Series:
    """Three weeks of hourly noise with five short excursions planted in it."""
    rng = np.random.default_rng(0)
    values = rng.normal(0.0, 1.0, 24 * 21)
    for start in (40, 150, 300, 380, 460):
        values[start : start + 3] += 6.0
    return series(values)


def debounceable() -> pd.Series:
    """Two weeks of hourly scores holding runs that a single quiet sample breaks.

    A run of 6.0 broken at one sample by 2.0, and a pair of 1.0 separated by one
    sample of 0.5. Everything else is a low, distinct ramp, so the search grid has
    candidates between each of those levels.
    """
    values = np.arange(336, dtype=float) * 1e-4
    values[50:55] = [6.0, 6.0, 2.0, 6.0, 6.0]
    values[200], values[201], values[202] = 1.0, 0.5, 1.0
    return series(values)


def two_columns() -> tuple[pd.DataFrame, pd.Series]:
    """Two score columns that both spike over one shared planted incident."""
    rng = np.random.default_rng(2)
    left = rng.uniform(0.0, 1.0, 120)
    right = rng.uniform(0.0, 1.0, 120)
    left[60:64] = 5.0
    right[60:64] = 7.0
    truth = np.zeros(120)
    truth[60:64] = 1.0
    index = pd.date_range("2024-01-01", periods=120, freq="h", name="time")
    return pd.DataFrame({"a": left, "b": right}, index=index), series(truth)


def single(result: Calibration | dict[str, Calibration]) -> Calibration:
    """Unwrap a one-column result, asserting that is what came back."""
    assert isinstance(result, Calibration)
    return result


def spread(result: Calibration | dict[str, Calibration]) -> dict[str, Calibration]:
    """Unwrap a multi-column result, asserting that is what came back."""
    assert isinstance(result, dict)
    return result


def rate_at(calibration: Calibration, cut: float) -> float:
    """Read one candidate's score straight off the curve."""
    return dict(calibration.curve)[cut]


# ---------------------------------------------------------------------------
# tune_threshold: the cut-off it finds
# ---------------------------------------------------------------------------


def test_a_cut_off_is_found_that_reproduces_the_planted_events_exactly() -> None:
    truth, scores = planted()
    best = single(tune_threshold(truth, scores))
    assert best.score == 1.0
    assert best.objective == "event f1"
    # Not merely a perfect score: the fence it chose flags the two planted outages
    # and nothing else, to the nanosecond.
    assert to_events(best.threshold.apply(scores)) == to_events(truth)


@pytest.mark.parametrize("metric", ["f1", "iou", "precision", "recall"])
def test_the_perfect_cut_off_scores_one_on_every_named_metric(metric: str) -> None:
    truth, scores = planted()
    best = single(tune_threshold(truth, scores, metric=metric))
    assert best.score == 1.0
    assert best.objective == f"event {metric}"
    assert to_events(best.threshold.apply(scores)) == to_events(truth)


def test_sample_scoring_can_land_on_a_different_cut_off_than_event_scoring() -> None:
    truth, scores = graded()
    by_event = single(tune_threshold(truth, scores))
    by_sample = single(tune_threshold(truth, scores, events=False))
    # Both are perfect and both are right about different questions: 6 of the
    # outage's 10 samples is enough for the outage to count as caught, so the
    # event search keeps the higher fence, while the sample search has to come
    # down far enough to pick up the remaining four.
    assert by_event.score == 1.0
    assert by_sample.score == 1.0
    assert by_event.cut_off > by_sample.cut_off
    assert by_event.objective == "event f1"
    assert by_sample.objective == "sample f1"


def test_metric_options_are_passed_through() -> None:
    truth, scores = graded()
    default = single(tune_threshold(truth, scores, metric="recall"))
    strict = single(tune_threshold(truth, scores, metric="recall", thresh=1.0))
    # Half the outage covered is a hit by default, so the fence can stay above the
    # four weaker samples; demanding all of it forces the fence below them.
    assert default.score == 1.0
    assert strict.score == 1.0
    assert strict.cut_off < default.cut_off


def test_ground_truth_may_be_events_instead_of_labels() -> None:
    truth, scores = planted()
    from_labels = single(tune_threshold(truth, scores))
    from_events = single(tune_threshold(to_events(truth), scores))
    assert from_events.cut_off == from_labels.cut_off
    assert from_events.score == from_labels.score
    # Element-wise and NaN-aware: a curve holds nan where the metric was
    # undefined, and nan is never equal to itself.
    np.testing.assert_array_equal(
        np.asarray(from_events.curve), np.asarray(from_labels.curve)
    )


def test_a_custom_callable_metric_is_used_and_named() -> None:
    def two_alerts(y_true: Any, y_pred: Any) -> float:
        assert isinstance(y_pred, Events)  # events=True, so intervals arrive
        return 1.0 if y_pred.n_events == 2 else 0.0

    truth, scores = planted()
    best = single(tune_threshold(truth, scores, metric=two_alerts))
    assert best.objective == "event two_alerts"
    assert best.score == 1.0
    assert to_events(best.threshold.apply(scores)).n_events == 2


# ---------------------------------------------------------------------------
# tune_threshold: ties and the curve
# ---------------------------------------------------------------------------


def test_a_tie_goes_to_the_higher_cut_off() -> None:
    truth, scores = graded()
    best = single(tune_threshold(truth, scores))
    cuts = [cut for cut, _ in best.curve]
    values = [value for _, value in best.curve]
    position = cuts.index(best.cut_off)

    # The two adjacent candidates either side of the four weaker samples both
    # score a perfect event F1, and the higher of the two is the one chosen.
    assert values[position - 1] == values[position] == best.score
    assert cuts[position - 1] < cuts[position]
    tied = [cut for cut, value in best.curve if value == best.score]
    assert len(tied) > 1
    assert best.cut_off == max(tied)


def test_a_metric_that_ties_everything_picks_the_highest_candidate() -> None:
    def flat(y_true: Any, y_pred: Any) -> float:
        return 0.25

    truth, scores = planted()
    best = single(tune_threshold(truth, scores, metric=flat))
    assert best.score == 0.25
    assert best.cut_off == max(cut for cut, _ in best.curve)


def test_the_curve_ascends_with_one_entry_per_distinct_candidate() -> None:
    truth, scores = planted()
    best = single(tune_threshold(truth, scores, candidates=50))
    cuts = [cut for cut, _ in best.curve]
    assert cuts == sorted(cuts)
    assert len(set(cuts)) == len(cuts)
    assert 2 <= len(cuts) <= 50
    assert best.cut_off in cuts


def test_the_best_finite_value_of_the_curve_is_the_score() -> None:
    truth, scores = planted()
    best = single(tune_threshold(truth, scores, metric="precision"))
    finite = [value for _, value in best.curve if math.isfinite(value)]
    # The top candidate flags nothing, so precision there is undefined rather
    # than zero, and an undefined value is never the winner.
    assert math.isnan(best.curve[-1][1])
    assert max(finite) == best.score
    assert dict(best.curve)[best.cut_off] == best.score


def test_a_finer_search_never_scores_worse_on_separable_data() -> None:
    # The quantile grids of two candidate counts are not nested, so this is a
    # property of separable data and not a guarantee of the search in general.
    truth, scores = graded()
    got = [
        single(tune_threshold(truth, scores, candidates=k)).score
        for k in (2, 3, 5, 10, 25, 50, 100, 200)
    ]
    assert got == sorted(got)
    assert got[-1] == 1.0


# ---------------------------------------------------------------------------
# tune_threshold: argument validation
# ---------------------------------------------------------------------------


def test_an_unknown_metric_name_lists_what_is_accepted() -> None:
    truth, scores = planted()
    with pytest.raises(KeyError, match="is not a metric tune_threshold knows"):
        tune_threshold(truth, scores, metric="fbeta")
    with pytest.raises(KeyError, match=r"\['f1', 'iou', 'precision', 'recall'\]"):
        tune_threshold(truth, scores, metric="fbeta")


def test_a_metric_that_is_neither_a_name_nor_a_callable_is_rejected() -> None:
    truth, scores = planted()
    with pytest.raises(TypeError, match="metric must be a name or a callable, got int"):
        tune_threshold(truth, scores, metric=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [1, 0, -3])
def test_a_search_of_fewer_than_two_candidates_is_rejected(bad: int) -> None:
    truth, scores = planted()
    with pytest.raises(ValueError, match="candidates must be at least 2"):
        tune_threshold(truth, scores, candidates=bad)
    with pytest.raises(ValueError, match="candidates must be at least 2"):
        budget_threshold(scores, candidates=bad)


@pytest.mark.parametrize(
    ("bad", "shown"), [(2.0, "float"), ("5", "str"), (True, "bool")]
)
def test_a_non_int_candidates_is_rejected(bad: Any, shown: str) -> None:
    truth, scores = planted()
    with pytest.raises(TypeError, match=f"candidates must be an int, got {shown}"):
        tune_threshold(truth, scores, candidates=bad)
    with pytest.raises(TypeError, match=f"candidates must be an int, got {shown}"):
        budget_threshold(scores, candidates=bad)


def test_scores_that_are_all_nan_leave_no_cut_off_to_choose() -> None:
    unscored = series([np.nan] * 10)
    labels = series([0, 0, 1, 1, 0, 0, 0, 0, 0, 0])
    with pytest.raises(ValueError, match="Every score is NaN"):
        tune_threshold(labels, unscored)
    with pytest.raises(ValueError, match="Every score is NaN"):
        budget_threshold(unscored)


# ---------------------------------------------------------------------------
# tune_threshold: several score columns
# ---------------------------------------------------------------------------


def test_several_score_columns_come_back_keyed_by_column() -> None:
    scores, truth = two_columns()
    results = spread(tune_threshold({"a": truth, "b": truth}, scores))
    assert sorted(results) == ["a", "b"]
    assert results["a"].score == 1.0
    assert results["b"].score == 1.0
    # Each column is searched on its own scale, so the two fences differ.
    assert results["a"].cut_off != results["b"].cut_off


def test_ground_truth_may_be_a_dict_a_frame_an_events_or_one_shared_series() -> None:
    scores, truth = two_columns()
    forms = {
        "dict": tune_threshold({"a": truth, "b": truth}, scores),
        "frame": tune_threshold(pd.DataFrame({"a": truth, "b": truth}), scores),
        "events": tune_threshold(to_events(truth), scores),
        "series": tune_threshold(truth, scores),
    }
    answers = {name: spread(result) for name, result in forms.items()}
    reference = answers["dict"]
    for got in answers.values():
        assert {name: value.cut_off for name, value in got.items()} == {
            name: value.cut_off for name, value in reference.items()
        }
        assert {name: value.score for name, value in got.items()} == {
            "a": 1.0,
            "b": 1.0,
        }


def test_a_dict_missing_a_score_column_is_rejected() -> None:
    scores, truth = two_columns()
    with pytest.raises(ValueError, match=r"no entry for score column 'b'.*\['a'\]"):
        tune_threshold({"a": truth}, scores)


def test_a_truth_frame_with_different_column_names_is_rejected() -> None:
    scores, truth = two_columns()
    mismatched = pd.DataFrame({"a": truth, "c": truth})
    with pytest.raises(ValueError, match=r"no column 'b' to match the scores"):
        tune_threshold(mismatched, scores)


def test_several_truth_columns_against_one_score_column_are_rejected() -> None:
    _, truth = two_columns()
    scores = series(np.arange(120.0))
    with pytest.raises(ValueError, match="expanded into several anomaly types"):
        tune_threshold(pd.DataFrame({"a": truth, "b": truth}), scores)


def test_a_cut_off_is_the_same_on_every_backend(native_factory: Any) -> None:
    # Two distinct score levels, so the grid is every distinct score and the
    # answer is exact rather than a quantile of one: the lower fence catches the
    # outage, the higher one flags nothing.
    truth = native_factory([0, 0, 0, 0, 1, 1, 1, 0, 0, 0])
    scores = native_factory([1, 1, 1, 1, 5, 5, 5, 1, 1, 1])
    best = single(tune_threshold(truth, scores))
    assert best.cut_off == 1.0
    assert best.score == 1.0
    assert [cut for cut, _ in best.curve] == [1.0, 5.0]

    budget = single(budget_threshold(scores, alerts=1, per="1h"))
    assert budget.cut_off == 1.0
    # One alert over the ten hours the series covers.
    assert budget.score == 0.1


# ---------------------------------------------------------------------------
# budget_threshold
# ---------------------------------------------------------------------------


def test_a_tighter_budget_raises_the_fence() -> None:
    scores = excursions()
    tight = single(budget_threshold(scores, alerts=1, per="7d"))
    loose = single(budget_threshold(scores, alerts=5, per="7d"))
    assert tight.cut_off > loose.cut_off
    assert tight.score <= 1.0
    assert loose.score <= 5.0
    assert tight.score < loose.score


def test_the_chosen_cut_off_is_the_last_one_inside_the_budget() -> None:
    scores = excursions()
    # Three weeks of hourly samples is exactly three seven-day periods.
    chosen = single(budget_threshold(scores, alerts=1, per="7d"))
    cuts = [cut for cut, _ in chosen.curve]
    position = cuts.index(chosen.cut_off)

    assert chosen.score <= 1.0
    assert to_events(chosen.threshold.apply(scores)).n_events == 3
    # Descend and stop: one candidate lower and the budget breaks, which is
    # exactly why the descent stopped where it did.
    assert position > 0
    assert chosen.curve[position - 1][1] > 1.0


def test_the_descent_stops_short_of_the_fence_that_flags_everything() -> None:
    scores = excursions()
    chosen = single(budget_threshold(scores, alerts=1, per="7d"))
    lowest_cut, lowest_rate = chosen.curve[0]
    values = scores.to_numpy()

    # The pathological candidate the descent exists to avoid: the lowest fence
    # flags all but one sample, which merges into a couple of enormous alerts and
    # so satisfies the budget outright.
    assert (values > lowest_cut).mean() > 0.99
    assert lowest_rate <= 1.0
    # Stopping at the first break from the top never reaches it.
    assert chosen.cut_off > lowest_cut
    assert (values > chosen.cut_off).mean() < 0.05


def test_a_quiet_sample_splits_an_alert_unless_gap_merges_it() -> None:
    scores = debounceable()
    plain = single(budget_threshold(scores, alerts=1, per="7d"))
    merged = single(budget_threshold(scores, alerts=1, per="7d", gap="1h"))
    # A fence just under the run of 6.0 flags four samples in two pieces, split by
    # the one quiet sample in the middle. Two weeks is two budget periods, so two
    # alerts is a rate of 1.0 and one alert is 0.5.
    cut = max(cut for cut, _ in plain.curve if cut < 6.0)
    assert to_events(FixedThreshold(high=cut).apply(scores)).n_events == 2
    assert rate_at(plain, cut) == 1.0
    assert rate_at(merged, cut) == 0.5


def test_gap_lets_the_fence_come_down() -> None:
    scores = debounceable()
    plain = single(budget_threshold(scores, alerts=1, per="7d"))
    merged = single(budget_threshold(scores, alerts=1, per="7d", gap="1h"))
    # Debouncing buys room inside the same budget, so the descent goes further.
    assert merged.cut_off < plain.cut_off
    assert plain.score <= 1.0
    assert merged.score <= 1.0


def test_gap_accepts_a_bare_int_of_nanoseconds() -> None:
    scores = debounceable()
    spelled = single(budget_threshold(scores, alerts=1, per="7d", gap="1h"))
    counted = single(budget_threshold(scores, alerts=1, per="7d", gap=HOUR))
    assert counted.cut_off == spelled.cut_off
    assert counted.score == spelled.score
    assert counted.curve == spelled.curve


@pytest.mark.parametrize("per", ["7d", timedelta(days=7), np.timedelta64(7, "D")])
def test_a_budget_period_may_be_given_in_any_duration_form(per: Any) -> None:
    scores = excursions()
    reference = single(budget_threshold(scores, alerts=1, per="7d"))
    got = single(budget_threshold(scores, alerts=1, per=per))
    assert got.cut_off == reference.cut_off
    assert got.score == reference.score
    assert got.objective == "alerts per 7d"


def test_each_score_column_gets_its_own_budget() -> None:
    scores, _ = two_columns()
    results = spread(budget_threshold(scores, alerts=2, per="1d"))
    assert sorted(results) == ["a", "b"]
    for calibration in results.values():
        assert calibration.score <= 2.0
        assert calibration.objective == "alerts per 1d"


@pytest.mark.parametrize("bad", [0, 0.0, -1.0])
def test_a_non_positive_budget_is_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="alerts must be positive"):
        budget_threshold(excursions(), alerts=bad)


@pytest.mark.parametrize(
    ("bad", "shown"), [("1", "str"), (None, "NoneType"), (True, "bool")]
)
def test_a_non_numeric_budget_is_rejected(bad: Any, shown: str) -> None:
    with pytest.raises(TypeError, match=f"alerts must be a number, got {shown}"):
        budget_threshold(excursions(), alerts=bad)


def test_a_rate_cannot_be_measured_over_a_single_observation() -> None:
    with pytest.raises(ValueError, match="rate cannot be measured over 1 observation"):
        budget_threshold(series([1.0]))


def test_the_objective_says_what_was_being_optimised() -> None:
    truth, scores = planted()
    assert single(tune_threshold(truth, scores)).objective == "event f1"
    assert single(tune_threshold(truth, scores, events=False)).objective == "sample f1"
    assert single(tune_threshold(truth, scores, metric="iou")).objective == "event iou"
    excursion = excursions()
    assert single(budget_threshold(excursion)).objective == "alerts per 7d"
    assert single(budget_threshold(excursion, per="1d")).objective == "alerts per 1d"


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def test_a_calibration_cannot_be_reassigned() -> None:
    calibration = Calibration(2.5, 0.8, "event f1", ((1.0, 0.5), (2.5, 0.8)))
    with pytest.raises(FrozenInstanceError, match="cannot assign to field 'cut_off'"):
        calibration.cut_off = 3.0  # type: ignore[misc]


def test_a_calibration_hands_back_its_cut_off_as_a_one_sided_fence() -> None:
    calibration = Calibration(2.5, 0.8, "event f1", ((1.0, 0.5), (2.5, 0.8)))
    fence = calibration.threshold
    assert isinstance(fence, FixedThreshold)
    assert fence.high == calibration.cut_off
    assert fence.low is None


def test_a_calibration_repr_names_the_objective() -> None:
    tuned = repr(Calibration(2.5, 0.8, "event f1", ((1.0, 0.5), (2.5, 0.8))))
    assert "event f1=0.8" in tuned
    assert "cut_off=2.5" in tuned
    assert "over 2 candidates" in tuned
    budgeted = repr(Calibration(2.5, 1.0, "alerts per 7d", ((2.5, 1.0),)))
    assert "alerts per 7d=1" in budgeted


def test_the_backend_of_the_scores_does_not_change_the_calibration() -> None:
    values = [1, 1, 1, 1, 5, 5, 5, 1, 1, 1]
    labels = [0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
    answers = {
        backend: single(
            tune_threshold(make_native(backend, labels), make_native(backend, values))
        )
        for backend in ("pandas", "polars", "pyarrow")
    }
    assert {name: value.cut_off for name, value in answers.items()} == {
        "pandas": 1.0,
        "polars": 1.0,
        "pyarrow": 1.0,
    }


# ---------------------------------------------------------------------------
# the corners the search can be driven into
# ---------------------------------------------------------------------------


def test_a_metric_that_is_never_defined_leaves_the_calibration_unknown() -> None:
    """Ground truth with nothing anomalous in it makes recall undefined everywhere.

    There is no cut-off that scores better than any other, so neither a cut-off
    nor a score can be reported — and reporting the last one tried would be
    presenting an arbitrary choice as a decision.
    """
    truth, scores = planted()
    quiet = truth * 0.0
    chosen = single(tune_threshold(quiet, scores, metric="recall"))
    assert math.isnan(chosen.cut_off)
    assert math.isnan(chosen.score)
    assert all(math.isnan(value) for _, value in chosen.curve)


def test_a_sub_second_budget_period_is_named_in_its_own_units() -> None:
    """``objective`` renders a duration that has no whole-unit name in nanoseconds."""
    _, scores = planted()
    chosen = single(budget_threshold(scores, alerts=1, per=np.timedelta64(1500, "ms")))
    assert chosen.objective == "alerts per 1500000000ns"


@pytest.mark.parametrize(
    ("per", "expected"),
    [
        (np.timedelta64(2, "D"), "alerts per 2d"),
        (np.timedelta64(6, "h"), "alerts per 6h"),
        (np.timedelta64(30, "m"), "alerts per 30min"),
        (np.timedelta64(45, "s"), "alerts per 45s"),
    ],
)
def test_a_budget_period_given_as_a_timedelta_is_named_in_whole_units(
    per: np.timedelta64, expected: str
) -> None:
    _, scores = planted()
    assert single(budget_threshold(scores, alerts=1, per=per)).objective == expected
