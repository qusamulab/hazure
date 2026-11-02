"""Choosing where to draw the line.

A scorer says how unusual each point is. Turning that into an alert needs a
number, and picking that number is the question people actually get stuck on:
every threshold in :mod:`hazure.thresholds` is parameterised by *something*, and
none of those somethings is "the answer I want".

Two ways out of it, depending on what you have.

If you have labelled history — even a handful of incidents somebody wrote down —
:func:`tune_threshold` searches for the cut-off that scores best against them.
Event-based F1 by default, because a monitor is judged on outages caught and
pages sent, not on samples classified.

If you have no labels, which is the case this library is mostly about, you still
have an alert budget: whatever the metric does, nobody is going to look at more
than one page a week. :func:`budget_threshold` finds the most sensitive cut-off
that stays inside that budget, which turns an unanswerable question about
distributions into an answerable one about how much attention you have.

Both hand back a :class:`Calibration`, which carries the whole curve it searched
and not only the winner — a threshold chosen off a flat peak is worth knowing
about, and so is one chosen off a spike.

Hyperparameters other than the cut-off need no machinery from here: components
follow the scikit-learn parameter conventions closely enough for
``sklearn.model_selection.GridSearchCV`` to drive them directly, and
:func:`~hazure.evaluation.split_train_test` supplies folds that respect time order.
"""

from __future__ import annotations

from hazure.calibration._budget import budget_threshold
from hazure.calibration._result import Calibration
from hazure.calibration._supervised import tune_threshold

__all__ = [
    "Calibration",
    "budget_threshold",
    "tune_threshold",
]


# Preserve the public class path used by existing pickles.
Calibration.__module__ = __name__
