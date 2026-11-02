"""Judging a model: metrics, and the folds to compute them over.

:func:`precision`, :func:`recall`, :func:`f1_score` and :func:`iou` accept either
label series (scored per sample) or :class:`~hazure.events.Events` (scored per
interval), and a dict of either to score several anomaly types at once. They say
whether an outage was caught. :func:`detection_delay` and
:func:`detection_delays` say how late, which is the other half of the question
and is not visible in any of the four.

:func:`average_precision` and :func:`roc_auc` need no threshold at all. They
score a continuous score directly, by how well it *ranks* the anomalous samples,
which is the only way to judge a :class:`~hazure.Scorer` without also judging
the fence you put around it.

:func:`split_train_test` builds time-ordered folds. Shuffling is not an option for
a time series, so the four schemes it offers differ in how the training window
grows and where the test block sits.

:func:`backtest` fits fresh detector clones on those folds and returns held-out
labels, metrics and comparison records. Optional notification policies add pulse
counts; optional stream history evaluates test observations as they arrive.
"""

from __future__ import annotations

from hazure.evaluation.backtest import BacktestFold, BacktestResult, backtest
from hazure.evaluation.delay import detection_delay, detection_delays
from hazure.evaluation.metrics import f1_score, iou, precision, recall
from hazure.evaluation.ranking import average_precision, roc_auc
from hazure.evaluation.split import split_train_test

__all__ = [
    "BacktestFold",
    "BacktestResult",
    "average_precision",
    "backtest",
    "detection_delay",
    "detection_delays",
    "f1_score",
    "iou",
    "precision",
    "recall",
    "roc_auc",
    "split_train_test",
]
