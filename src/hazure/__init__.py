"""hazure — finding anomalies in time series without labelled examples.

The case this is built for: you have a metric, you suspect it occasionally
misbehaves, and you have no record of when it did. There is nothing to train a
classifier on, so the model has to describe what normal looks like and report
departures from it.

Most of the time that is one call::

    >>> import numpy as np, pandas as pd
    >>> from hazure import detectors
    >>> index = pd.date_range("2024-01-01", periods=200, freq="h")
    >>> values = np.zeros(200)
    >>> values[120] = 9.0
    >>> flags = detectors.spike(window=24).fit_detect(pd.Series(values, index=index))
    >>> bool(flags.idxmax() == index[120])
    True

and what :mod:`hazure.detectors` hands back is always a :class:`Detector`: a
scorer and a threshold, held together.

    Scorer       series -> continuous score     .score()      .fit_score()
    Threshold    score  -> binary labels        .apply()      .fit_apply()
    Detector     a Scorer and a Threshold       .detect()     .fit_detect()
    Transformer  series -> series               .transform()  .fit_transform()
    Aggregator   several columns -> one         .aggregate()

Asking "how unusual is this point" and asking "is that unusual enough to report"
are different questions, so they are separate types. One threshold policy is then
reusable across every scorer, a scorer can be swapped without revisiting the
policy, and a score is useful on its own for ranking rather than flagging.

Any pandas, polars or pyarrow object with a time axis is accepted, and results
come back in the flavour they went in as. Labels are ``1.0`` anomalous, ``0.0``
normal and ``NaN`` unknown — a point whose score could not be computed is not
quietly called normal.

Runtime dependencies are ``narwhals`` and ``numpy``. SciPy, scikit-learn,
statsmodels, matplotlib, stumpy and ruptures are extras, imported only by the
components that need them.

This namespace holds the types, composition structures and monitoring helpers.
The remaining APIs live in modules named for what they hold:

``hazure.detectors``
    Ready-made detectors, one function per kind of anomaly.
``hazure.scorers``
    Continuous scores, for ranking or for pairing with a threshold.
``hazure.thresholds``
    Turning a score into labels.
``hazure.transformers``
    Feature engineering: rolling aggregates, lags, decomposition, projections.
``hazure.ensemble``
    Combining several verdicts, or several scores, into one.
``hazure.events``
    Moving between per-sample labels and anomalous intervals.
``hazure.evaluation``
    Metrics — how much was caught, how late, how well ranked — and time-ordered
    backtests that compare detectors on held-out folds.
``hazure.alerts``
    Notification decisions with consecutive-label rules and per-column cooldown.
``hazure.explanation``
    Actual scores, branch verdicts and fitted PCA column contributions.
``hazure.calibration``
    Choosing where to draw the line: from labelled events, or from an alert
    budget.
``hazure.datasets``
    Series to try things on — synthetic, or a benchmark fetched on demand.
``hazure.plotting``
    One :func:`~hazure.plotting.plot` function, for looking at the result.
"""

from __future__ import annotations

from hazure import (
    calibration,
    detectors,
    ensemble,
    evaluation,
    events,
    scorers,
    thresholds,
    transformers,
)
from hazure._core import (
    Aggregator,
    Component,
    Detector,
    Scorer,
    Threshold,
    TimeSeries,
    Transformer,
)
from hazure.alerts import AlertPolicy
from hazure.compose import Graph, Node, Pipeline
from hazure.explanation import Explanation, explain
from hazure.streaming import Stream

__version__ = "0.2.0"

__all__ = [
    "Aggregator",
    "AlertPolicy",
    "Component",
    "Detector",
    "Explanation",
    "Graph",
    "Node",
    "Pipeline",
    "Scorer",
    "Stream",
    "Threshold",
    "TimeSeries",
    "Transformer",
    "__version__",
    "calibration",
    "detectors",
    "ensemble",
    "evaluation",
    "events",
    "explain",
    "scorers",
    "thresholds",
    "transformers",
]
