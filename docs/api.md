# API reference

Every public name, grouped by the module it lives in. Component types,
`TimeSeries`, composition structures, `Stream`, `AlertPolicy` and explanation
helpers are importable straight from `hazure`; other names from the module
listed here.

## `hazure`

::: hazure
    options:
      members:
        - TimeSeries
        - Component
        - Scorer
        - Threshold
        - Detector
        - Transformer
        - Aggregator

## `hazure.detectors`

Ready-made detectors, one function per kind of anomaly. Each returns a
`Detector`.

::: hazure.detectors

## `hazure.scorers`

Continuous scores: how unusual each point is. Worth having on their own, for
ranking, and paired with a threshold to detect.

::: hazure.scorers

## `hazure.thresholds`

Where to draw the line, independently of what produced the score.

::: hazure.thresholds

## `hazure.transformers`

Series in, series out. These sit upstream of a scorer, or become one through
`AsScorer`. The rolling-window engine they are built on is exported here too.

::: hazure.transformers

## `hazure.ensemble`

Aggregators, for combining several detectors' verdicts or several scorers'
scores.

::: hazure.ensemble

## `hazure.compose`

Chaining and wiring components into models.

::: hazure.compose

## `hazure.events`

Moving between one-label-per-sample and the anomalous-interval view of the same
thing.

::: hazure.events

## `hazure.evaluation`

Point- and event-based metrics, how late each alert was, threshold-free ranking
quality, and time-ordered backtests comparing fresh detector clones.

::: hazure.evaluation

## `hazure.calibration`

Choosing the cut-off: from labelled incidents, or from how many alerts a week
anyone will read.

::: hazure.calibration

## `hazure.streaming`

Driving a fitted component one observation at a time, for a series that is still
being produced.

::: hazure.streaming

## `hazure.alerts`

Notification pulses from consecutive anomalous labels, with per-column cooldown
and resumable state. Detection labels remain separate from notification decisions.

::: hazure.alerts

## `hazure.explanation`

Actual scores, intermediate branch outputs and fitted PCA error contributions.

::: hazure.explanation

## `hazure.datasets`

Series to try things on: generated with a known answer, or fetched from the
Numenta Anomaly Benchmark.

::: hazure.datasets

## `hazure.plotting`

Drawing a series, its verdicts and its scores. Needs the `viz` extra.

::: hazure.plotting
