# From detection to notification

A useful monitor needs three separate answers: whether its detector works on
unseen history, why it labelled a point, and whether that verdict deserves a new
notification. These APIs keep those answers separate, so suppressing repeated
notifications never changes the labels used to judge detection quality.

## Suppress short runs and repeated notifications

`AlertPolicy` consumes detector labels and emits notification pulses: `1.0` only
when a notification should open, `0.0` on other known samples, and `NaN` when the
input is unknown. It does not send messages.

```python
import numpy as np
from hazure import AlertPolicy, TimeSeries

HOUR = 3_600_000_000_000
labels = TimeSeries.from_arrays(
    np.arange(7) * HOUR,
    [1, 1, 0, 1, 1, 1, 1],
)
policy = AlertPolicy(min_consecutive=2, cooldown="4h")
print(policy.apply(labels).values.ravel().tolist())
# [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
```

The first run notifies at hour 1. The second run meets the two-observation rule
at hour 4, but waits until hour 5 for the four-hour cooldown to expire. Staying
anomalous afterwards does not notify again. Normal or unknown observations
break continuity; unknown observations remain `NaN` in the output.

Cooldown is elapsed time, independently per column, measured from the previous
notification. Integer cooldowns are nanoseconds; use a duration string for
readability. Consecutive means consecutive observations, so an interval with no
observations does not itself break the run.

`apply` starts fresh on every call. For chunks or live labels, use `update_many`
or `update`; these retain state across calls and reject non-increasing timestamps
or changed column schemas. Both APIs use the same state machine. `reset` clears
it, and `to_dict` / `from_dict` preserve it through JSON.

## Compare detectors on held-out history

`backtest` connects chronological splits, fresh model clones, detection metrics
and optional notification policies. It leaves supplied models and policies
unchanged. Results use `TimeSeries` and ordinary records, so no dataframe library
is required.

```python
from hazure import detectors
from hazure.evaluation import backtest

time = np.arange(36) * HOUR
values = np.zeros(36)
values[14:17] = 9
values[20] = 9
values[28:31] = 9
values[34] = 9
data = TimeSeries.from_arrays(time, values)
truth = data.wrap((values > 5).astype(float))

comparison = backtest(
    {"limit": detectors.limits(high=5), "strict": detectors.limits(high=20)},
    data,
    truth,
    n_splits=2,
    alert_policy=AlertPolicy(min_consecutive=2, cooldown="6h"),
)
for row in comparison.summary():
    print(row["model"], row["recall"], row["n_alerts"], row["n_notifications"])
# limit 1.0 4.0 2.0
# strict 0.0 0.0 0.0
```

The default split expands the training window and uses disjoint test blocks.
Each fold fits a fresh clone, so held-out observations never enter fitting.
`to_records()` returns each fold and output column separately; `summary()` takes
unweighted means of defined fold metrics and sums counts. Undefined metrics stay
`NaN`. Other split modes are available through `mode` and `train_ratio`; modes
with overlapping test blocks count repeated observations in separate folds.

Ground truth may be labels on the full data's time axis, `Events`, or a mapping
from prediction column to either. One truth column can serve several prediction
columns. By default, precision, recall, F1 and IoU judge events; pass
`events=False` for sample metrics. Detection delay is in nanoseconds. Interval
truth is sampled and clipped to the test block, so an outage extending across a
fold boundary is evaluated only where that fold observes it.

Raw labels determine detection metrics and `n_alerts`. Notification policies add
`n_notifications` and per-fold `notifications`, without changing those metrics.
Notification history starts fresh at each test boundary.

Batch evaluation may use future observations **within the test block**, as an
offline change detector does, and rolling detectors have a test-block warmup.
For online evaluation, pass `history=48` or a duration such as `history="2d"`.
This primes `Stream` from training history and judges test samples as they arrive;
its buffer checks reject components that read forward.

## Inspect the actual verdict

`explain` inspects a fitted component. It returns the actual output and a tree of
intermediate computations, rather than fitting a second explanatory model.

```python
from hazure import Graph, Node, explain
from hazure.ensemble import OrAggregator

graph = Graph(
    [
        Node("limit", detectors.limits(high=5)),
        Node("strict", detectors.limits(high=20)),
        Node("either", OrAggregator(), inputs=("limit", "strict")),
    ]
).fit(data)
detail = explain(graph, data)
row = detail.at(int(time[14]))
print(row["steps"]["limit"]["output"], row["steps"]["strict"]["output"])
# {'value': 1.0} {'value': 0.0}
```

`detail.output` matches the component's normal output. A detector exposes its
pre-threshold `scores` and its threshold step. Pipeline steps, graph branches and
independently fitted columns appear under `steps`, preserving their hierarchy.
`at(timestamp)` returns named values for a single timestamp; an absent timestamp
raises instead of selecting a neighbour. All arrays remain `TimeSeries`, and
`.to_native()` can reconstruct their original backend.

For PCA reconstruction error, `contributions` contains the squared residual for
each column **from the same fitted subspace**. These add up to its anomaly score:

```python
base = np.arange(12.0)
normal = TimeSeries.from_arrays(
    np.arange(12) * HOUR,
    np.column_stack([base, 2 * base]),
    ["a", "b"],
)
pca = detectors.pca(k=1).fit(normal)
changed = normal.values.copy()
changed[6, 1] += 20
pca_detail = explain(pca, normal.wrap(changed))
print(
    bool(
        np.allclose(
            pca_detail.contributions.values.sum(axis=1),
            pca_detail.scores.values[:, 0],
        )
    )
)
# True
```

These contributions locate a disagreement between measurements, not its causal
culprit. Their units are squared input units; after a transformation, they refer
to transformed columns. Other model types expose their actual scores and branch
verdicts and leave `contributions` as `None`.

## Consume labels from a live detector

Keep notification state outside `Stream`'s observation buffer. This preserves
cooldown and active-incident history even when old observations leave the buffer.

```python
import json
from hazure import Stream

training = TimeSeries.from_arrays(time[:12], values[:12])
detector = detectors.limits(high=5).fit(training)
stream = Stream(detector, history=4).prime(training)
notifications = AlertPolicy(min_consecutive=2, cooldown="6h")
opened = []
for i in range(12, len(time)):
    label = stream.update(int(time[i]), float(values[i]))
    if notifications.update(int(time[i]), label) == 1:
        opened.append(i)
print(opened)
# [15, 29]

saved = json.loads(json.dumps(notifications.to_dict()))
resumed = AlertPolicy.from_dict(saved)
```

Persist the fitted `Stream` separately when resuming the whole monitor. The
notification policy stores counters and timestamps, not a second detector.
