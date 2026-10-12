<p align="center">
  <!-- Absolute, because this file is also PyPI's long description and PyPI
       does not resolve links relative to the repository. The image is on
       `main`, so the URL is live for readers of any branch. -->
  <img src="https://raw.githubusercontent.com/qusamulab/hazure/main/docs/assets/hero.png" alt="hazure" width="720">
</p>


You have a metric — requests per second, queue depth, a sensor reading — you
suspect it occasionally misbehaves, and you have no record of when it did.
`hazure` finds those moments: rule-based and unsupervised detectors that learn
what "normal" looks like from the series itself, and that say where it stopped
holding.

Any pandas, polars or pyarrow object with a time axis goes in, and the answer
comes back in the same flavour. At runtime it needs `narwhals` and `numpy` and
nothing else; every heavier dependency is an extra, imported lazily by the
algorithm that needs it.

```bash
pip install hazure[pandas]
```

```python
import numpy as np
import pandas as pd
from hazure import detectors
from hazure.events import to_events

# Hourly traffic with a daily rhythm — and one afternoon that went wrong.
rng = np.random.default_rng(0)
index = pd.date_range("2024-03-01", periods=24 * 21, freq="h", name="time")
daily = 40 * np.sin(2 * np.pi * np.arange(len(index)) / 24)
traffic = pd.Series(100 + daily + rng.normal(0, 3, len(index)), index=index, name="rps")
traffic.iloc[300:306] = 20.0

labels = detectors.seasonal(period=24).fit_detect(traffic)
print(to_events(labels))
# Events([2024-03-13T12:00:00..2024-03-13T17:59:59.999999999])
```

`labels` sits on `traffic`'s own index — `1.0` anomalous, `0.0` normal, `NaN`
unknown — and the six flagged hours are exactly the six that were planted.

`detectors.seasonal` is one of a set of ready-made detectors, one per kind of
anomaly — spikes, level shifts, broken seasonality, columns that stop agreeing.
Each is a function returning a `Detector`: a `Scorer` ("how unusual is each
point?") and a `Threshold` ("is that unusual enough to report?") held together.
Print one and it shows what it is made of; its parts are reachable by name, so
`set_params(threshold__factor=4.0)` reconfigures it and a grid search needs
nothing special. Build your own pairing with `Detector(scorer, threshold)`, chain
steps with `Pipeline`, and wire them with `Graph` when the model branches.

## On a series that is still arriving

Fit once on a period you are willing to call normal, then feed observations in as
they come. `Stream` runs the *same* fitted component over a buffer of recent
history, so the online answer is the batch answer rather than a second
implementation of it — and `prime` refuses to start if the buffer is too short for
what the component looks back over, instead of quietly computing from a window
that was never full.

```python
from hazure import Stream

detector = detectors.seasonal(period=24).fit(traffic)
stream = Stream(detector, history=48).prime(traffic)

stream.update("2024-03-22T00:00", 105.0)  # in line with the fortnight -> 0.0
stream.update("2024-03-22T01:00", 12.0)  # not in line with it -> 1.0
```

## Where to draw the line

Every threshold is parameterised by *something*, and none of those somethings is
"the answer I want". Two ways to get one. `PotThreshold` takes the false-alarm
probability directly — it fits a generalised Pareto to the tail of the training
scores, so the fence can be placed where exceedance has probability `1e-4`, beyond
the largest score ever observed. And `budget_threshold` needs no labels at all:
give it the number of alerts a week anyone will read, and it lowers the fence as
far as that allows and no further.

## Documentation

<https://qusamulab.github.io/hazure/>

- [Quickstart](https://qusamulab.github.io/hazure/quickstart/) — one planted anomaly
  end to end: detect, convert to intervals, score, plot.
- [Guide](https://qusamulab.github.io/hazure/guide/) — which detector suits which
  kind of anomaly, and the two behaviours that surprise people most.
- [How it works](https://qusamulab.github.io/hazure/algorithms/) — the mathematics of
  every scorer, threshold and metric, and where each one's assumptions run out.
- [API reference](https://qusamulab.github.io/hazure/api/) — every public name.

## Details

- Python 3.11 and newer.
- Fully typed; `mypy --strict` clean.
- MIT licensed.
