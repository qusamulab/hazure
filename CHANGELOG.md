# Changelog

Notable changes to `hazure`, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[semantic versioning](https://semver.org/spec/v2.0.0.html) — while the major
version is 0, a minor bump may still break an interface, and the changelog will
say so.

## 0.2.0 — 2026-09-25

A redesign of the public API and of the core. **This release breaks almost every
import**, deliberately: the 0.1 layout had grown two parallel detector
hierarchies and two names for most scores, and a pre-1.0 minor release is the
cheapest place to stop that. The algorithms, their defaults and their results are
unchanged; what changed is how they are assembled and where they are imported
from. A migration table follows the list.

### Changed

- **There is one detector class.** `Detector(scorer, threshold)` replaces
  `ScoreDetector`, `SignedScoreDetector`, `MultivariateScoreDetector`,
  `MultivariateSignedScoreDetector` and the 22 named detector classes. The
  ready-made detectors are now functions in `hazure.detectors` returning a
  `Detector` — `detectors.spike(window=24)` rather than `SpikeDetector(window=24)`
  — so a ready-made detector prints what it is made of, and its scorer and
  threshold are real parameters rather than private state rebuilt on every fit.
- **Direction is a threshold concern.** `SignedThreshold(threshold, side=...)`
  judges the magnitude of a signed score and keeps the requested direction; it is
  what the `side=` argument of the ready-made detectors builds, and it composes
  with any threshold.
- **A transformer whose output is a score is used as one through `AsScorer`.**
  `RollingAggregateScorer`, `DoubleRollingScorer`, `SeasonalResidualScorer`,
  `PcaReconstructionErrorScorer` and `RegressionResidualScorer` are gone; wrap
  `RollingAggregate`, `DoubleRollingAggregate`, `SeasonalDecomposition`,
  `PcaReconstructionError` or `RegressionResidual` instead. Their fitted
  attributes are on `.transformer` — `scorer.transformer.seasonal_`.
- **Modules are named for what they hold.** `hazure.scorers`, `hazure.thresholds`,
  `hazure.transformers` and `hazure.detectors` replace `hazure.scoring`,
  `hazure.features`, `hazure.detection` and `hazure.methods`, whose contents were
  split by history rather than by kind.
- **The top-level namespace holds the types and nothing else.** `hazure` exports
  `TimeSeries`, `Component`, `Scorer`, `Threshold`, `Detector`, `Transformer`,
  `Aggregator`, `Pipeline`, `Graph`, `Node` and `Stream`, down from over a
  hundred names. The base classes lost their `Base` prefix. The window engine
  (`rolling`, `double_rolling`, `parse_duration`, `AGGREGATIONS`) moved to
  `hazure.transformers`.
- **Nested parameters are reachable by name.** `get_params()` reports
  `scorer__window`, `threshold__factor` and, in a `Pipeline` or `Graph`,
  `step__parameter`; `set_params` accepts the same names. `get_params(deep=False)`
  is available for `sklearn.base.clone`.
- **An aggregator is an ordinary component.** It runs anywhere a component does,
  including at the end of a `Pipeline` whose previous step produced several
  columns, and no longer needs special-casing in `Graph`. `ScoreAggregator`
  reports its output as a score, so a graph ending in it answers to `score()`.
- **`output_kind` names what a component emits** — `"score"`, `"labels"` or
  `"series"` — rather than the verb that applies it. Composites derive it from
  their last part.
- **Whether a component needs fitting, or every column at once, is asked of the
  instance.** `is_trainable` and `is_multivariate` answer for a component
  assembled from others — a detector is trainable when either part is — so a
  `Pipeline` of untrainable steps no longer demands a `fit()`.
- **What `to_dict` stores is decided by convention.** Parameters, fitted
  attributes (public names ending in `_`), and private state a class declares in
  `_persisted`; caches are no longer written out. `Component.from_dict` is the
  generic loader.
- `RegressionResidual` copies the regressor it is given at fit time, as the
  scorer built on it always did, so the object passed stays unfitted.
- Internal refactoring throughout: the graph walk, the plotting entry point and
  the dispatch in `hazure.evaluation` are each described in one place rather than
  several, and the rules for reading a gap — in observations and in labels — now
  live together in one module. Modules inside the package import from
  `hazure._core` or their own subpackage, never from `hazure` itself, and a test
  keeps it that way.

### Fixed

- **A score panel no longer redraws the data.** `plot` decided what each panel
  showed by matching its column names against the data's, so a score named after
  the column it scored — which is what a scorer produces — was drawn as the
  series again. Panels now carry the series they came from, and names no longer
  have to be unique across the two.

### Migrating from 0.1

| 0.1 | 0.2 |
| --- | --- |
| `from hazure import SpikeDetector` | `from hazure import detectors` |
| `SpikeDetector(window=24)` | `detectors.spike(window=24)` |
| `LevelShiftDetector`, `VolatilityShiftDetector` | `detectors.level_shift`, `detectors.volatility_shift` |
| `SeasonalDetector`, `AutoregressionDetector` | `detectors.seasonal`, `detectors.autoregression` |
| `IqrDetector`, `QuantileDetector`, `EsdDetector` | `detectors.iqr`, `detectors.quantile`, `detectors.esd` |
| `ThresholdDetector(low, high)` | `detectors.limits(low, high)` |
| `RegressionDetector`, `PcaDetector` | `detectors.regression`, `detectors.pca` |
| `OutlierDetector`, `MinClusterDetector` | `detectors.outlier`, `detectors.min_cluster` |
| `HampelDetector`, `RollingQuantileDetector` | `detectors.hampel`, `detectors.rolling_quantile` |
| `SpectralResidualDetector`, `StlDetector`, `MstlDetector` | `detectors.spectral_residual`, `detectors.stl`, `detectors.mstl` |
| `PeltDetector`, `RupturesDetector` | `detectors.pelt`, `detectors.ruptures` |
| `MatrixProfileDetector`, `DampDetector` | `detectors.matrix_profile`, `detectors.damp` |
| `ScoreDetector(scorer, threshold)` | `Detector(scorer, threshold)` |
| `SignedScoreDetector(scorer, threshold, side=s)` | `Detector(scorer, SignedThreshold(threshold, side=s))` |
| `RollingAggregateScorer(window, agg, q=q)` | `AsScorer(RollingAggregate(window, agg, agg_params={"q": q}))` |
| `DoubleRollingScorer(window, diff=d)` | `AsScorer(DoubleRollingAggregate(window, agg="median", diff=d))` |
| `SeasonalResidualScorer(period)` | `AsScorer(SeasonalDecomposition(period))` |
| `PcaReconstructionErrorScorer(k)` | `AsScorer(PcaReconstructionError(k))` |
| `RegressionResidualScorer(target)` | `AsScorer(RegressionResidual(target))` |
| `hazure.scoring`, `hazure.methods` scorers | `hazure.scorers` |
| `hazure.features` | `hazure.transformers` |
| `BaseScorer`, `BaseThreshold`, `BaseTransformer`, `BaseAggregator` | `Scorer`, `Threshold`, `Transformer`, `Aggregator` |
| `BaseAggregator._combine` | `Aggregator._compute` |
| `detector.window`, `detector.factor` | `detector.scorer.transformer.window`, `detector.threshold.threshold.factor`, or `get_params()["scorer__transformer__window"]` |
| `Configurable.from_dict(payload)` | `Component.from_dict(payload)` |
| `hazure.rolling`, `hazure.parse_duration` | `hazure.transformers.rolling`, `hazure.transformers.parse_duration` |

`DoubleRollingScorer` summarised each window with the median by default, and
`DoubleRollingAggregate` with the mean, so pass `agg="median"` when migrating to
keep the same scores. Everything else in the table is a rename.

## 0.1.0 — 2026-07-31

The first release. Everything below is the initial development work, so there is
nothing to compare it against and no upgrade to plan; the list is what the library
contains.

### Added

- **The core.** `TimeSeries`, the dataframe boundary that reads any pandas,
  polars or pyarrow object with a time axis into a sorted `int64` array of UTC
  nanoseconds and returns results in the flavour they arrived in. `Component` and
  the five base classes, with automatic per-column fan-out for anything
  univariate. `rolling` and `double_rolling`, the window engine, in samples or in
  durations.
- **Detectors.** Distribution rules (`ThresholdDetector`, `QuantileDetector`,
  `IqrDetector`, `EsdDetector`), rolling comparisons (`SpikeDetector`,
  `LevelShiftDetector`, `VolatilityShiftDetector`), model residuals
  (`SeasonalDetector`, `AutoregressionDetector`, `RegressionDetector`,
  `PcaDetector`) and model adapters (`MinClusterDetector`, `OutlierDetector`).
- **Scorers and thresholds** for each of the above, usable separately:
  `ScoreDetector` pairs any scorer with any threshold.
- **Further method families** in `hazure.methods`: spectral residual, the Hampel
  filter, a rolling quantile band, PELT segmentation with a `ruptures` adapter,
  matrix-profile discords, and STL / MSTL residuals.
- **A detector for every method scorer**: `DampDetector`,
  `RollingQuantileDetector` and `RupturesDetector`, plus the `normalize` its
  scorer always took on `MatrixProfileDetector`.
- **Storing a fitted model.** `Component.to_dict` and
  `Configurable.from_dict` carry a fitted component — nested components, private
  learned state and all — through JSON, so a model fitted on a period you trust
  outlives the process that fitted it.
- **Per-column attribution for PCA anomalies.** `PcaColumnError` writes out the
  terms of the reconstruction error, one per column, so a flag can be taken apart
  without a second model that might disagree with the first.
- **How late, and how good the ranking is.** `detection_delay` and
  `detection_delays` say when an alert arrived rather than only whether it did,
  and `average_precision` and `roc_auc` score a scorer without a fence being
  chosen first.
- **Composition.** `Pipeline` for a chain, `Graph` for a directed acyclic graph
  of components, and the `And` / `Or` / `Vote` / customised aggregators.
  `ScoreAggregator` combines the scores themselves rather than the verdicts,
  normalising each input by rank or by its MAD first so that a scorer cannot
  dominate the ensemble by emitting larger numbers.
- **Events and evaluation.** `Events` as closed intervals in UTC nanoseconds,
  conversion to and from labels, point- and event-based precision, recall, F1 and
  IoU, and time-ordered `split_train_test` folds.
- **A fence at a false-alarm rate you choose.** `PotThreshold` fits a generalised
  Pareto distribution to the tail of the training scores by maximum likelihood, so
  the cut-off can be placed where exceedance has probability `1e-4` — beyond the
  largest score ever observed, which no quantile of a sample can reach. Its
  `update` drives the same fence online, absorbing each score into the tail and
  discarding the ones it flags, since an anomaly is not evidence about how normal
  behaves.
- **Detection on a series that is still arriving.** `Stream` keeps a buffer of the
  recent past and runs a fitted component over it once per observation, so the
  online answer is the batch answer rather than a second implementation that
  agrees with it on the tested cases. `Stream.prime` fills the buffer from history
  and *checks* it: too short for what the component looks back over, and it
  refuses to start rather than quietly computing from a window that was never
  full. A stream carries its buffer through `to_dict`, so a monitor can be stored
  and resumed.
- **Choosing the cut-off.** `tune_threshold` searches for the cut-off that scores
  best against labelled incidents, event-based by default.
  `budget_threshold` needs no labels at all: it lowers the fence as far as an
  alert budget — one page a week — allows, descending until the budget breaks so
  that the non-monotonicity of alert counts cannot flatter the result. Both return
  a `Calibration` carrying the whole curve, not only the winner.
- **Data to try things on**, in `hazure.datasets`. `make_series` plants anomalies
  of five shapes at a strength you set and hands back the ground truth with the
  series. `load_nab` fetches a labelled series from the Numenta Anomaly Benchmark
  on demand and caches it, needing nothing beyond the standard library.
  `compare` runs several detectors over either and lays the metrics out side by
  side, alert counts included.
- **Documentation** at <https://qusamulab.github.io/hazure/>: quickstart, guide, six
  pages on the mathematics of every scorer, threshold and metric, and a full API
  reference. Every example in every docstring is executed by the test suite, so a
  documented output that has drifted from the code is a test failure.
