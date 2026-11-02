# Contributing

## Package layout

The public imports are defined in each package's `__init__.py`. Start there to
find a component, then follow its import to the implementation. Files prefixed
with `_` are implementation details.

```text
src/hazure/
├── _core/
│   ├── series/
│   │   ├── _model.py       # TimeSeries, validation, selection and joins
│   │   └── _native.py      # Origin and native dataframe input/output
│   ├── time.py            # Shared timestamp conversion and frequency inference
│   ├── window/
│   │   ├── _bounds.py      # Duration parsing and window boundaries
│   │   ├── _aggregate.py   # Chunked numerical reductions
│   │   └── _rolling.py     # Rolling and adjacent-window operations
│   ├── component.py       # Scorer, Threshold, Transformer and Aggregator
│   ├── detector.py        # Scorer and threshold pairing
│   ├── params.py          # Configuration and cloning
│   └── persist.py         # JSON model storage
├── compose/
│   ├── _base.py           # Shared composite dispatch and validation
│   ├── _pipeline.py       # Linear chains
│   └── _graph.py          # Nodes, graph validation and execution
├── calibration/
│   ├── _result.py         # Calibration result and threshold conversion
│   ├── _common.py         # Candidate grids and per-column inputs
│   ├── _supervised.py     # Search against labelled history
│   └── _budget.py         # Search within an alert-rate budget
├── detectors/             # Ready-made scorer/threshold combinations
├── scorers/               # Continuous anomaly scores
├── thresholds/            # Scores to labels
├── transformers/          # Feature transformations
├── ensemble/              # Combining scores or labels
├── events/                # Intervals and per-sample label conversion
├── evaluation/            # Metrics, time-ordered splits and detector backtests
├── alerts/                # Stateful notification decisions from labels
├── datasets/              # Synthetic data and benchmark loaders
├── explanation.py         # Trace fitted components and PCA error contributions
├── streaming.py           # Buffered execution of a fitted component
└── plotting.py            # Optional visualisation
```

## Dependency boundaries

- Algorithms operate on `TimeSeries` and numpy arrays. Native dataframe
  conversion belongs to `_core/series/_native.py`.
- Time-axis conversions shared by series and events belong to `_core/time.py`.
- Window operations use bounds and aggregation; neither lower layer imports the
  operations that combine them.
- `Pipeline` and `Graph` share composite behaviour. A pipeline uses the graph
  representation for Mermaid output; graph execution does not depend on pipelines.
- Calibration policies share input handling and candidate generation, while each
  policy owns its objective and search.
- Notification policies consume labels independently of detection and streaming
  buffers. Backtests fit fresh clones and score raw labels before notification
  suppression. Explanations trace fitted components without refitting them.
- Internal modules import their owning subpackage or `_core`, never the `hazure`
  package root. Optional libraries are imported only when needed by an operation.

Keep existing public import paths when moving implementations. In particular,
component payloads and pickle reference class paths. The affected package
initialisers retain those paths through `__module__`; the compatibility fixture
in `tests/fixtures/` checks that a previously saved pipeline still runs.

## Tests and local checks

`tests/core/` covers the array model, window engine, component lifecycle and
package contracts. The remaining test modules cover the public features named
in their filenames. Shared dataframe factories live in `tests/helpers.py`;
`tests/conftest.py` registers pytest fixtures.

Install the same extras used by the main CI test jobs, plus the documentation
tools:

```bash
uv sync --locked --dev --group docs \
  --extra pandas --extra polars --extra pyarrow \
  --extra stats --extra sklearn --extra viz
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy
uv run --locked pytest --cov
uv run --locked python scripts/check_docs.py
uv run --locked mkdocs build --strict
```

Tests include docstring examples. Tests requiring `stumpy` or `ruptures` skip
when the corresponding extras are absent; CI also runs them with `mp` and `cpd`
installed on Python 3.12. The `minimal-deps` CI job separately runs detection
without optional dependencies.
