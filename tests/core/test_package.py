"""Guards on the package itself, rather than on anything it computes.

These exist because the failures they catch are invisible until a release: a
version string that disagrees with the distribution's, a name promised in
``__all__`` that nobody exports, or a module that reaches for pandas at import
time and so quietly breaks the "narwhals and numpy only" promise the README makes.
"""

from __future__ import annotations

import ast
import json
import pickle
import subprocess
import sys
from importlib import import_module
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pytest

import hazure

MODULES = (
    "hazure.calibration",
    "hazure.compose",
    "hazure.datasets",
    "hazure.detectors",
    "hazure.ensemble",
    "hazure.evaluation",
    "hazure.events",
    "hazure.scorers",
    "hazure.streaming",
    "hazure.thresholds",
    "hazure.transformers",
)

PACKAGE = Path(hazure.__file__).parent

OPTIONAL = (
    "matplotlib",
    "numba",
    "pandas",
    "polars",
    "pyarrow",
    "ruptures",
    "scipy",
    "sklearn",
    "statsmodels",
    "stumpy",
)


def test_dunder_version_matches_the_distribution() -> None:
    assert hazure.__version__ == version("hazure")


def test_every_promised_name_exists() -> None:
    missing = [name for name in hazure.__all__ if not hasattr(hazure, name)]
    assert missing == []


def test_all_has_no_duplicates() -> None:
    assert len(hazure.__all__) == len(set(hazure.__all__))


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("hazure._core.series", "TimeSeries"),
        ("hazure._core.series", "Origin"),
        ("hazure.compose", "Pipeline"),
        ("hazure.compose", "Graph"),
        ("hazure.compose", "Node"),
        ("hazure.calibration", "Calibration"),
    ],
)
def test_public_class_paths_stay_compatible_with_saved_objects(
    module: str, name: str
) -> None:
    cls = getattr(import_module(module), name)
    assert cls.__module__ == module
    assert pickle.loads(pickle.dumps(cls)) is cls


def test_a_pipeline_saved_before_the_package_reorganisation_still_runs() -> None:
    """Load a real 0.2 payload, rather than round-tripping the new layout."""
    fixture_path = Path(__file__).parents[1] / "fixtures" / "pipeline-0.2.json"
    fixture = json.loads(fixture_path.read_text())
    series = hazure.TimeSeries.from_arrays(
        np.asarray(fixture["time"], dtype=np.int64),
        np.asarray(fixture["values"]),
        fixture["columns"],
    )

    restored = hazure.Component.from_dict(fixture["model"])

    assert isinstance(restored, hazure.Pipeline)
    np.testing.assert_array_equal(restored.run(series).values, fixture["expected"])
    assert restored.to_dict() == fixture["model"]


@pytest.mark.parametrize("module", MODULES)
def test_every_name_a_submodule_promises_exists(module: str) -> None:
    imported = __import__(module, fromlist=["__all__"])
    missing = [name for name in imported.__all__ if not hasattr(imported, name)]
    assert missing == []
    assert len(imported.__all__) == len(set(imported.__all__))


def test_no_module_inside_the_package_imports_from_the_package_root() -> None:
    """Internal imports go through ``hazure._core`` or the owning subpackage.

    Importing from ``hazure`` itself would make a module's import depend on the
    order ``hazure/__init__.py`` lists things in, which is how a circular import
    appears after an innocent reordering. Docstring examples are exempt: they run
    after the package has loaded, and show what a user would write.
    """
    offenders = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if path == PACKAGE / "__init__.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module == "hazure":
                offenders.append(f"{path.relative_to(PACKAGE)}:{node.lineno}")
    assert offenders == []


def test_importing_hazure_pulls_in_nothing_optional() -> None:
    """The core promise, checked in a fresh interpreter.

    Checking ``sys.modules`` in *this* process would prove nothing: the test suite
    has already imported pandas. So this asks a subprocess, which has not.
    """
    names = ", ".join(repr(name) for name in OPTIONAL)
    script = (
        "import sys, hazure\n"
        f"leaked = {{{names}}} & set(sys.modules)\n"
        "assert not leaked, leaked\n"
        "print('clean')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "clean"
