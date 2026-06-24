"""The benchmark corpus (ADR-0076 §2): offline, deterministic, declarative.

Only sklearn built-ins and seeded synthetics — NO network in CI.
A dataset is a record; extending the corpus is adding a record here (+ a baseline
refresh via the CI job, see README.md). Metric orientation is NOT declared here:
the runner derives it from the library (``resolve_metric``) — a hand-written copy
would silently rot.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    task: str
    load: Callable[[], tuple[Any, Any]]


def _breast_cancer():
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    return data.data, data.target


def _wine():
    from sklearn.datasets import load_wine

    data = load_wine()
    return data.data, data.target


def _diabetes():
    from sklearn.datasets import load_diabetes

    data = load_diabetes()
    return data.data, data.target


def _synth_binary(**kw):
    def load():
        from sklearn.datasets import make_classification

        return make_classification(
            n_samples=600, n_features=15, n_informative=8, n_redundant=3, **kw
        )

    return load


def _synth_multiclass():
    from sklearn.datasets import make_classification

    return make_classification(
        n_samples=600,
        n_features=15,
        n_informative=8,
        n_redundant=3,
        n_classes=4,
        n_clusters_per_class=1,
        random_state=104,
    )


def _synth_regression():
    from sklearn.datasets import make_regression

    return make_regression(
        n_samples=600, n_features=15, n_informative=8, noise=10.0, random_state=105
    )


def _synth_regression_skewed():
    # heavy right-skewed target (ADR-0076 §2): an honesty stress for regression metrics
    import numpy as np
    from sklearn.datasets import make_regression

    X, y = make_regression(
        n_samples=600, n_features=15, n_informative=8, noise=5.0, random_state=106
    )
    scale = max(float(np.std(y)), 1e-9)
    return X, np.expm1((y - float(np.mean(y))) / scale * 1.5)


CORPUS: tuple[DatasetSpec, ...] = (
    DatasetSpec("breast_cancer", "binary", _breast_cancer),
    DatasetSpec("synth_binary", "binary", _synth_binary(random_state=101)),
    DatasetSpec(
        "synth_binary_imbalanced", "binary", _synth_binary(weights=[0.92], random_state=102)
    ),
    DatasetSpec("synth_binary_noisy", "binary", _synth_binary(flip_y=0.08, random_state=103)),
    DatasetSpec("wine", "multiclass", _wine),
    DatasetSpec("synth_multiclass", "multiclass", _synth_multiclass),
    DatasetSpec("diabetes", "regression", _diabetes),
    DatasetSpec("synth_regression", "regression", _synth_regression),
    DatasetSpec("synth_regression_skewed", "regression", _synth_regression_skewed),
)
