"""M8-2: the standalone inference boundary (ADR-0066, FR-SRV-2/5, NFR-SRV-1).

Lazy barrels (PEP 562) keep ``import honestml`` + ``load_artifact(...).predict(...)`` from executing the
training stack; the public surface is unchanged; the metric is resolved lazily (only ``score`` needs it);
the standalone path is byte-identical to the facade.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

import honestml
import honestml.adapters
import honestml.application
import honestml.composition
from honestml import AutoML, load_artifact, save_artifact

pytestmark = pytest.mark.unit

# training-only modules that must NOT be executed by load+predict (NFR-SRV-1)
_FORBIDDEN = [
    "optuna",
    "shap",
    "mlflow",
    "sklearn.cluster",
    "sklearn.ensemble",
    "honestml.adapters.tuning",
    "honestml.adapters.ensembling",
    "honestml.adapters.significance",
    "honestml.adapters.splitters",
    "honestml.adapters.feature_selectors",
    "honestml.adapters.feature_rankers",
    "honestml.adapters.tracking",
    # the AutoML metric machinery is deferred (ADR-0066 §2) — predict must not resolve it. (sklearn.metrics
    # itself may arrive via the estimator's own unpickle and is NOT asserted absent.)
    "honestml.adapters.metrics",
    "honestml.application.tuning",
    "honestml.application.ensemble",
]


def _fit_save(tmp_path):
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=60, n_features=6, n_informative=4, n_redundant=0, random_state=0
    )
    model = AutoML(task="binary", models=("baseline", "linear"), random_state=0).fit(X, y)
    art = tmp_path / "art"
    save_artifact(model.fitted_, art)
    return model, art, X


def test_predict_import_cone_excludes_training(tmp_path) -> None:
    """In a fresh process, load_artifact + predict must not pull the training stack (NFR-SRV-1)."""
    _, art, X = _fit_save(tmp_path)
    xfile = tmp_path / "X.npy"
    np.save(xfile, X)
    code = (
        "import sys, json, numpy as np, honestml\n"
        f"m = honestml.load_artifact(r'{art}')\n"
        f"m.predict(np.load(r'{xfile}'))\n"
        f"forbidden = {_FORBIDDEN!r}\n"
        "print(json.dumps([k for k in forbidden if k in sys.modules]))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    leaked = json.loads(out.stdout.strip().splitlines()[-1])
    assert leaked == [], f"training modules leaked into the predict cone: {leaked}"


def test_native_predict_cone_no_onnx_no_heavy(tmp_path) -> None:
    """M8b-1 (NFR-SER-3): loading a NATIVE artifact pulls only the model's own runtime — the heavy
    training adapters and the onnx tooling stay out of the predict cone."""
    pytest.importorskip("xgboost")
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=60, n_features=6, n_informative=4, n_redundant=0, random_state=0
    )
    model = AutoML(task="binary", models=("xgboost",), random_state=0).fit(X, y)
    art = tmp_path / "art"
    save_artifact(model.fitted_, art, model_format="native")
    xfile = tmp_path / "X.npy"
    np.save(xfile, X)
    forbidden = [*_FORBIDDEN, "onnx", "onnxmltools", "skl2onnx", "onnxruntime"]
    code = (
        "import sys, json, numpy as np, honestml\n"
        f"m = honestml.load_artifact(r'{art}')\n"
        f"m.predict(np.load(r'{xfile}'))\n"
        f"forbidden = {forbidden!r}\n"
        "print(json.dumps([k for k in forbidden if k in sys.modules]))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    leaked = json.loads(out.stdout.strip().splitlines()[-1])
    assert leaked == [], f"training/onnx modules leaked into the native predict cone: {leaked}"


def test_export_onnx_lazy_barrel() -> None:
    """M8b-2 (ADR-0071 §7): the three checkpoints — `import honestml`, resolving `honestml.export_onnx`
    and constructing `AutoML(...)` — must not import the onnx tooling/converter adapter; those are
    imported only when the function is called."""
    code = (
        "import sys, json, honestml\n"
        "fn = honestml.export_onnx\n"
        "assert callable(fn)\n"
        "honestml.AutoML(task='binary', models=('linear',))\n"
        "forbidden = ['onnx', 'onnxmltools', 'skl2onnx', 'onnxruntime', "
        "'honestml.adapters.onnx_export']\n"
        "print(json.dumps([k for k in forbidden if k in sys.modules]))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    leaked = json.loads(out.stdout.strip().splitlines()[-1])
    assert leaked == [], f"onnx tooling leaked on barrel/facade resolution: {leaked}"


def test_public_names_unchanged() -> None:
    """R-SRVDEP: every public name still resolves (lazily) and shows up in dir() for the 4 lazy barrels."""
    for mod in (honestml, honestml.adapters, honestml.application, honestml.composition):
        for name in mod.__all__:
            assert hasattr(mod, name), f"{mod.__name__}.{name} no longer importable"
        assert set(mod.__dir__()) >= set(mod.__all__)


def test_lazy_name_lists_in_sync() -> None:
    """R-SRVDEP drift guard: the lazy map and __all__ must not drift (a name in one but not the other
    silently breaks `import *`/__dir__ or __getattr__). honestml's __all__ is wider (eager core)."""
    for mod in (honestml.adapters, honestml.application, honestml.composition):
        assert set(mod._SUBMODULES) == set(mod.__all__), mod.__name__
    assert set(honestml._SUBMODULES) <= set(honestml.__all__)


def test_standalone_predict_equals_facade(tmp_path) -> None:
    model, art, X = _fit_save(tmp_path)
    loaded = load_artifact(art)
    assert np.array_equal(loaded.predict(X), model.predict(X))
    assert np.allclose(loaded.predict_proba(X), model.predict_proba(X))


def test_multiclass_metric_average_roundtrips(tmp_path) -> None:
    """The lazily-held ``metric_name``/``metric_average`` round-trip for multiclass, and the lazy resolve +
    score work end-to-end (exercises the multiclass path of the new metric fields, ADR-0066 §2)."""
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=90,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=0,
    )
    model = AutoML(
        task="multiclass", metric="roc_auc", models=("baseline", "linear"), random_state=0
    ).fit(X, y)
    art = tmp_path / "mc"
    save_artifact(model.fitted_, art)
    loaded = load_artifact(art)
    assert loaded.metric_name == model.fitted_.metric_name == "roc_auc"
    assert loaded.metric_average == model.fitted_.metric_average  # round-trips (value or None)
    assert np.array_equal(loaded.predict(X), model.predict(X))
    assert isinstance(loaded.score(X, y), float)  # lazy resolve of the multiclass metric works


def test_score_lazy_resolves_metric(tmp_path) -> None:
    """predict never resolves the metric; score triggers the one-time lazy resolution (ADR-0066 §2)."""
    model, art, X = _fit_save(tmp_path)
    from sklearn.datasets import make_classification

    _, y = make_classification(
        n_samples=60, n_features=6, n_informative=4, n_redundant=0, random_state=0
    )
    loaded = load_artifact(art)
    assert loaded._metric is None
    loaded.predict(X)
    assert loaded._metric is None  # predict did not touch the metric
    assert isinstance(loaded.score(X, y), float)
    assert loaded._metric is not None  # score resolved it
