"""M5-resume RC-a: the pure ``compute_run_fingerprint`` key (ADR-0035 §1, FR-RC-1/5, NFR-RC-1).

Pure: canonical JSON over the resolved config + task + metric identity + data-signature + estimators +
lib versions + honestml/fingerprint versions. A fail-closed key — every axis that changes the result must
change the digest. Synchronously testable, no I/O (data-signature is passed as a literal here).
"""

from __future__ import annotations

import pytest

from honestml.adapters import Accuracy, LogLoss, RocAuc
from honestml.application import FINGERPRINT_VERSION, collect_lib_versions, compute_run_fingerprint
from honestml.core import BudgetConfig, CVConfig, RunConfig, Task

pytestmark = pytest.mark.unit


def _fp(**over):
    base = dict(
        run_config=RunConfig(),
        task=Task(kind="binary"),
        metric=RocAuc(),
        data_signature="sig0",
        estimators=("catboost", "lightgbm"),
        lib_versions={"numpy": "1.26.0", "scikit-learn": "1.4.0"},
    )
    base.update(over)
    return compute_run_fingerprint(**base)


def test_same_inputs_same_fp() -> None:
    assert _fp() == _fp()


def test_digest_is_hex_sha256() -> None:
    fp = _fp()
    assert isinstance(fp, str) and len(fp) == 64
    int(fp, 16)


def test_canonical_order_independent() -> None:
    # canonical (sort_keys) + sorted estimators -> input order must not matter
    assert _fp(estimators=("catboost", "lightgbm")) == _fp(estimators=("lightgbm", "catboost"))


def test_seed_axis() -> None:
    assert _fp(run_config=RunConfig(seed=1)) != _fp(run_config=RunConfig(seed=2))


def test_cv_axis() -> None:
    assert _fp(run_config=RunConfig(cv=CVConfig(n_splits=3))) != _fp(
        run_config=RunConfig(cv=CVConfig(n_splits=5))
    )


def test_budget_axis_stricter_than_needed() -> None:
    # ADR-0035 §4: the whole RunConfig (incl. budget) is in the key -> stricter, never a false hit
    assert _fp(run_config=RunConfig(budget=BudgetConfig(mode="trials", n_trials=2))) != _fp()


def test_significance_axis() -> None:
    assert _fp(run_config=RunConfig(significance="off")) != _fp()


def test_metric_axis() -> None:
    assert _fp(metric=RocAuc()) != _fp(metric=Accuracy())


def test_metric_labels_axis() -> None:
    # two LogLoss differing only by class labels -> different identity -> different fp
    import numpy as np

    a = _fp(metric=LogLoss(classes=np.array([0, 1, 2])))
    b = _fp(metric=LogLoss(classes=np.array([0, 1, 2, 3])))
    assert a != b


def test_task_axis() -> None:
    assert _fp(task=Task(kind="binary")) != _fp(task=Task(kind="multiclass"))


def test_positive_label_axis() -> None:
    assert _fp(task=Task(kind="binary", positive_label=0)) != _fp(
        task=Task(kind="binary", positive_label=1)
    )


def test_native_cat_max_unique_axis() -> None:
    # NFR-6: the cardinality-gate cap is part of the task identity, so changing it (or the default)
    # changes the run-fingerprint -> no silent stale-cache reuse under a different routing.
    assert _fp(task=Task(kind="binary", native_cat_max_unique=8)) != _fp(
        task=Task(kind="binary", native_cat_max_unique=16)
    )
    assert _fp(task=Task(kind="binary", native_cat_max_unique=None)) != _fp(
        task=Task(kind="binary", native_cat_max_unique=16)
    )


def test_data_signature_axis() -> None:
    assert _fp(data_signature="A") != _fp(data_signature="B")


def test_estimators_axis() -> None:
    assert _fp(estimators=("catboost", "lightgbm")) != _fp(
        estimators=("catboost", "lightgbm", "xgboost")
    )


def test_lib_version_axis() -> None:
    assert _fp(lib_versions={"numpy": "1.26.0", "scikit-learn": "1.4.0"}) != _fp(
        lib_versions={"numpy": "2.0.0", "scikit-learn": "1.4.0"}
    )


def test_honestml_version_axis(monkeypatch: pytest.MonkeyPatch) -> None:
    a = _fp()
    monkeypatch.setattr("honestml.application.run_report.version", lambda pkg: "9.9.9-test")
    assert _fp() != a


def test_fingerprint_version_axis(monkeypatch: pytest.MonkeyPatch) -> None:
    a = _fp()
    monkeypatch.setattr(
        "honestml.application.run_report.FINGERPRINT_VERSION", FINGERPRINT_VERSION + 1
    )
    assert _fp() != a


# --- collect_lib_versions helper (ADR-0035 §1: missing package -> null, never raises) ---


def test_lib_versions_known_package() -> None:
    out = collect_lib_versions(["numpy"])
    assert out["numpy"] is not None  # numpy is installed


def test_lib_versions_missing_package_is_null() -> None:
    out = collect_lib_versions(["definitely-not-a-real-package-xyz"])
    assert out == {"definitely-not-a-real-package-xyz": None}  # fail-soft, not an exception
