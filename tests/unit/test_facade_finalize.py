"""M8-3: finalize / refit_full — ship the winner on all data under honest holdout (ADR-0068, FR-SRV-4).

finalize refits the shipped model on DEV+holdout AFTER the honest holdout score is taken from the DEV model;
the reported score stays the DEV estimate, ``shipped_on`` records the data the shipped model saw, and finalize
is post-selection so it never changes the run-fingerprint (off == M7).
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.base import clone

from honestml import AutoML
from honestml.composition.artifact import load_artifact, save_artifact
from honestml.core import CVConfig, EnsembleConfig, FeatureSelectionConfig, Task

pytestmark = pytest.mark.unit

_MODELS = ("baseline", "linear")


def _data(task: str = "binary", n: int = 150):
    if task == "regression":
        from sklearn.datasets import make_regression

        return make_regression(n_samples=n, n_features=6, n_informative=4, random_state=0)
    from sklearn.datasets import make_classification

    return make_classification(
        n_samples=n, n_features=6, n_informative=4, n_redundant=0, random_state=0
    )


def _fit(task: str = "binary", *, holdout: float = 0.3, finalize: bool = True, **kw):
    X, y = _data(task)
    cv = CVConfig(scheme="holdout", outer_holdout=holdout) if holdout else None
    model = AutoML(task=task, models=_MODELS, cv=cv, random_state=0, finalize=finalize, **kw).fit(
        X, y
    )
    return model, X, y


# --- finalize refits on all data, honest score preserved (ADR-0068 §1/§2) -------------------------


def test_finalize_refits_on_all_data() -> None:
    final, X, _ = _fit(finalize=True)
    dev, _, _ = _fit(finalize=False)
    assert final.shipped_on_ == "all" and dev.shipped_on_ == "dev"
    # the shipped model saw more rows -> its probabilities differ from the DEV-only refit
    assert not np.allclose(final.predict_proba(X), dev.predict_proba(X))


def test_holdout_score_unchanged_by_finalize() -> None:
    final, _, _ = _fit(finalize=True)
    dev, _, _ = _fit(finalize=False)
    # the reported holdout score is taken from the DEV model in BOTH cases (finalize runs after scoring)
    assert final.holdout_score_ == dev.holdout_score_


def test_shipped_on_all_in_manifest(tmp_path) -> None:
    final, _, _ = _fit(finalize=True)
    save_artifact(final.fitted_, tmp_path / "art")
    manifest = json.loads((tmp_path / "art" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["shipped_on"] == "all"
    assert load_artifact(tmp_path / "art").shipped_on == "all"


def test_finalize_false_keeps_dev_refit(tmp_path) -> None:
    dev, _, _ = _fit(finalize=False)
    assert dev.shipped_on_ == "dev"
    save_artifact(dev.fitted_, tmp_path / "art")
    manifest = json.loads((tmp_path / "art" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["shipped_on"] == "dev"


# --- no-op without holdout + fingerprint invariance (ADR-0068 §1, NFR-SRV-4) ----------------------


def test_no_holdout_is_noop() -> None:
    model, _, _ = _fit(holdout=0.0, finalize=True)  # default cv, no outer holdout
    assert model.shipped_on_ == "dev"  # finalize is a no-op when there is nothing carved out


def test_finalize_does_not_change_fingerprint() -> None:
    """finalize is post-selection -> not in the run-fingerprint: on/off give an identical hash (off == M7)."""
    on, _, _ = _fit(holdout=0.3, finalize=True)
    off, _, _ = _fit(holdout=0.3, finalize=False)
    assert on.run_report_["run_fingerprint"] == off.run_report_["run_fingerprint"]


# --- regression: finalize valid, calibrator invariant is a no-op (ADR-0068 §3) --------------------


def test_regression_finalize_refits_on_all_data() -> None:
    final, X, _ = _fit("regression", finalize=True)
    dev, _, _ = _fit("regression", finalize=False)
    assert final.shipped_on_ == "all" and dev.shipped_on_ == "dev"
    assert not np.allclose(final.predict(X), dev.predict(X))
    assert final.holdout_score_ == dev.holdout_score_


# --- ensemble + finalize: members re-shipped on all data (ADR-0068 §4) ----------------------------


def test_ensemble_with_finalize_ships_on_all() -> None:
    """An ensemble run under honest holdout finalizes on all data (members or the single fallback, §4)."""
    X, y = _data("binary", n=160)
    model = AutoML(
        task="binary",
        models=_MODELS,
        cv=CVConfig(scheme="holdout", outer_holdout=0.3),
        ensemble=EnsembleConfig(),
        random_state=0,
        finalize=True,
    ).fit(X, y)
    assert model.shipped_on_ == "all"
    # the run report's serving + ensemble blocks are coherent (gate decision is never silent)
    assert model.run_report_["serving"]["shipped_on"] == "all"
    assert model.run_report_["ensemble"] is not None  # ensemble block present (applied or not)


# --- serving provenance + clone (ADR-0068 §1/§5, NFR-SRV-5, FR-SRV-5) ------------------------------


def test_serving_provenance_in_report() -> None:
    model, _, _ = _fit(finalize=True)
    serving = model.run_report_["serving"]
    assert serving == {"finalize": True, "shipped_on": "all", "outer_holdout": pytest.approx(0.3)}


def test_serving_absent_when_selection() -> None:
    X, y = _data("binary")
    model = AutoML(task="binary", models=_MODELS, run_mode="selection", random_state=0).fit(X, y)
    assert model.run_report_["serving"] is None  # no model shipped -> no serving provenance


def test_clone_preserves_finalize_param() -> None:
    est = AutoML(task="binary", finalize=False)
    assert "finalize" in est.get_params()
    assert clone(est).finalize is False


# --- train==inference under finalize: feature-selection subset projected onto ds_full (ADR-0068 §5) ---


def test_finalize_with_feature_selection_predicts() -> None:
    """finalize must refit on the SELECTED subset (ds_full projected), else the shipped model's feature
    count diverges from its schema and predict breaks (ADR-0045 §2 train==inference; regression of the
    ds_full-projection blocker)."""
    from sklearn.datasets import make_classification

    # 3 informative + 3 pure-noise so the top-3 subset captures all signal: it lands comfortably
    # inside the significance band vs all-6 on every platform. The shared 4-informative data put the
    # gate's keep decision on the band boundary, which flipped under macOS Accelerate BLAS floats.
    X, y = make_classification(
        n_samples=200, n_features=6, n_informative=3, n_redundant=0, random_state=0
    )
    fs = FeatureSelectionConfig(strategy="random_probe", cutoff="top_k", top_k=3)
    model = AutoML(
        task="binary",
        models=("linear",),
        cv=CVConfig(scheme="holdout", outer_holdout=0.3),
        feature_selection=fs,
        random_state=0,
        finalize=True,
    ).fit(X, y)
    assert model.shipped_on_ == "all"
    assert model.schema_.selected_features is not None and len(model.schema_.selected_features) == 3
    model.predict(X)  # must not raise on the feature-count contract
    assert model.predict_proba(X).shape == (len(y), 2)


def test_finalize_deterministic_same_seed() -> None:
    a, X, _ = _fit(finalize=True)
    b, _, _ = _fit(finalize=True)
    assert np.array_equal(a.predict(X), b.predict(X))
    assert np.allclose(a.predict_proba(X), b.predict_proba(X))


def test_dev_unseen_class_detaches_calibrator() -> None:
    """ADR-0068 §3: a class absent from DEV invalidates the DEV-OOF calibrator -> detach + applied=False
    (no silent provenance, NFR-SRV-5)."""
    from honestml.adapters import Reader

    model, _, _ = _fit(finalize=False)
    model.fitted_.calibrator = object()  # pretend a calibrator was fit on DEV-OOF
    model.fitted_.calibration = {"applied": True, "method": "sigmoid"}
    Xd, yd = _data("binary", n=40)  # target is {0, 1} — a global class 2 is absent from DEV
    dev = Reader(Task(kind="multiclass")).read(Xd, yd)

    model._detach_dev_calibrator_if_unseen_class(dev, np.array([0, 1, 2]))
    assert model.fitted_.calibrator is None
    assert model.fitted_.calibration["applied"] is False
    assert model.fitted_.calibration["reason"] == "dev_unseen_class"
