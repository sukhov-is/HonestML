"""Regenerate the byte-frozen native artifacts (NFR-SER-5, M8b-1).

The committed artifacts pin the cross-version durability claim: a boosting-library bump that
breaks native ``load_model`` fails CI instead of passing silently (which a same-version
round-trip — or pickle — could not detect). Run from the repo root and commit the result:

    uv run python tests/fixtures/native_artifacts/generate.py

Then update README.md with the printed library versions.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_N_TREES = 8  # tiny bodies: the fixture pins the FORMAT, not model quality


def _data(task_kind: str) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    n = 50
    X = rng.normal(size=(n, 4))
    signal = X[:, 0] + 0.5 * X[:, 1]
    y = signal + 0.1 * rng.normal(size=n) if task_kind == "regression" else (signal > 0).astype(int)
    return X, y


def _fitted_model(lib: str, task_kind: str):
    from honestml.adapters import Reader
    from honestml.adapters.boosting import CATBOOST, LIGHTGBM, XGBOOST, build_boosting
    from honestml.application import design_matrix
    from honestml.composition.artifact import FittedModel
    from honestml.core import Task

    backend = {"xgboost": XGBOOST, "catboost": CATBOOST, "lightgbm": LIGHTGBM}[lib]
    task = Task(kind=task_kind)
    X, y = _data(task_kind)
    ds = Reader(task).read(X, y)
    est = build_boosting(
        backend, task=task, random_state=0, **{backend.n_estimators_kwarg: _N_TREES}
    )
    est.fit(design_matrix(ds), ds.target())
    fitted = FittedModel(
        estimator=est,
        schema=ds.schema,
        task=task,
        metric_name="roc_auc" if task_kind == "binary" else "rmse",
        classes=np.array([0, 1]) if task_kind == "binary" else None,
        leaderboard=[],
        best_model_id=lib,
    )
    return fitted, X


def main() -> None:
    import importlib.metadata as md

    from honestml.composition.artifact import load_artifact, save_artifact

    cases = [
        ("xgboost", "xgboost", "binary"),
        ("catboost", "catboost", "binary"),
        ("lightgbm_clf", "lightgbm", "binary"),
        ("lightgbm_reg", "lightgbm", "regression"),
    ]
    for name, lib, kind in cases:
        out = HERE / name
        shutil.rmtree(out, ignore_errors=True)
        fitted, X = _fitted_model(lib, kind)
        save_artifact(fitted, out / "artifact", honestml_version="fixture", model_format="native")
        loaded = load_artifact(out / "artifact")
        payload: dict[str, list] = {"X": X.tolist()}
        if kind == "binary":
            payload["proba"] = loaded.predict_proba(X).tolist()
        else:
            payload["pred"] = loaded.predict(X).tolist()
        (out / "expected.json").write_text(json.dumps(payload), encoding="utf-8")
        print(f"generated {name}")
    versions = {
        p: md.version(p) for p in ("xgboost", "catboost", "lightgbm", "scikit-learn", "numpy")
    }
    print(json.dumps(versions, indent=2))


if __name__ == "__main__":
    main()
