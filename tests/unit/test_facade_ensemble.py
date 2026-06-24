"""M7b-D: ensembling end-to-end through the AutoML facade (FR-ENS-3/4/5/6, ADR-0063/0064).

The honest gate and the BlendedEstimator wiring are exercised against the default built-in models
(baseline+linear, two candidates) and, for the deterministic "ships a blend" path, two complementary
fake plugins whose individual signals are partial so the blend strictly improves under significance='off'.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from honestml import AutoML, EnsembleConfig
from honestml.adapters import BlendedEstimator
from honestml.composition import registry as reg
from honestml.composition.artifact import load_artifact, save_artifact
from honestml.composition.registry import ComponentDescriptor
from honestml.core import Capabilities, ConfigError, ModelSpec, NotFittedError

pytestmark = pytest.mark.unit

_GATE_REASONS = {
    "significant_improvement",
    "equivalent_to_best",
    "worse_than_best",
    "no_proba_channel",
    "single_candidate",
    "degenerate_recipe",
    "insufficient_members_after_refit",
}


def _data(n: int = 80, seed: int = 0):
    from sklearn.datasets import make_classification

    return make_classification(
        n_samples=n, n_features=6, n_informative=4, n_redundant=0, random_state=seed
    )


def test_ensemble_absent_when_off() -> None:
    X, y = _data()
    m = AutoML(task="binary", models=("baseline", "linear"), random_state=0).fit(X, y)
    assert m.run_report_["ensemble"] is None


def test_ensemble_does_not_change_selection() -> None:
    # FR-ENS-4: the ensemble runs AFTER selection -> the leaderboard/winner is identical to ensemble=None
    X, y = _data()
    off = AutoML(task="binary", models=("baseline", "linear"), random_state=0).fit(X, y)
    on = AutoML(
        task="binary", models=("baseline", "linear"), random_state=0, ensemble=EnsembleConfig()
    ).fit(X, y)
    assert [e.model_id for e in on.leaderboard_] == [e.model_id for e in off.leaderboard_]
    assert on.best_model_id_ == off.best_model_id_
    assert [e.score for e in on.leaderboard_] == [e.score for e in off.leaderboard_]


def test_ensemble_block_in_report_has_reason() -> None:
    # NFR-M7-6: no silent gate — the decision + a concrete reason is always surfaced
    X, y = _data()
    m = AutoML(
        task="binary", models=("baseline", "linear"), random_state=0, ensemble=EnsembleConfig()
    ).fit(X, y)
    block = m.run_report_["ensemble"]
    assert block is not None and isinstance(block["applied"], bool)
    assert block["gate_reason"] in _GATE_REASONS
    assert pytest.approx(sum(block["weights"].values())) == 1.0


def test_ensemble_hard_label_metric_does_not_crash() -> None:
    # #9: a hard-label metric (accuracy) blends candidate PROBABILITIES; the stage must project the
    # blend to labels before scoring, not pass continuous proba into accuracy_score (which would raise
    # "mix of binary and continuous targets") and kill the whole fit after valid selection work.
    X, y = _data()
    m = AutoML(
        task="binary",
        metric="accuracy",
        models=("baseline", "linear"),
        random_state=0,
        ensemble=EnsembleConfig(),
    ).fit(X, y)
    block = m.run_report_["ensemble"]
    assert block is not None and block["gate_reason"] in _GATE_REASONS
    assert m.predict(X).shape[0] == X.shape[0]


def test_ensemble_stage_failure_degrades_to_single(monkeypatch) -> None:
    # #9(b): the ensemble is optional and post-selection — a blend-stage exception must degrade to
    # an honest "not applied" outcome with a surfaced reason, not abort a fit that already has a winner.
    from honestml.composition import facade as facade_mod

    def _boom(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("ensemble boom")

    monkeypatch.setattr(facade_mod, "ensemble_selection", _boom)
    X, y = _data()
    m = AutoML(
        task="binary", models=("baseline", "linear"), random_state=0, ensemble=EnsembleConfig()
    ).fit(X, y)
    block = m.run_report_["ensemble"]
    assert block is not None and block["applied"] is False
    assert block["gate_reason"].startswith("failed:")
    # the single honest winner still shipped and predicts
    assert m.predict(X).shape[0] == X.shape[0]


def test_ensemble_off_stable_and_changes_hash() -> None:
    # NFR-M7-4: off-hash stable between runs; turning the ensemble on shifts the fingerprint
    X, y = _data()
    ml = ("baseline", "linear")
    a = AutoML(task="binary", models=ml, random_state=0).fit(X, y)
    b = AutoML(task="binary", models=ml, random_state=0).fit(X, y)
    on = AutoML(
        task="binary", models=ml, random_state=0, ensemble=EnsembleConfig(method="weighted")
    ).fit(X, y)
    assert a.run_report_["run_fingerprint"] == b.run_report_["run_fingerprint"]
    assert on.run_report_["run_fingerprint"] != a.run_report_["run_fingerprint"]


def test_selection_mode_reports_recipe_no_model() -> None:
    # ADR-0064 §4: selection mode computes + reports the recipe but ships no model
    X, y = _data()
    m = AutoML(
        task="binary",
        models=("baseline", "linear"),
        random_state=0,
        ensemble=EnsembleConfig(),
        run_mode="selection",
    ).fit(X, y)
    assert m.run_report_["ensemble"] is not None
    with pytest.raises(NotFittedError):
        m.predict(X)


def test_clone_preserves_ensemble_param() -> None:
    cfg = EnsembleConfig(method="weighted", size=10, n_bags=4)
    est = AutoML(task="binary", ensemble=cfg)
    assert clone(est).get_params()["ensemble"] == cfg


def test_ensemble_rejects_bad_type() -> None:
    with pytest.raises(ConfigError):
        AutoML(task="binary", ensemble="caruana").fit(*_data())  # type: ignore[arg-type]


# -- deterministic "ships a blend" via two complementary partial-signal plugins ----


class _PartialClf:
    """A probabilistic classifier that only sees a subset of columns (partial signal)."""

    def __init__(self, *, cols: tuple[int, ...], random_state: int = 0) -> None:
        self.cols = cols
        self.random_state = random_state
        self.feature_names: list[str] = []
        self._m: LogisticRegression | None = None
        self.classes_ = None

    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None):  # noqa: ANN001
        self._m = LogisticRegression(max_iter=500, random_state=self.random_state)
        self._m.fit(X[:, self.cols], y, sample_weight=sample_weight)
        self.classes_ = self._m.classes_
        return self

    def predict(self, X):  # noqa: ANN001
        return self._m.predict(X[:, self.cols])

    def predict_proba(self, X):  # noqa: ANN001
        return self._m.predict_proba(X[:, self.cols])


def _descriptor(name: str, cols: tuple[int, ...]) -> ComponentDescriptor:
    return ComponentDescriptor(
        name=name,
        spec=ModelSpec(
            name=name, capabilities=Capabilities(tasks=("binary", "multiclass"), probabilistic=True)
        ),
        build=lambda *, task, random_state, **params: _PartialClf(
            cols=cols, random_state=random_state
        ),
    )


class _FakeEP:
    def __init__(self, d) -> None:
        self._d = d

    def load(self):
        return self._d


@pytest.fixture
def _complementary(monkeypatch):
    # pa sees only column 0 (x0), pb only column 1 (x1) — each a partial signal, so the blend improves
    eps = [_FakeEP(_descriptor("pa", (0,))), _FakeEP(_descriptor("pb", (1,)))]
    monkeypatch.setattr(reg, "entry_points", lambda group=None: eps)


def _complementary_data(n: int = 400):
    rng = np.random.default_rng(0)
    x0, x1 = rng.normal(size=n), rng.normal(size=n)
    y = (1.6 * x0 + 1.6 * x1 + rng.normal(size=n) > 0).astype(int)
    X = np.column_stack([x0, x1, rng.normal(size=(n, 4))])
    return X, y


def test_ensemble_applied_ships_blended(_complementary, tmp_path) -> None:
    # complementary partial models -> the blend strictly beats the best single; significance='off' applies it
    X, y = _complementary_data()
    m = AutoML(
        task="binary",
        metric="roc_auc",
        models=("pa", "pb"),
        random_state=0,
        significance="off",
        ensemble=EnsembleConfig(n_bags=1),
    ).fit(X, y)
    block = m.run_report_["ensemble"]
    assert block["applied"] is True and block["gate_reason"] == "significant_improvement"
    assert isinstance(m.best_estimator_, BlendedEstimator)
    assert m.predict(X).shape[0] == X.shape[0]
    assert m.predict_proba(X).shape == (X.shape[0], 2)

    art = tmp_path / "ens"
    save_artifact(m.fitted_, art)
    manifest = json.loads((art / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["best_model_id"] == "ensemble" and manifest["ensemble"]["applied"] is True
    loaded = load_artifact(art)
    assert np.allclose(loaded.predict_proba(X), m.predict_proba(X))
