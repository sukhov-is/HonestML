"""M7a-E: HPO end-to-end through the AutoML facade (ADR-0061/0062).

The HPO wiring is exercised against a FAKE tunable plugin model (sklearn-backed, with a declared
search_space) registered via entry-points — so the facade stage, write-back, report and fingerprint
are tested WITHOUT requiring a heavy boosting extra, consistent with the project's fake-based tests.
The real OptunaTuner does the searching (optuna is a dev dependency).
"""

from __future__ import annotations

import importlib.util

import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from honestml import AutoML, HPOConfig
from honestml.composition import registry as reg
from honestml.composition.build import build_default_components
from honestml.composition.registry import ComponentDescriptor
from honestml.core import (
    BudgetConfig,
    Capabilities,
    ConfigError,
    MissingDependencyError,
    ModelSpec,
    NotFittedError,
    Task,
)

pytestmark = pytest.mark.unit
pytest.importorskip("optuna")

_TUNABLE_SPACE = {"C": {"type": "float", "low": 0.01, "high": 100.0, "log": True}}


class _TunableClf:
    """A probabilistic, tunable sklearn-backed classifier (one knob: C)."""

    def __init__(self, *, random_state: int = 0, C: float = 1.0) -> None:
        self.random_state = random_state
        self.C = C
        self.feature_names: list[str] = []
        self._m: LogisticRegression | None = None
        self.classes_ = None

    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None):
        self._m = LogisticRegression(C=self.C, max_iter=500, random_state=self.random_state)
        self._m.fit(X, y, sample_weight=sample_weight)
        self.classes_ = self._m.classes_
        return self

    def predict(self, X):
        return self._m.predict(X)

    def predict_proba(self, X):
        return self._m.predict_proba(X)


def _build_tunable(*, task, random_state, **params):
    return _TunableClf(random_state=random_state, **params)


def _tunable_descriptor() -> ComponentDescriptor:
    return ComponentDescriptor(
        name="tunable",
        spec=ModelSpec(
            name="tunable",
            capabilities=Capabilities(tasks=("binary", "multiclass"), probabilistic=True),
            search_space=_TUNABLE_SPACE,
        ),
        build=_build_tunable,
    )


class _FakeEP:
    def __init__(self, d) -> None:
        self._d = d

    def load(self):
        return self._d


@pytest.fixture(autouse=True)
def _register_tunable(monkeypatch):
    monkeypatch.setattr(reg, "entry_points", lambda group=None: [_FakeEP(_tunable_descriptor())])


def _data(n: int = 200, seed: int = 0):
    return make_classification(
        n_samples=n, n_features=6, n_informative=4, n_redundant=0, random_state=seed
    )


def _fit(hpo=None, *, seed: int = 0, run_mode: str = "full", **kw):
    X, y = _data(seed=seed)
    return AutoML(models=("tunable",), random_state=seed, hpo=hpo, run_mode=run_mode, **kw).fit(
        X, y
    )


def test_hpo_tunes_and_writes_back() -> None:
    m = _fit(HPOConfig(n_trials=5, inner_cv=2))
    block = m.run_report_["hpo"]
    assert block is not None and "C" in block["tuned"]["tunable"]["chosen_params"]
    assert "tunable" in [e.model_id for e in m.leaderboard_]  # replaced in place (default)
    assert m.predict(_data()[0]).shape[0] == 200  # the tuned winner ships and predicts


def test_hpo_block_value_assertions() -> None:
    # FR-HPO-6 / NFR-M7-6/7: the block is content-checked, not just present
    m = _fit(HPOConfig(n_trials=5, inner_cv=3))
    block = m.run_report_["hpo"]
    assert block["selection_oof_is_post_tuning"] is True
    assert block["tuned_on_full_feature_space"] is False  # no FS in this run
    assert block["deterministic"] is True  # trials-mode (no timeout)
    assert block["cost_estimate_fits"] == 1 * 5 * 3  # Σ n_models × n_trials × inner_cv (NFR-M7-7)
    assert block["tuned"]["tunable"]["n_trials_run"] == 5


def test_deterministic_false_under_time_budget() -> None:
    # F042 / ADR-0062 §7: a time budget makes HPO non-deterministic (wall-clock decides how many trials
    # finish), even with hpo.timeout_s=None — the per-model fair-share cap imposes a finite Optuna
    # timeout. The report must surface deterministic=False, not silently claim reproducibility.
    m = _fit(
        HPOConfig(n_trials=5, inner_cv=2), budget=BudgetConfig(mode="time", time_budget_s=60.0)
    )
    assert m.run_report_["hpo"]["deterministic"] is False


def test_hpo_absent_when_off() -> None:
    assert _fit(None).run_report_["hpo"] is None


def test_hpo_none_stable_offhash() -> None:
    a, b = _fit(None), _fit(None)
    assert a.run_report_["run_fingerprint"] == b.run_report_["run_fingerprint"]
    assert _fit(HPOConfig()).run_report_["run_fingerprint"] != a.run_report_["run_fingerprint"]


def test_changed_hpo_changes_hash() -> None:
    a = _fit(HPOConfig(n_trials=3, inner_cv=2))
    b = _fit(HPOConfig(n_trials=6, inner_cv=2))
    assert a.run_report_["run_fingerprint"] != b.run_report_["run_fingerprint"]


def test_hpo_seed_deterministic() -> None:
    a = _fit(HPOConfig(n_trials=6, inner_cv=2), seed=11)
    b = _fit(HPOConfig(n_trials=6, inner_cv=2), seed=11)
    pa = a.run_report_["hpo"]["tuned"]["tunable"]["chosen_params"]
    pb = b.run_report_["hpo"]["tuned"]["tunable"]["chosen_params"]
    assert pa == pb  # trials-mode determinism (NFR-M7-2)


def test_selection_mode_runs_hpo() -> None:
    m = _fit(HPOConfig(n_trials=4, inner_cv=2), run_mode="selection")
    assert m.run_report_["hpo"] is not None  # HPO runs in selection mode too (ADR-0062 §2b)
    with pytest.raises(NotFittedError):
        m.predict(_data()[0])  # selection ships no model


def test_keep_baseline_augments() -> None:
    m = _fit(HPOConfig(n_trials=4, inner_cv=2, keep_baseline=True))
    ids = {e.model_id for e in m.leaderboard_}
    assert {"tunable", "tunable__tuned"} <= ids  # baseline kept alongside the tuned candidate


def test_clone_preserves_hpo_param() -> None:
    hpo = HPOConfig(n_trials=7, inner_cv=2)
    est = AutoML(models=("tunable",), hpo=hpo)
    assert clone(est).get_params()["hpo"] == hpo


def test_missing_optuna_raises_missing_dependency(monkeypatch) -> None:
    orig = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda n, *a, **k: None if n == "optuna" else orig(n, *a, **k)
    )
    with pytest.raises(MissingDependencyError):
        _fit(HPOConfig(n_trials=2, inner_cv=2))


def test_make_factory_rejects_unknown_key() -> None:
    # FR-HPO-3: a tuned key absent from the search_space fails loud at composition-time (not dropped)
    comp = build_default_components(
        Task(kind="binary"), random_state=0, models=("tunable",), hpo=HPOConfig()
    )
    assert comp.make_factory is not None
    comp.make_factory("tunable", {"C": 1.0})  # declared key: ok
    with pytest.raises(ConfigError):
        comp.make_factory("tunable", {"bogus": 1.0})


def test_explicit_hpo_seed_zero_in_config() -> None:
    # ADR-0062 §5/§7: an explicit tuning seed of 0 survives resolution into the fingerprinted config
    m = AutoML(
        models=("tunable",), random_state=42, hpo=HPOConfig(n_trials=2, inner_cv=2, random_state=0)
    )
    m.fit(*_data())
    assert m.run_report_["config"]["hpo"]["random_state"] == 0
