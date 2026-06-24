"""M7b-B/C: ensembler adapters + BlendedEstimator (FR-ENS-2/5, ADR-0063 §3 / ADR-0064 §1)."""

from __future__ import annotations

import numpy as np
import pytest

from honestml.adapters import BlendedEstimator, CaruanaEnsembler, WeightedEnsembler

pytestmark = pytest.mark.unit


def _neg_mse(y: np.ndarray):
    return lambda blended: -float(np.mean((blended - y) ** 2))


def _blend(oof: np.ndarray, recipe) -> np.ndarray:
    w = np.array([recipe.weights[m] for m in recipe.member_ids])
    return np.tensordot(w, oof, axes=([0], [0]))


# -- Caruana ----------------------------------------------------------------


def test_caruana_improves_over_best_single() -> None:
    # two members whose 0.5/0.5 blend has lower MSE than either single (FR-ENS-2)
    y = np.array([0.0, 0.0, 1.0, 1.0])
    oof = np.array([[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 1.0, 1.0]])
    score = _neg_mse(y)
    recipe = CaruanaEnsembler(size=8, n_bags=1).combine(
        oof, y, score=score, member_ids=("a", "b"), random_state=0
    )
    best_single = max(score(oof[0]), score(oof[1]))
    assert score(_blend(oof, recipe)) >= best_single - 1e-9
    assert score(_blend(oof, recipe)) > best_single  # strictly better here


def test_caruana_tiebreak_deterministic() -> None:
    # identical members -> every step ties; the smallest index always wins (ADR-0063 §3)
    y = np.array([0.0, 0.0])
    oof = np.array([[1.0, 1.0], [1.0, 1.0]])
    recipe = CaruanaEnsembler(size=5, n_bags=1).combine(
        oof, y, score=_neg_mse(y), member_ids=("a", "b"), random_state=0
    )
    assert recipe.weights == {"a": 1.0, "b": 0.0}


def test_ensembler_deterministic_seed() -> None:
    # bagged Caruana: same seed -> identical recipe (NFR-M7-2)
    rng = np.random.default_rng(7)
    y = rng.normal(size=40)
    oof = np.stack([y + rng.normal(scale=0.5, size=40) for _ in range(4)])
    ens = CaruanaEnsembler(size=10, n_bags=12)
    a = ens.combine(oof, y, score=_neg_mse(y), member_ids=tuple("abcd"), random_state=3)
    b = ens.combine(oof, y, score=_neg_mse(y), member_ids=tuple("abcd"), random_state=3)
    assert a.weights == b.weights


# -- weighted (SLSQP) -------------------------------------------------------


def test_weighted_slsqp_simplex() -> None:
    y = np.array([0.0, 0.0, 1.0, 1.0])
    oof = np.array([[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 1.0, 1.0]])
    score = _neg_mse(y)
    recipe = WeightedEnsembler().combine(oof, y, score=score, member_ids=("a", "b"), random_state=0)
    w = np.array(list(recipe.weights.values()))
    assert (w >= 0).all() and pytest.approx(w.sum(), abs=1e-6) == 1.0
    assert score(_blend(oof, recipe)) >= max(score(oof[0]), score(oof[1])) - 1e-9


def test_single_member_gets_unit_weight() -> None:
    y = np.array([0.0, 1.0])
    oof = np.array([[0.2, 0.8]])
    for ens in (CaruanaEnsembler(), WeightedEnsembler()):
        recipe = ens.combine(oof, y, score=_neg_mse(y), member_ids=("solo",), random_state=0)
        assert recipe.weights == {"solo": 1.0}


# -- BlendedEstimator -------------------------------------------------------


class _FakeMember:
    """A member whose ``predict_proba`` columns follow a configurable ``classes_`` order."""

    def __init__(self, proba: np.ndarray, classes: np.ndarray) -> None:
        self._proba = np.asarray(proba, dtype=np.float64)
        self.classes_ = np.asarray(classes)
        self.feature_names: list[str] = []

    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None):  # noqa: ANN001
        return self

    def predict(self, X):  # noqa: ANN001
        return self.classes_[np.argmax(self._proba, axis=1)]

    def predict_proba(self, X):  # noqa: ANN001
        return self._proba


def test_blended_classes_equal_global_order_and_aligns_members() -> None:
    # member B stores P in REVERSED class order; BlendedEstimator must align by label, not column (ADR-0064 §1)
    a = _FakeMember(np.array([[0.8, 0.2]]), classes=np.array([0, 1]))
    b = _FakeMember(np.array([[0.3, 0.7]]), classes=np.array([1, 0]))  # P(1)=0.3, P(0)=0.7
    blended = BlendedEstimator([a, b], np.array([0.5, 0.5]), classes=np.array([0, 1]))
    proba = blended.predict_proba(np.zeros((1, 4)))
    assert np.array_equal(blended.classes_, np.array([0, 1]))
    assert proba.shape == (1, 2) and pytest.approx(proba.sum()) == 1.0
    # aligned: A=[0.8,0.2], B->[0.7,0.3]; mean = [0.75,0.25]
    assert proba[0, 1] == pytest.approx(0.25)
    assert blended.predict(np.zeros((1, 4)))[0] == 0  # argmax -> class 0


def test_blended_regression_is_weighted_mean_no_proba() -> None:
    class _RegMember:
        feature_names: list[str] = []

        def __init__(self, v: float) -> None:
            self.v = v

        def fit(self, X, y, X_val=None, y_val=None, sample_weight=None):  # noqa: ANN001
            return self

        def predict(self, X):  # noqa: ANN001
            return np.full(X.shape[0], self.v)

    blended = BlendedEstimator(
        [_RegMember(2.0), _RegMember(4.0)], np.array([0.25, 0.75]), classes=None
    )
    assert blended.predict(np.zeros((3, 2))).tolist() == [3.5, 3.5, 3.5]
    with pytest.raises(ValueError):
        blended.predict_proba(np.zeros((3, 2)))
