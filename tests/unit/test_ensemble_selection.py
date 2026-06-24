"""M7b-D: the ``ensemble_selection`` honest choose_better gate + refit_members (FR-ENS-3/4, ADR-0063 §5)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_regression

from honestml.adapters import CaruanaEnsembler, Reader, resolve_metric
from honestml.application import ensemble_selection, refit_members
from honestml.core import Candidate, EnsembleRecipe, NoSignificanceTest, SelectionPolicy, Task

pytestmark = pytest.mark.unit


class _Rmse:
    """A minimal value metric (lower is better) for the regression blend tests."""

    name = "rmse"
    greater_is_better = False
    needs = "value"
    optimum = 0.0
    average = None
    proper_proba = False

    def score(self, y_true, y_pred, sample_weight=None):  # noqa: ANN001
        return float(np.sqrt(np.average((y_true - y_pred) ** 2, weights=sample_weight)))


class _FakeSig:
    """A ``SignificanceTest`` whose verdict is fixed, to drive the gate deterministically."""

    seed = 0
    n_boot = 0

    def __init__(self, equivalent: bool) -> None:
        self._equivalent = equivalent

    def equivalent(self, a, b, y, *, alpha, block_index=None, sample_weight=None):  # noqa: ANN001
        return self._equivalent


class _AllOnFirst:
    """A degenerate ensembler: all weight on the first member."""

    name = "degenerate"

    def combine(self, oof, y, *, score, member_ids, random_state, sample_weight=None):  # noqa: ANN001
        ids = tuple(member_ids)
        return EnsembleRecipe(
            {m: (1.0 if i == 0 else 0.0) for i, m in enumerate(ids)}, self.name, ids
        )


class _FixedEqual:
    """An ensembler that always returns equal weights (so a worse blend is not auto-degenerated)."""

    name = "fixed"

    def combine(self, oof, y, *, score, member_ids, random_state, sample_weight=None):  # noqa: ANN001
        ids = tuple(member_ids)
        w = 1.0 / len(ids)
        return EnsembleRecipe({m: w for m in ids}, self.name, ids)


_TASK = Task(kind="regression")
_POLICY = SelectionPolicy(greater_is_better=False)
# a + b blended 0.5/0.5 has lower RMSE than either single (the blend genuinely improves)
_Y = np.array([0.0, 0.0, 1.0, 1.0])
_A = Candidate(
    id="a", score=-0.707, oof_pred=np.array([0.0, 0.0, 2.0, 2.0]), oof_mask=np.ones(4, bool)
)
_B = Candidate(
    id="b", score=-0.707, oof_pred=np.array([1.0, 1.0, 1.0, 1.0]), oof_mask=np.ones(4, bool)
)


def _select(*, sig, mode, ensembler=None, candidates=(_A, _B), best="a"):
    return ensemble_selection(
        list(candidates),
        _TASK,
        y=_Y,
        best_model_id=best,
        ensembler=ensembler or CaruanaEnsembler(size=5, n_bags=1),
        metric=_Rmse(),
        significance_test=sig,
        policy=_POLICY,
        significance_mode=mode,
        random_state=0,
    )


def test_significant_ensemble_applied() -> None:
    out = _select(sig=_FakeSig(equivalent=False), mode="bootstrap")
    assert out.applied is True and out.gate_reason == "significant_improvement"
    assert pytest.approx(sum(out.weights.values())) == 1.0 and out.oof_delta is not None


def test_equivalent_ensemble_not_applied() -> None:
    out = _select(sig=_FakeSig(equivalent=True), mode="bootstrap")
    assert out.applied is False and out.gate_reason == "equivalent_to_best"


def test_significance_off_strict_gt_gate() -> None:
    # off mode ignores the test: ship iff the blend strictly beats the best single (legacy semantics)
    better = _select(sig=NoSignificanceTest(), mode="off")
    assert better.applied is True and better.gate_reason == "significant_improvement"


def test_significance_off_worse_blend_not_applied() -> None:
    # a fixed-equal blend of a perfect + a poor member is worse than the perfect single -> not shipped
    strong = Candidate(id="a", score=0.0, oof_pred=_Y.copy(), oof_mask=np.ones(4, bool))
    weak = Candidate(
        id="b", score=-1.0, oof_pred=np.array([1.0, 1.0, 0.0, 0.0]), oof_mask=np.ones(4, bool)
    )
    out = _select(
        sig=NoSignificanceTest(),
        mode="off",
        ensembler=_FixedEqual(),
        candidates=(strong, weak),
        best="a",
    )
    assert out.applied is False and out.gate_reason == "worse_than_best"


def test_single_candidate_skips_with_reason() -> None:
    out = _select(sig=_FakeSig(False), mode="bootstrap", candidates=(_A,))
    assert out.applied is False and out.gate_reason == "single_candidate"


def test_no_proba_channel_skips_with_warning() -> None:
    # classification candidates without a proba channel -> blending labels is incorrect, skip (ADR-0063 §2)
    c0 = Candidate(id="a", score=0.6, oof_pred=np.array([0, 1, 0, 1]), oof_mask=np.ones(4, bool))
    c1 = Candidate(id="b", score=0.5, oof_pred=np.array([1, 0, 1, 0]), oof_mask=np.ones(4, bool))
    out = ensemble_selection(
        [c0, c1],
        Task(kind="binary"),
        y=np.array([0, 1, 0, 1]),
        best_model_id="a",
        ensembler=CaruanaEnsembler(),
        metric=_Rmse(),
        significance_test=_FakeSig(False),
        policy=SelectionPolicy(),
        significance_mode="bootstrap",
    )
    assert out.applied is False and out.gate_reason == "no_proba_channel"


def test_degenerate_recipe_ships_single() -> None:
    out = _select(sig=_FakeSig(False), mode="bootstrap", ensembler=_AllOnFirst())
    assert out.applied is False and out.gate_reason == "degenerate_recipe"


class _LogLoss:
    """A minimal multiclass proba metric (lower is better) over an (n, K) matrix."""

    name = "log_loss"
    greater_is_better = False
    needs = "proba"
    optimum = 0.0
    average = None
    proper_proba = True

    def score(self, y_true, y_pred, sample_weight=None):  # noqa: ANN001
        p = np.clip(y_pred, 1e-15, 1.0)
        return float(-np.mean(np.log(p[np.arange(len(y_true)), y_true])))


_CLASS_GATES = {
    "significant_improvement",
    "equivalent_to_best",
    "worse_than_best",
    "degenerate_recipe",
}


def test_binary_accuracy_blend_projects_to_labels() -> None:
    # ADR-0063 + #9: a hard-label metric must score the blend on labels, not continuous proba.
    # The real sklearn-backed accuracy raises "mix of binary and continuous targets" without the
    # projection, so this faithfully reproduces the crash and proves the fix.
    y = np.array([0, 0, 1, 1, 0, 1])
    a = Candidate(
        id="a",
        score=0.83,
        oof_proba=np.array([0.2, 0.1, 0.9, 0.8, 0.4, 0.6]),
        oof_mask=np.ones(6, bool),
    )
    b = Candidate(
        id="b",
        score=0.83,
        oof_proba=np.array([0.1, 0.3, 0.7, 0.95, 0.45, 0.55]),
        oof_mask=np.ones(6, bool),
    )
    out = ensemble_selection(
        [a, b],
        Task(kind="binary"),
        y=y,
        best_model_id="a",
        ensembler=CaruanaEnsembler(size=5, n_bags=1),
        metric=resolve_metric("accuracy", classes=np.array([0, 1])),
        significance_test=_FakeSig(False),
        policy=SelectionPolicy(),
        significance_mode="bootstrap",
    )
    assert out.gate_reason in _CLASS_GATES


def test_multiclass_accuracy_blend_projects_to_labels() -> None:
    # multiclass hard-label metric: the (n, K) blend must argmax to labels before scoring.
    y = np.array([0, 1, 2, 0, 1, 2])
    conf, flat = 0.8, 1 / 3
    a = np.array(
        [[conf if i == y[r] else (1 - conf) / 2 for i in range(3)] for r in range(3)]
        + [[flat, flat, flat]] * 3
    )
    b = np.array(
        [[flat, flat, flat]] * 3
        + [[conf if i == y[r + 3] else (1 - conf) / 2 for i in range(3)] for r in range(3)]
    )
    out = ensemble_selection(
        [
            Candidate(id="a", score=0.5, oof_proba=a, oof_mask=np.ones(6, bool)),
            Candidate(id="b", score=0.5, oof_proba=b, oof_mask=np.ones(6, bool)),
        ],
        Task(kind="multiclass"),
        y=y,
        best_model_id="a",
        ensembler=CaruanaEnsembler(size=5, n_bags=1),
        metric=resolve_metric("accuracy", classes=np.array([0, 1, 2])),
        significance_test=_FakeSig(False),
        policy=SelectionPolicy(),
        significance_mode="bootstrap",
    )
    assert out.gate_reason in _CLASS_GATES


def test_multiclass_blend_channel_applies() -> None:
    # FR-ENS-2/ADR-0063 §2: the 3-D (m,n,K) proba stack blends to (n,K) and the blend improves log loss
    y = np.array([0, 1, 2, 0, 1, 2])
    conf, flat = 0.8, 1 / 3
    a = np.array(
        [[conf if i == y[r] else (1 - conf) / 2 for i in range(3)] for r in range(3)]
        + [[flat, flat, flat]] * 3
    )
    b = np.array(
        [[flat, flat, flat]] * 3
        + [[conf if i == y[r + 3] else (1 - conf) / 2 for i in range(3)] for r in range(3)]
    )
    cands = [
        Candidate(id="a", score=-0.65, oof_proba=a, oof_mask=np.ones(6, bool)),
        Candidate(id="b", score=-0.65, oof_proba=b, oof_mask=np.ones(6, bool)),
    ]
    out = ensemble_selection(
        cands,
        Task(kind="multiclass"),
        y=y,
        best_model_id="a",
        ensembler=CaruanaEnsembler(size=5, n_bags=1),
        metric=_LogLoss(),
        significance_test=_FakeSig(False),
        policy=SelectionPolicy(greater_is_better=False),
        significance_mode="bootstrap",
    )
    assert out.applied is True and out.gate_reason == "significant_improvement"
    assert pytest.approx(sum(out.weights.values())) == 1.0


# -- refit_members (ADR-0064 §4 drop-and-renormalize) -----------------------


class _GoodMember:
    def __init__(self) -> None:
        self.feature_names: list[str] = []

    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None):  # noqa: ANN001
        return self

    def predict(self, X):  # noqa: ANN001
        return np.zeros(X.shape[0])


class _BadMember(_GoodMember):
    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None):  # noqa: ANN001
        raise RuntimeError("member refit boom")


def test_member_refit_failure_drops_and_renormalizes() -> None:
    X, y = make_regression(n_samples=30, n_features=3, random_state=0)
    ds = Reader(Task(kind="regression")).read(X, y)
    members, kept, dropped = refit_members(
        ds,
        Task(kind="regression"),
        member_ids=("g", "b"),
        factories={"g": lambda: _GoodMember(), "b": lambda: _BadMember()},
    )
    assert kept == ("g",) and dropped == ("b",) and len(members) == 1
