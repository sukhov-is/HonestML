"""M1b: domain ports are structural contracts (Protocol), satisfied by fakes."""

from __future__ import annotations

import numpy as np
import pytest

from honestml.core import (
    Budget,
    Capabilities,
    Estimator,
    Metric,
    ModelSpec,
    NoSignificanceTest,
    ProbabilisticEstimator,
    SignificanceTest,
    SupportsFeatureImportance,
    SupportsShap,
)

pytestmark = pytest.mark.unit


class _RocAuc:
    name = "roc_auc"
    greater_is_better = True
    needs = "proba"
    optimum = 1.0
    average = None  # ADR-0021: the Metric port carries the multiclass averaging mode
    proper_proba = False  # ADR-0031 §2: not a proper loss (refinement no-op by this gate)

    def score(self, y_true, y_pred, sample_weight=None):
        return 0.5


class _Regressor:
    def __init__(self):
        self.feature_names = ["a", "b"]

    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None):
        return self

    def predict(self, X):
        return np.zeros(len(X))


class _Classifier(_Regressor):
    classes_ = np.array([0, 1])

    def predict_proba(self, X):
        return np.zeros((len(X), 2))

    @property
    def feature_importances(self):
        return np.ones(2)

    def shap_values(self, X):
        return np.zeros((len(X), 2))


def test_metric_protocol() -> None:
    m = _RocAuc()
    assert isinstance(m, Metric)
    assert m.needs == "proba" and m.greater_is_better


def test_estimator_role_interfaces() -> None:
    reg, clf = _Regressor(), _Classifier()
    assert isinstance(reg, Estimator)
    # base Estimator is interchangeable; proba is a separate role-interface (R-4)
    assert not isinstance(reg, ProbabilisticEstimator)
    assert isinstance(clf, ProbabilisticEstimator)
    assert isinstance(clf, SupportsFeatureImportance)
    assert isinstance(clf, SupportsShap)
    assert not isinstance(reg, SupportsShap)


def test_model_spec_capabilities() -> None:
    spec = ModelSpec(
        name="catboost",
        capabilities=Capabilities(
            tasks=("binary", "multiclass", "regression"), handles_cat=True, handles_missing=True
        ),
    )
    assert spec.supports("regression")
    assert spec.supports("binary")
    assert spec.capabilities.handles_missing


def test_no_significance_is_a_significance_test() -> None:
    nst = NoSignificanceTest()
    assert isinstance(nst, SignificanceTest)
    assert nst.equivalent(np.zeros(3), np.ones(3), np.array([0, 1, 0]), alpha=0.05) is False


def test_budget_protocol_with_fake() -> None:
    class _Budget:
        def time_left(self):
            return 10.0

        def consume(self, seconds):
            pass

        @property
        def exhausted(self):
            return False

        @property
        def exhausted_reason(self):
            return None

        def memory_left(self):
            return None

    assert isinstance(_Budget(), Budget)
