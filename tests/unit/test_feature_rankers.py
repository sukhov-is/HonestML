"""M6b: default feature-ranker adapters (ADR-0044 §2) — importance + random_probe."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from honestml.adapters import (
    ImportanceRanker,
    NullImportanceRanker,
    RandomProbeRanker,
    ShapRanker,
    make_ranker_fit_predict,
)
from honestml.core import FeatureRanker, MissingDependencyError, Task

pytestmark = pytest.mark.unit

_BIN = Task(kind="binary")
_NOCAT = np.zeros(2, dtype=bool)


def _signal_noise(n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """col 0 perfectly determines y; col 1 is pure noise."""
    rng = np.random.default_rng(0)
    signal = rng.normal(size=n)
    x = np.column_stack([signal, rng.normal(size=n)])
    return x, (signal > 0).astype(int)


def test_rankers_satisfy_port() -> None:
    assert isinstance(ImportanceRanker(_BIN), FeatureRanker)
    assert isinstance(RandomProbeRanker(_BIN), FeatureRanker)


def test_importance_ranks_signal_above_noise() -> None:
    x, y = _signal_noise()
    imp = ImportanceRanker(_BIN).rank(x, y, categorical=_NOCAT, random_state=0)
    assert imp.shape == (2,) and bool(np.all(imp >= 0))
    assert imp[0] > imp[1]


def test_importance_regression_task() -> None:
    # task-kind-agnostic: importances are defined for regression too (ADR-0043 §5)
    rng = np.random.default_rng(1)
    x = rng.normal(size=(150, 2))
    y = 3.0 * x[:, 0] + rng.normal(scale=0.1, size=150)
    imp = ImportanceRanker(Task(kind="regression")).rank(x, y, categorical=_NOCAT, random_state=0)
    assert imp[0] > imp[1]


def test_importance_empty_x_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        ImportanceRanker(_BIN).rank(
            np.empty((0, 2)), np.empty(0), categorical=_NOCAT, random_state=0
        )


def test_random_probe_margin_signal_beats_baseline() -> None:
    x, y = _signal_noise()
    margin = RandomProbeRanker(_BIN, n_probes=5).rank(x, y, categorical=_NOCAT, random_state=0)
    assert margin.shape == (2,)
    assert margin[0] > 0.0  # signal beats the strongest probe
    assert margin[0] > margin[1]  # signal margin above noise margin


def test_random_probe_seeded_deterministic() -> None:
    x, y = _signal_noise()
    r = RandomProbeRanker(_BIN)
    a = r.rank(x, y, categorical=_NOCAT, random_state=7)
    b = r.rank(x, y, categorical=_NOCAT, random_state=7)
    assert np.allclose(a, b)


def test_auto_thresholds() -> None:
    assert ImportanceRanker(_BIN).auto_threshold(4) == 0.25
    assert RandomProbeRanker(_BIN).auto_threshold(4) == 0.0


# --- M6c null_importance (ADR-0047 §1) ---


def test_null_importance_satisfies_port_and_threshold() -> None:
    r = NullImportanceRanker(_BIN, n_runs=10)
    assert isinstance(r, FeatureRanker)
    assert r.auto_threshold(4) == 0.0  # signed margin


def test_null_importance_signal_beats_permuted_null() -> None:
    x, y = _signal_noise()
    margin = NullImportanceRanker(_BIN, n_runs=20).rank(x, y, categorical=_NOCAT, random_state=0)
    assert margin.shape == (2,)
    assert margin[0] > 0.0  # signal importance exceeds its permuted-target null
    assert margin[0] > margin[1]  # signal margin above noise margin


def test_null_importance_seeded_deterministic() -> None:
    x, y = _signal_noise()
    r = NullImportanceRanker(_BIN, n_runs=15)
    a = r.rank(x, y, categorical=_NOCAT, random_state=3)
    b = r.rank(x, y, categorical=_NOCAT, random_state=3)
    assert np.allclose(a, b)


# --- M6d structure-aware null_importance for group/timeseries (ADR-0050, FR-FSH-1) ---

_REG = Task(kind="regression")


def _group_spurious(
    n_groups: int = 20, per_group: int = 15
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """col0 genuine iid predictor; col1 = per-group constant (group identity) spurious for within-group y."""
    rng = np.random.default_rng(0)
    n = n_groups * per_group
    groups = np.repeat(np.arange(n_groups), per_group)
    intercept = rng.normal(size=n_groups)
    genuine = rng.normal(size=n)
    s_g = rng.normal(size=n_groups)
    spurious = s_g[groups]
    y = 0.6 * genuine + intercept[groups] + 0.3 * rng.normal(size=n)
    return np.column_stack([genuine, spurious]), y, groups


def test_null_importance_groups_none_matches_m6c() -> None:
    # back-compat: groups=None keeps the M6c uniform permutation behavior exactly
    x, y = _signal_noise()
    r = NullImportanceRanker(_BIN, n_runs=15)
    base = r.rank(x, y, categorical=_NOCAT, random_state=3)
    none = r.rank(x, y, categorical=_NOCAT, random_state=3, groups=None)
    assert np.allclose(base, none)


def test_null_importance_within_group_rejects_spurious_group_id() -> None:
    # FR-FSH-1 discriminator (group, decisive): uniform permutation FALSELY keeps the group-id feature;
    # within-group permutation correctly rejects it while keeping the genuine one (SPIKE-M6d-validity).
    x, y, groups = _group_spurious()
    r = NullImportanceRanker(_REG, n_runs=30, null_percentile=95.0)
    uniform = r.rank(x, y, categorical=_NOCAT, random_state=0)
    within = r.rank(x, y, categorical=_NOCAT, random_state=0, groups=groups)
    assert uniform[1] > 0.0  # uniform: spurious group-id kept (invalid null)
    assert within[1] <= 0.0  # within-group: spurious rejected (valid null)
    assert within[0] > 0.0  # genuine survives the structure-aware null


def test_null_importance_within_group_deterministic() -> None:
    x, y, groups = _group_spurious()
    r = NullImportanceRanker(_REG, n_runs=10)
    a = r.rank(x, y, categorical=_NOCAT, random_state=5, groups=groups)
    b = r.rank(x, y, categorical=_NOCAT, random_state=5, groups=groups)
    assert np.allclose(a, b)


# --- M6d shap ranker (ADR-0051, FR-FSH-4/5) ---


def test_shap_ranker_missing_dependency_raises() -> None:
    if importlib.util.find_spec("shap") is not None:
        pytest.skip("shap installed; the missing-dependency path is not exercised in this env")
    with pytest.raises(MissingDependencyError, match="shap"):
        ShapRanker(_BIN)


def test_shap_ranker_binary_ranks_signal_above_noise() -> None:
    pytest.importorskip("shap")
    x, y = _signal_noise()
    r = ShapRanker(_BIN)
    assert isinstance(r, FeatureRanker)
    imp = r.rank(x, y, categorical=_NOCAT, random_state=0)
    assert imp.shape == (2,) and bool(np.all(imp >= 0))
    assert imp[0] > imp[1]


def test_shap_ranker_regression_and_multiclass_aggregate() -> None:
    pytest.importorskip("shap")
    rng = np.random.default_rng(1)
    xr = rng.normal(size=(120, 2))
    yr = 3.0 * xr[:, 0] + rng.normal(scale=0.1, size=120)
    impr = ShapRanker(_REG).rank(xr, yr, categorical=_NOCAT, random_state=0)
    assert impr.shape == (2,) and impr[0] > impr[1]
    xc = rng.normal(size=(150, 3))
    yc = (xc[:, 0] > 0).astype(int) + (xc[:, 1] > 0).astype(int)  # 3 classes
    impc = ShapRanker(Task(kind="multiclass")).rank(
        xc, yc, categorical=np.zeros(3, dtype=bool), random_state=0
    )
    assert impc.shape == (3,) and bool(np.all(impc >= 0))  # |shap| aggregated over classes -> 1-D


def test_shap_ranker_max_samples_caps_and_threshold() -> None:
    pytest.importorskip("shap")
    x, y = _signal_noise(200)
    imp = ShapRanker(_BIN, max_samples=50).rank(x, y, categorical=_NOCAT, random_state=0)
    assert imp.shape == (2,)
    assert ShapRanker(_BIN).auto_threshold(4) == 0.25


# --- M6e interventional shap background (ADR-0056) ---


def test_background_is_evenly_spaced_and_deterministic() -> None:
    # _background needs no shap: linspace indices spread over the whole row order (not a head-slice), and
    # return x unchanged when the cap covers every row or is None (ADR-0056 §2, fix R2).
    from honestml.adapters.feature_rankers import _background

    x = np.arange(100).reshape(100, 1).astype(float)
    bg = _background(x, 5)
    assert bg[:, 0].tolist() == [0.0, 24.0, 49.0, 74.0, 99.0]  # spans 0..99, not the leading 5
    assert _background(x, 200) is x and _background(x, None) is x  # k>=n / None -> full, no copy


def test_shap_interventional_is_deterministic_and_default_is_tpd() -> None:
    pytest.importorskip("shap")
    x, y = _signal_noise(200)
    r = ShapRanker(_BIN, perturbation="interventional", background_samples=40)
    a = r.rank(x, y, categorical=_NOCAT, random_state=0)
    b = r.rank(x, y, categorical=_NOCAT, random_state=0)
    assert a.shape == (2,) and bool(np.all(a >= 0))
    assert np.allclose(
        a, b
    )  # deterministic linspace background -> reproducible without a seed (NFR-FSE-5)
    # default stays tree_path_dependent (M6d); both produce a valid non-negative score vector
    tpd = ShapRanker(_BIN).rank(x, y, categorical=_NOCAT, random_state=0)
    assert tpd.shape == (2,) and bool(np.all(tpd >= 0))


def test_shap_explainer_wiring_passes_data_only_for_interventional(monkeypatch) -> None:
    # FR-FSE-9 acceptance: interventional builds TreeExplainer WITH data=<background>; tree_path_dependent
    # builds it WITHOUT data= (тождественно M6d). Spy on shap.TreeExplainer to capture the kwargs.
    shap = pytest.importorskip("shap")
    captured: list[dict] = []
    real = shap.TreeExplainer

    def _spy(model, **kw):
        captured.append(kw)
        return real(model, **kw)

    monkeypatch.setattr(shap, "TreeExplainer", _spy)
    x, y = _signal_noise(120)
    ShapRanker(_BIN).rank(x, y, categorical=_NOCAT, random_state=0)
    ShapRanker(_BIN, perturbation="interventional", background_samples=20).rank(
        x, y, categorical=_NOCAT, random_state=0
    )
    assert (
        "data" not in captured[0] and captured[0]["feature_perturbation"] == "tree_path_dependent"
    )
    assert "data" in captured[1] and captured[1]["feature_perturbation"] == "interventional"
    assert captured[1]["data"].shape == (
        20,
        x.shape[1],
    )  # linspace-sampled background of the requested size


# --- M6f kmeans interventional background (ADR-0060) ---


def test_kmeans_background_is_deterministic() -> None:
    # seeded KMeans centroids -> reproducible (SPIKE-M6f-shap-bg deterministic=True); full x when k>=n/None.
    from honestml.adapters.feature_rankers import _kmeans_background

    x = np.random.default_rng(0).standard_normal((200, 4))
    a = _kmeans_background(x, 8, seed=0)
    assert a.shape == (8, 4) and np.allclose(a, _kmeans_background(x, 8, seed=0))
    assert _kmeans_background(x, 500, 0) is x and _kmeans_background(x, None, 0) is x


def test_kmeans_explainer_wiring_uses_centroids(monkeypatch) -> None:
    # FR-FSF-4: shap_background="kmeans" feeds CENTROIDS (not linspace rows) as the interventional background,
    # seeded by the rank-time random_state (no __init__ seed).
    shap = pytest.importorskip("shap")
    from honestml.adapters.feature_rankers import _kmeans_background

    captured: list[dict] = []
    real = shap.TreeExplainer
    monkeypatch.setattr(
        shap, "TreeExplainer", lambda model, **kw: (captured.append(kw), real(model, **kw))[1]
    )
    x, y = _signal_noise(150)
    ShapRanker(
        _BIN, perturbation="interventional", background_samples=12, shap_background="kmeans"
    ).rank(x, y, categorical=_NOCAT, random_state=3)
    assert captured[0]["data"].shape == (12, x.shape[1])
    assert np.allclose(
        captured[0]["data"], _kmeans_background(x, 12, seed=3)
    )  # centroids @ rank-time seed


def test_mean_abs_per_feature_normalizes_all_shap_shapes() -> None:
    # the risky aggregation (no shap needed): per-class list, single 2-D, and 3-D all -> (p,) >= 0
    from honestml.adapters.feature_rankers import _mean_abs_per_feature

    p = 3
    arr2d = np.array([[1.0, -2.0, 0.0], [-1.0, 2.0, 4.0], [0.5, -0.5, -3.0]])  # (n=3, p=3)
    out2 = _mean_abs_per_feature(arr2d, p)
    out_list = _mean_abs_per_feature([arr2d, -arr2d], p)  # binary/multiclass list per class
    out3 = _mean_abs_per_feature(np.stack([arr2d, -arr2d], axis=2), p)  # (n, p, n_classes)
    for out in (out2, out_list, out3):
        assert out.shape == (p,) and bool(np.all(out >= 0))
    assert np.allclose(out2, np.abs(arr2d).mean(axis=0))  # 2-D path = mean(|.|) over rows


def test_ranker_fit_predict_classification_and_regression() -> None:
    x, y = _signal_noise(120)
    proba, pred, classes = make_ranker_fit_predict(_BIN)(x[:80], y[:80], x[80:], None, 0)
    assert (
        proba is not None and proba.shape == (40, 2) and classes is not None and pred.shape == (40,)
    )
    rng = np.random.default_rng(2)
    xr = rng.normal(size=(100, 2))
    yr = xr[:, 0] + rng.normal(scale=0.1, size=100)
    rproba, rpred, rclasses = make_ranker_fit_predict(Task(kind="regression"))(
        xr[:70], yr[:70], xr[70:], None, 0
    )
    assert rproba is None and rclasses is None and rpred.shape == (30,)
