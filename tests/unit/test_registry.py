"""M3a (ADR-0019): component registry — discovery, dedup, determinism, laziness."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from honestml.adapters import BaselineClassifier
from honestml.adapters.boosting import CATBOOST, LIGHTGBM, XGBOOST
from honestml.composition import AutoML
from honestml.composition import registry as reg
from honestml.composition.registry import (
    ComponentDescriptor,
    ComponentRegistry,
    available_models,
    model_registry,
)
from honestml.core import (
    Capabilities,
    MissingDependencyError,
    ModelSpec,
    PluginConflictError,
    Task,
    parse_search_space,
)

pytestmark = pytest.mark.unit


# --- M7a HPO: per-component SearchSpace declaration (ADR-0061 §4) ------------


def test_boosting_declares_search_space() -> None:
    specs = {d.name: d.spec for d in model_registry().descriptors()}
    for name in ("catboost", "lightgbm", "xgboost"):
        space = parse_search_space(specs[name].search_space)  # valid + non-empty
        assert space, f"{name} should declare a non-empty search_space"
    # baseline/linear are not tuned in M7a -> empty space => HPO skips them
    assert specs["baseline"].search_space == {}
    assert specs["linear"].search_space == {}


@pytest.mark.parametrize("backend", [CATBOOST, LIGHTGBM, XGBOOST])
def test_search_space_tree_key_matches_backend_kwarg(backend) -> None:
    # the tree-count key MUST equal the backend's n_estimators_kwarg so a tuned tree count overrides
    # the fixed default rather than colliding (catboost='iterations'; lgbm/xgb='n_estimators').
    assert backend.n_estimators_kwarg in backend.search_space
    # and no foreign tree-count alias leaks in (catboost must NOT declare 'n_estimators')
    aliases = {"iterations", "n_estimators"} - {backend.n_estimators_kwarg}
    assert aliases.isdisjoint(backend.search_space)


class _FakeEP:
    """Stand-in for ``importlib.metadata.EntryPoint`` whose ``load`` returns a descriptor."""

    def __init__(self, descriptor: ComponentDescriptor) -> None:
        self._descriptor = descriptor

    def load(self) -> ComponentDescriptor:
        return self._descriptor


def _patch_entry_points(monkeypatch, descriptors: list[ComponentDescriptor]) -> None:
    monkeypatch.setattr(reg, "entry_points", lambda group=None: [_FakeEP(d) for d in descriptors])


_BUILTIN_NAMES = {"baseline", "linear", "catboost", "lightgbm", "xgboost"}


def _descriptor(
    name: str, *, build=None, api_version: int = 1, dist: str = "ext"
) -> ComponentDescriptor:
    return ComponentDescriptor(
        name=name,
        spec=ModelSpec(name=name, capabilities=Capabilities(tasks=("binary",), probabilistic=True)),
        build=build or (lambda **kw: BaselineClassifier()),
        api_version=api_version,
        dist=dist,
    )


def _xy(n: int = 40, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] > 0).astype(int)
    return X, y


# --- discovery -------------------------------------------------------------


def test_builtins_discovered_without_plugins() -> None:
    names = set(model_registry().by_name())
    assert names == _BUILTIN_NAMES


def test_third_party_plugin_discovered(monkeypatch) -> None:
    _patch_entry_points(monkeypatch, [_descriptor("ext_model")])
    names = set(model_registry().by_name())
    assert "ext_model" in names and {"baseline", "linear"} <= names


def test_duplicate_name_raises_plugin_conflict(monkeypatch) -> None:
    _patch_entry_points(monkeypatch, [_descriptor("baseline")])  # collides with the built-in
    with pytest.raises(PluginConflictError, match="duplicate component name"):
        model_registry().descriptors()


def test_enumeration_is_deterministic(monkeypatch) -> None:
    _patch_entry_points(monkeypatch, [_descriptor("zeta"), _descriptor("alpha")])
    names = [d.name for d in model_registry().descriptors()]
    assert names == sorted(names)  # by name, independent of source order


# --- laziness & missing extras --------------------------------------------


def test_discovery_does_not_call_build(monkeypatch) -> None:
    def _boom(**kwargs):
        raise RuntimeError("build must not run during discovery")

    _patch_entry_points(monkeypatch, [_descriptor("ext_lazy", build=_boom)])
    names = set(model_registry().by_name())  # discovery only
    assert "ext_lazy" in names  # listed without building


def test_build_missing_extra_raises_missing_dependency(monkeypatch) -> None:
    def _needs_extra(**kwargs):
        raise ImportError("No module named 'catboost'")

    _patch_entry_points(monkeypatch, [_descriptor("ext_boost", build=_needs_extra)])
    registry = model_registry()
    with pytest.raises(MissingDependencyError):
        registry.build("ext_boost", task=Task(kind="binary"), random_state=0)


def test_unknown_api_version_skipped_with_warning(monkeypatch, caplog) -> None:
    _patch_entry_points(monkeypatch, [_descriptor("ext_future", api_version=99)])
    with caplog.at_level(logging.WARNING, logger="honestml"):
        names = set(model_registry().by_name())
    assert "ext_future" not in names
    assert any("ext_future" in r.getMessage() for r in caplog.records)


# --- read-only listing API -------------------------------------------------


def test_available_models_lists_capabilities() -> None:
    listing = available_models()
    assert set(listing) == _BUILTIN_NAMES
    assert all(isinstance(c, Capabilities) for c in listing.values())


def test_handles_cat_only_for_native_boostings() -> None:
    # FR-2 / ADR-0087: native categorical handling is catboost/lightgbm only; the per-backend caps
    # are sourced from `_Backend.handles_categorical`, so the registry no longer shares one constant.
    caps = available_models()
    assert caps["catboost"].handles_cat and caps["lightgbm"].handles_cat
    assert not caps["xgboost"].handles_cat
    assert not caps["linear"].handles_cat and not caps["baseline"].handles_cat
    assert CATBOOST.handles_categorical and LIGHTGBM.handles_categorical
    assert not XGBOOST.handles_categorical


def test_available_models_filtered_by_task() -> None:
    # every built-in family spans all kinds now -> a regression task lists them all
    # (listing is install-agnostic; the install gate only affects default selection)
    assert set(available_models(Task(kind="regression"))) == _BUILTIN_NAMES


# --- plugin participates end-to-end (no core/composition edit) --------------


def test_plugin_participates_in_leaderboard(monkeypatch) -> None:
    _patch_entry_points(monkeypatch, [_descriptor("ext_model")])
    X, y = _xy()
    model = AutoML(models=("ext_model",)).fit(X, y)
    assert "ext_model" in [e.model_id for e in model.leaderboard_]


def test_registry_build_returns_estimator() -> None:
    est = model_registry().build("linear", task=Task(kind="binary"), random_state=7)
    assert est.random_state == 7


def test_proba_metric_drops_nonprobabilistic_without_materialization(monkeypatch) -> None:
    """A proba metric filters out a ``probabilistic=False`` plugin via the static tag,
    WITHOUT calling its ``build`` (ADR-0019 §2 — no materialization)."""
    from honestml.composition.build import build_default_components

    def _boom(**kwargs):
        raise AssertionError("build must not be called for a filtered-out component")

    non_proba = ComponentDescriptor(
        name="ext_nonproba",
        spec=ModelSpec(name="ext_nonproba", capabilities=Capabilities(tasks=("binary",))),
        build=_boom,
    )
    _patch_entry_points(monkeypatch, [non_proba])
    components = build_default_components(Task(kind="binary"), random_state=0, metric="roc_auc")
    assert "ext_nonproba" not in components.estimators  # dropped by the static proba filter
    assert {"baseline", "linear"} <= set(components.estimators)  # builtins still selected


def test_direct_construction_with_explicit_builtins() -> None:
    registry = ComponentRegistry("honestml.models", [_descriptor("solo")])
    # no entry-points patched -> real (empty) group; only the explicit builtin
    assert "solo" in registry.by_name()


def test_handles_cat_plugin_without_marker_warns(monkeypatch, caplog) -> None:
    # FR-2: a plugin declaring handles_cat=True but NOT implementing SupportsNativeCategorical is warned
    # and falls back to the codes path (the marker drives native routing, not the static flag).
    from honestml.composition.build import build_default_components

    plugin = ComponentDescriptor(
        name="ext_cat",
        spec=ModelSpec(
            name="ext_cat",
            capabilities=Capabilities(tasks=("binary",), probabilistic=True, handles_cat=True),
        ),
        build=lambda **kw: BaselineClassifier(),  # no SupportsNativeCategorical marker
        dist="ext",  # a plugin (not "<builtin>") — built-ins are skipped, aligned by construction
    )
    _patch_entry_points(monkeypatch, [plugin])
    with caplog.at_level(logging.WARNING, logger="honestml"):
        build_default_components(Task(kind="binary"), random_state=0, metric="roc_auc")
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "ext_cat" in msg and "SupportsNativeCategorical" in msg
