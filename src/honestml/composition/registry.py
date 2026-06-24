"""Component registry: built-in + entry-point plugins (ADR-0019, extended ADR-0020).

Discovery is **lazy**: third-party descriptors are read from entry-points, but a
component's (possibly heavy) adapter is imported only when its ``build`` runs. Built-in
descriptors are declared here — composition is the one place that names concrete adapters
(ADR-0009) — referencing adapter classes/factories so ``adapters`` never imports composition
(layers contract). The ``ComponentDescriptor`` type also lives here; third-party plugins
import it from this module (the plugin contract).

Determinism (NFR-4): duplicate names are rejected **before** sorting (``PluginConflictError``);
survivors are ordered by ``name`` (unique after dedup), so enumeration does not depend on
``sys.path`` traversal.

Extras availability (ADR-0020 §5, find_spec amendment to ADR-0019 §1): a descriptor declares
its runtime ``requires`` (top-level module names). :meth:`ComponentRegistry.is_available`
checks them with ``importlib.util.find_spec`` — **without importing** the heavy library — so
the default selection (``models=None``) auto-includes a boosting component only when its extra
is installed, while explicit selection of a missing one fails with ``MissingDependencyError``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from importlib.metadata import entry_points
from importlib.util import find_spec
from typing import Any

from honestml.adapters.boosting import CATBOOST, LIGHTGBM, XGBOOST, build_boosting
from honestml.adapters.estimators import (
    BaselineClassifier,
    BaselineRegressor,
    LinearClassifier,
    LinearRegressor,
)
from honestml.core import (
    Capabilities,
    Estimator,
    MissingDependencyError,
    ModelSpec,
    PluginConflictError,
    Task,
    get_logger,
)

MODELS_GROUP = "honestml.models"
REGISTRY_API_VERSION = 1
_BUILTIN_DIST = "<builtin>"

logger = get_logger("composition.registry")

# Light builtins (sklearn, always present): one descriptor per family spans all kinds;
# `build` picks the classifier/regressor variant. `probabilistic=True` means the
# classification branch is probabilistic (regression uses a value metric — ADR-0020 §3).
_LIGHT_CAPS = Capabilities(
    tasks=("binary", "multiclass", "regression"),
    probabilistic=True,
    handles_cat=False,
    # a median SimpleImputer prefixes the linear/baseline Pipeline (ADR-0078), so raw NaN no longer
    # evicts them from the zoo (finding #6); the gate still fires for plugins declaring False.
    handles_missing=True,
)


# Boosting (extras): NaN handled natively & identically on train/inference (ADR-0020 §2);
# supports_early_stopping=True -> composition carves an es tail for it (ADR-0080). `handles_cat` is
# per-backend (native categories: catboost/lightgbm only, ADR-0087), sourced from the adapter's
# `_Backend.handles_categorical` so the static capability cannot drift from the runtime marker.
def _boost_caps(*, handles_cat: bool) -> Capabilities:
    return Capabilities(
        tasks=("binary", "multiclass", "regression"),
        probabilistic=True,
        handles_cat=handles_cat,
        handles_missing=True,
        supports_early_stopping=True,
    )


@dataclass(frozen=True)
class ComponentDescriptor:
    """A registered component: its name, declared ``ModelSpec`` and a lazy factory.

    ``build`` must not import a heavy dependency at module load — only inside the call
    (ADR-0019 §1). ``api_version`` lets the registry skip future-contract plugins instead of
    crashing. ``requires`` lists the runtime module(s) the component needs; an empty tuple
    means always available (core/sklearn). The registry checks ``requires`` via ``find_spec``
    for default selection (ADR-0020 §5).
    """

    name: str
    spec: ModelSpec
    build: Callable[..., Any]
    api_version: int = 1
    dist: str = _BUILTIN_DIST
    requires: tuple[str, ...] = ()


def _build_baseline(*, task: Task, random_state: int) -> Estimator:
    return BaselineClassifier() if task.is_classification else BaselineRegressor()


def _build_linear(*, task: Task, random_state: int, **params: Any) -> Estimator:
    # `**params` carries tuned hyperparameters (ADR-0061 §4); linear's search_space is empty in M7a
    # (boosting is the HPO target — task-aware C/alpha linear HPO is a follow-up), so params is empty.
    if task.is_classification:
        return LinearClassifier(random_state=random_state, **params)
    return LinearRegressor(random_state=random_state, **params)


def _builtin_models() -> list[ComponentDescriptor]:
    return [
        ComponentDescriptor(
            name="baseline",
            spec=ModelSpec(name="baseline", capabilities=_LIGHT_CAPS),
            build=_build_baseline,
        ),
        ComponentDescriptor(
            name="linear",
            spec=ModelSpec(name="linear", capabilities=_LIGHT_CAPS),
            build=_build_linear,
        ),
        ComponentDescriptor(
            name="catboost",
            spec=ModelSpec(
                name="catboost",
                capabilities=_boost_caps(handles_cat=CATBOOST.handles_categorical),
                search_space=CATBOOST.search_space,
            ),
            build=partial(build_boosting, CATBOOST),
            requires=("catboost",),
        ),
        ComponentDescriptor(
            name="lightgbm",
            spec=ModelSpec(
                name="lightgbm",
                capabilities=_boost_caps(handles_cat=LIGHTGBM.handles_categorical),
                search_space=LIGHTGBM.search_space,
            ),
            build=partial(build_boosting, LIGHTGBM),
            requires=("lightgbm",),
        ),
        ComponentDescriptor(
            name="xgboost",
            spec=ModelSpec(
                name="xgboost",
                capabilities=_boost_caps(handles_cat=XGBOOST.handles_categorical),
                search_space=XGBOOST.search_space,
            ),
            build=partial(build_boosting, XGBOOST),
            requires=("xgboost",),
        ),
    ]


class ComponentRegistry:
    """Discover and materialize components for one entry-point group."""

    def __init__(self, group: str, builtins: list[ComponentDescriptor]) -> None:
        self._group = group
        self._builtins = list(builtins)
        self._cache: dict[str, ComponentDescriptor] | None = None

    def descriptors(self) -> list[ComponentDescriptor]:
        """Built-in + third-party descriptors: deduped, version-filtered, name-sorted."""
        return list(self._resolved().values())

    def by_name(self) -> dict[str, ComponentDescriptor]:
        return dict(self._resolved())

    def is_available(self, name: str) -> bool:
        """True if every module in the descriptor's ``requires`` is importable (no import)."""
        return all(_module_present(m) for m in self._resolved()[name].requires)

    def build(self, name: str, **kwargs: Any) -> Any:
        """Materialize ``name``; a missing optional extra surfaces as MissingDependencyError."""
        descriptor = self._resolved()[name]
        try:
            return descriptor.build(**kwargs)
        except (ImportError, ModuleNotFoundError) as exc:
            raise MissingDependencyError(name) from exc

    # -- internals ----------------------------------------------------------

    def _resolved(self) -> dict[str, ComponentDescriptor]:
        if self._cache is None:
            collected = self._builtins + self._discover()
            supported = [d for d in collected if self._version_ok(d)]
            self._reject_duplicates(supported)
            ordered = sorted(supported, key=lambda d: (d.name, d.dist))
            self._cache = {d.name: d for d in ordered}
        return self._cache

    def _discover(self) -> list[ComponentDescriptor]:
        # entry_points reads metadata only; ep.load() imports the (light) descriptor
        # module, not the heavy adapter — the heavy import lives inside build().
        return [ep.load() for ep in entry_points(group=self._group)]

    def _version_ok(self, descriptor: ComponentDescriptor) -> bool:
        if descriptor.api_version > REGISTRY_API_VERSION:
            logger.warning(
                "skipping plugin %r: api_version %d > supported %d",
                descriptor.name,
                descriptor.api_version,
                REGISTRY_API_VERSION,
            )
            return False
        return True

    def _reject_duplicates(self, descriptors: list[ComponentDescriptor]) -> None:
        dupes = sorted(n for n, c in Counter(d.name for d in descriptors).items() if c > 1)
        if dupes:
            raise PluginConflictError(
                f"duplicate component name(s) in group {self._group!r}: {dupes}"
            )


def _module_present(module: str) -> bool:
    """Whether *module* is importable, without importing it (find_spec only)."""
    try:
        return find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def model_registry() -> ComponentRegistry:
    """The default model registry (built-in models + ``honestml.models`` plugins)."""
    return ComponentRegistry(MODELS_GROUP, _builtin_models())


def available_models(task: Task | None = None) -> dict[str, Capabilities]:
    """Read-only listing of discoverable models (no materialization, ADR-0019 §7).

    Lists every registered component (including extras-gated boosting), so a user sees what
    *can* be installed; the install check (``is_available``) only gates default *selection*.
    """
    out: dict[str, Capabilities] = {}
    for descriptor in model_registry().descriptors():
        caps = descriptor.spec.capabilities
        if task is not None and task.kind not in caps.tasks:
            continue
        out[descriptor.name] = caps
    return out
