# Plugin contract — third-party models

A third-party package adds a model to honestml without editing the core: it ships a
`ComponentDescriptor` and declares an entry-point in the `honestml.models` group. This page is
the contract — what to implement, the rules you must hold, and what honestml guarantees in
return.

## What you ship

A light **descriptor module** exposing one `ComponentDescriptor` per model, and an entry-point
pointing at it:

```python
# honestml_tabnet/plugin.py — must stay import-light (see "The import-light rule")
from honestml.composition.registry import ComponentDescriptor
from honestml.core import Capabilities, ModelSpec

def _build(*, task, random_state, **params):
    from .adapter import TabNetAdapter  # the heavy import lives HERE, never at module top
    return TabNetAdapter(task=task, random_state=random_state, **params)

DESCRIPTOR = ComponentDescriptor(
    name="tabnet",
    spec=ModelSpec(name="tabnet", capabilities=Capabilities(
        tasks=("binary", "multiclass", "regression"),
        probabilistic=True,        # the classification branch exposes predict_proba
        handles_missing=False,
    )),
    build=_build,
    api_version=1,
    requires=("pytorch_tabnet",),  # runtime module(s); gates default selection
)
```

```toml
# pyproject.toml of the plugin package
[project.entry-points."honestml.models"]
tabnet = "honestml_tabnet.plugin:DESCRIPTOR"
```

`honestml.models` is currently the only entry-point group. Pick a `name` not taken by a
built-in (`baseline`, `linear`, `catboost`, `lightgbm`, `xgboost`) or by any installed plugin —
a duplicate name fails discovery with `honestml.core.PluginConflictError` (a `ConfigError`
subclass).

## `ComponentDescriptor` fields

| field | type | meaning |
|---|---|---|
| `name` | `str` | unique component id — the value users pass to `models=(...)`. Must be unique across built-ins and plugins (see "Determinism & conflicts"). |
| `spec` | `ModelSpec` | `name` + `Capabilities` + an optional declarative `search_space` (see "Capabilities semantics" and "Declaring a search space"). |
| `build` | `Callable[..., Estimator]` | **lazy factory**, called as `build(task=Task, random_state=int, **params)`. The heavy import happens here. |
| `api_version` | `int` (default `1`) | plugin-contract version; newer than the installed registry supports → skipped with a WARNING (see "Versioning & deprecation"). |
| `dist` | `str` (default `"<builtin>"`) | informational; a stable secondary sort key only (does not affect determinism). |
| `requires` | `tuple[str, ...]` (default `()`) | top-level runtime module(s) the component needs. Empty = always available. See "Extras availability". |

## The import-light rule

The descriptor module **must not import its heavy dependency at module load** — only inside
`build()`. Discovery loads the descriptor module (`entry_points(...).load()`) to read its
capabilities; if that import pulls the heavy library, laziness is broken for everyone. This is a
**plugin responsibility**: honestml holds this rule for its own built-ins but does not sandbox
plugins. A descriptor that imports a heavy package at top level violates the contract.

## `Capabilities` semantics

- `tasks: tuple[TaskKind, ...]` — which of `binary`/`multiclass`/`regression` the model serves.
  One descriptor may span all three; `build` picks the per-kind implementation.
- `probabilistic: bool` — a **static tag** read *without materializing* the model, so a proba
  metric can filter candidates cheaply. It means **the classification branch exposes
  `predict_proba`**. On a regression task the value metric does not consult it (a regression +
  proba/class metric is rejected up front by the task↔metric guard).
- `handles_missing: bool` — declares whether your model tolerates raw NaN. Currently
  informational: honestml neither imputes nor filters candidates on it, so on NaN-bearing
  data a model that cannot handle NaN fails at `fit` and is recorded as a failed candidate.
  Handle NaN identically on train and inference.
- `handles_cat: bool` — native categorical handling. honestml feeds categorical **codes as
  numeric** to a model that declares `False`; a model that declares `True` is additionally handed
  the categorical column indices to consume natively (see `SupportsNativeCategorical` below).
  Built-in: catboost/lightgbm declare `True`, xgboost/linear/baseline `False`. Declaring `True`
  **without** implementing the marker logs a warning and falls back to the codes path.
- `supports_early_stopping: bool` — declares that your estimator early-stops on a validation tail
  (the `SupportsEarlyStopping` marker — `fit(..., X_val=, y_val=)`). honestml **reads this** (it is
  not inert): when `True`, composition carves an early-stopping tail from each fold's train block
  and hands it to your `fit`. Leave it `False` unless your `fit` actually consumes `X_val`/`y_val`.

`Capabilities` also accepts `needs_scaling`, `gpu`, `max_rows`, `max_cols` (default
off/`None`) — reserved declarations, currently not consulted by selection; leave them at
defaults.

## The estimator your `build` returns

Implement the `Estimator` port (numpy boundary):

```python
feature_names: list[str]
def fit(self, X, y, X_val=None, y_val=None, sample_weight=None) -> Self: ...
def predict(self, X) -> np.ndarray: ...          # 1-D labels (classification) or values (regression)
```

honestml assigns `feature_names` to your estimator before each `fit` (and re-assigns it after
feature selection) — it must be a plain writable attribute, not a read-only property.

`X_val`/`y_val` carry the early-stopping validation tail (ADR-0080): the pipeline passes them
to an early-stopping-capable model when a fold has a carved es tail, and passes `None` otherwise
— your `fit` must accept them and may ignore them if it does not early-stop.

Opt-in role-interfaces:

- **`ProbabilisticEstimator`** (classification): add `classes_: np.ndarray` (the column order of
  `predict_proba`) and `def predict_proba(self, X) -> np.ndarray` returning `(n, len(classes_))`.
- **`SupportsFeatureImportance`**: a `feature_importances` property → **1-D** `np.ndarray` of
  length `n_features` (for multiclass, aggregate across classes).
- **`SupportsShap`**: `def shap_values(self, X) -> np.ndarray`.
- **`SupportsNativeCategorical`** (native categorical handling, pairs with `handles_cat=True`):
  set `supports_native_categorical: bool = True` and accept an injected
  `categorical_indices: list[int]` — the positions of categorical columns in the design matrix,
  assigned by the pipeline before each `fit` (like `feature_names`), and re-used on `predict`.
  Materialize those columns through your library's native categorical API. An empty list is a
  valid no-op (a dataset with no categories). A model that declares `handles_cat=True` but does
  not implement this marker is logged a warning and trains on the numeric codes instead. Note: a
  `handles_cat=True` plugin has its `build()` called once during model selection (before CV) to
  verify this marker via `isinstance`, so keep adapter construction cheap and side-effect-free.

A saved artifact persists a plugin estimator through the default pickle serializer, so the
plugin package must be installed wherever the artifact is loaded.

## Declaring a search space

`ModelSpec.search_space` optionally declares hyperparameters for tuning — one validated dict per
parameter:

```python
spec=ModelSpec(
    name="tabnet",
    capabilities=...,
    search_space={
        "n_steps": {"type": "int", "low": 3, "high": 10},  # optional "step" (default 1)
        "learning_rate": {"type": "float", "low": 1e-3, "high": 0.1, "log": True},
        "mask_type": {"type": "categorical", "choices": ["sparsemax", "entmax"]},
    },
)
```

An invalid entry (unknown `type`, `low >= high`, empty `choices`) fails with `ConfigError`
rather than being silently dropped. Tuned parameters are validated as a **subset of the declared
space** — a stray key is a `ConfigError` — and are passed to your `build` as `**params`, so
`build` must accept every declared parameter as a keyword. A model with an empty `search_space`
is simply not tuned.

## Extras availability — default vs explicit selection

`requires` declares the runtime module(s). The registry checks them with
`importlib.util.find_spec` — **without importing** the heavy library:

- **Default run** (`models=None`): a component is auto-included only when every `requires`
  module is importable; otherwise it is silently skipped, so a lightweight install never fails
  on models it cannot run.
- **Explicit run** (`models=("catboost",)`): a name no descriptor provides raises `ConfigError`
  listing the available models; a known but uninstalled model fails fast with
  `MissingDependencyError` (`pip install honestml[catboost]`). The install hint names the
  *component* — for a built-in that matches a honestml extra, but for a third-party plugin it
  will not match your package's install command.
- **Listing** (`AutoML.available_models()`): lists **every** registered component regardless of
  install state, so a user sees what *can* be installed.
- If `build` still raises `ImportError` at materialization, the registry maps it to
  `MissingDependencyError`.

## Determinism & conflicts

Discovery is deterministic: duplicate names — across built-ins and plugins alike — are rejected
**before** sorting (`PluginConflictError`, no "last wins"); survivors are ordered by `name`,
independent of `sys.path` traversal order.

## Versioning & deprecation

- The contract is additive: new `Capabilities`/descriptor fields land with defaults, so an older
  plugin keeps loading.
- Bump your descriptor's `api_version` only when you rely on a newer contract; an older
  `honestml` then skips your plugin with a WARNING rather than crashing — provided the
  descriptor itself still constructs under the older contract. Do not pass descriptor or
  `Capabilities` fields the older honestml does not have (guard them, or declare a minimum
  honestml version in your package metadata): a descriptor module that fails to import
  crashes discovery for every run; it is not skipped.
- Removing/renaming a component `name` is a breaking change for users' `models=(...)` configs —
  deprecate first.

## Security / trust model

Loading a plugin runs installed package code: `entry_points(...).load()` imports the descriptor
module, and `build()` imports the adapter. This is the standard Python plugin model — the
entry-point **group is a constant, never user input**, and untrusted *data* is never executed as
code. Install only honestml plugins you trust, exactly as you would any dependency. honestml
does not sandbox plugin imports.
