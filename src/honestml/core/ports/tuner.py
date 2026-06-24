"""The ``Tuner`` port + backend-neutral ``SearchSpace``.

HPO behind a port: a component declares a backend-neutral ``SearchSpace`` (carried on
``ModelSpec.search_space``) and a :class:`Tuner` adapter searches it. The port is a
**Humble Object** (like :class:`FeatureSubsetSelector`): ``tune`` receives an injected
``score(params) -> float`` and never sees raw rows/folds, so the domain stays free of optuna and
the search is leakage-safe by construction. ``ParamSpec`` is a declarative, validated, JSON-stable
schema; the Optuna adapter translates it to ``trial.suggest_*``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator

from ..exceptions import ConfigError


class _ParamBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class IntParam(_ParamBase):
    """An integer hyperparameter searched in ``[low, high]`` with a ``step`` grid."""

    type: Literal["int"] = "int"
    low: int
    high: int
    step: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _check_bounds(self) -> IntParam:
        if self.low >= self.high:
            raise ValueError(f"int param requires low < high (got {self.low} >= {self.high})")
        return self


class FloatParam(_ParamBase):
    """A float hyperparameter searched in ``[low, high]``, optionally on a log scale."""

    type: Literal["float"] = "float"
    low: float
    high: float
    log: bool = False

    @model_validator(mode="after")
    def _check_bounds(self) -> FloatParam:
        if self.low >= self.high:
            raise ValueError(f"float param requires low < high (got {self.low} >= {self.high})")
        if self.log and self.low <= 0:
            raise ValueError("log-scale float param requires low > 0")
        return self


class CategoricalParam(_ParamBase):
    """A categorical hyperparameter chosen from a non-empty ``choices`` list."""

    type: Literal["categorical"] = "categorical"
    choices: tuple[str | int | float | bool, ...]

    @model_validator(mode="after")
    def _check_choices(self) -> CategoricalParam:
        if not self.choices:
            raise ValueError("categorical param requires a non-empty choices list")
        return self


ParamSpec = Annotated[IntParam | FloatParam | CategoricalParam, Field(discriminator="type")]
SearchSpace = Mapping[str, ParamSpec]

_PARAM_ADAPTER: TypeAdapter[ParamSpec] = TypeAdapter(ParamSpec)


def parse_search_space(raw: Mapping[str, Any]) -> dict[str, ParamSpec]:
    """Validate a raw ``ModelSpec.search_space`` dict into typed :data:`ParamSpec` entries.

    An unknown ``type``, bad bounds (``low >= high``) or an empty ``choices`` fails loud with
    :class:`ConfigError` (not silently dropped). Returns ``{}`` for an empty space (the component
    is then skipped by HPO).
    """
    out: dict[str, ParamSpec] = {}
    for name, spec in raw.items():
        try:
            out[name] = _PARAM_ADAPTER.validate_python(spec)
        except ValidationError as exc:
            raise ConfigError(f"invalid search_space entry {name!r}: {exc}") from exc
    return out


@dataclass(frozen=True)
class TuneOutcome:
    """A tuning result: the best params, how many trials ran, the best inner score.

    ``best_params`` is normalized to python-native scalars (int/float/str/bool) at the adapter
    boundary so report emission and any fingerprint inclusion are byte-stable.
    """

    best_params: dict[str, Any]
    n_trials_run: int
    best_score: float


@runtime_checkable
class Tuner(Protocol):
    """Search a ``SearchSpace`` to maximize an injected scalar score (Humble Object)."""

    name: str

    def tune(
        self,
        search_space: Mapping[str, ParamSpec],
        score: Callable[[Mapping[str, Any]], float],
        *,
        max_trials: int,
        timeout_s: float | None,
        greater_is_better: bool,
        random_state: int,
    ) -> TuneOutcome:
        """Return the best params found.

        ``score(params)`` is an application-provided pure scorer returning the **metric value in its
        own orientation**; the adapter sets the search direction from ``greater_is_better`` (so the
        application never flips). The adapter sees only a scalar per candidate config, never raw rows
        (leakage-safe). ``max_trials``/``timeout_s`` are budget scalars computed by the application
        from the run :class:`Budget`; the adapter owns the search loop and must be
        deterministic given ``random_state`` when ``timeout_s`` is ``None`` (single-thread).
        """
        ...
