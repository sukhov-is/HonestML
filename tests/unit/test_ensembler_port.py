"""M7b-A: the ``Ensembler`` port + ``EnsembleRecipe`` contract (FR-ENS-1, ADR-0063 §1)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pytest

from honestml.core import Ensembler, EnsembleRecipe

pytestmark = pytest.mark.unit


class _FakeEnsembler:
    """A minimal structural ``Ensembler``: equal weights over the members."""

    name = "fake"

    def combine(
        self,
        oof: np.ndarray,
        y: np.ndarray,
        *,
        score: Callable[[np.ndarray], float],
        member_ids: Sequence[str],
        random_state: int,
        sample_weight: np.ndarray | None = None,
    ) -> EnsembleRecipe:
        w = 1.0 / len(member_ids)
        return EnsembleRecipe(
            weights={m: w for m in member_ids}, method=self.name, member_ids=tuple(member_ids)
        )


def test_runtime_checkable() -> None:
    assert isinstance(_FakeEnsembler(), Ensembler)


def test_recipe_weights_are_simplex() -> None:
    # keys must equal member_ids, weights >= 0 and sum ~= 1 (ADR-0063 §1)
    with pytest.raises(ValueError):
        EnsembleRecipe(weights={"a": 0.7, "b": 0.7}, method="caruana", member_ids=("a", "b"))
    with pytest.raises(ValueError):
        EnsembleRecipe(weights={"a": 1.2, "b": -0.2}, method="caruana", member_ids=("a", "b"))
    with pytest.raises(ValueError):
        EnsembleRecipe(weights={"a": 1.0}, method="caruana", member_ids=("a", "b"))
    ok = EnsembleRecipe(weights={"a": 0.25, "b": 0.75}, method="caruana", member_ids=("a", "b"))
    assert pytest.approx(sum(ok.weights.values())) == 1.0


def test_recipe_weights_native_float() -> None:
    # numpy weights are coerced to python float for byte-stable report/manifest emission (ADR-0063 §1)
    r = EnsembleRecipe(
        weights={"a": np.float64(0.5), "b": np.float64(0.5)},
        method="weighted",
        member_ids=("a", "b"),
    )
    assert all(type(v) is float for v in r.weights.values())
