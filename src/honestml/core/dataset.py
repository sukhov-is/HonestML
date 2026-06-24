"""The ``Dataset`` port — internal data representation at the model boundary.

A narrow Protocol exposing numeric values and categorical *codes* (via the
schema-owned table), but deliberately **no ``to_pandas()``**: the model boundary is
numpy + codes, far cheaper than materializing string pandas and keeping both
pandas and polars out of the domain. The default implementation (polars-backed)
lives in ``honestml.adapters``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from .schema import FeatureSchema


@runtime_checkable
class Dataset(Protocol):
    """Domain view over tabular data: numeric block, categorical codes, target."""

    @property
    def schema(self) -> FeatureSchema: ...

    @property
    def n_rows(self) -> int: ...

    @property
    def columns(self) -> list[str]: ...

    def select(self, columns: Sequence[str]) -> Dataset:
        """Return a dataset restricted to *columns* (schema updated accordingly)."""
        ...

    def take(self, indices: Sequence[int] | np.ndarray) -> Dataset:
        """Return a dataset with only the given row indices (fold slicing)."""
        ...

    def with_selected_features(self, names: Sequence[str]) -> Dataset:
        """Return a dataset whose schema carries the feature-selection subset.

        Same rows/frame; only ``schema.selected_features`` is set, so ``design_matrix`` projects the
        model input to *names* on refit and inference (train==inference by construction).
        """
        ...

    def to_numpy(self) -> np.ndarray:
        """Numeric feature block as ``float64`` with shape ``(n_rows, n_numeric)``."""
        ...

    def categorical_codes(self) -> np.ndarray:
        """Categorical feature codes as ``int64`` with shape ``(n_rows, n_categorical)``.

        Codes come from the schema-owned category table, so they are identical on
        train and inference.
        """
        ...

    def target(self) -> np.ndarray | None:
        """Target values, or ``None`` for an inference dataset."""
        ...

    def sample_weight(self) -> np.ndarray | None:
        """Per-row sample weights, or ``None``."""
        ...

    def groups(self) -> np.ndarray | None:
        """Group-column values in row order, or ``None`` when there is no group role.

        The single source of group labels for group-aware CV: the splitter
        and ``validate_fold`` both read this array, index-aligned with ``design_matrix``,
        so the group/fold/feature ordering cannot drift.
        """
        ...

    def time(self) -> np.ndarray | None:
        """TIME-role column values in row order, or ``None``.

        The single, index-aligned source of the CV time axis for ``TimeSeriesSplitter`` and the
        value-based ``validate_fold`` (same contract as ``groups()``), so the splitter never reads
        a reserved column name from the frame. Distinct from DATETIME features.
        """
        ...

    def label_time(self) -> np.ndarray | None:
        """Optional per-row label-end-time ``t1`` for full de Prado purge, or ``None``.

        Name-based secondary metadata (like ``sample_weight``), present only when declared; used by
        the splitter to drop train rows whose label window overlaps the test interval.
        """
        ...
