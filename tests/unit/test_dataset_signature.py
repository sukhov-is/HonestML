"""M5-resume RC-a: the pure ``dataset_signature`` content digest (ADR-0035 §2, FR-RC-1, NFR-RC-7).

A Humble Object: a fake ``Dataset`` (numpy + a real ``FeatureSchema``), no polars/I/O. The digest must
be deterministic, sensitive to data values/metadata, stable for object/string targets (per-element
``repr`` canonicalization, not raw-address hashing), and never leak the raw data (digest-only).
"""

from __future__ import annotations

import numpy as np
import pytest

from honestml.application import dataset_signature
from honestml.core import ColumnRole, FeatureSchema

pytestmark = pytest.mark.unit


class _FakeDataset:
    """Minimal Dataset over numpy with a real FeatureSchema; all port metadata accessors."""

    def __init__(
        self,
        *,
        numeric,
        codes=None,
        y=None,
        sample_weight=None,
        groups=None,
        time=None,
        label_time=None,
    ) -> None:
        self._numeric = np.asarray(numeric, dtype=np.float64)
        n = self._numeric.shape[0]
        self._codes = (
            np.asarray(codes, dtype=np.int64) if codes is not None else np.empty((n, 0), np.int64)
        )
        self._y = y
        self._sw = sample_weight
        self._groups = groups
        self._time = time
        self._label_time = label_time
        roles: dict[str, ColumnRole] = {
            f"num{i}": ColumnRole.NUMERIC for i in range(self._numeric.shape[1])
        }
        roles.update({f"cat{i}": ColumnRole.CATEGORICAL for i in range(self._codes.shape[1])})
        self._schema = FeatureSchema(roles=roles)

    @property
    def schema(self):
        return self._schema

    @property
    def n_rows(self):
        return self._numeric.shape[0]

    def to_numpy(self):
        return self._numeric

    def categorical_codes(self):
        return self._codes

    def target(self):
        return self._y

    def sample_weight(self):
        return self._sw

    def groups(self):
        return self._groups

    def time(self):
        return self._time

    def label_time(self):
        return self._label_time


def test_same_data_same_digest() -> None:
    a = _FakeDataset(numeric=[[1.0], [2.0], [3.0]], y=np.array([0, 1, 0]))
    b = _FakeDataset(numeric=[[1.0], [2.0], [3.0]], y=np.array([0, 1, 0]))
    assert dataset_signature(a) == dataset_signature(b)


def test_value_change_changes_digest() -> None:
    # same n_rows + schema, a single changed value -> different digest (defeats schema-only legacy)
    a = _FakeDataset(numeric=[[1.0], [2.0], [3.0]], y=np.array([0, 1, 0]))
    b = _FakeDataset(numeric=[[1.0], [2.0], [3.5]], y=np.array([0, 1, 0]))
    assert dataset_signature(a) != dataset_signature(b)


def test_target_change_changes_digest() -> None:
    a = _FakeDataset(numeric=[[1.0], [2.0]], y=np.array([0, 1]))
    b = _FakeDataset(numeric=[[1.0], [2.0]], y=np.array([1, 0]))
    assert dataset_signature(a) != dataset_signature(b)


def test_string_target_stable_digest() -> None:
    # object/str target with None: per-element repr canonicalization -> stable across constructions
    a = _FakeDataset(numeric=[[1.0], [2.0], [3.0]], y=np.array(["cat", "dog", None], dtype=object))
    b = _FakeDataset(numeric=[[1.0], [2.0], [3.0]], y=np.array(["cat", "dog", None], dtype=object))
    assert dataset_signature(a) == dataset_signature(b)
    c = _FakeDataset(
        numeric=[[1.0], [2.0], [3.0]], y=np.array(["cat", "dog", "fish"], dtype=object)
    )
    assert dataset_signature(a) != dataset_signature(c)


def test_metadata_axis_changes_digest() -> None:
    base = dict(numeric=[[1.0], [2.0], [3.0], [4.0]], y=np.array([0, 1, 0, 1]))
    a = _FakeDataset(**base, groups=np.array([0, 0, 1, 1]))
    b = _FakeDataset(**base, groups=np.array([0, 1, 0, 1]))
    assert dataset_signature(a) != dataset_signature(b)
    # sample_weight is part of the signature too
    c = _FakeDataset(**base, sample_weight=np.array([1.0, 1.0, 1.0, 1.0]))
    d = _FakeDataset(**base, sample_weight=np.array([1.0, 2.0, 1.0, 1.0]))
    assert dataset_signature(c) != dataset_signature(d)


def test_time_axis_changes_digest() -> None:
    # F045: the time axis is hashed into the signature (run_report.py); a changed time column must
    # change the digest — else a time-series rerun could get a false cache hit on stale results.
    base = dict(numeric=[[1.0], [2.0], [3.0], [4.0]], y=np.array([0, 1, 0, 1]))
    a = _FakeDataset(**base, time=np.array([10.0, 20.0, 30.0, 40.0]))
    b = _FakeDataset(**base, time=np.array([10.0, 20.0, 30.0, 99.0]))
    assert dataset_signature(a) != dataset_signature(b)


def test_label_time_axis_changes_digest() -> None:
    # F045: the label_time (t1) axis is hashed too; a changed label-horizon column must change the digest.
    base = dict(numeric=[[1.0], [2.0], [3.0], [4.0]], y=np.array([0, 1, 0, 1]))
    a = _FakeDataset(**base, label_time=np.array([11.0, 21.0, 31.0, 41.0]))
    b = _FakeDataset(**base, label_time=np.array([11.0, 21.0, 31.0, 99.0]))
    assert dataset_signature(a) != dataset_signature(b)


def test_digest_is_hex_sha256_no_raw_leak() -> None:
    sig = dataset_signature(_FakeDataset(numeric=[[1.0], [2.0]], y=np.array([0, 1])))
    assert isinstance(sig, str) and len(sig) == 64
    int(sig, 16)  # valid hex; raw values are not recoverable from a one-way digest


def test_single_pass_over_design_matrix() -> None:
    # NFR-RC-7 gating invariant: the design matrix is materialized exactly once for the signature
    ds = _FakeDataset(numeric=[[1.0], [2.0], [3.0]], y=np.array([0, 1, 0]))
    calls = {"n": 0}
    original = ds.to_numpy

    def counting():
        calls["n"] += 1
        return original()

    ds.to_numpy = counting  # type: ignore[method-assign]
    dataset_signature(ds)
    assert calls["n"] == 1
