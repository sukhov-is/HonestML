"""Pure run-report assembler + run-fingerprint (ADR-0033 / ADR-0035, G-O1).

A tracker-independent summary of a run: the resolved config (scheme/purge/embargo, seed,
budget, significance), stage timings, the winner/leaderboard, the honesty band, and the
budget outcome. It is a Humble Object — no I/O, synchronously testable on a fake
``SliceResult``/``RunConfig`` (NFR-M5-3) — and emits **JSON primitives only**: the numpy-
carrying ``SliceResult`` fields (``oof_fold_index``, candidate OOF arrays) never enter it.

Provenance is read from the **resolved** ``RunConfig`` (truthful, NFR-M5-6): the budget
mode and significance mode are what actually ran; the degradation outcome (exhausted/
skipped) comes from the ``SliceResult``. Schema is versioned by ``RUN_MANIFEST_VERSION``,
separate from the artifact's ``ARTIFACT_VERSION``.

This module also hosts the **run-fingerprint** (ADR-0035): :func:`compute_run_fingerprint` and
:func:`dataset_signature` are pure (numpy + hashlib + ``importlib.metadata``, no polars, no I/O), so
the cache key is synchronously testable. The digest is over canonical JSON (``sort_keys``) of the
resolved run inputs — the key is full (everything that affects a per-candidate OOF/score) and
fail-closed (any uncertainty must change it). ``FINGERPRINT_VERSION`` is an emergency manual
invalidator for key-composition changes shipped without a release bump.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

import numpy as np

from honestml.core import RunConfig

from .slice import SliceResult, design_matrix

if TYPE_CHECKING:
    from honestml.core import Dataset, Metric, Task

RUN_MANIFEST_VERSION = 1
FINGERPRINT_VERSION = 1

# holdout-optimism diagnostic (finding #11c): a holdout better than the winner's OOF by more than this
# RELATIVE margin signals split dependence (the outer holdout is not independent of DEV). Relative so it
# is metric-scale invariant; tunable as real runs accumulate.
_OPTIMISM_REL = 0.10


def build_run_report(
    *,
    run_config: RunConfig,
    timings: dict[str, dict[str, float]],
    result: SliceResult,
    run_fingerprint: str = "",
    cache_enabled: bool = False,
    fs_resolution: Mapping[str, str] | None = None,
    hpo: Mapping[str, Any] | None = None,
    ensemble: Mapping[str, Any] | None = None,
    serving: Mapping[str, Any] | None = None,
    preset: Mapping[str, Any] | None = None,
    task: str | None = None,
    metric: str | None = None,
) -> dict[str, Any]:
    """Assemble the run report (schema v1 + additive RC keys, JSON primitives only); ADR-0033/0037.

    ``run_fingerprint`` and the ``cache`` block are **additive** top-level keys (ADR-0037 §3):
    ``RUN_MANIFEST_VERSION`` is NOT bumped (new keys, not a changed semantics of existing ones). The
    fingerprint is present whenever the facade supplies it (computed even when the cache is off; the
    default ``""`` keeps tracker-independent callers working). The cache block is truthful — empty
    lists when disabled, the resolved reused/computed ids when enabled.
    """
    return {
        "run_manifest_version": RUN_MANIFEST_VERSION,
        "honestml_version": _honestml_version(),
        # task/metric identity (ADR-0075 §2): they live OUTSIDE RunConfig (facade-level domain
        # choice), so without these keys the report was not self-describing (F4.5). Additive
        # top-level keys; RUN_MANIFEST_VERSION unchanged (None when the caller has no facade).
        "task": task,
        "metric": metric,
        "config": run_config.model_dump(mode="json"),
        "timings": timings,
        "winner": result.best_model_id,
        # per-candidate failures (F4.2): the isolation outcome must be visible in the report,
        # not only in logging (NullHandler by default). [] on a clean run. Additive; version 1.
        "failed": [{"model_id": f.id, "reason": f.reason} for f in result.failed],
        # honest final estimate (ADR-0072 §5): the untouched outer-holdout score of the winner; None in
        # selection mode / outer_holdout=0. Additive top-level key; RUN_MANIFEST_VERSION unchanged.
        "holdout_score": None if result.holdout_score is None else float(result.holdout_score),
        # split-dependence diagnostic (ADR-0029, finding #11c): None unless the holdout scores implausibly
        # better than the winner's OOF (the outer holdout may not be independent). Additive; version unchanged.
        "holdout_optimism": _holdout_optimism(result),
        "leaderboard": [
            {"model_id": e.model_id, "score": float(e.score), "rank": e.rank}
            for e in result.leaderboard
        ],
        "band": {
            "member_ids": list(result.band_member_ids),
            "unstable": bool(result.band_unstable),
            "width": int(result.band_width),
            "winner_by_tiebreak": bool(result.winner_by_tiebreak),
        },
        "budget": {
            "mode": run_config.budget.mode,
            "exhausted": bool(result.budget.exhausted),
            "skipped": list(result.budget.skipped),
            # which axis exhausted the run — outcome (ADR-0039 §5). memory_limit_mb is the INPUT and lives
            # in config["budget"] via the config dump; it is NOT duplicated here (fix m4).
            "exhausted_by": result.budget.exhausted_by,
        },
        "significance": run_config.significance,
        # period-CV split diagnostics (ADR-0096 §4): the densified period/fold/dropped-empty counts of a
        # timeseries_period run — the computed truth the input config dump cannot carry. None otherwise.
        # Additive top-level key; RUN_MANIFEST_VERSION unchanged.
        "cv": dict(result.cv_split) if result.cv_split is not None else None,
        # feature-selection outcome (ADR-0045 §3 / ADR-0049 §3): None when off (report like M6a), else the
        # strategy + kept subset, plus the M6c compare record (strategies_evaluated/per_strategy/winner)
        # when several were compared. Additive top-level key; RUN_MANIFEST_VERSION unchanged.
        "feature_selection": (
            _feature_selection_report(run_config, result, fs_resolution)
            if result.feature_selection is not None
            else None
        ),
        # HPO outcome (ADR-0062 §7): None when off (report like M6), else per-model chosen params + cost +
        # honesty disclosures. Additive top-level key; RUN_MANIFEST_VERSION unchanged.
        # native-categorical routing verdict (ADR-0095, FR-5): None when the cardinality gate demoted
        # nothing (or is off), else the per-column native/high_cardinality decision — a demotion is never
        # silent. Additive top-level key; RUN_MANIFEST_VERSION unchanged.
        "native_routing": (
            _native_routing_report(result.native_routing)
            if result.native_routing is not None
            else None
        ),
        "hpo": dict(hpo) if hpo else None,
        # ensemble outcome (ADR-0064 §5): None when off, else applied/method/member_ids/weights/oof_delta/
        # gate_reason — the gate decision is never silent (NFR-M7-6). Additive; RUN_MANIFEST_VERSION unchanged.
        "ensemble": dict(ensemble) if ensemble else None,
        # serving outcome (ADR-0068 §5): None in selection mode (no model shipped), else
        # {finalize, shipped_on (dev/all), outer_holdout} — finalize is post-selection so it is NOT in the
        # run-fingerprint. Additive top-level key; RUN_MANIFEST_VERSION unchanged.
        "serving": dict(serving) if serving else None,
        # preset provenance (ADR-0074 §3): None when no preset was requested, else
        # {name (None for a custom Mapping), applied: [...]} — input sugar, NOT in the
        # fingerprint (the resolved config above is). Additive; RUN_MANIFEST_VERSION unchanged.
        "preset": dict(preset) if preset else None,
        "run_fingerprint": run_fingerprint,
        "cache": {
            "enabled": cache_enabled,
            "reused": list(result.reused) if cache_enabled else [],
            "computed": list(result.computed) if cache_enabled else [],
        },
    }


def _holdout_optimism(result: SliceResult) -> dict[str, Any] | None:
    """Flag a holdout that scores implausibly BETTER than the winner's OOF (finding #11c).

    An untouched outer holdout should be roughly unbiased — within noise of the OOF, often slightly
    worse, never markedly better. A holdout beating the winner's cross-validated estimate by more than
    ``_OPTIMISM_REL`` (relative) signals that the holdout is NOT independent of DEV: group-structured
    rows plus target encoding carry a group's outcome across a row-wise carve (the #11 mechanism), which
    no within-DEV honesty can repair. Diagnostic only — orientation is inferred from the ranked
    leaderboard (rank 1 = best), so no ``Metric`` object is needed. ``None`` when there is no holdout,
    fewer than two distinct OOF scores (orientation unknown), or the gap is benign.
    """
    if result.holdout_score is None or len(result.leaderboard) < 2:
        return None
    best, worst = result.leaderboard[0].score, result.leaderboard[-1].score
    if best == worst:
        return None
    greater_is_better = best > worst
    winner = next((e for e in result.leaderboard if e.model_id == result.best_model_id), None)
    if winner is None or winner.score == 0.0:
        return None
    holdout = float(result.holdout_score)
    optimism = (holdout - winner.score) if greater_is_better else (winner.score - holdout)
    relative = optimism / abs(winner.score)
    if relative <= _OPTIMISM_REL:
        return None
    return {
        "winner_oof": float(winner.score),
        "holdout": holdout,
        "relative_optimism": float(relative),
        "message": (
            f"holdout ({holdout:.4g}) scores {relative:.0%} better than the winner's OOF "
            f"({winner.score:.4g}); the outer holdout may not be independent of DEV — suspect "
            "group-structured rows + target encoding leaking across a row-wise carve (finding #11)"
        ),
    }


def _native_routing_report(routing: Mapping[str, str]) -> dict[str, Any]:
    """The ``native_routing`` report block (ADR-0095, FR-5): which categoricals went native vs to codes.

    ``routing`` is the post-gate per-column verdict carried on ``SliceResult.native_routing`` (present
    only when >=1 column was demoted). Demoted columns carry the reason token (``high_cardinality``) so
    the run-report makes the cardinality gate's decision auditable, not silent.
    """
    native = sorted(c for c, r in routing.items() if r == "native")
    demoted = sorted(c for c, r in routing.items() if r != "native")
    return {
        "native": native,
        "demoted_to_codes": [{"column": c, "reason": routing[c]} for c in demoted],
    }


def _feature_selection_report(
    run_config: RunConfig, result: SliceResult, fs_resolution: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """The ``feature_selection`` report block (ADR-0049 §3); single-path == M6b, compare adds the record."""
    fs = result.feature_selection
    assert fs is not None
    winner = fs.selected_strategy or (run_config.fs.strategy if run_config.fs is not None else None)
    block: dict[str, Any] = {
        "strategy": winner,
        "n_selected": len(fs.selected_features),
        "selected": list(fs.selected_features),
    }
    # no-selection honest gate verdict (finding #10): present whenever the gate ran. "no_selection_better"
    # means the subset was dropped and all features shipped — the gate is never silent (ADR-0063 §5).
    if fs.selection_gate is not None:
        block["no_selection_gate"] = fs.selection_gate
    if fs.per_strategy is not None:
        std = dict(fs.per_strategy_std or ())
        mean_feat = dict(fs.per_strategy_mean_features or ())
        block["strategies_evaluated"] = [name for name, _, _ in fs.per_strategy]
        block["per_strategy"] = {
            name: {
                "n_selected": k,
                "arb_score": (s if math.isfinite(s) else None),
                **({"arb_score_std": std[name]} if name in std else {}),
                **({"mean_n_features": mean_feat[name]} if name in mean_feat else {}),
            }
            for name, k, s in fs.per_strategy
        }
        block["winner"] = winner
        # M6d nested/significance observability (ADR-0052/0053): why this winner + the equivalence band
        if fs.winner_rule is not None:
            block["winner_rule"] = fs.winner_rule
        if fs.band_members is not None:
            block["band_members"] = list(fs.band_members)
        # in-sequential band of the winning wrapper selector (ADR-0086 §1): SEPARATE from the
        # strategy-arbitration band above; present only when sequential won with significance on.
        if fs.seq_band is not None:
            block["seq_band"] = fs.seq_band
        # M6e per-fold re-selection observability (ADR-0054 §6): whether the honest procedure ran or degraded,
        # and the winner's per-fold subset stability (mean pairwise Jaccard).
        if fs.arbitration_effective is not None:
            block["arbitration_effective"] = fs.arbitration_effective
            block["per_fold_reselection"] = fs.arbitration_effective in (
                "nested_per_fold",
                "per_fold_partial_c5_inner",
            )
        if fs.fold_subset_jaccard is not None:
            block["fold_subset_jaccard"] = fs.fold_subset_jaccard
    # M6d structure-aware null diagnostics (ADR-0050 §5): block stats for ts/group null_importance. M6f
    # (ADR-0059) adds per-fold degenerate aggregates here (merged into null_block_stats in run_slice).
    if fs.null_block_stats is not None:
        block["null_block_stats"] = fs.null_block_stats
    # M6f resolve provenance (ADR-0058 §4): why arbitration/block resolved as they did (auto / cost_budget)
    if fs_resolution:
        block["fs_resolution"] = dict(fs_resolution)
    return block


def _honestml_version() -> str:
    try:
        return version("honestml")
    except PackageNotFoundError:
        return "0+unknown"


def collect_lib_versions(packages: Iterable[str]) -> dict[str, str | None]:
    """Installed version per package, ``None`` when not installed (fail-soft, ADR-0035 §1).

    ``importlib.metadata`` (stdlib) only — a ``PackageNotFoundError`` (e.g. an editable/dev compute
    stack) yields ``None`` deterministically instead of raising, so the fingerprint never fails on a
    missing package.
    """
    out: dict[str, str | None] = {}
    for pkg in sorted(set(packages)):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


def compute_run_fingerprint(
    *,
    run_config: RunConfig,
    task: Task,
    metric: Metric,
    data_signature: str,
    estimators: Iterable[str],
    lib_versions: Mapping[str, str | None],
) -> str:
    """Deterministic, fail-closed run key — hex SHA-256 over canonical JSON (ADR-0035 §1).

    The key includes everything that affects a per-candidate OOF/score, including facade params
    outside ``RunConfig``: the resolved ``task`` and ``metric`` identity (``metric``/``task`` are
    separate ``AutoML`` params, not in ``RunConfig``). ``estimators`` is the resolved name set,
    ``lib_versions`` the resolved compute-stack versions; both are gathered by composition. The
    whole ``RunConfig`` is in the key (stricter than the OOF strictly needs, but never a false hit).
    """
    parts = {
        "config": run_config.model_dump(mode="json"),
        "task": task.model_dump(mode="json"),
        "metric": _metric_identity(metric),
        "data_signature": data_signature,
        "estimators": sorted(estimators),
        "lib_versions": dict(lib_versions),
        "honestml_version": _honestml_version(),
        "fingerprint_version": FINGERPRINT_VERSION,
    }
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode("utf-8")).hexdigest()


def _metric_identity(metric: Metric) -> dict[str, Any]:
    """The metric attributes that change the result: name + direction + averaging + class labels.

    ``name``/``greater_is_better``/``average`` are ``Metric``-port fields (always present), read
    directly. Only ``labels`` is concrete-metric-only (not in the port), so it alone is read
    defensively via ``getattr`` (ADR-0035 §1/§3)."""
    labels = getattr(metric, "labels", None)
    return {
        "name": metric.name,
        "greater_is_better": bool(metric.greater_is_better),
        "average": metric.average,
        "labels": np.asarray(labels).tolist() if labels is not None else None,
    }


def dataset_signature(dataset: Dataset) -> str:
    """Content digest of the model input + target/metadata — pure, no polars, no leak (ADR-0035 §2).

    Hashed over the materialized ``design_matrix`` (built exactly once, NFR-RC-7), the ``target`` and
    any present ``sample_weight``/``groups``/``time``/``label_time``, the serialized ``FeatureSchema``
    and ``n_rows``. Computed over the DEV dataset that ``run_slice`` actually trains on (post-carve),
    so a changed carve/value/metadata yields a different digest — never a false hit. Only the one-way
    digest leaves; raw values never enter the key (R-LEAK).
    """
    h = hashlib.sha256()
    _update_array(h, design_matrix(dataset))
    for marker, arr in (
        (b"target", dataset.target()),
        (b"sample_weight", dataset.sample_weight()),
        (b"groups", dataset.groups()),
        (b"time", dataset.time()),
        (b"label_time", dataset.label_time()),
    ):
        h.update(marker)
        if arr is None:
            h.update(b"\x00none")
        else:
            _update_array(h, np.asarray(arr))
    h.update(b"schema")
    h.update(dataset.schema.model_dump_json().encode("utf-8"))
    h.update(b"n_rows")
    h.update(str(dataset.n_rows).encode("ascii"))
    return h.hexdigest()


def _update_array(h: Any, arr: np.ndarray) -> None:
    """Fold an array into the digest deterministically, without requiring mutual comparability.

    Numeric/bool: contiguous ``tobytes`` + dtype token. Non-numeric (object/str/datetime/mixed):
    per-element ``repr`` UTF-8 bytes in positional order (no sort/dedup, which crashes on mixed
    object), with a fixed ``\\x00NA`` marker for ``None``/``NaN`` (ADR-0035 §2)."""
    if arr.dtype.kind in ("b", "i", "u", "f"):
        h.update(arr.dtype.str.encode("ascii"))
        h.update(np.ascontiguousarray(arr).tobytes())
        return
    h.update(b"\x00obj")
    for v in arr.ravel(order="C").tolist():
        if v is None or (isinstance(v, float) and math.isnan(v)):
            h.update(b"\x00NA")
        else:
            h.update(repr(v).encode("utf-8"))
        h.update(b"\x01")
