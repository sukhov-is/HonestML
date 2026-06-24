"""Default feature-ranker adapters (ADR-0044 §2) — dependency-free, behind the ``FeatureRanker`` port.

Two cheap strategies usable without any optional extra (NFR-FS-5): ``importance`` reads a fitted tree
ensemble's impurity importances; ``random_probe`` augments the matrix with seeded random probe columns
and keeps features whose importance beats the probe baseline (signed margin). Both fit a single cheap
sklearn ``ExtraTrees`` model per fold — the application spine drives the per-fold loop (ADR-0044 §1),
so this stays a pure ``rank(one matrix) -> scores`` adapter. The ranker-model is **separate** from the
candidate estimators (estimator-agnostic subset, ADR-0043 §4). sklearn is a hard dependency, so the
default catalog pulls no boosting extra and ``import honestml`` never grows (the impl-note refinement of
ADR-0044 §2's illustrative "LightGBM": ExtraTrees is lighter and extra-free).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

from honestml.core import MissingDependencyError, Task

# cheap fixed budget for a ranker-model (NFR-FS-5): far lighter than the candidate boosting budget
_N_TREES = 100


def _fit_ranker_model(
    task: Task, x: np.ndarray, y: np.ndarray, random_state: int, sample_weight: np.ndarray | None
):
    """Fit the cheap estimator-agnostic ranker-model (ExtraTrees) on ``x`` (ADR-0043 §4)."""
    model = (
        ExtraTreesClassifier(n_estimators=_N_TREES, random_state=random_state, n_jobs=1)
        if task.is_classification
        else ExtraTreesRegressor(n_estimators=_N_TREES, random_state=random_state, n_jobs=1)
    )
    model.fit(x, y, sample_weight=sample_weight)
    return model


def _check_non_empty(x: np.ndarray) -> None:
    if x.shape[0] == 0:
        raise ValueError("feature ranking requires a non-empty training matrix")


def _permute_target(
    y: np.ndarray, block_indices: list[np.ndarray] | None, rng: np.random.RandomState
) -> np.ndarray:
    """Permute the target i.i.d. (``block_indices=None``, M6c) or WITHIN each structure block (M6d, ADR-0050).

    Within-structure shuffling keeps the null realistic for autocorrelated/grouped targets: only the
    intra-block exchangeable part is permuted, so a feature that merely tracks the block structure no
    longer beats an unrealistically white null (SPIKE-M6d-validity). ``block_indices`` is precomputed once
    per fold (block structure is constant across the ``n_runs`` permutations).
    """
    if block_indices is None:
        return rng.permutation(y)
    out = y.copy()
    for idx in block_indices:
        out[idx] = rng.permutation(y[idx])
    return out


def _fit_importances(
    task: Task,
    x: np.ndarray,
    y: np.ndarray,
    random_state: int,
    sample_weight: np.ndarray | None,
) -> np.ndarray:
    """Impurity importances of a cheap ExtraTrees fit on ``x`` (codes fed as numeric, like the models)."""
    model = _fit_ranker_model(task, x, y, random_state, sample_weight)
    return np.asarray(model.feature_importances_, dtype=np.float64)


class ImportanceRanker:
    """Rank features by a tree-ensemble's impurity importance (ADR-0044 §2); non-negative scores."""

    name = "importance"

    def __init__(self, task: Task) -> None:
        self._task = task

    def rank(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        categorical: np.ndarray,
        random_state: int,
        sample_weight: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> np.ndarray:
        _check_non_empty(x)
        return _fit_importances(self._task, x, y, random_state, sample_weight)

    def auto_threshold(self, n_features: int) -> float:
        # above the uniform share -> "more important than an average feature" (ADR-0044 §3)
        return 1.0 / n_features


class NullImportanceRanker:
    """Score features by their importance margin over a permuted-target null (ADR-0047 §1).

    Refit the cheap ranker-model on ``n_runs`` target permutations to build a per-feature null
    distribution, then score ``importance - percentile(null, p)`` — a signed margin that beats the
    random background (auto-threshold ``0``). Only the **train** target is permuted (a valid null on
    ``fit ⊕ es``); the spine never passes test rows. With ``groups`` (M6d, ADR-0050) the permutation is
    restricted to WITHIN each structure block (time-block / group), keeping the null valid for
    autocorrelated/grouped targets; ``groups=None`` keeps the M6c i.i.d. permutation. Deterministic given
    ``random_state``; extra-free.
    """

    name = "null_importance"

    def __init__(self, task: Task, n_runs: int = 30, null_percentile: float = 95.0) -> None:
        self._task = task
        self._n_runs = n_runs
        self._null_percentile = null_percentile

    def rank(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        categorical: np.ndarray,
        random_state: int,
        sample_weight: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> np.ndarray:
        _check_non_empty(x)
        imp_real = _fit_importances(self._task, x, y, random_state, sample_weight)
        rng = np.random.RandomState(random_state)
        # block structure is constant across permutations -> precompute the per-block row indices once
        # (`groups` (M6d) restricts the shuffle to within structure blocks, ADR-0050; None = i.i.d.)
        block_indices = (
            [np.flatnonzero(groups == g) for g in np.unique(groups)] if groups is not None else None
        )
        null = np.empty((self._n_runs, x.shape[1]), dtype=np.float64)
        for r in range(self._n_runs):
            # permute the training target -> the null hypothesis; the model seed stays fixed so the
            # null reflects target shuffling, not model randomness (deterministic via rng order).
            y_perm = _permute_target(y, block_indices, rng)
            null[r] = _fit_importances(self._task, x, y_perm, random_state, sample_weight)
        return imp_real - np.percentile(null, self._null_percentile, axis=0)

    def auto_threshold(self, n_features: int) -> float:
        return 0.0  # signed margin: a positive mean beats the permuted-target background


def make_ranker_fit_predict(
    task: Task,
) -> Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, int],
    tuple[np.ndarray | None, np.ndarray, np.ndarray | None],
]:
    """Build the estimator-agnostic ``fit_predict`` the compare scorer/arbiter inject (ADR-0046/0048).

    Fits the cheap ranker-model (``ExtraTrees``, like the ranker, estimator-agnostic R-FS-RANKER-MODEL)
    on one matrix and returns ``(proba, pred, classes)`` for classification (``(None, pred, None)`` for
    regression) — the application aligns/projects to the metric. The leakage-critical fold loop stays in
    the application (this only fits one matrix).
    """

    def fit_predict(
        x_tr: np.ndarray,
        y_tr: np.ndarray,
        x_te: np.ndarray,
        sample_weight: np.ndarray | None,
        random_state: int,
    ) -> tuple[np.ndarray | None, np.ndarray, np.ndarray | None]:
        if task.is_classification:
            clf = ExtraTreesClassifier(n_estimators=_N_TREES, random_state=random_state, n_jobs=1)
            clf.fit(x_tr, y_tr, sample_weight=sample_weight)
            return clf.predict_proba(x_te), clf.predict(x_te), clf.classes_
        reg = ExtraTreesRegressor(n_estimators=_N_TREES, random_state=random_state, n_jobs=1)
        reg.fit(x_tr, y_tr, sample_weight=sample_weight)
        return None, reg.predict(x_te), None

    return fit_predict


class RandomProbeRanker:
    """Score features by their importance margin over ``n_probes`` seeded random columns (ADR-0044 §2).

    The margin is signed (a feature below every probe is negative); the spine's ``auto`` cutoff keeps
    a positive mean margin (beats the noise baseline). The margin is fold-relative (same fit, same
    scale), so the spine does not re-normalize it.
    """

    name = "random_probe"

    def __init__(self, task: Task, n_probes: int = 3) -> None:
        self._task = task
        self._n_probes = n_probes

    def rank(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        categorical: np.ndarray,
        random_state: int,
        sample_weight: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> np.ndarray:
        _check_non_empty(x)
        rng = np.random.RandomState(random_state)
        probes = rng.random((x.shape[0], self._n_probes))
        imp = _fit_importances(self._task, np.hstack([x, probes]), y, random_state, sample_weight)
        n = x.shape[1]
        probe_max = float(imp[n:].max())
        return imp[:n] - probe_max

    def auto_threshold(self, n_features: int) -> float:
        return 0.0


def _background(x: np.ndarray, k: int | None) -> np.ndarray:
    """Deterministic, evenly-spaced background for interventional SHAP (ADR-0056 §2).

    Uses ``np.linspace`` indices over the row order rather than a biased head-slice ``x[:k]`` (the first rows
    mis-represent group/time-ordered data and would shift the interventional attribution). Spread is uniform
    for any ``k``; ``x`` unchanged when ``k`` is ``None`` or covers every row (no determinism risk — no seed).
    """
    if k is None or k >= x.shape[0]:
        return x
    return x[np.linspace(0, x.shape[0] - 1, k).astype(np.int64)]


def _kmeans_background(x: np.ndarray, k: int | None, seed: int) -> np.ndarray:
    """Deterministic kmeans-centroid background for interventional SHAP (ADR-0060).

    Cluster centroids cover density modes with ``k`` points better than ``k`` evenly-spaced rows on
    multimodal data (SPIKE-M6f-shap-bg); ``KMeans(n_init=10, random_state=seed)`` is reproducible. Full
    ``x`` when ``k`` is ``None`` or covers every row (same guard as :func:`_background`).
    """
    if k is None or k >= x.shape[0]:
        return x
    return np.asarray(KMeans(n_clusters=k, n_init=10, random_state=seed).fit(x).cluster_centers_)


def _mean_abs_per_feature(shap_values: object, n_features: int) -> np.ndarray:
    """Aggregate TreeExplainer output to a non-negative per-feature score ``mean(|shap|)`` (ADR-0051 §1).

    Normalizes both shapes ``shap>=0.44`` emits on sklearn ensembles: a per-class **list** of ``(n, p)``
    arrays (binary/multiclass) or a single ``(n, p)`` / 3-D ``(n, p, n_classes)`` ndarray (regression /
    newer versions). Averages ``|shap|`` over rows and classes -> ``(n_features,)``.
    """
    if isinstance(shap_values, list):
        arr = np.mean([np.abs(np.asarray(s)) for s in shap_values], axis=0)
    else:
        a = np.abs(np.asarray(shap_values, dtype=np.float64))
        arr = a.mean(axis=2) if a.ndim == 3 else a
    return np.asarray(arr, dtype=np.float64).mean(axis=0).reshape(n_features)


class ShapRanker:
    """Rank features by mean absolute SHAP value of a cheap tree ensemble (ADR-0051; lazy ``shap`` extra).

    Fits the same estimator-agnostic ExtraTrees ranker-model as ``importance`` and scores features by
    ``mean(|TreeExplainer.shap_values|)`` (``tree_path_dependent`` -> exact, background-free, deterministic).
    Non-negative scores pass the spine's L1 normalization like ``importance`` (auto-threshold ``1/n``).
    ``shap`` is imported lazily (fail-fast at construction with :class:`MissingDependencyError` so the
    config error surfaces at build time); ``import honestml`` never pulls it (NFR-FSH-4).
    """

    name = "shap"

    def __init__(
        self,
        task: Task,
        max_samples: int | None = None,
        *,
        perturbation: str = "tree_path_dependent",
        background_samples: int | None = None,
        shap_background: str = "linspace",
    ) -> None:
        try:
            import shap  # noqa: F401
        except ImportError as exc:
            raise MissingDependencyError("shap") from exc
        self._task = task
        self._max_samples = max_samples
        self._perturbation = perturbation
        self._background_samples = background_samples
        self._shap_background = shap_background

    def rank(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        categorical: np.ndarray,
        random_state: int,
        sample_weight: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> np.ndarray:
        import shap

        _check_non_empty(x)
        model = _fit_ranker_model(self._task, x, y, random_state, sample_weight)
        x_explain = x if self._max_samples is None else x[: self._max_samples]
        if self._perturbation == "interventional":
            # interventional Tree SHAP integrates over a background distribution (causally cleaner
            # attribution, ADR-0056). M6f (ADR-0060): "kmeans" uses seeded cluster centroids (better
            # mode-coverage on small backgrounds), else deterministic evenly-spaced "linspace". Both
            # reproducible; kmeans seed = the rank-time random_state (no __init__ seed).
            background = (
                _kmeans_background(x, self._background_samples, random_state)
                if self._shap_background == "kmeans"
                else _background(x, self._background_samples)
            )
            explainer = shap.TreeExplainer(
                model, data=background, feature_perturbation="interventional"
            )
        else:
            explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        return _mean_abs_per_feature(explainer.shap_values(x_explain), x.shape[1])

    def auto_threshold(self, n_features: int) -> float:
        return 1.0 / n_features  # above the uniform share, like importance (ADR-0051 §1)
