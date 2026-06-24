"""Exception taxonomy.

The hierarchy is part of the public API (SemVer): callers catch specific types,
and it cannot be widened later without breaking.
"""

from __future__ import annotations


class AutoMLError(Exception):
    """Base class for every error raised by the library."""


class ConfigError(AutoMLError):
    """Invalid configuration (wraps validation failures at the boundary)."""


class UnsupportedSchemeError(ConfigError):
    """A CV scheme/parameter is valid but not implemented yet.

    Subclass of :class:`ConfigError` (catchable as both): separates "valid but
    not yet implemented" from "invalid configuration", instead of silently
    substituting a shuffling split.
    """


class PluginConflictError(ConfigError):
    """Two registered components share a name within a registry group.

    Subclass of :class:`ConfigError` (catchable as both). Fail-fast on duplicate
    plugin names instead of silently letting "the last one win".
    """


class MissingDependencyError(AutoMLError):
    """An optional extra is required but not installed.

    Raised by adapters (not by core imports) so a missing boosting/tracking
    library surfaces as an actionable message instead of an ``ImportError`` deep
    in an import chain.
    """

    def __init__(self, extra: str, *, package: str | None = None) -> None:
        self.extra = extra
        self.package = package or extra
        super().__init__(
            f"Optional dependency '{self.package}' for feature '{extra}' is not "
            f"installed. Install it with: pip install honestml[{extra}]"
        )


class SchemaValidationError(AutoMLError):
    """Input does not satisfy the ``FeatureSchema``/``Task`` contract.

    Covers X/y length mismatch, unknown/missing columns, targets outside
    ``Task.kind``, empty or all-NaN inputs, and dtype drift.
    Also the artifact/serialization format boundary: an unknown
    ``model_type``/``model_format`` and a non-exportable estimator are the same
    kind of contract violation, not a new exception type.
    """


class NativeCategoricalONNXUnsupportedError(SchemaValidationError):
    """A natively-categorical boosting model cannot be exported to ONNX (FR-6, ADR-0091).

    Subclass of :class:`SchemaValidationError` (catchable as both): non-ONNX-exportability is the same
    serialization-boundary contract violation as baseline/ensemble, but a specific type lets callers
    distinguish "native categorical → no ONNX" from other export rejections. CatBoost native cat is
    impossible (catboost#863, ONNX-ML is numeric-only); LightGBM native cat is deferred to an ONNX
    re-spike. The gate fires BEFORE any converter, so there is never a silently-wrong graph.
    """


class ArtifactIntegrityError(AutoMLError):
    """An artifact failed integrity verification before deserialization.

    ``reason`` is one of ``missing_checksums`` (no checksums block under
    ``require_integrity``), ``missing_file`` (a checksummed file is absent or its name
    escapes the artifact directory), ``digest_mismatch`` (a file's sha256 differs —
    corruption or naive tampering) or ``signature_mismatch`` (the optional signature
    hook rejected the artifact). Integrity detects corruption/naive substitution, NOT
    authenticity: a malicious author can embed code with a matching digest — use a
    signature (and load only from a trusted source) for that.
    """

    def __init__(self, reason: str, *, file: str | None = None) -> None:
        self.reason = reason
        self.file = file
        detail = f" ({file})" if file else ""
        super().__init__(f"artifact integrity check failed: {reason}{detail}")


class InputError(AutoMLError):
    """Loading the input from a path failed.

    Missing/unreadable file, encoding error or unsupported format in the optional
    ``load_table`` loader. A folder whose files disagree on schema raises
    :class:`SchemaValidationError` instead (a contract violation, not I/O).
    """


class NotFittedError(AutoMLError):
    """A fitted artifact was used before ``fit`` (e.g. ``predict`` on a fresh model)."""


class FitFailedError(AutoMLError):
    """Every candidate failed to fit in a run.

    Per-candidate failures are isolated, logged and collected into
    ``SliceResult.failed``; this terminal error is raised only when *no* candidate
    survived, so an empty leaderboard never passes silently.
    """

    def __init__(self, failures: list[tuple[str, str]]) -> None:
        self.failures = failures
        detail = "; ".join(f"{name}: {reason}" for name, reason in failures)
        super().__init__(f"all {len(failures)} candidate(s) failed to fit ({detail})")


class FeatureSelectionError(AutoMLError):
    """A feature-selection strategy failed during compare (fail-fast).

    Raised when any strategy in ``FeatureSelectionConfig.compare`` raises while selecting its subset:
    the offending strategy name is reported and the original error chained, instead of silently
    dropping a strategy from the comparison (no silent defaults).
    """

    def __init__(self, strategy: str, cause: object) -> None:
        self.strategy = strategy
        super().__init__(f"feature-selection strategy {strategy!r} failed: {cause}")


class BudgetExhaustedError(AutoMLError):
    """The run budget was exhausted before any candidate completed.

    Raised only when the budget skipped candidates and *none* finished — distinct from
    :class:`FitFailedError` (every candidate that started failed on its own). Carries the
    budget mode and the completed/skipped/failed counts for an actionable message.
    """

    def __init__(self, mode: str, *, completed: int, skipped: int, failed: int) -> None:
        self.mode = mode
        self.completed = completed
        self.skipped = skipped
        self.failed = failed
        super().__init__(
            f"budget ({mode}) exhausted: {completed} completed before exhaustion; "
            f"{skipped} skipped, {failed} failed"
        )
