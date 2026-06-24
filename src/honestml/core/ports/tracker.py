"""The ``ExperimentTracker`` port.

A post-fit, one-shot observability boundary: the facade hands the COMPLETED run's
report (the tracker-independent run report — JSON primitives only, versioned by
``run_manifest_version``) to whatever tracking backend the user opted into. The
tracker is a pure consumer of that report (single source of provenance); it never
receives raw data rows. No null-object is defined: the single call site treats
``None`` as off (the ``budget``/``cache`` pattern), unlike ``SignificanceTest``
which flows deep into the use-cases.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ExperimentTracker(Protocol):
    """Logs one completed run from its report.

    One call per ``fit``. ``report`` is a deep copy owned by the callee — mutating
    it cannot corrupt the facade's ``run_report_``. Implementations must ignore
    unknown keys (the report evolves additively). A raised exception is
    downgraded to a WARNING by the facade: a tracking failure must not destroy a
    finished fit.
    """

    def log_run(self, report: Mapping[str, Any]) -> None: ...
