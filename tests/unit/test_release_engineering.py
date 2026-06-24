"""M9-3: release engineering invariants (ADR-0077, FR-DLV-5/6, NFR-DLV-5/6).

Machine-checkable halves of the release contour: the triple version gate, the
id-token isolation of the publish job, the audit gate wiring, the license, the
release checklist and the README anti-legacy pin.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]


def _load_check_tag_version():
    spec = importlib.util.spec_from_file_location(
        "check_tag_version", _ROOT / "scripts" / "check_tag_version.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.check_tag_version


def test_triple_version_gate() -> None:
    """FR-DLV-5: tag == pyproject == honestml.__version__ — any mismatch fails."""
    check = _load_check_tag_version()
    pyproject = '[project]\nname = "honestml"\nversion = "0.1.0"\n'
    init = '__version__ = "0.1.0"\n'
    assert check("refs/tags/v0.1.0", pyproject, init) == "0.1.0"
    assert check("v0.1.0", pyproject, init) == "0.1.0"
    with pytest.raises(ValueError, match="version mismatch"):
        check("v0.2.0", pyproject, init)
    with pytest.raises(ValueError, match="version mismatch"):
        check("v0.1.0", pyproject.replace("0.1.0", "0.2.0"), init)
    with pytest.raises(ValueError, match="version mismatch"):
        check("v0.1.0", pyproject, init.replace("0.1.0", "0.2.0"))
    with pytest.raises(ValueError, match="no __version__"):
        check("v0.1.0", pyproject, "")


def test_gate_matches_real_repo_files() -> None:
    """The real pyproject and __init__ agree with each other (tag-independent half)."""
    check = _load_check_tag_version()
    pyproject = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    init = (_ROOT / "src" / "honestml" / "__init__.py").read_text(encoding="utf-8")
    import honestml

    assert check(f"v{honestml.__version__}", pyproject, init) == honestml.__version__


def _jobs(workflow: str) -> dict:
    doc = yaml.safe_load((_ROOT / ".github" / "workflows" / workflow).read_text(encoding="utf-8"))
    return doc["jobs"]


def test_release_workflow_id_token_only_in_publish() -> None:
    """NFR-DLV-5: OIDC permission is isolated to the publish job; audit gates publish."""
    doc = yaml.safe_load(
        (_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    )
    assert "id-token" not in (doc.get("permissions") or {}), "id-token at workflow level"
    jobs = doc["jobs"]
    for name, job in jobs.items():
        permissions = job.get("permissions") or {}
        if name == "publish":
            assert permissions.get("id-token") == "write"
        else:
            assert "id-token" not in permissions, f"id-token leaked into job {name!r}"
    assert set(jobs["publish"]["needs"]) == {"check", "build", "audit"}
    assert jobs["publish"]["environment"] == "pypi"


def test_benchmark_workflow_is_dispatch_only() -> None:
    """ADR-0076 §4: the benchmark runs on dispatch, never on push/PR."""
    doc = yaml.safe_load(
        (_ROOT / ".github" / "workflows" / "benchmark.yml").read_text(encoding="utf-8")
    )
    triggers = doc.get("on", doc.get(True))  # yaml 1.1 parses bare `on` as True
    assert set(triggers) == {"workflow_dispatch"}


def test_audit_workflow_and_ignore_valve_exist() -> None:
    jobs = _jobs("audit.yml")
    assert "pip-audit" in jobs
    ignore = _ROOT / "audits" / "pip-audit-ignore.txt"
    assert ignore.exists()
    # the valve is reviewed: every non-comment line must carry a justification comment
    for line in ignore.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            assert "#" in stripped, f"unjustified ignore entry: {line!r}"


def test_license_is_mit() -> None:
    """FR-DLV-5 (G-L1): the LICENSE file exists and matches the pyproject SPDX field."""
    text = (_ROOT / "LICENSE").read_text(encoding="utf-8")
    assert text.startswith("MIT License")
    assert 'license = "MIT"' in (_ROOT / "pyproject.toml").read_text(encoding="utf-8")


def test_releasing_checklist_is_complete() -> None:
    text = (_ROOT / "docs" / "releasing.md").read_text(encoding="utf-8")
    for required in (
        "check_tag_version",
        "Trusted Publisher",
        "protection rules",
        "benchmark.yml",
        "CHANGELOG",
        "pip-audit-ignore",
    ):
        assert required in text, f"releasing.md misses checklist item {required!r}"


# source of the marker list: the legacy sketch API documented by the pre-M9 README
# (run_automl/load_dataset/retrain_best, S00..S07 scenarios, composite score)
_LEGACY_MARKERS = (
    r"\brun_automl\b",
    r"\bload_dataset\b",
    r"\bretrain_best\b",
    r"\bS0\d\b",
    r"\bcomposite score\b",
)


def test_readme_has_no_legacy_api() -> None:
    """NFR-DLV-6 (anti-drift): README documents the facade, not the retired sketch."""
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    for marker in _LEGACY_MARKERS:
        assert re.search(marker, readme) is None, f"legacy marker {marker} in README"
    for expected in (
        "AutoML(",
        "export_onnx",
        "TrackerConfig",
        "preset",
        "load_artifact",
        "extras",
    ):
        assert expected in readme, f"README misses {expected!r}"


def test_api_docs_cover_public_surface() -> None:
    """NFR-DLV-6: every pinned public name is present in docs/api.md (no doc drift)."""
    import honestml

    api = (_ROOT / "docs" / "api.md").read_text(encoding="utf-8")
    missing = [name for name in honestml.__all__ if name not in api]
    assert missing == [], f"public names missing from docs/api.md: {missing}"


def test_llms_txt_covers_every_nav_page(tmp_path: Path) -> None:
    """llms.txt indexes — and llms-full.txt inlines — every page of the mkdocs nav."""
    spec = importlib.util.spec_from_file_location(
        "build_llms_txt", _ROOT / "scripts" / "build_llms_txt.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    llms, llms_full = module.build(tmp_path)

    config = yaml.safe_load((_ROOT / "mkdocs.yml").read_text(encoding="utf-8"))
    pages = module._nav_pages(config["nav"])
    assert len(pages) >= 6, "mkdocs nav lost its pages"
    index = llms.read_text(encoding="utf-8")
    full = llms_full.read_text(encoding="utf-8")
    assert config["site_url"] in index
    for title, path in pages:
        assert f"[{title}](" in index, f"llms.txt misses nav page {title!r}"
        content = (_ROOT / "docs" / path).read_text(encoding="utf-8").strip()
        assert content in full, f"llms-full.txt misses the content of {path!r}"


def test_docs_deploy_id_token_isolated_to_deploy_job() -> None:
    """Same OIDC policy as release.yml: id-token only where the deployment happens."""
    doc = yaml.safe_load(
        (_ROOT / ".github" / "workflows" / "docs-deploy.yml").read_text(encoding="utf-8")
    )
    assert "id-token" not in (doc.get("permissions") or {}), "id-token at workflow level"
    jobs = doc["jobs"]
    for name, job in jobs.items():
        permissions = job.get("permissions") or {}
        if name == "deploy":
            assert permissions.get("id-token") == "write"
        else:
            assert "id-token" not in permissions, f"id-token leaked into job {name!r}"
    assert jobs["deploy"]["environment"]["name"] == "github-pages"
    assert jobs["deploy"]["needs"] == "build"
