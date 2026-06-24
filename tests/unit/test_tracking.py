"""M8c: the ``ExperimentTracker`` port, facade opt-in and MLflow adapter (ADR-0072/0073).

mlflow-independent properties (stub through the facade, fail-fast gate, fail-soft
logging, opt-in forms, fingerprint neutrality, mapping helpers) run in the plain
suite; the MLflow happy path is gated on the installed extra (file store under
``tmp_path``) — the onnx pattern.
"""

from __future__ import annotations

import ast
import json
import logging
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from sklearn.datasets import make_classification

from honestml import AutoML, ConfigError, CVConfig, MissingDependencyError, TrackerConfig
from honestml.adapters import tracking

pytestmark = pytest.mark.unit


def _data():
    return make_classification(
        n_samples=60, n_features=6, n_informative=4, n_redundant=0, random_state=0
    )


class _StubTracker:
    def __init__(self) -> None:
        self.reports: list[dict[str, Any]] = []

    def log_run(self, report) -> None:
        self.reports.append(report)


class _RaisingTracker:
    def log_run(self, report) -> None:
        raise RuntimeError("store down")


# --- port + facade opt-in (FR-TRK-1/3/5, NFR-TRK-5) — no mlflow required -----------------


def test_stub_instance_receives_deepcopy_of_report() -> None:
    """FR-TRK-3 (instance form) + ADR-0072 §1: the payload equals run_report_ but is a deep
    copy — a mutating tracker cannot corrupt the facade's report."""
    X, y = _data()
    stub = _StubTracker()
    model = AutoML(task="binary", models=("baseline", "linear"), random_state=0, tracker=stub)
    model.fit(X, y)
    assert len(stub.reports) == 1
    report = stub.reports[0]
    assert report == model.run_report_
    assert report is not model.run_report_
    assert report["config"] is not model.run_report_["config"]
    report["winner"] = "corrupted"
    assert model.run_report_["winner"] != "corrupted"


def test_raising_tracker_is_fail_soft(caplog) -> None:
    """FR-TRK-5: an exception AFTER the finished training is a WARNING, not a lost fit."""
    X, y = _data()
    model = AutoML(
        task="binary", models=("baseline", "linear"), random_state=0, tracker=_RaisingTracker()
    )
    with caplog.at_level(logging.WARNING, logger="honestml"):
        model.fit(X, y)
    assert model.run_report_["winner"] == model.best_model_id_
    assert model.predict(X) is not None
    warnings = [r for r in caplog.records if "experiment tracking failed" in r.message]
    assert len(warnings) == 1


def test_unknown_tracker_string_fails_before_training() -> None:
    X, y = _data()
    model = AutoML(task="binary", models=("baseline",), tracker="wandb")
    with pytest.raises(ConfigError, match="unknown tracker"):
        model.fit(X, y)
    assert not hasattr(model, "run_report_")


def test_non_tracker_object_rejected() -> None:
    """ADR-0072 §3: garbage and an attribute-not-method ``log_run`` are ConfigError, not a
    silent fail-soft TypeError after the expensive fit."""
    X, y = _data()
    for bad in (123, SimpleNamespace(log_run="not-callable")):
        with pytest.raises(ConfigError, match="tracker must be"):
            AutoML(task="binary", models=("baseline",), tracker=bad).fit(X, y)


def test_missing_mlflow_fails_fast(monkeypatch) -> None:
    """FR-TRK-4: tracker requested by config form + mlflow absent -> MissingDependencyError
    BEFORE training (and from the adapter constructor for the instance form)."""
    monkeypatch.setattr(tracking, "find_spec", lambda name: None)
    monkeypatch.setitem(sys.modules, "mlflow", None)  # a moved gate would surface as ImportError
    with pytest.raises(MissingDependencyError) as exc:
        tracking.MlflowTracker()
    assert exc.value.extra == "mlflow"
    X, y = _data()
    model = AutoML(task="binary", models=("baseline",), tracker="mlflow")
    with pytest.raises(MissingDependencyError):
        model.fit(X, y)
    assert not hasattr(model, "run_report_")


def test_mlflow_string_and_config_build_adapter(monkeypatch) -> None:
    """FR-TRK-3 positive branch (plain suite): find_spec is patched truthy — construction is
    safe because mlflow is imported only inside log_run (ADR-0073 §1)."""
    monkeypatch.setattr(tracking, "find_spec", lambda name: object())
    sugar = AutoML(tracker="mlflow")._resolve_tracker()
    assert isinstance(sugar, tracking.MlflowTracker)
    assert sugar._experiment == "honestml" and sugar._run_name is None
    cfg = TrackerConfig(
        experiment="exp", tracking_uri="file:./x", run_name="r", tags={"team": "ds"}
    )
    configured = AutoML(tracker=cfg)._resolve_tracker()
    assert configured._experiment == "exp"
    assert configured._tracking_uri == "file:./x"
    assert configured._run_name == "r"
    assert configured._tags == {"team": "ds"}
    assert AutoML()._resolve_tracker() is None


def test_holdout_score_wired_into_report_via_fit() -> None:
    """FR-TRK-2 / ADR-0072 §5: the honest holdout estimate reaches run_report_ (and the
    tracker payload) through a REAL fit — the wiring is order-sensitive (the holdout is
    scored before the report is assembled); selection mode stays None."""
    X, y = _data()
    stub = _StubTracker()
    model = AutoML(
        task="binary",
        models=("baseline", "linear"),
        random_state=0,
        cv=CVConfig(outer_holdout=0.25),
        tracker=stub,
    ).fit(X, y)
    assert isinstance(model.run_report_["holdout_score"], float)
    assert model.run_report_["holdout_score"] == model.holdout_score_
    assert stub.reports[0]["holdout_score"] == model.holdout_score_
    selection = AutoML(
        task="binary", models=("baseline",), random_state=0, run_mode="selection"
    ).fit(X, y)
    assert selection.run_report_["holdout_score"] is None


def test_tracker_not_in_fingerprint_or_config() -> None:
    """NFR-TRK-5: tracking is post-selection observability — same fingerprint with and
    without it, and no tracker key in the reproducibility config dump."""
    X, y = _data()
    plain = AutoML(task="binary", models=("baseline", "linear"), random_state=0).fit(X, y)
    tracked = AutoML(
        task="binary", models=("baseline", "linear"), random_state=0, tracker=_StubTracker()
    ).fit(X, y)
    assert plain.run_report_["run_fingerprint"] == tracked.run_report_["run_fingerprint"]
    assert "tracker" not in plain.run_report_["config"]


def test_tracker_config_neutral_defaults() -> None:
    """NFR-TRK-6: defaults are data-independent (the backend generates the run name)."""
    cfg = TrackerConfig()
    assert cfg.backend == "mlflow"
    assert cfg.experiment == "honestml"
    assert cfg.tracking_uri is None and cfg.run_name is None and cfg.tags == {}


def test_tracker_lazy_three_checkpoints() -> None:
    """NFR-TRK-1 (ADR-0072 §3 / ADR-0073 §1): `import honestml`, resolving TrackerConfig,
    constructing AutoML(tracker='mlflow') and a fit WITHOUT a tracker import neither mlflow
    nor the tracking adapter."""
    code = (
        "import sys, json, honestml\n"
        "honestml.TrackerConfig()\n"
        "honestml.AutoML(task='binary', models=('linear',), tracker='mlflow')\n"
        "from sklearn.datasets import make_classification\n"
        "X, y = make_classification(n_samples=40, n_features=5, n_informative=3,"
        " n_redundant=0, random_state=0)\n"
        "honestml.AutoML(task='binary', models=('baseline',), random_state=0).fit(X, y)\n"
        "forbidden = ['mlflow', 'honestml.adapters.tracking']\n"
        "print(json.dumps([k for k in forbidden if k in sys.modules]))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    leaked = json.loads(out.stdout.strip().splitlines()[-1])
    assert leaked == [], f"tracking leaked outside log_run: {leaked}"


# --- adapter mapping helpers (NFR-TRK-3/4) — no mlflow required ---------------------------


def test_adapter_source_has_no_fluent_calls() -> None:
    """NFR-TRK-3 (static half): no global-state mlflow calls anywhere in the adapter."""
    tree = ast.parse(Path(tracking.__file__).read_text(encoding="utf-8"))
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not called & {"set_tracking_uri", "set_experiment", "start_run"}


def test_param_and_tag_truncation_warns(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="honestml"):
        param = tracking._truncate("x" * 7000, tracking._MAX_PARAM_VALUE, "param", "k")
        tag = tracking._truncate("y" * 9000, tracking._MAX_TAG_VALUE, "tag", "t")
    assert len(param) == 6000 and len(tag) == 8000
    assert sum("truncated" in r.message for r in caplog.records) == 2
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="honestml"):
        assert tracking._truncate("short", tracking._MAX_PARAM_VALUE, "param", "k") == "short"
    assert not caplog.records


def test_key_sanitized_to_mlflow_alphabet() -> None:
    assert tracking._sanitize_key("score.my model(v2)=x") == "score.my model_v2__x"
    assert tracking._sanitize_key("time.run/refit") == "time.run/refit"  # already legal
    assert len(tracking._sanitize_key("k" * 300)) == tracking._MAX_KEY
    # mlflow's path-normalization rule (run-verified): path-like keys must not survive as-is
    assert tracking._sanitize_key("/lead") == "_lead"
    assert tracking._sanitize_key("a//b") == "a__b"
    assert tracking._sanitize_key("a/../b") == "a_.._b"
    assert tracking._sanitize_key("..") == "__"


def test_flatten_dot_joins_nested_config() -> None:
    flat = tracking._flatten({"cv": {"n_splits": 5, "scheme": "auto"}, "seed": 42})
    assert flat == {"cv.n_splits": 5, "cv.scheme": "auto", "seed": 42}
    # a NESTED empty mapping stays visible as a leaf; an empty top-level flattens to nothing
    assert tracking._flatten({"x": {}}) == {"x": {}}
    assert tracking._flatten({}) == {}


def test_params_chunked_under_batch_limit() -> None:
    chunks = list(tracking._chunks(list(range(250)), tracking._BATCH))
    assert [len(c) for c in chunks] == [100, 100, 50]


def test_report_metrics_mapping_skips_non_finite() -> None:
    report = {
        "leaderboard": [
            {"model_id": "a", "score": 0.9, "rank": 1},
            {"model_id": "b", "score": float("nan"), "rank": 2},
        ],
        "winner": "a",
        "holdout_score": 0.85,
        "timings": {"run": {"selection": 1.5}},
    }
    metrics = tracking._report_metrics(report)
    assert metrics == {
        "score.a": 0.9,
        "winner_score": 0.9,
        "holdout_score": 0.85,
        "time.run.selection": 1.5,
    }


def test_bad_user_input_fails_fast_at_construction(monkeypatch) -> None:
    """ADR-0073 §1: bad user input must fail at construction, not lose tracking after the fit."""
    monkeypatch.setattr(tracking, "find_spec", lambda name: object())
    with pytest.raises(ConfigError, match="tag key"):
        tracking.MlflowTracker(tags={"": "x"})
    with pytest.raises(ConfigError, match="tag key"):
        tracking.MlflowTracker(tags={"k" * 300: "x"})
    # the provenance namespace is reserved — a user tag must not silently shadow it
    with pytest.raises(ConfigError, match="honestml"):
        tracking.MlflowTracker(tags={"honestml.winner": "x"})
    # two keys collapsing to one after sanitization would silently drop a tag
    with pytest.raises(ConfigError, match="collide"):
        tracking.MlflowTracker(tags={"a(b)": "1", "a[b]": "2"})
    with pytest.raises(ConfigError, match="experiment"):
        tracking.MlflowTracker(experiment="  ")
    with pytest.raises(ConfigError, match="run_name"):
        tracking.MlflowTracker(run_name=42)
    with pytest.raises(Exception, match="experiment"):
        TrackerConfig(experiment="")  # pydantic min_length mirrors the constructor guard


# --- MLflow happy path (importorskip; runs under the `mlflow` extra) ----------------------


@pytest.fixture()
def file_store(tmp_path: Path, monkeypatch) -> str:
    """A hermetic tmp_path file store. mlflow >=3.13 gates the filesystem backend behind
    this env var (maintenance mode; sqlite is the recommended local backend) — for unit
    tests the throwaway file store is exactly right (run-verified, ADR-0073 §4)."""
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    return tmp_path.joinpath("mlruns").as_uri()


def _winner_score(report: dict[str, Any]) -> float:
    return next(e["score"] for e in report["leaderboard"] if e["model_id"] == report["winner"])


def test_mlflow_file_store_end_to_end(file_store) -> None:
    """FR-TRK-2: the logged run carries all four field groups (tags/params/metrics/artifact)."""
    pytest.importorskip("mlflow")
    from mlflow.artifacts import download_artifacts
    from mlflow.tracking import MlflowClient

    uri = file_store
    X, y = _data()
    cfg = TrackerConfig(
        experiment="exp-m8c", tracking_uri=uri, run_name="run-m8c", tags={"team": "ds"}
    )
    model = AutoML(task="binary", models=("baseline", "linear"), random_state=0, tracker=cfg).fit(
        X, y
    )

    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name("exp-m8c")
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    run = runs[0]
    assert run.info.status == "FINISHED"
    assert run.info.run_name == "run-m8c"
    report = model.run_report_
    assert run.data.tags["honestml.winner"] == report["winner"]
    assert run.data.tags["honestml.fingerprint"] == report["run_fingerprint"]
    assert run.data.tags["team"] == "ds"
    assert run.data.params["cv.n_splits"] == str(report["config"]["cv"]["n_splits"])
    assert run.data.params["seed"] == "0"
    assert run.data.metrics[f"score.{report['winner']}"] == pytest.approx(_winner_score(report))
    assert run.data.metrics["winner_score"] == pytest.approx(_winner_score(report))
    assert "time.run.selection" in run.data.metrics
    artifact = download_artifacts(
        run_id=run.info.run_id, artifact_path="run_report.json", tracking_uri=uri
    )
    assert json.loads(Path(artifact).read_text(encoding="utf-8")) == report


def test_mlflow_global_state_untouched(file_store) -> None:
    """NFR-TRK-3: logging through the adapter never moves the user's global tracking URI."""
    mlflow = pytest.importorskip("mlflow")

    before = mlflow.get_tracking_uri()
    tracker = tracking.MlflowTracker(experiment="g", tracking_uri=file_store)
    tracker.log_run(
        {
            "run_manifest_version": 1,
            "honestml_version": "0.1.0",
            "config": {"seed": 0},
            "timings": {},
            "winner": "a",
            "leaderboard": [{"model_id": "a", "score": 0.5, "rank": 1}],
            "holdout_score": None,
            "run_fingerprint": "f",
        }
    )
    assert mlflow.get_tracking_uri() == before


def test_experiment_create_then_get_race(tmp_path) -> None:
    """ADR-0073 §2: a raced ``create_experiment`` re-reads instead of losing the run."""
    pytest.importorskip("mlflow")
    from mlflow.exceptions import MlflowException

    class _RaceClient:
        def get_experiment_by_name(self, name):
            if not getattr(self, "_asked", False):
                self._asked = True
                return None  # we lost the race: not visible yet...
            return SimpleNamespace(experiment_id="42")  # ...but the winner created it

    def _create(name):
        raise MlflowException("already exists")

    client = _RaceClient()
    client.create_experiment = _create
    tracker = tracking.MlflowTracker(
        experiment="raced", tracking_uri=tmp_path.joinpath("mlruns").as_uri()
    )
    assert tracker._experiment_id(client) == "42"


def test_mlflow_failed_run_marked_failed(file_store) -> None:
    """ADR-0073 §2 cleanup: a mid-logging failure marks the run FAILED (no eternal RUNNING),
    and the facade downgrades it to a WARNING (FR-TRK-5)."""
    pytest.importorskip("mlflow")
    from mlflow.tracking import MlflowClient

    uri = file_store
    tracker = tracking.MlflowTracker(experiment="boom", tracking_uri=uri)

    def _explode(self, client, run_id, report):
        raise RuntimeError("mid-logging failure")

    tracker._log_payload = _explode.__get__(tracker)
    with pytest.raises(RuntimeError, match="mid-logging"):
        tracker.log_run({"winner": "a"})
    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name("boom")
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1 and runs[0].info.status == "FAILED"
