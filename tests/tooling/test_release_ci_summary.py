"""Release evidence must reject partial matrices and mismatched source claims."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "ci_summary", ROOT / "notes/scripts/summarize_release_ci.py"
)
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)
PRIOR = json.loads(
    (
        ROOT / "notes/benchmarks/portable-flow-ci-qualification-2026-11-25.json"
    ).read_text()
)
LOCAL = json.loads(
    (
        ROOT / "notes/benchmarks/portable-flow-local-qualification-2026-11-25.json"
    ).read_text()
)


def run_metadata():
    return {
        "event": "workflow_dispatch",
        "status": "completed",
        "conclusion": "success",
        "databaseId": PRIOR["workflow_run_id"],
        "headSha": PRIOR["candidate_commit"],
        "url": PRIOR["url"],
        "jobs": copy.deepcopy(PRIOR["jobs"]),
    }


@pytest.fixture
def artifacts(tmp_path):
    for entry in PRIOR["qualification_receipts"]:
        folder = tmp_path / entry["artifact"]
        folder.mkdir()
        name = (
            "source-qualification.json"
            if entry["artifact"] == "qualification-source"
            else "wheel-qualification.json"
        )
        (folder / name).write_text(json.dumps(entry["receipt"]))
    return tmp_path


def test_real_historical_matrix_and_source_proof_are_accepted(artifacts):
    assert len(SUMMARY.validate_jobs(run_metadata())) == 22
    assert len(SUMMARY.validate_artifacts(artifacts)) == 20
    count, fingerprint = SUMMARY.source_proof(PRIOR["candidate_commit"], LOCAL)
    assert count == 500
    assert fingerprint == PRIOR["production_source_tree_sha256"]


@pytest.mark.parametrize(
    "mutation", ["incomplete", "missing", "duplicate", "failed", "published"]
)
def test_partial_failed_and_published_runs_never_become_qualification(mutation):
    run = run_metadata()
    if mutation == "incomplete":
        run["status"] = "in_progress"
    elif mutation == "missing":
        run["jobs"].pop()
    elif mutation == "duplicate":
        run["jobs"].append(copy.deepcopy(run["jobs"][0]))
    elif mutation == "failed":
        run["jobs"][0]["conclusion"] = "failure"
    else:
        next(job for job in run["jobs"] if job["name"] == "publish")[
            "conclusion"
        ] = "success"
    with pytest.raises(ValueError):
        SUMMARY.validate_jobs(run)


@pytest.mark.parametrize("mutation", ["missing", "scope", "python", "hash", "version"])
def test_missing_and_mislabeled_artifacts_are_rejected(artifacts, mutation):
    path = (
        artifacts / "qualification-x86_64-apple-darwin-cp3.9/wheel-qualification.json"
    )
    report = json.loads(path.read_text())
    if mutation == "missing":
        path.unlink()
    else:
        if mutation == "scope":
            report["artifacts"][0]["tests"] = "maintained"
        elif mutation == "python":
            report["python"] = "3.12.0"
        elif mutation == "hash":
            report["artifacts"][0]["sha256"] = "not-a-hash"
        else:
            report["artifacts"][0]["smoke"]["version"] = "0.0.0"
        path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        SUMMARY.validate_artifacts(artifacts)


def test_newer_production_code_cannot_use_old_source_qualification():
    with pytest.raises(ValueError, match="production bytes differ"):
        SUMMARY.source_proof("6ec32ed1959543036f638abedf0dbe23df21a6f7", LOCAL)


@pytest.mark.parametrize("mutation", ["fingerprint", "configuration"])
def test_altered_local_source_or_build_receipts_are_rejected(mutation):
    local = copy.deepcopy(LOCAL)
    if mutation == "fingerprint":
        local["production_source_tree_sha256"] = "0" * 64
    else:
        local["cargo_inputs_sha256"]["Cargo.toml"] = "0" * 64
    with pytest.raises(ValueError):
        SUMMARY.source_proof(PRIOR["candidate_commit"], local)


@pytest.mark.parametrize(
    "log", [b"test result: FAILED", b"test result: ok. 1 passed; 0 failed; 0 ignored"]
)
def test_failed_or_incomplete_native_logs_cannot_qualify(artifacts, log):
    with pytest.raises(ValueError, match="Native log"):
        SUMMARY.summarize(run_metadata(), artifacts, log, LOCAL)


def test_cli_never_overwrites_an_existing_report(tmp_path):
    import subprocess
    import sys

    output = tmp_path / "receipt.json"
    output.write_text("preserve this receipt")
    command = [sys.executable, str(ROOT / "notes/scripts/summarize_release_ci.py")]
    for option in ("--run", "--artifacts", "--log", "--local-receipt"):
        command.extend([option, str(tmp_path / "absent")])
    result = subprocess.run(
        command + ["--output", str(output)], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "Report already exists" in result.stderr
    assert output.read_text() == "preserve this receipt"
