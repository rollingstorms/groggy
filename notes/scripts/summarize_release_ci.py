"""Validate downloaded manual release CI evidence against a pinned local receipt.

Read-only inputs: `gh run view ID --json databaseId,headSha,event,status,conclusion,url,jobs`,
`gh run download ID --pattern 'qualification-*' --dir ARTIFACTS`, and
`gh run view ID --log`. Run only after completion. This script never dispatches,
publishes, downloads or replaces an existing report. The expected matrix is the
current publish.yml contract: 19 wheels, source rebuild and native gates.
"""

import argparse
import hashlib
import io
import json
import re
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGETS = (
    "x86_64-apple-darwin",
    "aarch64-apple-darwin",
    "x86_64-pc-windows-msvc",
    "x86_64-unknown-linux-gnu",
)
VERSIONS = ("3.8", "3.9", "3.10", "3.11", "3.12")
COMBINATIONS = tuple(
    (version, target)
    for version in VERSIONS
    for target in TARGETS
    if not (version == "3.8" and target == "aarch64-apple-darwin")
)
SOURCES = ("src", "python-groggy/src", "python-groggy/python/groggy")
CONFIGS = (
    "Cargo.toml",
    "python-groggy/Cargo.toml",
    "pyproject.toml",
    "python-groggy/pyproject.toml",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def validate_jobs(run):
    require(
        run.get("event") == "workflow_dispatch", "Expected a manual qualification run"
    )
    require(
        run.get("status") == "completed" and run.get("conclusion") == "success",
        "Run has not completed successfully",
    )
    jobs = run["jobs"]
    by_name = {job["name"]: job for job in jobs}
    expected = {"native-gates", "build-sdist", "publish"} | {
        f"Qualify CPython {version} ({target})" for version, target in COMBINATIONS
    }
    require(
        len(by_name) == len(jobs) and set(by_name) == expected,
        "Unexpected or duplicate CI job matrix",
    )
    for name, job in by_name.items():
        require(job["status"] == "completed", "Incomplete job: " + name)
        require(
            job["conclusion"] == ("skipped" if name == "publish" else "success"),
            "Unqualified job: " + name,
        )
    return [
        {key: job[key] for key in ("name", "status", "conclusion", "url")}
        for job in jobs
    ]


def validate_artifacts(folder):
    expected = {
        f"qualification-{target}-cp{version}/wheel-qualification.json": (
            version,
            (
                "maintained"
                if target == "x86_64-unknown-linux-gnu" and version == "3.12"
                else "ownership"
            ),
        )
        for version, target in COMBINATIONS
    }
    expected["qualification-source/source-qualification.json"] = ("3.12", "none")
    paths = {
        p.relative_to(folder).as_posix(): p for p in folder.rglob("*qualification.json")
    }
    require(
        set(paths) == set(expected),
        "Missing, duplicate or unexpected artifact receipts",
    )
    receipts = {}
    versions = set()
    for name, (python, tests) in expected.items():
        data = json.loads(paths[name].read_text())
        require(
            data["python"].split()[0].startswith(python + "."),
            "Wrong interpreter receipt: " + name,
        )
        require(len(data["artifacts"]) == 1, "Expected one wheel per receipt: " + name)
        wheel = data["artifacts"][0]
        require(wheel["tests"] == tests, "Wrong test scope: " + name)
        require(
            re.fullmatch(r"[0-9a-f]{64}", wheel["sha256"]) is not None,
            "Invalid wheel hash: " + name,
        )
        require(wheel["wheel"].endswith(".whl"), "Missing wheel name: " + name)
        versions.add(wheel["smoke"]["version"])
        receipts[name] = {"sha256": digest(paths[name].read_bytes()), "report": data}
    require(len(versions) == 1, "Artifact versions disagree")
    return receipts


def archive(revision):
    require(
        re.fullmatch(r"[0-9a-f]{40}", revision) is not None,
        "Expected a full immutable commit SHA",
    )
    content = subprocess.check_output(
        ["git", "archive", revision, *SOURCES, *CONFIGS], cwd=ROOT
    )
    with tarfile.open(fileobj=io.BytesIO(content)) as tar:
        return {
            m.name: tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()
        }


def source_proof(candidate, local):
    current, qualified = archive(candidate), archive(local["runtime"])
    paths = sorted(
        Path(name) for name in qualified if Path(name).suffix in (".rs", ".py", ".pyi")
    )
    candidate_paths = sorted(
        Path(name) for name in current if Path(name).suffix in (".rs", ".py", ".pyi")
    )
    require(paths == candidate_paths, "Candidate production inventory differs")
    require(
        all(current[str(p)] == qualified[str(p)] for p in paths),
        "Candidate production bytes differ from local qualification",
    )
    fingerprint = digest(
        "".join(
            str(p) + "\0" + digest(qualified[str(p)]) + "\n" for p in paths
        ).encode()
    )
    require(
        len(paths) == local["production_source_file_count"]
        and fingerprint == local["production_source_tree_sha256"],
        "Local source receipt mismatch",
    )
    for name in CONFIGS:
        require(
            digest(current[name]) == local["cargo_inputs_sha256"][name],
            "Candidate build input mismatch: " + name,
        )
    return len(paths), fingerprint


def summarize(run, artifacts, log, local):
    jobs = validate_jobs(run)
    receipts = validate_artifacts(artifacts)
    count, fingerprint = source_proof(run["headSha"], local)
    require(b"test result: FAILED" not in log, "Native log contains a failed suite")
    rows = re.findall(
        rb"test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored", log
    )
    require(
        len(rows) >= local["native"]["suite_results"]
        and sum(int(row[0]) for row in rows) >= local["native"]["passed"]
        and all(int(row[1]) == 0 for row in rows),
        "Native log lacks passing suite evidence",
    )
    version = local["artifact"]["wheel"].split("-")[1]
    require(
        all(
            entry["report"]["artifacts"][0]["smoke"]["version"] == version
            for entry in receipts.values()
        ),
        "Artifact version differs from local qualification",
    )
    return {
        "version": version,
        "build_configuration_boundary": "Tracked manifests match the local receipt. Cargo.lock files are ignored and regenerated by CI; matching dependency lock graphs is not claimed.",
        "workflow_run_id": run["databaseId"],
        "url": run["url"],
        "candidate_commit": run["headSha"],
        "qualified_runtime_source_equivalent": local["runtime"],
        "production_source_file_count": count,
        "production_source_tree_sha256": fingerprint,
        "result": "success",
        "required_jobs_passed": 21,
        "wheel_combinations": 19,
        "sdist_rebuild": "passed",
        "publish": "skipped",
        "maintained_suite_boundary": "Only Linux CPython 3.12 runs the maintained full Python/docs suite; other wheel jobs run ownership/application smoke checks; source rebuild runs smoke. Artifact receipts are CI evidence, not independent wheel downloads.",
        "jobs": jobs,
        "qualification_receipts": receipts,
        "native_log_suite_result_count": len(rows),
        "run_log_sha256": digest(log),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--local-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Report already exists; choose a fresh path")
    result = summarize(
        json.loads(args.run.read_text()),
        args.artifacts,
        args.log.read_bytes(),
        json.loads(args.local_receipt.read_text()),
    )
    result["local_receipt"] = str(args.local_receipt)
    result["collector_sha256"] = digest(Path(__file__).read_bytes())
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"Verified {result['required_jobs_passed']} jobs, {result['wheel_combinations']} wheels and {result['production_source_file_count']} production sources"
    )


if __name__ == "__main__":
    main()
