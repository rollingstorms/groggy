"""Validate and render the curated capability matrix without importing Groggy.

--check verifies checked-in output, coverage and evidence links. --verify-runtime
also checks the local production tree against the pinned qualification receipt;
later source changes are never silently credited with that older test evidence.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "docs/api/capabilities.json"
OUTPUT = ROOT / "docs/api/capability-matrix.md"
CHECKLIST = ROOT / "notes/planning/graph-engine-flat-checklist.md"
LABELS = {
    "bounded": "Verified within stated bounds",
    "partial": "Partial",
    "native_only": "Native only",
    "not_implemented": "Not implemented for this target",
    "deferred": "Deferred",
}
SUPPORT = {"available", "partial", "unsupported", "not_provided", "not_applicable"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return path.read_text(encoding="utf-8")


def repository_file(path):
    candidate = (ROOT / path).resolve()
    try:
        candidate.relative_to(ROOT)
    except ValueError:
        raise ValueError("Evidence path escapes repository: " + path)
    require(candidate.is_file(), "Missing evidence/source: " + path)
    return candidate


def public_exports():
    tree = ast.parse(read(ROOT / "python-groggy/python/groggy/__init__.py"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            return set(ast.literal_eval(node.value))
    raise ValueError("Public __all__ inventory is missing")


def checklist_rows():
    rows = re.findall(
        r"^- \[([ x])\] \*\*([A-Za-z0-9.]+) — ([^\n]+)", read(CHECKLIST), re.M
    )
    require(rows, "No checklist rows found")
    require(len({row[1] for row in rows}) == len(rows), "Duplicate checklist ID")
    return rows


def covers(prefix, item):
    return prefix == item or item.startswith(prefix + ".")


def validate(data, verify_runtime=False):
    require(data.get("schema_version") == 1, "Unsupported capability schema")
    require(
        re.fullmatch(r"[0-9a-f]{40}", data.get("reviewed_runtime", "")),
        "Runtime must be a full immutable Git SHA",
    )
    rows = checklist_rows()
    ids = set()
    exports = set()
    for record in data["features"]:
        key = record["id"]
        require(
            key not in ids and re.fullmatch(r"[a-z0-9-]+", key),
            "Duplicate/invalid feature ID: " + key,
        )
        ids.add(key)
        require(record["status"] in LABELS, "Invalid status: " + key)
        for field in ["feature", "section", "persistence", "limits"]:
            require(bool(record[field].strip()), "Missing " + field + ": " + key)
        for prefix in record["checklist_prefixes"]:
            require(
                any(covers(prefix, item) for _, item, _ in rows),
                "Unknown checklist prefix: " + prefix,
            )
        for axis in ["rust", "python"]:
            state = record[axis]
            require(state["support"] in SUPPORT, "Invalid support state: " + key)
            require(state["surface"].strip(), "Missing API scope: " + key)
            if state["support"] == "available":
                require(state["sources"], "Available API needs source links: " + key)
            for source in state["sources"]:
                repository_file(source)
        evidence = record["evidence"]
        require(evidence["contracts"], "Missing contract/limits document: " + key)
        for path in evidence["contracts"]:
            repository_file(path)
        for path in evidence["native_tests"]:
            require(
                "#[test]" in read(repository_file(path)),
                "Native evidence has no test case: " + path,
            )
        for path in evidence["python_tests"]:
            tree = ast.parse(read(repository_file(path)))
            require(
                any(
                    isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and n.name.startswith("test_")
                    for n in ast.walk(tree)
                ),
                "Python evidence has no test case: " + path,
            )
        if record["status"] == "bounded":
            require(
                all(record[a]["support"] == "available" for a in ["rust", "python"]),
                "Bounded support requires both APIs: " + key,
            )
            require(
                evidence["native_tests"] and evidence["python_tests"],
                "Bounded support needs both native and Python evidence: " + key,
            )
        if record["status"] == "native_only":
            require(
                record["python"]["support"] == "not_provided"
                and evidence["native_tests"],
                "Native-only support misclassified: " + key,
            )
        exports.update(record["python_exports"])
    expected_exports = public_exports()
    require(
        exports == expected_exports,
        "Export coverage differs: missing="
        + str(sorted(expected_exports - exports))
        + "; unknown="
        + str(sorted(exports - expected_exports)),
    )
    uncovered = [
        item
        for _, item, _ in rows
        if not any(
            covers(p, item) for r in data["features"] for p in r["checklist_prefixes"]
        )
    ]
    require(not uncovered, "Unclassified checklist rows: " + str(uncovered))
    receipt = json.loads(read(repository_file(data["local_receipt"])))
    if verify_runtime:
        paths = sorted(
            p
            for base in ["src", "python-groggy/src", "python-groggy/python/groggy"]
            for p in (ROOT / base).rglob("*")
            if p.suffix in [".rs", ".py", ".pyi"]
        )
        digest = hashlib.sha256(
            "".join(
                str(p.relative_to(ROOT))
                + "\0"
                + hashlib.sha256(p.read_bytes()).hexdigest()
                + "\n"
                for p in paths
            ).encode()
        ).hexdigest()
        require(
            digest == receipt["production_source_tree_sha256"],
            "Runtime differs from the pinned qualification receipt; do not reuse old test evidence",
        )
    return rows, len(expected_exports), receipt


def link(path, label, data):
    relative = Path(path)
    if path.startswith("docs/api/"):
        target = relative.name
    elif path.startswith("docs/"):
        target = "../" + relative.relative_to("docs").as_posix()
    else:
        ref = (
            data["publication_ref"]
            if path.startswith("notes/planning/")
            else data["reviewed_runtime"]
        )
        target = data["repository_url"] + "/blob/" + ref + "/" + relative.as_posix()
    return "[" + (label or relative.name) + "](" + target + ")"


def cell(text):
    return text.replace("|", "\\|").replace("\n", " ")


def render(data, rows, export_count, receipt):
    def reference(path, label=None):
        return link(path, label, data)

    checked = sum(state == "x" for state, _, _ in rows)
    lines = [
        "# Groggy capability matrix",
        "",
        "Reviewed "
        + data["updated"]
        + " against runtime `"
        + data["reviewed_runtime"]
        + "`.",
        "",
        "This is a feature-family map of the public library and original roadmap. "
        "It classifies **all "
        + str(export_count)
        + " declared Python exports and all "
        + str(len(rows))
        + " stable checklist rows**. "
        "Nested methods, detailed domains and compatibility aliases are specified by the linked contracts; this is not a new method-count completion claim.",
        "",
        "**"
        + str(checked)
        + " checked / "
        + str(len(rows) - checked)
        + " open checklist rows.** "
        "Rows overlap. A verified bounded feature does not close its broader parent, promise every backend/domain, or imply production readiness.",
        "",
        "## How to read support",
        "",
        "- **Verified within stated bounds:** Rust and Python surfaces, applicable persistence behavior, contract and native/Python correctness evidence are present for exactly this row's scope.",
        "- **Partial:** working components exist, with remaining target scope explicitly stated.",
        "- **Native only:** native functionality has evidence; a general Python interface is not claimed.",
        "- **Not implemented for this target / Deferred:** declarations, other primitives or future plans do not establish a completed feature.",
        "- **Persistence** distinguishes graph changes from transient views, indexes, numerical outputs and presentation. A successful computation does not itself record a delta.",
        "",
        "Evidence is pinned to the runtime above. Later code changes do not inherit those results automatically. "
        "Source links locate implementations; they are not evidence by themselves.",
        "",
    ]
    sections = list(dict.fromkeys(r["section"] for r in data["features"]))
    for section in sections:
        lines += [
            "## " + section,
            "",
            "| Feature / status | Rust / Python | Persistence | Evidence and limits |",
            "| --- | --- | --- | --- |",
        ]
        for r in data["features"]:
            if r["section"] != section:
                continue
            api = []
            for axis in ["rust", "python"]:
                value = r[axis]
                sources = ", ".join(reference(p) for p in value["sources"])
                api.append(
                    "**"
                    + axis.title()
                    + " ("
                    + value["support"].replace("_", " ")
                    + "):** "
                    + value["surface"]
                    + (" — " + sources if sources else "")
                )
            evidence = r["evidence"]
            refs = ", ".join(reference(p) for p in evidence["contracts"])
            tests = ", ".join(
                reference(p)
                for p in evidence["native_tests"] + evidence["python_tests"]
            )
            proof = refs + (
                "<br>**Tests:** " + tests
                if tests
                else "<br>No feature-completion test claim."
            )
            proof += "<br>**Limits / remaining:** " + r["limits"]
            prefix = ", ".join(r["checklist_prefixes"])
            lines.append(
                "| "
                + " | ".join(
                    cell(c)
                    for c in [
                        '<a id="'
                        + r["id"]
                        + '"></a>**'
                        + r["feature"]
                        + "**<br>"
                        + LABELS[r["status"]]
                        + "<br>Checklist: "
                        + prefix,
                        "<br>".join(api),
                        r["persistence"],
                        proof,
                    ]
                )
                + " |"
            )
        lines.append("")
    lines += [
        "## Qualification and remaining original vision",
        "",
        "Pinned local evidence: **"
        + str(receipt["native"]["passed"])
        + " native passed / "
        + str(receipt["native"]["ignored"])
        + " ignored**, separate serial checkpoint recovery, strict Clippy; "
        "fresh unpacked wheel **"
        + str(receipt["python"]["maintained_passed"])
        + " Python passed / one skipped**, **"
        + str(receipt["python"]["documentation_examples"])
        + " executable docs**. "
        + reference(data["local_receipt"], "Machine-readable qualification")
        + ".",
        "",
        "These counts cover the recorded runtime and bounded tests, not all promised domains. "
        "Production latency/memory/scale and full external-engine parity remain open. "
        "The pending provisional-entity retained-history, schema-history and deployment-budget questions are not answered by this matrix.",
        "",
        "Current open requirements, including deferred work, remain explicit in "
        + reference(
            "notes/planning/graph-engine-flat-checklist.md", "the flat checklist"
        )
        + ". "
        "No original feature is retired by publishing this document. Broader documentation synchronization remains 34.2.",
        "",
        "## Declared Python export mapping",
        "",
        "An export's existence establishes no behavioral guarantee beyond its feature row. "
        "Every public `__all__` name has an explicit classification; private/native-only declarations are not inferred to be Python features.",
        "",
        "| Feature | Declared exports |",
        "| --- | --- |",
    ]
    for r in data["features"]:
        if r["python_exports"]:
            lines.append(
                "| ["
                + r["feature"]
                + "](#"
                + r["id"]
                + ") | "
                + ", ".join("`" + n + "`" for n in r["python_exports"])
                + " |"
            )
    lines += [
        "",
        "## Reproducibility",
        "",
        "Curated record: "
        + reference("docs/api/capabilities.json")
        + ". Render with `python notes/scripts/build_capability_matrix.py`; "
        "verify links, source/test references, full export/checklist coverage and generated output with `--check`. "
        "Use `--check --verify-runtime` to additionally compare current production sources with the pinned local receipt. "
        "The validator never imports the extension or upgrades a status because a method exists.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--verify-runtime", action="store_true")
    args = parser.parse_args()
    try:
        data = json.loads(read(MANIFEST))
        rows, exports, receipt = validate(data, args.verify_runtime)
        output = render(data, rows, exports, receipt)
        if args.check:
            require(
                OUTPUT.is_file() and read(OUTPUT) == output,
                "Capability matrix is stale; run the generator after reviewing changed claims",
            )
        else:
            OUTPUT.write_text(output, encoding="utf-8")
        print(
            "Verified",
            len(data["features"]),
            "capability families,",
            exports,
            "exports,",
            len(rows),
            "checklist rows",
        )
    except (ValueError, KeyError, TypeError) as error:
        print("Capability matrix:", error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
