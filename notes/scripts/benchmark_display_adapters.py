"""Matched historical/native dictionary-display timing and Python allocation receipt.

Run with the in-tree extension after native builds have stopped. No dependencies
are installed. Baseline Python adapters are read from the specified Git revision.
Fixture construction, imports, output verification and allocation measurement are
excluded from timing. tracemalloc measures Python allocations, not native/RSS.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc

ROOT = Path(__file__).resolve().parents[2]


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def load_baseline(revision, directory):
    prefix = "python-groggy/python/groggy/display/"
    paths = subprocess.check_output(
        ["git", "ls-tree", "--name-only", revision, prefix], cwd=ROOT, text=True
    ).splitlines()
    hashes = {}
    for path in paths:
        if not path.endswith(".py"):
            continue
        source = subprocess.check_output(
            ["git", "show", revision + ":" + path], cwd=ROOT
        )
        (directory / Path(path).name).write_bytes(source)
        hashes[path] = hashlib.sha256(source).hexdigest()
    spec = importlib.util.spec_from_file_location(
        "groggy_display_baseline",
        directory / "__init__.py",
        submodule_search_locations=[str(directory)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, hashes


def observe(function, data, options):
    output = function(data, **options)
    samples = []
    for _ in range(7):
        started = time.perf_counter()
        candidate = function(data, **options)
        samples.append(time.perf_counter() - started)
        assert candidate == output
    tracemalloc.start()
    candidate = function(data, **options)
    retained, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert candidate == output
    return {
        "seconds": samples,
        "median_seconds": statistics.median(samples),
        "python_retained_bytes_with_output": retained,
        "python_peak_traced_bytes": peak,
        "output_sha256": digest(output),
        "output_bytes": len(output.encode()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="37f1ff2d")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["NO_COLOR"] = "1"
    import groggy.display as current
    import groggy._groggy as native

    cases = []
    with tempfile.TemporaryDirectory(prefix="groggy-display-baseline-") as scratch:
        previous, baseline_hashes = load_baseline(args.baseline, Path(scratch))
        for kind in ("array", "matrix", "table"):
            for count in (1, 10, 1000, 100_000):
                if kind == "array":
                    data = {
                        "data": [1.23456789] * count,
                        "shape": (count,),
                        "dtype": "f64",
                    }
                elif kind == "matrix":
                    data = {
                        "data": [[1.25, -0.0, "猫😺", True, None, 2**70]] * count,
                        "shape": (count, 6),
                        "dtype": "object",
                    }
                else:
                    columns = ["number", "text", "bool", "null", "integer", "date"]
                    data = {
                        "data": [
                            [1.25, "猫😺abcdefghijk", True, None, 2**70, "2026-11-18"]
                        ]
                        * count,
                        "columns": columns,
                        "dtypes": {
                            "number": "float64",
                            "text": "string",
                            "date": "date",
                        },
                        "shape": (count, 6),
                    }
                before = observe(getattr(previous, "format_" + kind), data, {})
                after = observe(getattr(current, "format_" + kind), data, {})
                equal = before["output_sha256"] == after["output_sha256"]
                # Arrays now correctly distinguish omission rows from literal values.
                assert equal or kind == "array" and count > 10
                cases.append(
                    {
                        "kind": kind,
                        "input_rows": count,
                        "max_rows": 10,
                        "outputs_equal": equal,
                        "before": before,
                        "after": after,
                    }
                )
    current_paths = list((ROOT / "python-groggy/python/groggy/display").glob("*.py"))
    current_paths += list((ROOT / "src/display").rglob("*.rs"))
    current_paths += [ROOT / "python-groggy/src/ffi/display/mod.rs"]
    result = {
        "scope": "12 bounded dictionary-display cases, visible-row budget 10; seven warm timing observations and one separate tracemalloc call per implementation. Fixtures/imports/output checks excluded from time. Python traced peak excludes native allocation, allocator retention and RSS. No production latency or whole-engine performance promise.",
        "python": sys.version,
        "platform": platform.platform(),
        "baseline_revision": subprocess.check_output(
            ["git", "rev-parse", args.baseline], cwd=ROOT, text=True
        ).strip(),
        "baseline_python_sha256": baseline_hashes,
        "current_source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(current_paths)
        },
        "native_module_sha256": hashlib.sha256(
            Path(native.__file__).read_bytes()
        ).hexdigest(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": cases,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
