"""Matched valid-input normalization pipelines against an explicit unpacked wheel.

Setup, compilation and warm-up are excluded; transient execution, attachment and
Python output marshalling are timed. Alias controls retain the same loop frame.
No storage synchronization, allocator, RSS or production latency claim is made.
"""

import argparse
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

import groggy
from groggy import Graph, Pipeline
from groggy.builder import step_pipeline


def step(kind, **params):
    return {"id": kind, "params": params}


def measure(nodes, iterations, method, observations):
    graph = Graph()
    graph.add_nodes(nodes)
    values = {node: 1.0 + (node % 7) / 4.0 for node in range(nodes)}
    graph.writer().set_node_attrs({"score": values})
    raw = list(values.values())
    total, maximum, minimum = math.fsum(raw), max(raw), min(raw)
    mean = total / nodes
    std = math.sqrt(math.fsum((value - mean) ** 2 for value in raw) / nodes)
    if method == "control":
        operation = step("alias", source="values", target="out")
        expected = raw
    elif method in ("sum", "max", "minmax"):
        operation = step(
            "core.normalize_values", source="values", target="out", method=method
        )
        expected = [
            (
                value / total
                if method == "sum"
                else (
                    value / maximum
                    if method == "max"
                    else (value - minimum) / (maximum - minimum)
                )
            )
            for value in raw
        ]
    elif method == "standardize":
        operation = step("core.standardize", source="values", target="out")
        expected = [(value - mean) / std for value in raw]
    else:
        operation = step("core.clip", source="values", target="out", min=1.25, max=2.0)
        expected = [min(2.0, max(1.25, value)) for value in raw]
    program = Pipeline(
        [
            step_pipeline(
                "valid normalization",
                [
                    step("core.load_node_attr", attr="score", target="values"),
                    step("iter.loop", iterations=iterations, body=[operation]),
                    step("core.attach_node_attr", source="out", attr="answer"),
                ],
                atomic=False,
            )
        ]
    )
    selected, head = graph.view(), graph.current_state_id
    samples = []
    for observation in range(observations + 1):
        start = time.perf_counter()
        _, profile = program.run(selected, persist=False, return_profile=True)
        elapsed = time.perf_counter() - start
        rows = dict(profile["outputs"]["node_attributes.answer"])
        assert len(rows) == nodes
        assert all(
            math.isclose(rows[node], value, rel_tol=1e-12, abs_tol=1e-12)
            for node, value in enumerate(expected)
        )
        assert graph.current_state_id == head
        if observation:
            samples.append(elapsed)
    return {
        "method": method,
        "nodes": nodes,
        "iterations": iterations,
        "observations_seconds": samples,
        "median_seconds": statistics.median(samples),
        "checksum": math.fsum(rows.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--package-root", required=True, type=Path)
    parser.add_argument("--observations", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    assert args.package_root.resolve() in Path(groggy.__file__).resolve().parents
    assert (
        args.package_root.resolve() in Path(groggy._groggy.__file__).resolve().parents
    )
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "package": str(Path(groggy.__file__).resolve()),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": __doc__,
        "cases": [
            measure(nodes, iterations, method, args.observations)
            for nodes in (1000, 10000)
            for iterations in (1, 10)
            for method in ("control", "sum", "max", "minmax", "standardize", "clip")
        ],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            [
                {k: row[k] for k in ("method", "nodes", "iterations", "median_seconds")}
                for row in report["cases"]
            ]
        )
    )


if __name__ == "__main__":
    main()
