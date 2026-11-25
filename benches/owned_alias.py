"""Matched owned-map alias workload; run against an explicit unpacked wheel.

Setup/compilation and warm-up are excluded; execution and final Python output
marshalling are included. This is a transient memory workload, not durability,
allocator or RSS evidence. No package installation is needed.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import groggy
from groggy import Graph, Pipeline
from groggy.builder import step_pipeline


def step(kind, **params):
    return {"id": kind, "params": params}


def measure(nodes, iterations, observations, aliases):
    graph = Graph()
    graph.add_nodes(nodes)
    steps = [step("core.init_nodes", target="original", value=7)]
    if aliases:
        steps.append(
            step(
                "iter.loop",
                iterations=iterations,
                body=[step("alias", source="original", target="copy")],
            )
        )
    steps.append(
        step(
            "core.attach_node_attr",
            source="copy" if aliases else "original",
            attr="answer",
        )
    )
    pipeline = Pipeline([step_pipeline("owned aliases", steps, atomic=False)])
    selected = graph.view()
    head = graph.current_state_id
    samples = []
    for observation in range(observations + 1):
        start = time.perf_counter()
        _, profile = pipeline.run(selected, persist=False, return_profile=True)
        elapsed = time.perf_counter() - start
        rows = dict(profile["outputs"]["node_attributes.answer"])
        assert rows == dict.fromkeys(range(nodes), 7)
        assert graph.current_state_id == head
        if observation:
            samples.append(elapsed)
    return {
        "kind": "alias" if aliases else "control",
        "nodes": nodes,
        "iterations": iterations,
        "observations_seconds": samples,
        "median_seconds": statistics.median(samples),
        "checksum": sum(rows.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--observations", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    report = {
        "package": str(Path(groggy.__file__).resolve()),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": __doc__,
        "cases": [
            measure(nodes, iterations, args.observations, aliases)
            for nodes, iterations in [
                (1000, 10),
                (10000, 10),
                (1000, 100),
                (10000, 100),
            ]
            for aliases in [False, True]
        ],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            [
                {k: row[k] for k in ("kind", "nodes", "iterations", "median_seconds")}
                for row in report["cases"]
            ]
        )
    )


if __name__ == "__main__":
    main()
