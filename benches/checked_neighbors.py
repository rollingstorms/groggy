"""Matched public collection/mode benchmark for an explicitly selected wheel.

Run with an unpacked wheel first on PYTHONPATH. No installation is performed.
Graph construction, pipeline compilation and one warm-up run are excluded.
Reported time includes checked step-body execution and Python output marshalling.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import time

import groggy
from groggy import Graph, GraphBatch, Pipeline
from groggy.builder import step_pipeline


def step(kind, **params):
    return {"id": kind, "params": params}


def measure(nodes, iterations, observations, kind):
    graph = Graph(directed=True)
    batch = GraphBatch(effective_at=0)
    for _ in range(nodes):
        batch.add_node()
    rows = [[(i + 1) % nodes] for i in range(nodes)]
    for i, neighbors in enumerate(rows):
        batch.add_edge(i, neighbors[0])
        if i % 10 == 0:
            batch.add_edge(i, neighbors[0])
            neighbors.append(neighbors[0])
        if i % 4 == 0:
            batch.add_edge(i, i)
            neighbors.append(i)
    graph.writer().apply_batch(batch)
    selected = graph.view()
    expected = []
    for neighbors in rows:
        counts = Counter(neighbors)
        most = max(counts.values())
        expected.append(min(n for n, count in counts.items() if count == most))
    if kind == "native":
        loop = step(
            "iter.loop",
            iterations=iterations,
            body=[
                step(
                    "core.collect_neighbor_values",
                    source="values",
                    target="lists",
                    include_self=False,
                ),
                step(
                    "core.mode_list", source="lists", target="mode", tie_break="lowest"
                ),
            ],
        )
    else:
        loop = step(
            "iter.loop",
            iterations=iterations,
            body=[],
            batch_plan={
                "slot_count": 2,
                "carried_slots": [],
                "instructions": [
                    {"type": "load_node_prop", "dst": 0, "var_name": "values"},
                    {
                        "type": "neighbor_mode",
                        "src": 0,
                        "dst": 1,
                        "tie_break": "lowest",
                        "direction": "out",
                    },
                    {"type": "store_node_prop", "src": 1, "var_name": "mode"},
                ],
            },
        )
    p = Pipeline(
        [
            step_pipeline(
                "checked neighbors",
                [
                    step("core.init_nodes_with_index", target="values"),
                    loop,
                    step("core.attach_node_attr", source="mode", attr="answer"),
                ],
                atomic=False,
            )
        ]
    )
    samples = []
    before = graph.current_state_id
    for observation in range(observations + 1):
        started = time.perf_counter()
        _, profile = p.run(selected, persist=False, return_profile=True)
        elapsed = time.perf_counter() - started
        result = dict(profile["outputs"]["node_attributes.answer"])
        assert [result[i] for i in range(nodes)] == expected
        assert graph.current_state_id == before
        if observation:
            samples.append(elapsed)
    return {
        "kind": kind,
        "nodes": nodes,
        "edges": sum(map(len, rows)),
        "iterations": iterations,
        "observations_seconds": samples,
        "median_seconds": statistics.median(samples),
        "checksum": sum(expected),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--observations", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    report = {
        "package": str(Path(groggy.__file__).resolve()),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Directed ring with parallel edges and self-loops; pinned memory view; collection/lowest mode in a native step-body or equivalent explicit-out neighbor_mode register loop, then transient attachment. Register mode is the unported vector interpreter before migration and checked native lowering afterward. Includes output marshalling; excludes setup/compilation/warm-up. No atomic/durable acceptance, allocator, RSS or production budget claim.",
        "cases": [
            measure(n, i, args.observations, kind)
            for n, i in [(1000, 10), (10000, 10), (1000, 100), (10000, 100)]
            for kind in ["native", "register"]
        ],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "cases": [
                    {
                        "kind": c["kind"],
                        "nodes": c["nodes"],
                        "iterations": c["iterations"],
                        "median_seconds": c["median_seconds"],
                    }
                    for c in report["cases"]
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
