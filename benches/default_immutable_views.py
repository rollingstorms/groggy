"""In-tree release timings: cold immutable capture, warm default/live cache, owner degree.

Run after native builds are quiet; graph construction is excluded.
"""
import argparse
import gc
import json
import statistics
import time
from pathlib import Path

from groggy import Graph


def timed(call, iterations):
    wall = time.perf_counter_ns()
    cpu = time.process_time_ns()
    for _ in range(iterations):
        result = call()
    return {
        "wall_ns_per_call": (time.perf_counter_ns() - wall) / iterations,
        "cpu_ns_per_call": (time.process_time_ns() - cpu) / iterations,
    }, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = []
    for count in [1_000, 100_000]:
        graph = Graph()
        graph.add_nodes([{"score": float(node)} for node in range(count)])
        graph.add_edges([(node - 1, node) for node in range(1, count)])
        cold = []
        for _ in range(5):
            sample, view = timed(graph.view, 1)
            cold.append(sample)
            assert view.node_count() == count
            del view
            gc.collect()
        frozen = graph.view()
        live = graph.connected_view()
        graph.degree(0)
        methods = {
            "warm_default_view": graph.view,
            "warm_explicit_connected_view": graph.connected_view,
            "warm_owner_degree": lambda: graph.degree(0),
        }
        warm = {}
        for name, call in methods.items():
            samples = []
            for _ in range(7):
                sample, result = timed(call, 1024)
                samples.append(sample)
                if name == "warm_default_view":
                    assert result is frozen
                elif name == "warm_explicit_connected_view":
                    assert result is live
                else:
                    assert result == 1
            warm[name] = samples
        meta = graph.induced_subgraph([0]).collapse()
        expansion = []
        for _ in range(7):
            sample, result = timed(meta.expand, 1024)
            expansion.append(sample)
            assert result.node_count() == 1
        cases.append({"nodes": count, "edges": count - 1, "cold_capture": cold, "warm_meta_expansion": expansion, **warm})
    args.output.write_text(json.dumps({
        "fixture": "undirected chain with one f64 score per node; release in-tree extension; construction excluded; cold views dropped between captures; warm captured/live views held; process CPU separate from elapsed wall time",
        "cases": cases,
    }, indent=2) + "\n")
    for case in cases:
        print(case["nodes"], {key: statistics.median(row["wall_ns_per_call"] for row in value) for key, value in case.items() if isinstance(value, list)})


if __name__ == "__main__":
    main()
