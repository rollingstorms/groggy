"""Captured collection column projection, with separate wall/process CPU samples."""
import argparse
import json
import time
from pathlib import Path

from groggy import Graph


def measure(call, verify):
    samples = []
    for _ in range(5):
        wall = time.perf_counter_ns()
        cpu = time.process_time_ns()
        result = call()
        cpu = time.process_time_ns() - cpu
        wall = time.perf_counter_ns() - wall
        verify(result)
        samples.append({"wall_ns": wall, "cpu_ns": cpu})
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    cases = []
    for count, groups in [(1000, 1), (100000, 1), (100000, 100)]:
        graph = Graph()
        graph.add_nodes([{"group": node % groups, "x": node, "y": 2 * node} for node in range(count)])
        selected = graph.view().group_by("group", "nodes")
        before = graph.current_state_id
        expected = count * (count - 1) // 2

        def verify_tables(tables):
            assert len(tables) == groups
            assert sum(len(tables[i]) for i in range(groups)) == count
            assert sum(tables[i].column("x").sum() for i in range(groups)) == expected
            assert sum(tables[i].column("y").sum() for i in range(groups)) == 2 * expected

        def verify_arrays(arrays):
            assert len(arrays) == groups
            assert sum(arrays[i].sum() for i in range(groups)) == expected

        cases.append({"nodes": count, "groups": groups,
                      "tables": measure(lambda: selected[["x", "y"]], verify_tables),
                      "observations": measure(lambda: selected["x"], verify_arrays)})
        assert graph.current_state_id == before
    args.output.write_text(json.dumps({"label": args.label, "scope": "release in-tree extension; captured grouped nodes held; graph/selection construction and output validation excluded; no owner mutations; x/y exact i64 values; wall and process CPU are distinct; no RSS/allocation claim", "cases": cases}, indent=2) + "\n")


if __name__ == "__main__":
    main()
