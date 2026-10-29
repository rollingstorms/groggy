"""Absolute frozen visualization capture cost; initial graph capture separate."""
import argparse
import json
from pathlib import Path
from time import perf_counter_ns

from groggy import Graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = []
    for count in (1_000, 100_000):
        graph = Graph(manual=True, directed=True)
        graph.add_nodes([{"score": 0.123456789012345} for _ in range(count)])
        graph.add_edges([(node - 1, node) for node in range(1, count)])
        started = perf_counter_ns()
        root = graph.read_view()
        capture = perf_counter_ns() - started
        for size in (16, count):
            selected = root.select_nodes(list(range(size)))
            times = []
            for _ in range(5):
                started = perf_counter_ns()
                accessor = selected.viz
                times.append(perf_counter_ns() - started)
                del accessor
            cases.append({"owner_nodes": count, "selected_nodes": size, "selected_edges": size - 1, "owner_capture_nanoseconds": capture, "viz_capture_nanoseconds": times})
    args.output.write_text(json.dumps({"fixture": "release extension; five frozen visualization captures; directed chain with f64 node properties; source setup/capture separate; selected payload construction and Python accessor included; no server/layout or comparison to incorrect whole-owner capture", "cases": cases}, indent=2) + "\n")


if __name__ == "__main__":
    main()
