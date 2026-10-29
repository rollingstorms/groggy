"""Explicit selected-copy cost; source setup/capture excluded from copy timings."""
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
        view = graph.read_view()
        capture = perf_counter_ns() - started
        timings = []
        for _ in range(5):
            started = perf_counter_ns()
            copy = view.to_graph()
            timings.append(perf_counter_ns() - started)
            assert copy.node_count() == count
            assert copy.edge_count() == count - 1
            assert copy.current_state_id == 1
            assert copy.get_node_attr(count - 1, "score") == 0.123456789012345
            del copy
        cases.append({"nodes": count, "edges": count - 1, "source_capture_nanoseconds": capture, "copy_nanoseconds": timings})
    args.output.write_text(json.dumps({"fixture": "release extension; directed chain with exact f64 node property; five independent publishing selected copies; source setup excluded and capture reported separately; copy includes destination installation and one accepted initial state; no comparison with the former empty placeholder or RSS/GIL claim", "cases": cases}, indent=2) + "\n")


if __name__ == "__main__":
    main()
