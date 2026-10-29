"""Measure the actual Python cached full-view call, without immutable capture.

Run with the in-tree extension, after native builds are quiet:
    python benches/current_full_view_cache.py --output /tmp/full-view.json
"""
import argparse
import json
import time
from pathlib import Path

from groggy import Graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = []
    for count in [1_000, 100_000]:
        graph = Graph()
        graph.add_nodes(count)
        graph.add_edges([(node - 1, node) for node in range(1, count)])
        call = graph.view
        first = call()
        assert first.node_count() == count
        assert first.edge_count() == count - 1
        samples = []
        for _ in range(7):
            start = time.perf_counter_ns()
            for _ in range(128):
                last = call()
            samples.append((time.perf_counter_ns() - start) / 128)
            assert last is first
        cases.append({"nodes": count, "edges": count - 1, "nanoseconds_per_call": samples})
    args.output.write_text(json.dumps({
        "fixture": "unchanged connected full view on a chain, no immutable capture; release extension; 7 samples of 128 Python calls; source construction excluded",
        "cases": cases,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
