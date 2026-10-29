"""Actual connected adjacency export, including Python dictionary marshalling."""
import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter_ns

from groggy import Graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1_000, 5_000, 100_000])
    args = parser.parse_args()
    cases = []
    for count in args.sizes:
        graph = Graph(manual=True)
        graph.add_nodes(count)
        graph.add_edges([(node - 1, node) for node in range(1, count)])
        read = graph.view().adjacency_list
        read()
        samples = []
        for _ in range(5):
            start = perf_counter_ns()
            rows = read()
            samples.append(perf_counter_ns() - start)
            assert len(rows) == count
            for node, neighbors in rows.items():
                expected = ([node - 1] if node else []) + ([node + 1] if node + 1 < count else [])
                assert sorted(neighbors) == expected
        cases.append({"nodes": count, "edges": count - 1, "nanoseconds": samples, "median_nanoseconds": median(samples)})
    args.output.write_text(json.dumps({"fixture": "release-extension connected undirected chain; five warmed bound-method exports; dictionary/list construction included; setup, initial warmup and analytic validation excluded; old unordered rows validated as multisets", "cases": cases}, indent=2) + "\n")


if __name__ == "__main__":
    main()
