"""Warm connected sparse export; source preparation and scalar checks excluded."""
import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter_ns

from groggy import Graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = []
    for count in (1_000, 100_000):
        graph = Graph(manual=True, directed=True)
        graph.add_nodes(count)
        graph.add_edges([(node - 1, node) for node in range(1, count)])
        read = graph.view().adjacency_matrix
        read(sparse=True)
        samples = []
        for _ in range(5):
            start = perf_counter_ns()
            matrix = read(sparse=True)
            samples.append(perf_counter_ns() - start)
            assert matrix.shape == (count, count)
            assert matrix.is_sparse
            assert matrix.get(0, 1) == 1
            assert matrix.get(count - 1, 0) == 0
        cases.append({"nodes": count, "edges": count - 1, "nanoseconds": samples, "median_nanoseconds": median(samples)})
    args.output.write_text(json.dumps({"fixture": "release extension; directed chain; five warmed connected sparse matrix exports; source setup/warmup/scalar validation excluded; sparse construction and Python wrapper included; no dense intermediate", "cases": cases}, indent=2) + "\n")


if __name__ == "__main__":
    main()
