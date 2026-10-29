"""Time connected whole-selection column reads; setup and validation excluded."""
import argparse
import json
from pathlib import Path
from time import perf_counter_ns
from statistics import median

from groggy import Graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = []
    value = 0.123456789012345
    for count in (1_000, 100_000):
        graph = Graph(manual=True)
        graph.add_nodes([{"score": value} for _ in range(count)])
        selected = graph.view()
        read = selected.__getitem__
        read("score")
        samples = []
        for _ in range(7):
            start = perf_counter_ns()
            column = read("score")
            samples.append(perf_counter_ns() - start)
            assert column.to_list() == [value] * count
        cases.append({"nodes": count, "nanoseconds": samples, "median_nanoseconds": median(samples)})
    args.output.write_text(json.dumps({"fixture": "release extension; homogeneous exact f64 score column; connected selection; seven Python bound method calls; setup, initial warmup and to_list validation excluded; output BaseArray construction included", "cases": cases}, indent=2) + "\n")


if __name__ == "__main__":
    main()
