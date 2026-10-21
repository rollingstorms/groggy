"""Profile the checked numerical slice against an in-tree Python extension.

Example: PYTHONPATH=/path/to/in-tree-package python3 benches/matrix_statistics_profile.py
Reports wall time and whole-process peak RSS, not per-operation allocation.
"""
import argparse
import json
import resource
import statistics
import sys
import time

from groggy import GraphMatrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--cols", type=int, default=384)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    if min(args.rows, args.cols, args.repeat) <= 1:
        parser.error("rows, cols and repeat must exceed one")
    row = [float(i % 17) for i in range(args.cols)]
    # Outer inputs are columns in the existing factory.
    matrix = GraphMatrix.from_data([[value] * args.rows for value in row])
    assert matrix.shape == (args.rows, args.cols)
    expected = statistics.stdev(row)
    assert abs(matrix.std_axis(0).to_list()[0] - expected) < 1e-12
    timings = {}
    for axis in (0, 1):
        matrix.std_axis(axis)  # warmup
        samples = []
        for _ in range(args.repeat):
            start = time.perf_counter()
            result = matrix.std_axis(axis)
            samples.append((time.perf_counter() - start) * 1000)
        assert len(result.to_list()) == (args.rows if axis == 0 else args.cols)
        timings[str(axis)] = {"median_ms": statistics.median(samples), "samples_ms": samples}
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        rss *= 1024
    print(json.dumps({"shape": [args.rows, args.cols], "dtype": "f64", "ddof": 1,
                      "axis_0": "one result per row", "axis_1": "one result per column",
                      "timings": timings, "process_peak_rss_bytes": rss}, indent=2))


if __name__ == "__main__":
    main()
