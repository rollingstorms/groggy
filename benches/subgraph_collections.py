"""Bounded matched-wheel collection benchmark; fresh owners stay outside timing.

No third-party graph engine or package installation is involved. Reports per-call
wall/process CPU samples; it does not measure allocations/RSS or establish an SLA.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import tempfile
import time
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 1000])
    parser.add_argument("--samples", type=int, default=11)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if args.samples < 3 or any(n < 2 for n in args.sizes):
        parser.error("need at least three samples and at least two nodes")
    with tempfile.TemporaryDirectory(prefix="groggy-collection-bench-") as directory:
        with zipfile.ZipFile(args.wheel) as archive:
            archive.extractall(directory)
        sys.path.insert(0, directory)
        from groggy import Graph
        from groggy.builder import step_pipeline

        def step(kind, **params):
            return {"id": kind, "params": params}

        workloads = {
            "per_node_passthrough": [
                step("sample.iterate_nodes", target="seeds"),
                step("sample.for_each", source="seeds", target="mapped", body=[]),
            ],
            "overlapping_edges_write": [
                step("sample.iterate_edges", target="seeds"),
                step(
                    "sample.for_each",
                    source="seeds",
                    target="mapped",
                    body=[
                        step(
                            "core.load_node_attr",
                            attr="counter",
                            default=0,
                            target="old",
                        ),
                        step("core.init_scalar", value=1, target="one"),
                        step("core.add", left="old", right="one", target="next"),
                        step("core.attach_node_attr", source="next", attr="counter"),
                    ],
                ),
            ],
        }
        rows = []
        for size in args.sizes:
            for name, steps in workloads.items():
                program = step_pipeline("collection benchmark", steps, atomic=True)
                wall, cpu = [], []
                for sample in range(args.samples + 3):
                    graph = Graph(directed=True)
                    graph.add_nodes(size)
                    graph.add_edges([(i, i + 1) for i in range(size - 1)])
                    before = graph.current_state_id
                    start_cpu, start_wall = (
                        time.process_time_ns(),
                        time.perf_counter_ns(),
                    )
                    graph.apply(program)
                    elapsed_wall, elapsed_cpu = (
                        time.perf_counter_ns() - start_wall,
                        time.process_time_ns() - start_cpu,
                    )
                    if name == "overlapping_edges_write":
                        assert graph.get_node_attr(0, "counter") == 1
                        assert graph.get_node_attr(size - 1, "counter") == 1
                        if size > 2:
                            assert graph.get_node_attr(1, "counter") == 2
                        assert graph.current_state_id == before + 1
                    else:
                        assert graph.current_state_id == before
                    if sample >= 3:
                        wall.append(elapsed_wall)
                        cpu.append(elapsed_cpu)
                rows.append(
                    {
                        "nodes": size,
                        "edges": size - 1,
                        "workload": name,
                        "wall_ns": wall,
                        "cpu_ns": cpu,
                        "median_wall_ns": statistics.median(wall),
                        "median_cpu_ns": statistics.median(cpu),
                    }
                )
        args.output.write_text(
            json.dumps(
                {
                    "runtime": args.runtime,
                    "wheel": args.wheel.name,
                    "wheel_sha256": hashlib.sha256(args.wheel.read_bytes()).hexdigest(),
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "scope": "Directed path, fresh memory owner outside timing, explicit atomic step builder; three warmups. No allocation/RSS or production scale claim.",
                    "samples_per_case": args.samples,
                    "results": rows,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
