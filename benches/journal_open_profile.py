"""Bounded warm-open replay baseline; uses only the in-tree Groggy API."""
import argparse
import json
import platform
import statistics
import tempfile
import time
from pathlib import Path

from groggy import DurableGraph


def profile(updates, samples):
    with tempfile.TemporaryDirectory(prefix="groggy-open-profile-") as temporary:
        path = Path(temporary) / "history.ggj"
        with DurableGraph.create(path) as graph:
            writer = graph.writer(author="opening-profile")
            node = writer.add_node()
            for value in range(updates):
                writer.set_node_attr(node, "score", value)
            state = graph.current_state_id
        del writer, graph
        size = path.stat().st_size
        wall_samples, cpu_samples = [], []
        # Warmup is explicit: this is not a cold-device or reboot benchmark.
        with DurableGraph.open(path) as warmup:
            assert warmup.current_state_id == state
        del warmup
        for _ in range(samples):
            cpu_start = time.process_time_ns()
            wall_start = time.perf_counter_ns()
            graph = DurableGraph.open(path)
            elapsed_wall = time.perf_counter_ns() - wall_start
            elapsed_cpu = time.process_time_ns() - cpu_start
            try:
                assert graph.current_state_id == state
                assert graph.get_node_attr(node, "score") == (
                    updates - 1 if updates else None
                )
            finally:
                graph.close()
            del graph
            wall_samples.append(elapsed_wall)
            cpu_samples.append(elapsed_cpu)
        return {
            "live_nodes": 1,
            "live_edges": 0,
            "property_updates": updates,
            "accepted_batches": updates + 1,
            "journal_bytes": size,
            "open_wall_ns": wall_samples,
            "open_process_cpu_ns": cpu_samples,
            "median_wall_ns": statistics.median(wall_samples),
            "median_process_cpu_ns": statistics.median(cpu_samples),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--updates", type=int, nargs="+", default=[0, 100, 1000])
    args = parser.parse_args()
    if args.samples < 1 or any(count < 0 for count in args.updates):
        parser.error("samples must be positive and update counts nonnegative")
    result = {
        "profile": "development in-tree native Python extension",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "sample_count": args.samples,
        "scope": (
            "Warm filesystem cache, fixed one-node graph, growing checked journal. "
            "Open includes lock/header/checksum/replay through API return; setup, "
            "validation and close excluded. Process CPU is not kernel-only compute. "
            "No cold-open, release, memory, scale or SLA claim."
        ),
        "cases": [profile(count, args.samples) for count in args.updates],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for case in result["cases"]:
        print(
            f"{case['accepted_batches']} batches, {case['journal_bytes']} bytes: "
            f"wall={case['median_wall_ns'] / 1e6:.3f} ms, "
            f"CPU={case['median_process_cpu_ns'] / 1e6:.3f} ms"
        )


if __name__ == "__main__":
    main()
