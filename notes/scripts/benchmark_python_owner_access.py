"""Measure public access paths against a selected unpacked package directory.

Run with python -I -S SCRIPT PACKAGE_ROOT; useful for matched old-wheel/current
tree comparison without installation. Timers include Python/FFI call overhead;
they are not measurements of FFI overhead alone. No external packages required.
"""
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
import groggy as gr


def measure(operation, iterations, samples=7):
    values = []
    for _ in range(samples):
        started = time.perf_counter_ns()
        for _ in range(iterations):
            operation()
        values.append((time.perf_counter_ns() - started) / iterations)
    return {"iterations": iterations, "samples_ns_per_call": values,
            "median_ns_per_call": statistics.median(values)}


graph = gr.Graph()
graph.add_nodes([{"score": float(i)} for i in range(1000)])
view = graph.read_view()
writer = graph.writer()
columns = {"score": [(i, float(i)) for i in range(1000)]}
results = {
    "package": str(Path(gr.__file__).resolve()), "version": gr.__version__,
    "scope": "Seven samples per public operation, 1k scalar nodes, unchanged captured source; steady warm cache; timing includes Python loop, dispatch, FFI and native work. Bulk includes Python batch staging plus checked native publication. No allocation/RSS/GIL-concurrency or isolated 100ns FFI-budget claim.",
    "live_property": measure(lambda: graph.get_node_attr(0, "score"), 100000),
    "captured_property": measure(lambda: view.node_attr(0, "score"), 100000),
    "warm_capture": measure(graph.read_view, 100000),
    "bulk_1000_properties": measure(lambda: writer.set_node_attrs(columns), 10),
}
assert view.node_attr(0, "score") == graph.get_node_attr(0, "score") == 0.0
print(json.dumps(results, indent=2))
