"""Matched writer-coordinate overhead; run against one rebuilt in-tree extension."""
import json
import subprocess
import time

from groggy import Graph
from groggy.writer import GraphWriter

CONTROL = "4c68b7df"
source = subprocess.check_output(
    ["git", "show", f"{CONTROL}:python-groggy/python/groggy/writer.py"], text=True
)
namespace = {"__name__": "groggy._writer_benchmark_control", "__package__": "groggy"}
exec(compile(source, "previous-writer-control", "exec"), namespace)
PreviousWriter = namespace["GraphWriter"]


def measure(writer_type, nodes, iterations):
    graph = Graph()
    graph.add_nodes(nodes)
    writer = writer_type(graph)
    # Identical setup/input and kernel warmup outside both measured intervals.
    writer.set_node_attr(0, "value", -1)
    clock, cpu = time.perf_counter(), time.process_time()
    if nodes == 1:
        for value in range(iterations):
            writer.set_node_attr(0, "value", value)
    else:
        for value in range(iterations):
            rows = [(node, node + value) for node in range(nodes)]
            writer.set_node_attrs({"value": rows})
    elapsed, cpu = time.perf_counter() - clock, time.process_time() - cpu
    assert graph.get_node_attr(0, "value") == iterations - 1
    assert writer.state_id == graph.current_state_id
    return {"wall_seconds": elapsed, "cpu_seconds": cpu}


cases = []
for nodes, iterations in [(1, 5000), (1000, 20)]:
    samples = []
    for repeat in range(5):
        order = [("previous", PreviousWriter), ("coordinate", GraphWriter)]
        if repeat % 2:
            order.reverse()
        samples.append({name: measure(kind, nodes, iterations) for name, kind in order})
    cases.append({"nodes_per_batch": nodes, "operations": iterations, "samples": samples})
print(json.dumps({
    "benchmark": "python_write_coordinate_overhead", "control_commit": CONTROL,
    "scope": "five alternating-order matched samples on the same loaded extension; fresh owner per sample; current branch unchanged; setup and scalar warmup excluded; bulk Python row preparation included; no RSS/whole-engine guarantee",
    "cases": cases,
}, indent=2))
