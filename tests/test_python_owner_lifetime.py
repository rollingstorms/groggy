"""Foreign-thread GC must free owners without granting foreign live access."""
import json
import os
import subprocess
import sys

import pytest


PROGRAM = r'''
import gc, json, sys, threading
from pathlib import Path
from groggy import Graph, DurableGraph, GraphMatrix, num_array

kind, path, creator_exits = sys.argv[1], Path(sys.argv[2]), sys.argv[3] == "yes"
errors = []
sys.unraisablehook = lambda event: errors.append(str(event.exc_value))
gc.disable()

def build():
    if kind == "durable":
        graph = DurableGraph.create(path)
        graph.writer().add_node(score=1.25)
        return graph
    graph = Graph()
    graph.add_nodes([{"score": 1.25}, {}])
    graph.add_edge(0, 1)
    choices = {
        "graph": lambda: graph,
        "writer": graph.writer,
        "nodes": lambda: graph.nodes,
        "edges": lambda: graph.edges,
        "node": lambda: graph.nodes[0],
        "edge": lambda: graph.edges[0],
        "connected": graph.connected_view,
        "frozen": graph.view,
        "view": graph.read_view,
        "table": lambda: graph.read_view().table(),
        "matrix": lambda: GraphMatrix.from_data([[1., 2.], [3., 4.]]),
        "array": lambda: num_array([1., 2.]),
        "snapshot": lambda: graph.snapshot_at_commit(graph.current_state_id),
    }
    return choices[kind]()

if creator_exits:
    values = []
    creator = threading.Thread(target=lambda: values.append(build()))
    creator.start(); creator.join()
    value = values.pop()
else:
    value = build()

cycle = [value]
cycle.append(cycle)
del value, cycle
if creator_exits:
    gc.collect()
else:
    worker = threading.Thread(target=gc.collect)
    worker.start(); worker.join()
assert not errors, errors
if kind == "durable":
    # This requires the native destructor to release the file lock immediately,
    # without close(), creator-thread polling or waiting for thread shutdown.
    with DurableGraph.open(path) as reopened:
        assert reopened.get_node_attr(0, "score") == 1.25
print(json.dumps(errors))
'''


@pytest.mark.parametrize("kind", [
    "graph", "writer", "nodes", "edges", "node", "edge", "connected",
    "frozen", "view", "table", "matrix", "array", "snapshot", "durable",
])
@pytest.mark.parametrize("creator_exits", [False, True])
def test_foreign_cyclic_collection_releases_graph_family_without_unraisable_errors(
    tmp_path, kind, creator_exits
):
    run = subprocess.run(
        [sys.executable, "-c", PROGRAM, kind, str(tmp_path / "graph.ggj"),
         "yes" if creator_exits else "no"],
        env=os.environ.copy(), capture_output=True, text=True, timeout=30,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert json.loads(run.stdout) == []


@pytest.mark.parametrize("kind", [
    "graph", "node", "connected", "durable", "durable-close", "durable-status",
])
def test_foreign_live_mutation_rejects_and_creator_can_continue(tmp_path, kind):
    from concurrent.futures import ThreadPoolExecutor
    from groggy import DurableGraph, Graph

    durable = kind.startswith("durable")
    graph = DurableGraph.create(tmp_path / "graph.ggj") if durable else Graph()
    try:
        graph.writer().add_node(score=1)
        if kind == "durable-close":
            operation = graph.close
        elif kind == "durable-status":
            operation = lambda: graph.is_closed
        elif kind == "node":
            target = graph.nodes[0]
            operation = lambda: target.__setitem__("score", 2)
        elif kind == "connected":
            target = graph.connected_view()
            operation = lambda: target.nodes[0].__setitem__("score", 2)
        else:
            operation = lambda: graph.writer().set_node_attr(0, "score", 2)
        with ThreadPoolExecutor(max_workers=1) as pool:
            with pytest.raises(BaseException, match="creating thread"):
                pool.submit(operation).result(timeout=10)
        assert graph.get_node_attr(0, "score") == 1
        graph.writer().set_node_attr(0, "score", 3)
        assert graph.get_node_attr(0, "score") == 3
    finally:
        if durable:
            graph.close()
