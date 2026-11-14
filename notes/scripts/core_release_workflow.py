"""Executable embedded-core acceptance workflow; uses no external graph library.

Run with the in-tree extension:
PYTHONPATH=python-groggy/python python notes/scripts/core_release_workflow.py
"""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from groggy import DurableGraph, Graph, GraphBatch


PRECISE = 1.0000000000000002


def exercise(graph):
    """Combine batch identity, table edits, time reads and tracked restoration."""
    batch = GraphBatch(effective_at=100)
    left = batch.add_node(score=PRECISE, payload=b"\x00\xff")
    right = batch.add_node(score=2.0)
    batch.add_node(score=0.0)  # An isolate is part of the graph too.
    batch.set_node_external_id(left, "left")
    batch.set_node_external_id(right, "right")
    batch.add_edge(left, right, weight=2.5)
    batch.add_edge(left, right, weight=1.0)  # Distinct parallel fact.
    batch.add_edge(left, left, weight=0.5)
    graph.writer().apply_batch(batch)
    initial = graph.current_state_id
    original = graph.read_view()
    assert original.node_ids == [0, 1, 2]
    assert original.edge_ids == [0, 1, 2]

    # Sorted, projected rows retain source IDs without an ID column.
    table = (original.select_nodes([0, 1]).nodes_table().base_table()
             .sort_by("score", ascending=False).select(["score"]))
    assert table.source_entity_ids == [1, 0]
    table["score"] = [20.0, 10.0]
    updated = table.update_source_graph(graph, ["score"], effective_at=200)
    edited = graph.current_state_id
    assert edited == initial + 1
    assert updated["view"].node_attr(0, "score") == 10.0
    assert original.node_attr(0, "score") == PRECISE
    assert graph.read_view_at(150).node_attr(0, "score") == PRECISE
    assert graph.read_view_at(250).node_attr(0, "score") == 10.0

    # A failed mixed mutation must not partially publish its valid assignment.
    invalid = GraphBatch(effective_at=250)
    invalid.set_node_attr(0, "score", -999.0)
    invalid.set_edge_attr(999, "weight", 7.0)
    try:
        graph.writer().apply_batch(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid edge assignment must reject the batch")
    assert graph.current_state_id == edited
    assert graph.get_node_attr(0, "score") == 10.0

    # Exact edge selection excludes the second parallel edge; numerics consume
    # that same immutable selection, not the owner's broader topology.
    selected = original.select_edges([0, 2])
    matrix = selected.adjacency_matrix(sparse=False, weight_attr="weight")
    assert matrix.shape == (2, 2)
    assert matrix.get(0, 0) == 0.5
    assert matrix.get(0, 1) == matrix.get(1, 0) == 2.5
    assert abs(matrix.determinant() + 6.25) < 1e-12
    assert graph.current_state_id == edited  # Analysis creates no delta.

    graph.checkpoint("before restoration", "release-workflow")
    assert graph.current_state_id == edited
    restored = graph.writer().restore_state(initial, mode="tracked", effective_at=300)
    assert restored == edited + 1
    assert graph.read_view_at(250).node_attr(0, "score") == 10.0
    assert graph.read_view_at(350).node_attr(0, "score") == PRECISE
    assert updated["view"].node_attr(0, "score") == 10.0
    graph.check_integrity()
    return {"initial": initial, "edited": edited, "restored": restored}, original


def verify_reopened(graph, states):
    assert graph.current_state_id == states["restored"]
    assert graph.node_by_external_id("left") == 0
    assert graph.get_node_attr(0, "payload") == b"\x00\xff"
    assert graph.read_view().edge_ids == [0, 1, 2]
    assert graph.read_view_at(250).node_attr(0, "score") == 10.0
    assert graph.read_view_at(350).node_attr(0, "score") == PRECISE
    assert len(graph.commit_history()) == 3
    assert graph.checkpoints()[0]["message"] == "before restoration"
    graph.check_integrity()


def run_workflow(folder, durable=False):
    folder = Path(folder)
    path = folder / ("workflow.ggj" if durable else "workflow.gga")
    if durable:
        with DurableGraph.create(path) as graph:
            states, original = exercise(graph)
            graph.verify_storage()
        with DurableGraph.open(path) as recovered:
            verify_reopened(recovered, states)
            recovered.verify_storage()
    else:
        graph = Graph()
        states, original = exercise(graph)
        graph.save_archive(path)
        del graph
        recovered = Graph.load_archive(path)
        verify_reopened(recovered, states)
        del recovered
    # A capture survives owner destruction or durable close/reopen.
    assert original.node_attr(0, "score") == PRECISE
    assert original.edge_endpoints(0) == (0, 1)
    return {"storage": "journal" if durable else "archive", **states}


if __name__ == "__main__":
    with TemporaryDirectory(prefix="groggy-core-workflow-") as folder:
        print(json.dumps([run_workflow(folder), run_workflow(folder, durable=True)], indent=2))
