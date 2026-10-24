"""Explicit graph-table mappings accept mixed property columns in one batch."""
import datetime as dt

import pytest

from groggy import EdgesTable, Graph, GraphTable, NodesTable


def fixture(directed=True):
    graph = Graph(directed=directed)
    graph.add_nodes([{"x": 1}, {"x": 2}, {"x": 3}])
    graph.add_edges([(2, 0, {"w": 1}), (0, 1, {"w": 2})])
    nodes = NodesTable.from_dict({"node_id": [100, 200, 300], "x": [30, None, 20]})
    edges = EdgesTable.from_dict({
        "edge_id": [30, 20], "source": [100, 200], "target": [200, 300],
        "w": [1.0000000001, 2.0000000001],
    })
    options = dict(node_mapping={100: 2, 200: 0, 300: 1}, edge_mapping={30: 0, 20: 1},
                   node_columns=["x"], edge_columns=["w"])
    return graph, GraphTable(nodes, edges), options


def test_mixed_writeback_publishes_one_version_with_explicit_clock_and_null():
    graph, table, options = fixture()
    before = graph.current_state_id
    prior = graph.read_view()
    state = table.update_graph(graph, before, effective_at=123, **options)
    assert state == before + 1
    assert graph.get_node_attr(2, "x") == 30
    assert graph.get_node_attr(0, "x") is None
    assert graph.get_node_attr(1, "x") == 20
    assert graph.get_edge_attr(0, "w") == 1.0000000001
    assert graph.get_edge_attr(1, "w") == 2.0000000001
    assert prior.node_attr(2, "x") == 3
    snapshot = graph.snapshot_at_commit(state)
    assert snapshot.node_attr(2, "x").value == 30
    assert snapshot.node_attr(0, "x").value is None
    assert not graph.has_uncommitted_changes()
    metadata = graph.publication_metadata(state)
    assert metadata["effective_at"] == 123
    assert all(row["effective_at"] == 123 for row in metadata["change_times"])
    assert len(graph.commit_history()) == 3
    assert graph.commit("table checkpoint", "test") == state


@pytest.mark.parametrize("case", ["stale", "missing_mapping", "extra_mapping", "duplicate_target", "inactive_node", "inactive_edge", "edge_endpoint", "missing_column", "reserved_node", "reserved_edge", "duplicate_column"])
def test_rejected_table_update_changes_neither_side_or_consumes_ids(case):
    graph, table, options = fixture()
    state = graph.current_state_id
    if case == "stale":
        state -= 1
    elif case == "missing_mapping":
        del options["node_mapping"][100]
    elif case == "extra_mapping":
        options["edge_mapping"][999] = 0
    elif case == "duplicate_target":
        options["node_mapping"][100] = 0
    elif case == "inactive_node":
        options["node_mapping"][100] = 999
    elif case == "inactive_edge":
        options["edge_mapping"][30] = 999
    elif case == "edge_endpoint":
        options["edge_mapping"] = {30: 1, 20: 0}
    elif case == "missing_column":
        options["edge_columns"].append("missing")
    elif case == "reserved_node":
        options["node_columns"].append("entity_type")
    elif case == "reserved_edge":
        options["edge_columns"].append("source")
    elif case == "duplicate_column":
        options["edge_columns"].append("w")
    before = graph.current_state_id
    with pytest.raises(ValueError):
        table.update_graph(graph, state, **options)
    assert graph.current_state_id == before
    assert [graph.get_node_attr(n, "x") for n in sorted(graph.node_ids)] == [1, 2, 3]
    assert graph.get_edge_attr(0, "w") == 1
    assert graph.get_edge_attr(1, "w") == 2
    assert not graph.has_uncommitted_changes()
    assert graph.add_node() == 3
    assert graph.add_edge(0, 1) == 2


@pytest.mark.parametrize("bad_id", [None, -1, 1.5, "100"])
def test_bad_structural_rows_are_not_silently_dropped(bad_id):
    graph = Graph()
    graph.add_nodes(1)
    table = GraphTable(NodesTable.from_dict({"node_id": [bad_id], "x": [9]}), EdgesTable([]))
    with pytest.raises(ValueError, match="row 0"):
        table.update_graph(graph, graph.current_state_id, node_mapping={100: 0}, edge_mapping={},
                           node_columns=["x"], edge_columns=[])
    assert graph.current_state_id == 1
    assert graph.get_node_attr(0, "x") is None


def test_joined_duplicate_rows_require_explicit_reduction():
    graph = Graph()
    graph.add_node()
    table = GraphTable(NodesTable.from_dict({"node_id": [10, 10], "x": [1, 2]}), EdgesTable([]))
    with pytest.raises(ValueError, match="duplicate table ID"):
        table.update_graph(graph, 1, node_mapping={10: 0}, edge_mapping={},
                           node_columns=["x"], edge_columns=[])
    assert graph.current_state_id == 1


def test_undirected_mapping_accepts_reversed_storage_endpoints():
    graph, table, options = fixture(directed=False)
    options["node_mapping"] = {100: 0, 200: 2, 300: 1}
    # Both original edges must match the chosen mapping.
    edges = EdgesTable.from_dict({"edge_id": [30, 20], "source": [100, 100],
                                 "target": [200, 300], "w": [3, 4]})
    table = GraphTable(NodesTable.from_dict({"node_id": [100, 200, 300], "x": [3, 4, 5]}), edges)
    assert table.update_graph(graph, 2, **options) == 3
    assert graph.get_edge_attr(0, "w") == 3


def test_empty_selection_is_noop_and_unused_columns_are_unchanged():
    graph, table, options = fixture()
    state = graph.current_state_id
    options.update(node_columns=[], edge_columns=[])
    assert table.update_graph(graph, state, **options) == state
    options["edge_columns"] = ["w"]
    assert table.update_graph(graph, state, **options) == state + 1
    assert [graph.get_node_attr(n, "x") for n in sorted(graph.node_ids)] == [1, 2, 3]


def test_typed_values_remain_lossless_and_compute_clock_is_default():
    graph = Graph()
    graph.add_node()
    instant = dt.datetime(2026, 10, 23, tzinfo=dt.timezone.utc)
    table = GraphTable(NodesTable.from_dict({"node_id": [10], "instant": [instant],
                                            "vector": [[1.0000000001, 2.0]]}), EdgesTable([]))
    state = table.update_graph(graph, 1, node_mapping={10: 0}, edge_mapping={},
                               node_columns=["instant", "vector"], edge_columns=[])
    assert graph.get_node_attr(0, "instant") == instant
    assert graph.get_node_attr(0, "vector") == [1.0000000001, 2.0]
    metadata = graph.publication_metadata(state)
    assert metadata["effective_at"] == metadata["recorded_at"]


def test_manual_graph_requires_clean_head_then_explicit_table_batch_publishes():
    graph = Graph(manual=True)
    graph.add_node()
    table = GraphTable(NodesTable.from_dict({"node_id": [10], "x": [9]}), EdgesTable([]))
    options = dict(node_mapping={10: 0}, edge_mapping={}, node_columns=["x"], edge_columns=[])
    with pytest.raises(ValueError, match="uncommitted"):
        table.update_graph(graph, 0, **options)
    graph.commit("initial", "test")
    assert table.update_graph(graph, 1, **options) == 2
    assert graph.get_node_attr(0, "x") == 9
    assert not graph.automatic_publication
    assert not graph.has_uncommitted_changes()
