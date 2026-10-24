"""Checked table materialization keeps every supported row and property."""
import datetime as dt

import pytest

from groggy import EdgesTable, Graph, GraphTable, NodesTable


def make_table(nodes=None, edges=None):
    if nodes is None:
        nodes = {"node_id": [200, 100], "name": ["Ada", "Lin"]}
    if edges is None:
        edges = {"edge_id": [900], "source": [100], "target": [200], "weight": [1.0000000001]}
    return GraphTable(NodesTable.from_dict(nodes), EdgesTable.from_dict(edges))


@pytest.mark.parametrize("directed", [False, True])
def test_row_order_direction_and_properties_are_accepted_in_one_state(directed):
    table = make_table()
    graph = table.to_graph(directed=directed)
    assert graph.is_directed == directed
    assert graph.current_state_id == 1
    assert graph.automatic_publication
    assert not graph.has_uncommitted_changes()
    assert graph.get_node_attr(0, "name") == "Ada"
    assert graph.get_node_attr(1, "name") == "Lin"
    assert graph.edge_endpoints(0) == (1, 0)
    assert graph.get_edge_attr(0, "weight") == 1.0000000001
    snapshot = graph.snapshot_at_commit(1)
    assert snapshot.node_attr(0, "name").value == "Ada"
    assert snapshot.edge_attr(0, "weight").value == 1.0000000001
    metadata = graph.publication_metadata(1)
    assert all(change["effective_at"] == metadata["recorded_at"] for change in metadata["change_times"])
    assert graph.checkpoints() == []
    assert graph.commit("materialized", "test") == 1
    graph.set_node_attr(0, "name", "Grace")
    assert graph.current_state_id == 2
    assert snapshot.node_attr(0, "name").value == "Ada"


def test_managed_identities_survive_graph_table_materialization():
    source = Graph(directed=True)
    a, b = source.upsert_nodes([{"external_id": "a", "labels": ["Person"]}, {"external_id": "b"}])
    source.upsert_edges([{"external_id": "rel", "source": "b", "target": "a", "weight": 2}])
    copied = source.table().to_graph(directed=True)
    assert copied.node_by_external_id("a") == a
    assert copied.node_by_external_id("b") == b
    assert copied.edge_by_external_id("rel") == 0
    assert copied.get_edge_attr(0, "weight") == 2
    assert copied.get_node_attr(a, "labels") == ["Person"]
    assert copied.get_node_attr(b, "labels") is None
    assert copied.edge_endpoints(0) == (b, a)
    assert copied.upsert_nodes([{"external_id": "a"}]) == [a]
    assert copied.current_state_id == 1


def test_typed_properties_null_parallel_edges_and_loops_survive():
    instant = dt.datetime(2026, 10, 23, tzinfo=dt.timezone.utc)
    table = make_table(
        {"node_id": [10], "when": [instant], "vector": [[1.0000000001, 2.0]], "value": [None]},
        {"edge_id": [90, 80], "source": [10, 10], "target": [10, 10], "w": [1, 2]},
    )
    graph = table.to_graph()
    assert graph.get_node_attr(0, "when") == instant
    assert graph.get_node_attr(0, "vector") == [1.0000000001, 2.0]
    assert graph.snapshot_at_commit(1).node_attr(0, "value").value is None
    assert [graph.get_edge_attr(edge, "w") for edge in [0, 1]] == [1, 2]
    assert graph.edge_endpoints(0) == graph.edge_endpoints(1) == (0, 0)


@pytest.mark.parametrize("column", ["node_id", "edge_id", "source", "target"])
@pytest.mark.parametrize("bad", [-1, 1.5, "10"])
def test_malformed_structural_rows_reject_instead_of_disappearing(column, bad):
    nodes = {"node_id": [10]}
    edges = {"edge_id": [20], "source": [10], "target": [10]}
    (nodes if column == "node_id" else edges)[column] = [bad]
    table = make_table(nodes, edges)
    with pytest.raises(RuntimeError, match="row 0"):
        table.to_graph()


@pytest.mark.parametrize("case", ["duplicate_node", "duplicate_edge", "missing_endpoint", "meta", "pooled", "labels", "relationship", "external_type", "external_conflict"])
def test_invalid_properties_and_identity_reject_the_whole_materialization(case):
    nodes = {"node_id": [10, 20], "name": ["a", "b"]}
    edges = {"edge_id": [90, 80], "source": [10, 20], "target": [20, 10]}
    if case == "duplicate_node":
        nodes["node_id"] = [10, 10]
    elif case == "duplicate_edge":
        edges["edge_id"] = [90, 90]
    elif case == "missing_endpoint":
        edges["source"] = [10, 99]
    elif case == "meta":
        nodes["entity_type"] = ["base", "meta"]
    elif case == "pooled":
        nodes["contains_subgraph"] = [None, 7]
    elif case == "labels":
        nodes["labels"] = [["ok"], ["duplicate", "duplicate"]]
    elif case == "relationship":
        edges["relationship_type"] = ["ok", 42]
    elif case == "external_type":
        nodes["external_id"] = ["a", 42]
    elif case == "external_conflict":
        nodes["external_id"] = ["a", "b"]
        edges["external_id"] = ["a", "edge"]
    table = make_table(nodes, edges)
    with pytest.raises(RuntimeError):
        table.to_graph()
    # Repeating validation does not mutate/repair the source table in place.
    with pytest.raises(RuntimeError):
        table.to_graph()


def test_meta_projection_never_silently_downgrades_to_base():
    source = Graph()
    source.add_node()
    source.view().collapse(node_aggs={"size": "count"})
    table = source.table()
    with pytest.raises(RuntimeError, match="pool"):
        table.to_graph()
    assert source.current_state_id == 2


def test_null_edge_ids_keep_compatibility_without_mutating_the_table():
    table = make_table({"node_id": [10]}, {"edge_id": [None, 50], "source": [10, 10], "target": [10, 10], "w": [3, 4]})
    for _ in range(2):
        graph = table.to_graph()
        assert graph.current_state_id == 1
        assert [graph.get_edge_attr(edge, "w") for edge in [0, 1]] == [3, 4]

    assert len(table.auto_assign_edge_ids().to_graph().edge_ids) == 2


def test_null_edge_id_overflow_rejects_before_returning_a_graph():
    table = make_table({"node_id": [10]}, {"edge_id": [2**63 - 1, None], "source": [10, 10], "target": [10, 10]})
    with pytest.raises(RuntimeError, match="integer range"):
        table.to_graph()
    with pytest.raises(RuntimeError, match="integer range"):
        table.auto_assign_edge_ids()
