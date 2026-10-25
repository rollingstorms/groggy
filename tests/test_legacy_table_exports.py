"""Legacy entry points now return source-bound immutable table captures."""
import pytest
from groggy import Graph, GraphTable


def fixture():
    graph = Graph()
    graph.add_nodes([{"x": 1.0000000000000002}, {"explicit_null": None}, {}])
    graph.add_edges([(0, 1, {"w": 1}), (0, 1, {"w": 2}), (1, 2, {"w": 3})])
    return graph


def column(table, name):
    value = table[name]
    return value.to_list() if hasattr(value, "to_list") else [value]


@pytest.mark.parametrize("entry", ["graph", "nodes", "edges"])
def test_direct_table_exports_preserve_frozen_capture_and_exact_row_identity(entry):
    graph = fixture()
    if entry == "graph":
        table = graph.table().nodes.base_table()
    elif entry == "nodes":
        table = graph.nodes.table().base_table()
    else:
        table = graph.edges.table().base_table()
    assert table.source_entity_ids == [0, 1, 2]
    assert table.source_entity_kind == ("edge" if entry == "edges" else "node")
    assert table.source_state_id == graph.current_state_id
    graph.set_node_attr(0, "x", 99)
    assert table.source_views()[0].node_attr(0, "x") == 1.0000000000000002


def test_graph_bundle_export_shares_one_capture_for_mixed_publication():
    graph = fixture()
    table = graph.table()
    nodes = table.nodes.with_attributes("score", {0: 10, 1: 20, 2: 30})
    edges = table.edges.with_attributes("w", {0: 4, 1: 5, 2: 6})
    before = graph.current_state_id
    result = GraphTable(nodes, edges).update_source_graph(graph, node_columns=["score"], edge_columns=["w"])
    assert result["state_id"] == before + 1
    assert graph.get_node_attr(1, "score") == 20
    assert graph.get_edge_attr(1, "w") == 5
    assert column(table.edges.base_table(), "w") == [1, 2, 3]


@pytest.mark.parametrize("kind", ["nodes", "edges"])
def test_constrained_accessor_exports_are_sorted_deduplicated_and_source_bound(kind):
    graph = fixture()
    accessor = graph.nodes[[2, 0, 2]].nodes if kind == "nodes" else graph.edges[[2, 0, 2]].edges
    table = accessor.table().base_table()
    assert table.source_entity_ids == [0, 2]
    assert column(table, "node_id" if kind == "nodes" else "edge_id") == [0, 2]
    assert len(table.source_views()) == 1


def test_selected_explicit_null_property_is_retained_as_present():
    graph = fixture()
    table = graph.nodes[[1]].nodes.table().base_table()
    assert "explicit_null" in table.columns
    assert column(table, "explicit_null") == [None]
    assert table.source_views()[0].has_node_attribute(1, "explicit_null")


@pytest.mark.parametrize("kind", ["nodes", "edges"])
def test_stale_constrained_accessors_reject_without_synthetic_rows(kind):
    graph = fixture()
    accessor = graph.nodes[[2]].nodes if kind == "nodes" else graph.edges[[2]].edges
    old = accessor.table().base_table()
    if kind == "nodes":
        graph.remove_node(2)
    else:
        graph.remove_edge(2)
    with pytest.raises(ValueError):
        accessor.table()
    assert old.source_entity_ids == [2]


def test_manual_working_tables_are_pinned_but_read_only_until_clean_publication():
    graph = Graph(manual=True)
    graph.add_node(x=3)
    table = graph.nodes.table().base_table()
    assert table.source_state_id is None
    assert column(table, "x") == [3]
    with pytest.raises(ValueError, match="read-only"):
        table.update_source_graph(graph, ["x"])


def test_legacy_subgraph_bundle_keeps_capture_after_parent_changes():
    graph = fixture()
    subgraph = graph.to_subgraph()
    table = subgraph.table()
    assert table.nodes.base_table().source_entity_ids == [0, 1, 2]
    assert table.edges.base_table().source_entity_ids == [0, 1, 2]
    graph.set_node_attr(0, "x", 99)
    assert table.nodes.base_table().source_views()[0].node_attr(0, "x") == 1.0000000000000002
    assert column(subgraph.table().nodes.base_table(), "x")[0] == 99


def test_separately_exported_components_do_not_forge_one_shared_capture():
    graph = fixture()
    table = GraphTable(graph.nodes.table(), graph.edges.table())
    with pytest.raises(ValueError, match="exact source capture"):
        table.update_source_graph(graph, node_columns=[], edge_columns=[])
