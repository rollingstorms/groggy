"""Graph table projections preserve row identity and captured values."""
from concurrent.futures import ThreadPoolExecutor

import groggy as gr
import pytest


def values(table, name):
    column = table[name]
    return column.to_list() if hasattr(column, "to_list") else [column]


def test_sparse_properties_are_aligned_across_graph_subgraph_and_frozen_tables():
    graph = gr.Graph()
    nodes = graph.add_nodes(4)
    first = graph.add_edge(nodes[0], nodes[2])
    second = graph.add_edge(nodes[2], nodes[3])
    graph.set_node_attr(nodes[3], "late", 77)
    graph.set_node_attr(nodes[0], "first", True)
    graph.set_node_attr(nodes[2], "explicit_null", None)
    graph.set_edge_attr(second, "late", 88)
    legacy = graph.to_subgraph()
    frozen = graph.read_view()
    for table in [graph.table(), legacy.table(), frozen.table()]:
        assert values(table.nodes, "node_id") == nodes
        assert values(table.nodes, "late") == [None, None, None, 77]
        assert values(table.nodes, "first") == [True, None, None, None]
        assert values(table.nodes, "explicit_null") == [None] * 4
        assert values(table.edges, "edge_id") == [first, second]
        assert values(table.edges, "late") == [None, 88]
        assert values(table.edges, "source") == [nodes[0], nodes[2]]
        assert values(table.edges, "target") == [nodes[2], nodes[3]]
    graph.set_node_attr(nodes[3], "late", 999)
    assert values(frozen.nodes_table(), "late") == [None, None, None, 77]
    assert values(legacy.table().nodes, "late") == [None, None, None, 999]
    graph.remove_node(nodes[3])
    with pytest.raises((ValueError, RuntimeError)):
        legacy.table()
    assert values(frozen.edges_table(), "edge_id") == [first, second]


def test_selected_tables_retain_types_and_scope_after_owner_destruction():
    graph = gr.Graph(directed=True)
    a, b, isolated = graph.add_nodes(3)
    first = graph.add_edge(a, b)
    excluded = graph.add_edge(a, b)
    graph.set_node_attr(b, "large", 2**63 - 1)
    graph.set_node_attr(b, "vector", [1.0, 2.0])
    graph.set_edge_attr(excluded, "excluded", 99)
    view = graph.read_view().select_edges([first])
    del graph
    table = view.table()
    assert values(table.nodes, "node_id") == [a, b]
    assert values(table.nodes, "large") == [None, 2**63 - 1]
    assert values(table.nodes, "vector") == [None, [1.0, 2.0]]
    assert values(table.edges, "edge_id") == [first]
    with pytest.raises((KeyError, ValueError, RuntimeError)):
        table.edges["excluded"]
    assert isolated not in values(table.nodes, "node_id")
    empty = view.select_nodes([]).table()
    assert values(empty.nodes, "node_id") == []
    assert values(empty.edges, "edge_id") == []
    with ThreadPoolExecutor(max_workers=2) as workers:
        assert list(workers.map(lambda _: values(view.nodes_table(), "large"), range(4))) == [[None, 2**63 - 1]] * 4


@pytest.mark.parametrize("name", ["edge_id", "source", "target"])
def test_structural_property_collisions_fail_explicitly(name):
    graph = gr.Graph()
    a, b = graph.add_nodes(2)
    edge = graph.add_edge(a, b)
    graph.set_edge_attr(edge, name, 999)
    view = graph.read_view()
    for export in [graph.table, view.table, view.edges_table]:
        with pytest.raises((ValueError, RuntimeError), match="structural"):
            export()
    assert view.edge_endpoints(edge) == (a, b)
    assert view.edge_attr(edge, name) == 999


def test_node_identity_collision_is_an_error_only_within_selected_rows():
    graph = gr.Graph()
    a, b = graph.add_nodes(2)
    graph.set_node_attr(a, "node_id", 999)
    view = graph.read_view()
    with pytest.raises((ValueError, RuntimeError), match="structural"):
        view.nodes_table()
    assert values(view.select_nodes([b]).nodes_table(), "node_id") == [b]
