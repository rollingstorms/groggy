"""Collection projections read captured columns, never current owner mappings."""
import pytest
from groggy import Graph


def fixture():
    graph = Graph()
    graph.add_nodes([
        {"x": 10, "y": 100}, {}, {"x": None, "y": 200}, {}, {"y": 400},
    ])
    graph.remove_nodes([1, 3])
    graph.add_edges([(0, 2, {"weight": 1}), (2, 4, {"weight": 2})])
    return graph, graph.view().neighborhood([0], hops=2)


def test_multicolumn_projection_keeps_captured_values_missing_cells_and_id_order():
    graph, selected = fixture()
    graph.set_node_attr(0, "x", 99)
    graph.set_node_attr(2, "y", 999)
    graph.remove_node(4)
    table = selected[["x", "y", "absent"]][0]
    assert table.column("x").to_list() == [10, None, None]
    assert table.column("y").to_list() == [100, 200, 400]
    assert table.column("absent").to_list() == [None, None, None]
    assert selected[0].node_ids.to_list() == [0, 2, 4]


def test_single_column_preserves_present_nulls_and_legacy_edge_fallback():
    graph, selected = fixture()
    graph.set_node_attr(0, "x", 99)
    graph.set_edge_attr(0, "weight", 9)
    graph.remove_edge(1)
    assert selected["x"][0].to_list() == [10, None]
    assert selected["weight"][0].to_list() == [1, 2]
    graph.set_node_attr(0, "late", 7)
    with pytest.raises(KeyError):
        selected["late"]


def test_projection_from_retained_meta_source_survives_original_removal():
    graph, _ = fixture()
    meta = graph.induced_subgraph([0, 2, 4]).collapse(node_strategy="collapse")
    selected = meta.expand().neighborhood([0], hops=2)
    assert sorted(graph.node_ids) == [meta.id]
    assert selected[["x", "y"]][0].column("y").to_list() == [100, 200, 400]
    assert selected["x"][0].to_list() == [10, None]
    assert selected["weight"][0].to_list() == [1, 2]


def test_present_null_node_column_takes_precedence_over_edge_column():
    graph = Graph()
    graph.add_nodes([{"fact": None}, {}])
    graph.add_edge(0, 1, fact="edge")
    selected = graph.view().neighborhood([0], hops=1)
    graph.set_node_attr(0, "fact", "changed")
    assert selected["fact"][0].to_list() == [None]
    assert selected[["fact"]][0].column("fact").to_list() == [None, None]


def test_array_merge_keeps_exact_captured_sparse_ids_direction_and_parallel_edges():
    graph = Graph(directed=True)
    graph.add_nodes([{"x": node} for node in range(5)])
    graph.add_edges([(0, 4), (1, 3), (1, 3), (3, 3)])
    selected = graph.subgraph_from_edges([1, 2, 3]).neighborhood([1, 3], hops=1)
    graph.remove_node(1)
    graph.set_node_attr(3, "x", 999)
    before = graph.current_state_id
    merged = selected.merge()
    assert merged.is_directed
    assert sorted(merged.node_ids) == [1, 3]
    assert sorted(merged.edge_ids) == [1, 2, 3]
    assert merged.edge_endpoints(1) == (1, 3)
    assert merged.edge_endpoints(2) == (1, 3)
    assert merged.edge_endpoints(3) == (3, 3)
    assert merged.get_node_attr(3, "x") == 3
    assert graph.current_state_id == before
    merged.set_node_attr(3, "x", -1)
    assert selected[["x"]][0].column("x").to_list() == [1, 3]
    assert graph.get_node_attr(3, "x") == 999


def test_array_grouping_invalid_domains_and_component_callback_errors_propagate():
    graph, selected = fixture()
    with pytest.raises(ValueError, match="element_type"):
        selected.group_by("x", "invalid")
    components = graph.connected_components()
    before = graph.current_state_id

    def fail(_):
        raise LookupError("caller failure")

    with pytest.raises(LookupError, match="caller failure"):
        components.filter(fail)

    class BadTruth:
        def __bool__(self):
            raise RuntimeError("truth failure")

    with pytest.raises(RuntimeError, match="truth failure"):
        components.filter(lambda _: BadTruth())
    with pytest.raises(TypeError, match="callable"):
        components.filter(3)
    assert graph.current_state_id == before
    assert len(components.filter(lambda _: True)) == len(components)


def test_array_summary_uses_captured_directed_density():
    graph = Graph(directed=True)
    graph.add_nodes(2)
    graph.add_edge(0, 1)
    selected = graph.view().neighborhood([0], hops=1)
    graph.remove_edge(0)
    summary = selected.summary()
    assert summary.column("density").to_list() == [0.5]
    assert summary.column("edge_count").to_list() == [1]


def test_array_merge_retains_meta_source_properties_and_group_payloads():
    graph = Graph()
    graph.add_nodes([{"x": 1}, {"x": 2}])
    meta = graph.induced_subgraph([0, 1]).collapse(
        node_aggs={"total": ("sum", "x")}, node_strategy="collapse",
    )
    selected = graph.view().neighborhood([meta.id], hops=0)
    graph.set_node_attr(meta.id, "total", 99)
    merged = selected.merge()
    assert list(merged.node_ids) == [meta.id]
    assert merged.get_node_attr(meta.id, "total") == 3
    restored = merged.read_view().meta_subgraph(meta.id)
    assert sorted(restored.node_ids) == [0, 1]
    assert [restored.node_attr(node, "x") for node in [0, 1]] == [1, 2]


def test_collection_validation_and_late_callback_errors_do_not_return_partial_results():
    empty = Graph().view().group_by("missing", "nodes")
    with pytest.raises(ValueError, match="element_type"):
        empty.group_by("x", "invalid")
    with pytest.raises(TypeError, match="callable"):
        Graph().connected_components().filter(3)
    graph = Graph()
    graph.add_nodes(2)
    components = graph.connected_components()
    calls = []

    def fail_late(component):
        calls.append(component.node_count())
        if len(calls) == 2:
            raise RuntimeError("late failure")
        return True

    before = graph.current_state_id
    with pytest.raises(RuntimeError, match="late failure"):
        components.filter(fail_late)
    assert calls == [1, 1]
    assert len(components) == 2 and graph.current_state_id == before
