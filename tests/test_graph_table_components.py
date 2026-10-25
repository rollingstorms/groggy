"""GraphTable owns two rectangles and induced node-subset topology."""
import pytest
from groggy import BaseTable, EdgesTable, Graph, GraphTable, NodesTable


def fixture():
    return GraphTable(
        NodesTable.from_base_table(BaseTable.from_dict({
            "node_id": [70, 12, 99, 44], "group": [1, 1, 2, 2],
        })),
        EdgesTable([(900, 70, 12), (901, 70, 12), (902, 12, 12),
                    (903, 12, 99), (904, 99, 44), (905, 44, 70)]),
    )


def test_node_subsets_keep_internal_parallel_edges_and_loops_and_can_materialize():
    table = fixture()
    head = table.head(2)
    assert head.nodes.node_ids().to_list() == [70, 12]
    assert head.edges.as_tuples() == [(900, 70, 12), (901, 70, 12), (902, 12, 12)]
    assert table.tail(2).edges.as_tuples() == [(904, 99, 44)]
    assert table.slice(1, 3).edges.as_tuples() == [(902, 12, 12), (903, 12, 99)]
    assert repr(head.viz) == "VizAccessor(GraphTable)"
    assert len(head.to_graph()) == 2
    assert table.edges.nrows() == 6


def test_component_shape_schema_and_converters_preserve_independent_ownership():
    table = fixture()
    assert table.shape() == {"nodes": (4, 2), "edges": (6, 3)}
    assert set(table.columns) == {"nodes", "edges"}
    assert set(table.columns["nodes"]) == {"node_id", "group"}
    assert table.columns["edges"] == ["edge_id", "source", "target"]
    nodes = table.to_nodes()
    edges = table.to_edges()
    assert isinstance(nodes, NodesTable)
    assert isinstance(edges, EdgesTable)
    edited = nodes.base_table()
    edited["group"] = [9, 9, 9, 9]
    assert table.nodes.base_table()["group"].to_list() == [1, 1, 2, 2]
    assert edges.as_tuples() == table.edges.as_tuples()


@pytest.mark.parametrize("method,args", [("head", [0]), ("tail", [0]), ("slice", [99, 100]), ("slice", [3, 1])])
def test_empty_subsets_preserve_both_schemas(method, args):
    table = getattr(fixture(), method)(*args)
    assert table.is_empty()
    assert table.shape() == {"nodes": (0, 2), "edges": (0, 3)}


def test_dangling_or_duplicate_inputs_raise_before_disappearing_in_a_subset():
    invalid = GraphTable(NodesTable([1]), EdgesTable([(1, 1, 2)]))
    with pytest.raises(ValueError, match="endpoint outside"):
        invalid.head(0)
    duplicate = GraphTable(NodesTable([1, 1]), EdgesTable([]))
    with pytest.raises(ValueError, match="Duplicate node_id"):
        duplicate.tail(1)


def test_induced_component_converters_keep_frozen_source_identity_for_one_mixed_update():
    graph = Graph()
    graph.add_nodes([{}, {}, {}])
    graph.add_edges([(0, 1), (1, 2), (0, 1)])
    original = graph.table()
    head = original.head(2)
    nodes = head.to_nodes().base_table()
    edges = head.to_edges().base_table()
    assert nodes.source_entity_ids == [0, 1]
    assert edges.source_entity_ids == [0, 2]
    nodes["x"] = [1, 2]
    edges["w"] = [3, 4]
    edited = GraphTable(NodesTable.from_base_table(nodes), EdgesTable.from_base_table(edges))
    before = graph.current_state_id
    edited.update_source_graph(graph, node_columns=["x"], edge_columns=["w"], effective_at=123)
    assert graph.current_state_id == before + 1
    assert graph.get_edge_attr(2, "w") == 4
    assert "x" not in original.nodes.base_table().columns


def test_node_grouping_retains_internal_edges_and_source_rows():
    groups = fixture().group_by(["group"])
    assert sorted([group.edges.edge_ids().to_list() for group in groups]) == [[900, 901, 902], [904]]
    assert all(repr(group.viz) == "VizAccessor(GraphTable)" for group in groups)
