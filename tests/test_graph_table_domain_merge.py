"""Specialized graph merges retain properties, domain identity and source snapshots."""
import pytest
from groggy import BaseTable, EdgesTable, Graph, GraphTable, NodesTable


def column(table, name):
    value = table[name]
    return value.to_list() if hasattr(value, "to_list") else [value]


def bundle(offset=0, value=1):
    nodes = NodesTable.from_base_table(BaseTable.from_dict({"node_id": [offset + 20, offset + 10], "x": [value, 1.0000000000000002]}))
    edges = EdgesTable.from_base_table(BaseTable.from_dict({"edge_id": [offset + 90], "source": [offset + 10], "target": [offset + 20], "weight": [value + 0.125]}))
    return GraphTable(nodes, edges)


def test_domain_prefix_retains_attributes_and_exact_endpoint_domains():
    result = GraphTable.merge([bundle(value=1), bundle(value=2)])
    assert column(result.nodes.base_table(), "node_id") == [0, 1, 2, 3]
    assert column(result.nodes.base_table(), "x") == [1, 1.0000000000000002, 2, 1.0000000000000002]
    assert column(result.edges.base_table(), "source") == [1, 3]
    assert column(result.edges.base_table(), "target") == [0, 2]
    assert column(result.edges.base_table(), "weight") == [1.125, 2.125]
    assert column(result.edges.base_table(), "_domain") == ["domain_0", "domain_1"]


@pytest.mark.parametrize("strategy,expected", [("keep_first", 1), ("keep_second", 2), ("merge_attributes", 2)])
def test_collision_policies_retain_real_property_rows(strategy, expected):
    result = GraphTable.merge([bundle(value=1), bundle(value=2)], strategy=strategy)
    assert column(result.nodes.base_table(), "node_id") == [20, 10]
    assert column(result.nodes.base_table(), "x")[0] == expected
    assert column(result.edges.base_table(), "weight") == [expected + 0.125]


def test_fail_and_auto_remap_have_distinct_id_contracts():
    result = GraphTable.merge([bundle(), bundle(100)], strategy="fail")
    assert column(result.nodes.base_table(), "node_id") == [20, 10, 120, 110]
    with pytest.raises(RuntimeError, match="collision"):
        GraphTable.merge([bundle(), bundle()], strategy="fail")
    result = GraphTable.merge([bundle(), bundle(), bundle(100)], strategy="auto_remap")
    assert column(result.nodes.base_table(), "node_id") == [20, 10, 0, 1, 120, 110]
    assert column(result.edges.base_table(), "source") == [10, 1, 110]


@pytest.mark.parametrize("strategy", ["fail", "keep_first", "keep_second", "merge_attributes", "domain_prefix", "auto_remap"])
def test_single_input_is_validated_instead_of_bypassing_merge_checks(strategy):
    invalid = GraphTable(NodesTable.from_base_table(BaseTable.from_dict({"node_id": [1]})), EdgesTable.from_base_table(BaseTable.from_dict({"edge_id": [2], "source": [1], "target": [999]})))
    with pytest.raises(RuntimeError, match="input domain"):
        GraphTable.merge([invalid], strategy=strategy)


def test_conflicting_edge_endpoints_require_explicit_resolution():
    reversed_edges = EdgesTable.from_base_table(BaseTable.from_dict({"edge_id": [90], "source": [20], "target": [10]}))
    with pytest.raises(RuntimeError, match="endpoint conflict"):
        GraphTable.merge([bundle(), GraphTable(bundle().nodes, reversed_edges)], strategy="merge_attributes")


def test_merged_components_retain_original_captures_without_row_writeback_identity():
    owner = Graph()
    owner.add_nodes([{"x": 1}, {"x": 2}])
    owner.add_edge(0, 1)
    other = Graph()
    other.add_node(x=3)
    merged = GraphTable.merge([owner.read_view().table(), other.read_view().table()])
    nodes = merged.nodes.base_table()
    assert len(nodes.source_views()) == 2
    assert nodes.source_entity_ids is None
    owner.set_node_attr(0, "x", 99)
    assert nodes.source_views()[0].node_attr(0, "x") == 1
    with pytest.raises(ValueError, match="explicit identity"):
        nodes.update_source_graph(owner, ["x"])
