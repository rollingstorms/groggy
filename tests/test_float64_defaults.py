"""Ordinary property floats are f64; explicit f32 survives all storage paths."""
import json
import math

import numpy as np
from groggy import AttrValue, BaseArray, BaseTable, DurableGraph, Graph


def test_scalar_collection_inference_preserves_precision_and_explicit_widths():
    x = 1 + 2**-40
    for value in [x, 1e300, np.float64(x)]:
        typed = AttrValue(value)
        assert typed.type_name == "float64" and typed.value == value
    for value in [[x, 1e300], [x, 1], [], np.array([x, 1e300], dtype=np.float64)]:
        assert AttrValue(value).type_name == "float64_vec"
        assert AttrValue(value).value == list(value)
    assert AttrValue([1, 2]).type_name == "int_vec"
    assert AttrValue([True, False]).type_name == "bool_vec"
    assert AttrValue(np.float32(0.1)).type_name == "float"
    assert AttrValue(np.array([], dtype=np.float32)).type_name == "float_vec"
    assert AttrValue(np.array([0.1], dtype=np.float32)).type_name == "float_vec"
    assert AttrValue([np.float32(0.1)]).type_name == "float_vec"
    assert AttrValue.float32(0.1).value == float(np.float32(0.1))
    assert math.isnan(AttrValue(math.nan).value)


def test_defaults_across_direct_bulk_tables_history_and_active_export(tmp_path):
    x = 1 + 2**-40
    graph = Graph()
    node = graph.add_node(x=x, embedding=[x, 1e300], labels=[])
    edge = graph.add_edge(node, node, x=x)
    graph.set_node_attrs({"bulk": {node: x}})
    graph.set_edge_attrs({"bulk": {edge: x}})
    state = graph.commit("f64 defaults", "test")
    snapshot = graph.snapshot_at_commit(state)
    for kind, identifier in [(snapshot.node_attr, node), (snapshot.edge_attr, edge)]:
        assert kind(identifier, "x").type_name == "float64"
        assert kind(identifier, "bulk").value == x
    assert snapshot.node_attr(node, "embedding").type_name == "float64_vec"
    assert snapshot.node_attr(node, "labels").type_name == "text_vec"
    assert graph.nodes.table().column("x").to_list() == [x]
    assert BaseArray([x]).dtype() == "Float64"
    path = tmp_path / "graph.json"
    graph.save_to_path(path)
    assert Graph.load_from_path(path).get_node_attr(node, "embedding") == [x, 1e300]
    value = json.loads(path.read_text())["nodes"][0]["attributes"]["x"]
    assert value["type"] == "float64"
    graph.set_node_attr(node, "x", x + 1)
    assert snapshot.node_attr(node, "x").value == x


def test_default_floats_recover_losslessly_through_checked_journal(tmp_path):
    path = tmp_path / "defaults.ggj"
    x = 1 + 2**-40
    with DurableGraph.create(path) as graph:
        node = graph.writer().add_node(x=x, vector=[x, 1e300], old=AttrValue.float32(0.1))
        state = graph.current_state_id
    with DurableGraph.open(path) as graph:
        assert graph.get_node_attr(node, "x") == x
        snapshot = graph.snapshot_at_commit(state)
        assert snapshot.node_attr(node, "x").type_name == "float64"
        assert snapshot.node_attr(node, "vector").type_name == "float64_vec"
        assert snapshot.node_attr(node, "old").type_name == "float"
        assert graph.get_node_attr(node, "old") == float(np.float32(0.1))


def test_table_schema_floating_family_and_explicit_widths():
    source = BaseTable.from_dict({"x": [1 + 2**-40]})
    assert source.validate_schema({"x": "float"}).column("type_match").to_list() == [True]
    assert source.validate_schema({"x": "float64"}).column("type_match").to_list() == [True]
    assert source.validate_schema({"x": "float32"}).column("type_match").to_list() == [False]
    vectors = BaseTable.from_dict({"x": [[1 + 2**-40]]})
    for expected, matches in [("float_vec", True), ("float64_vec", True),
                              ("float32_vec", False)]:
        assert vectors.validate_schema({"x": expected}).column("type_match").to_list() == [matches]


def test_preexisting_f32_export_tags_remain_f32_after_default_switch(tmp_path):
    # Fixed older-format payload: do not generate it through the new translator.
    path = tmp_path / "old.json"
    path.write_text(json.dumps({
        "format": "groggy.graph", "version": 2,
        "config": {"graph_type": "undirected"},
        "nodes": [{"id": 0, "attributes": {
            "x": {"type": "float", "value": 1036831949},
            "v": {"type": "float_vec", "value": [1036831949]},
        }}], "edges": [],
    }))
    graph = Graph.load_from_path(path)
    state = graph.commit("old data", "test")
    snapshot = graph.snapshot_at_commit(state)
    assert snapshot.node_attr(0, "x").type_name == "float"
    assert snapshot.node_attr(0, "v").type_name == "float_vec"
    assert graph.get_node_attr(0, "x") == float(np.float32(0.1))
    assert graph.get_node_attr(0, "v") == [float(np.float32(0.1))]
    graph.save_to_path(path)
    attrs = json.loads(path.read_text())["nodes"][0]["attributes"]
    assert attrs["x"] == {"type": "float", "value": 1036831949}
    assert attrs["v"] == {"type": "float_vec", "value": [1036831949]}
