import json
import math

import pytest

from groggy import AttrValue, Graph


def test_typed_export_round_trips_graph_python_values_and_snapshots(tmp_path):
    graph = Graph(directed=True)
    values = {
        "null": None,
        "bool": True,
        "int": 2**60,
        "float": -0.0,
        "text": 'Unicode λ "quoted"',
        "bytes": b"\x00\xff",
        "ints": [1, 2],
        "floats": [0.5, 2.0],
        "flags": [True, False],
        "names": ["a", "b"],
        "nested": {"rows": [1, None, "text"], "large_int": 10**400},
        "empty_flags": AttrValue.bool_vec([]),
        "large_text": "x" * 10000,
        "large_vector": [0.0] * 512,
    }
    node = graph.add_node(**values)
    edge = graph.add_edge(node, node, **values)
    path = tmp_path / "typed.json"
    graph.save_to_path(path)
    loaded = Graph.load_from_path(path)
    assert loaded.is_directed
    assert loaded.edge_endpoints(edge) == (node, node)
    for name, expected in values.items():
        expected = expected.value if isinstance(expected, AttrValue) else expected
        assert loaded.get_node_attr(node, name) == expected
        assert loaded.get_edge_attr(edge, name) == expected
        assert loaded.nodes[name][0] == expected
    assert math.copysign(1, loaded.get_node_attr(node, "float")) == -1
    state = loaded.commit("loaded properties", "test")
    snapshot = loaded.snapshot_at_commit(state)
    assert snapshot.node_attr(node, "empty_flags").type_name == "bool_vec"
    assert snapshot.node_attr(node, "null").value is None
    assert snapshot.node_attr(node, "absent") is None
    assert json.loads(path.read_text())["version"] == 2


def test_export_keeps_nonfinite_float_values(tmp_path):
    graph = Graph()
    node = graph.add_node(nan=float("nan"), vector=[float("inf"), -0.0])
    path = tmp_path / "float.json"
    graph.save_to_path(path)
    loaded = Graph.load_from_path(path)
    assert math.isnan(loaded.get_node_attr(node, "nan"))
    assert loaded.get_node_attr(node, "vector")[0] == float("inf")
    assert math.copysign(1, loaded.get_node_attr(node, "vector")[1]) == -1


def test_legacy_export_and_future_version_rejection(tmp_path):
    path = tmp_path / "graph.json"
    path.write_text(json.dumps({
        "nodes": [{"id": 3, "attributes": {"name": "legacy"}}], "edges": []
    }))
    loaded = Graph.load_from_path(path)
    assert loaded.get_node_attr(3, "name") == "legacy"
    path.write_text(json.dumps({
        "format": "groggy.graph", "version": 999, "nodes": [], "edges": []
    }))
    with pytest.raises(ValueError, match="format/version"):
        Graph.load_from_path(path)
