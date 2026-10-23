"""Explicit f64 tags are lossless before changing ordinary float defaults."""
import json
import math
import struct

import groggy as gr
import pytest


def bits(value):
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def test_explicit_scalar_vector_history_and_json_keep_precision(tmp_path):
    scalar = 1 + 2**-40
    vector = [scalar, 1e300, -0.0, float("inf")]
    graph = gr.Graph()
    node = graph.add_node(scalar=gr.AttrValue.float64(scalar),
                          vector=gr.AttrValue.float64_vec(vector),
                          old=gr.AttrValue.float_vec([0.1]))
    state = graph.commit("f64", "test")
    frozen = graph.read_view()
    graph.set_node_attr(node, "scalar", gr.AttrValue.float64(2))
    historical = graph.snapshot_at_commit(state)
    assert historical.node_attr(node, "scalar").type_name == "float64"
    assert historical.node_attr(node, "scalar").value == scalar
    assert historical.node_attr(node, "vector").type_name == "float64_vec"
    assert frozen.node_attr(node, "scalar") == scalar
    path = tmp_path / "wide.json"
    graph.save_to_path(path)
    loaded = gr.Graph.load_from_path(path)
    assert loaded.get_node_attr(node, "scalar") == 2
    assert [bits(v) for v in loaded.get_node_attr(node, "vector")] == [bits(v) for v in vector]
    payload = json.loads(path.read_text())["nodes"][0]["attributes"]
    assert payload["scalar"]["type"] == "float64"
    assert payload["vector"]["type"] == "float64_vec"
    assert payload["old"]["type"] == "float_vec"
    assert loaded.get_node_attr(node, "old") != [0.1]  # Explicit f32 stays f32.
    assert math.copysign(1, loaded.get_node_attr(node, "vector")[2]) == -1


def test_logical_float_schema_accepts_both_widths_and_explicit_width_is_checked():
    graph = gr.Graph()
    graph.add_node(labels=["Entity"], scalar=gr.AttrValue.float64(1e300))
    schema = lambda kind: {"node_labels": {"Entity": {"scalar": {"type": kind}}}}
    assert graph.validate_schema(schema("float")) == []
    assert graph.validate_schema(schema("float64")) == []
    assert graph.validate_schema(schema("float32"))


def test_f64_weight_and_personalization_are_not_narrowed():
    graph = gr.Graph(directed=True)
    a, b = graph.add_nodes(2)
    length = 1 + 2**-40
    graph.add_edge(a, b, w=gr.AttrValue.float64(length))
    assert dict(graph.read_view().dijkstra_distances(a, "w"))[b] == length
    graph.set_node_attr(a, "p", gr.AttrValue.float64(1e300))
    graph.set_node_attr(b, "p", gr.AttrValue.float64(0))
    assert dict(graph.read_view().pagerank(damping=0, personalization_attr="p").scores) == {a: 1, b: 0}


def test_empty_and_nonfinite_f64_vectors_preserve_type():
    assert gr.AttrValue.float64_vec([]).type_name == "float64_vec"
    assert gr.AttrValue.float64_vec([]).value == []
    assert math.isnan(gr.AttrValue.float64(float("nan")).value)
    with pytest.raises((ValueError, RuntimeError)):
        graph = gr.Graph(directed=True)
        a, b = graph.add_nodes(2)
        graph.add_edge(a, b, w=gr.AttrValue.float64(float("inf")))
        graph.read_view().dijkstra_distances(a, "w")
