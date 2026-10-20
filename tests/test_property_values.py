import math

import pytest

from groggy import AttrValue, Graph, GraphBatch


@pytest.mark.parametrize("value,kind", [
    (None, "null"),
    (True, "bool"),
    (2**40, "int"),
    (0.5, "float"),
    ("text with \"quotes\", \\ and unicode λ", "text"),
    ([1, 2, 3], "int_vec"),
    ([0.5, 1.0, 2], "float_vec"),
    ([True, False], "bool_vec"),
    (["a", "b\"\\"], "text_vec"),
    (b"\x00\xffbinary", "bytes"),
    ({"nested": [1, None, True, {"name": "value"}]}, "json"),
    ([1, "mixed", None], "json"),
    ([], "float_vec"),
])
def test_values_keep_types_across_graph_columns_tables_and_history(value, kind):
    graph = Graph()
    node = graph.add_node(payload=value)
    edge = graph.add_edge(node, node, payload=value)
    state = graph.commit("properties", "test")
    snapshot = graph.snapshot_at_commit(state)
    assert graph.get_node_attr(node, "payload") == value
    assert graph.get_edge_attr(edge, "payload") == value
    assert graph.nodes["payload"][0] == value
    assert graph.edges["payload"][0] == value
    assert graph.nodes.table()["payload"] == value
    assert graph.edges.table()["payload"] == value
    assert snapshot.node_attr(node, "payload").value == value
    assert snapshot.edge_attr(edge, "payload").value == value
    assert snapshot.node_attr(node, "payload").type_name == kind
    assert AttrValue(value).type_name == kind
    assert AttrValue(value).value == value
    graph.set_node_attr(node, "copied", snapshot.node_attr(node, "payload"))
    assert graph.get_node_attr(node, "copied") == value
    graph.set_node_attrs({"bulk": {node: value}})
    graph.set_edge_attrs({"bulk": {edge: value}})
    assert graph.get_node_attr(node, "bulk") == value
    assert graph.get_edge_attr(edge, "bulk") == value
    graph.commit("bulk properties", "test")
    batch = GraphBatch()
    batch.set_node_attr(node, "payload", {"after": True})
    result = graph.apply_batch(batch, graph.current_state_id, "change", "test")
    assert snapshot.node_attr(node, "payload").value == value
    change = next(
        row for row in graph.diff_commits(state, result["state_id"])["attribute_changes"]
        if row["entity_type"] == "node" and row["name"] == "payload"
    )
    assert change["old"] == value
    assert change["new"] == {"after": True}


def test_compressed_text_and_vectors_return_actual_values():
    graph = Graph()
    text = "x" * 10000
    vector = [0.0] * 512
    node = graph.add_node(text=text, vector=vector)
    edge = graph.add_edge(node, node, text=text, vector=vector)
    state = graph.commit("compressed", "test")
    snapshot = graph.snapshot_at_commit(state)
    assert graph.get_node_attr(node, "text") == text
    assert graph.get_node_attr(node, "vector") == vector
    assert graph.get_edge_attr(edge, "text") == text
    assert graph.get_edge_attr(edge, "vector") == vector
    assert graph.nodes["text"][0] == text
    assert graph.nodes["vector"][0] == vector
    assert snapshot.node_attr(node, "text").value == text
    assert snapshot.node_attr(node, "vector").value == vector


def test_optimized_values_have_consistent_wrapper_equality_and_hashes():
    graph = Graph()
    node = graph.add_node(number=1, text="short", long_text="x" * 10000)
    state = graph.commit("optimized", "test")
    snapshot = graph.snapshot_at_commit(state)
    for name, value in [("number", 1), ("text", "short"), ("long_text", "x" * 10000)]:
        stored = snapshot.node_attr(node, name)
        direct = AttrValue(value)
        assert stored == direct
        assert hash(stored) == hash(direct)
        assert len({stored, direct}) == 1
    nan = AttrValue(float("nan"))
    assert nan == AttrValue(float("nan"))
    assert hash(nan) == hash(AttrValue(float("nan")))
    assert math.isnan(nan.value)
    assert AttrValue(-0.0) == AttrValue(0.0)
    assert hash(AttrValue(-0.0)) == hash(AttrValue(0.0))


def test_json_mapping_order_is_canonical_and_non_string_keys_are_rejected():
    assert AttrValue({"a": 1, "b": 2}) == AttrValue({"b": 2, "a": 1})
    with pytest.raises(TypeError, match="keys must be strings"):
        AttrValue({"nested": {1: "one"}})
    cycle = {}
    cycle["self"] = cycle
    with pytest.raises(ValueError, match="Circular"):
        AttrValue(cycle)


@pytest.mark.parametrize("value", [2**80, [2**80]])
def test_oversized_integers_raise_instead_of_becoming_lossy_floats(value):
    graph = Graph()
    with pytest.raises(OverflowError):
        AttrValue(value)
    with pytest.raises(ValueError):
        graph.add_node(payload=value)
    assert graph.add_node() == 0


def test_null_and_absence_are_distinct_in_snapshots():
    graph = Graph()
    node = graph.add_node(explicit=None)
    state = graph.commit("null", "test")
    snapshot = graph.snapshot_at_commit(state)
    assert snapshot.node_attr(node, "explicit") is not None
    assert snapshot.node_attr(node, "explicit").value is None
    assert snapshot.node_attr(node, "absent") is None


def test_structured_property_errors_reject_public_creation_before_allocation():
    graph = Graph()
    with pytest.raises(ValueError):
        graph.add_node(valid=1, bad={"nested": object()})
    with pytest.raises(ValueError):
        graph.add_nodes([{"valid": 1}, {"bad": {"nested": object()}}])
    assert graph.add_node() == 0


def test_typed_vector_constructors_disambiguate_empty_vectors_and_casts():
    for name, factory in [
        ("float_vec", AttrValue.float_vec),
        ("int_vec", AttrValue.int_vec),
        ("text_vec", AttrValue.text_vec),
        ("bool_vec", AttrValue.bool_vec),
    ]:
        value = factory([])
        assert value.type_name == name
        assert value.value == []
        graph = Graph()
        node = graph.add_node(payload=value)
        state = graph.commit("typed empty", "test")
        assert graph.snapshot_at_commit(state).node_attr(node, "payload").type_name == name
    assert AttrValue.float_vec([1, 2]).value == [1.0, 2.0]


def test_returned_structures_and_binary_values_do_not_mutate_graph_storage():
    graph = Graph()
    original = {"nested": [1]}
    binary = bytearray(b"original")
    node = graph.add_node(payload=original, binary=binary)
    original["nested"].append(2)
    binary[0] = 0
    returned = graph.get_node_attr(node, "payload")
    returned["nested"].append(3)
    assert graph.get_node_attr(node, "payload") == {"nested": [1]}
    assert graph.get_node_attr(node, "binary") == b"original"


def test_single_element_numpy_vectors_keep_their_shape_and_integer_type():
    numpy = pytest.importorskip("numpy")
    assert AttrValue(numpy.array([1], dtype=numpy.int64)).type_name == "int_vec"
    assert AttrValue(numpy.array([0.5], dtype=numpy.float64)).value == [0.5]
