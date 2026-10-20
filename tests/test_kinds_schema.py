import pytest

from groggy import Graph, GraphBatch


def test_multiple_labels_relationship_types_history_and_export(tmp_path):
    graph = Graph(directed=True)
    node = graph.add_node(labels=["Person", "Entity"], name="Ada")
    first = graph.add_edge(node, node, relationship_type="KNOWS")
    second = graph.add_edge(node, node, relationship_type="KNOWS")
    assert graph.node_labels(node) == ["Entity", "Person"]
    assert graph.nodes["labels"][0] == ["Entity", "Person"]
    assert graph.get_node_attr(node, "entity_type") == "base"
    assert graph.nodes_with_label("Entity") == [node]
    assert graph.nodes_with_label("entity") == []
    assert graph.edges_with_relationship_type("KNOWS") == [first, second]
    state = graph.commit("kinds", "test")
    snapshot = graph.snapshot_at_commit(state)
    graph.set_node_labels(node, ["Episodic"])
    graph.set_edge_relationship_type(first, "SUPPORTS")
    graph.set_edge_relationship_type(second)
    assert snapshot.node_attr(node, "labels").value == ["Entity", "Person"]
    assert snapshot.edge_attr(first, "relationship_type").value == "KNOWS"
    assert graph.edge_relationship_type(second) is None
    path = tmp_path / "kinds.json"
    graph.save_to_path(path)
    restored = Graph.load_from_path(path)
    assert restored.node_labels(node) == ["Episodic"]
    assert restored.edges_with_relationship_type("SUPPORTS") == [first]
    graph.remove_node(node)
    assert graph.nodes_with_label("Episodic") == []
    assert graph.edges_with_relationship_type("SUPPORTS") == []
    with pytest.raises(ValueError):
        graph.node_labels(node)
    with pytest.raises(ValueError):
        graph.edge_relationship_type(first)


@pytest.mark.parametrize("value", [None, "Entity", [1], [""], [" "], ["Entity", "Entity"]])
def test_invalid_labels_reject_creation_before_ids_are_allocated(value):
    graph = Graph()
    with pytest.raises(ValueError, match="labels"):
        graph.add_nodes([{"labels": ["Entity"]}, {"labels": value}])
    assert graph.add_node() == 0


@pytest.mark.parametrize("value", [3, [], "", " "])
def test_invalid_relationship_types_reject_creation_before_edge_allocation(value):
    graph = Graph()
    node = graph.add_node()
    with pytest.raises(ValueError, match="relationship_type"):
        graph.add_edge(node, node, relationship_type=value)
    assert graph.add_edge(node, node) == 0


def test_defaults_empty_labels_and_explicit_clearing():
    graph = Graph()
    node = graph.add_node(labels=[])
    assert graph.node_labels(node) == []
    assert graph.nodes["labels"][0] == []
    graph.set_node_attrs({"labels": {node: ["Person", "Entity"]}})
    assert graph.node_labels(node) == ["Entity", "Person"]
    graph.set_node_attr(node, "labels", [])
    assert graph.node_labels(node) == []
    edge = graph.add_edge(node, node)
    assert graph.edge_relationship_type(edge) is None
    with pytest.raises(ValueError):
        graph.set_node_attr(node, "relationship_type", "bad kind")
    with pytest.raises(ValueError):
        graph.set_edge_attr(edge, "labels", ["bad kind"])


def test_invalid_kind_updates_preserve_all_bulk_values_and_batch_allocation():
    graph = Graph()
    node = graph.add_node(labels=["Entity"])
    edge = graph.add_edge(node, node, relationship_type="KNOWS")
    graph.commit("initial", "test")
    with pytest.raises(ValueError, match="labels"):
        graph.set_node_attrs({"name": {node: "partial"}, "labels": {node: ["Entity", "Entity"]}})
    with pytest.raises(ValueError, match="relationship_type"):
        graph.set_edge_attrs({"weight": {edge: 1}, "relationship_type": {edge: 3}})
    assert graph.get_node_attr(node, "name") is None
    assert graph.get_edge_attr(edge, "weight") is None
    batch = GraphBatch()
    batch.add_node(labels=None)
    with pytest.raises(ValueError, match="labels"):
        graph.apply_batch(batch, graph.current_state_id, "bad", "test")
    assert not graph.has_uncommitted_changes()
    assert graph.add_node() == 1


def test_kind_upserts_are_checked_and_idempotent_with_unsorted_labels():
    graph = Graph()
    row = {"external_id": "ada", "labels": ["Person", "Entity"]}
    node = graph.upsert_nodes([row])[0]
    graph.commit("initial", "test")
    assert graph.upsert_nodes([row]) == [node]
    assert not graph.has_uncommitted_changes()
    with pytest.raises(ValueError, match="labels"):
        graph.upsert_nodes([{"external_id": "bad", "labels": [""]}])
    assert graph.add_node() == 1


def test_schema_validation_is_explicit_structured_and_does_not_change_state():
    graph = Graph()
    node = graph.add_node(labels=["Person", "Entity"], age=None, free_form="allowed")
    edge = graph.add_edge(node, node, relationship_type="KNOWS", weight="bad")
    state = graph.commit("fixture", "test")
    schema = {
        "node_labels": {
            "Person": {"name": {"type": "text"}, "age": {"type": "int", "required": False}},
            "Entity": {"id": {"type": "int", "required": False, "nullable": True}},
        },
        "relationship_types": {"KNOWS": {"weight": {"type": "float"}}},
    }
    issues = graph.validate_schema(schema)
    assert [(row["entity_type"], row["entity_id"], row["property"]) for row in issues] == [
        ("node", node, "age"), ("node", node, "name"), ("edge", edge, "weight")
    ]
    assert "null" in issues[0]["message"]
    assert "absent" in issues[1]["message"]
    assert not graph.has_uncommitted_changes()
    assert graph.current_state_id == state
    graph.set_node_attrs({"name": {node: "x" * 10000}, "age": {node: 42}, "id": {node: None}})
    graph.set_edge_attr(edge, "weight", 1.0)
    assert graph.validate_schema(schema) == []
    assert graph.validate_schema({}) == []
    # Validation does not install or enforce this schema on subsequent mutations.
    graph.add_node(labels=["Person"])
    assert graph.validate_schema(schema)


@pytest.mark.parametrize("schema", [
    {"unknown": {}},
    {"node_labels": {"": {}}},
    {"node_labels": {"Person": {"age": {"type": "unknown"}}}},
    {"node_labels": {"Person": {"age": {"type": "int", "unknown": True}}}},
])
def test_invalid_schema_definitions_fail_explicitly(schema):
    with pytest.raises(ValueError, match="[Ss]chema"):
        Graph().validate_schema(schema)


def test_all_matching_labels_apply_and_unknown_kinds_stay_flexible():
    graph = Graph()
    node = graph.add_node(labels=["A", "B", "Unknown"], value=1)
    schema = {"node_labels": {
        "A": {"value": {"type": "int"}},
        "B": {"value": {"type": "text"}},
    }}
    issues = graph.validate_schema(schema)
    assert len(issues) == 1
    assert issues[0]["entity_id"] == node
    assert issues[0]["selector"] == "B"


def test_successful_mixed_batch_publishes_kinds_together():
    graph = Graph()
    batch = GraphBatch()
    node = batch.add_node(labels=["Person", "Entity"])
    batch.add_edge(node, node, relationship_type="KNOWS")
    result = graph.apply_batch(batch, graph.current_state_id, "kinds", "test")
    assert graph.nodes_with_label("Entity") == result["added_nodes"]
    assert graph.edges_with_relationship_type("KNOWS") == result["added_edges"]
    snapshot = graph.snapshot_at_commit(result["state_id"])
    assert snapshot.node_attr(result["added_nodes"][0], "labels").value == ["Entity", "Person"]
