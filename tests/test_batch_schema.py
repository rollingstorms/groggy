import pytest

from groggy import Graph, GraphBatch


SCHEMA = {
    "node_labels": {
        "Entity": {
            "name": {"type": "text"},
            "entity_type": {"type": "text"},
            "labels": {"type": "text_vec"},
        }
    },
    "relationship_types": {
        "KNOWS": {
            "weight": {"type": "float"},
            "note": {"type": "text", "required": False, "nullable": True},
        }
    },
}


def people_batch():
    batch = GraphBatch()
    first = batch.add_node(labels=["Person", "Entity"], name="Ada")
    second = batch.add_node()
    batch.add_edge(first, second, relationship_type="KNOWS", weight=0.5)
    return batch


def test_schema_batch_publishes_once_and_keeps_prior_snapshot():
    graph = Graph(directed=True)
    result = graph.apply_batch(people_batch(), 0, "people", "test", schema=SCHEMA)
    assert result["added_nodes"] == [0, 1]
    assert result["added_edges"] == [0]
    assert len(graph.commit_history()) == 1
    assert graph.validate_schema(SCHEMA) == []
    prior = graph.snapshot_at_commit(result["state_id"])
    update = GraphBatch()
    update.set_node_attr(0, "name", "Grace")
    graph.apply_batch(update, result["state_id"], "rename", "test", schema=SCHEMA)
    assert prior.node_attr(0, "name").value == "Ada"
    assert graph.get_node_attr(0, "name") == "Grace"
    assert not graph.has_uncommitted_changes()
    assert len(graph.commit_history()) == 2


def test_schema_errors_aggregate_before_allocating_nodes_or_edges():
    graph = Graph()
    batch = GraphBatch()
    node = batch.add_node(labels=["Entity"])
    batch.add_edge(node, node, relationship_type="KNOWS", weight=2, note=False)
    with pytest.raises(ValueError, match="Schema rejected batch") as error:
        graph.apply_batch(batch, 0, "invalid", "test", schema=SCHEMA)
    text = str(error.value)
    assert "node 0 [Entity].name" in text
    assert "edge 0 [KNOWS].weight" in text
    assert "edge 0 [KNOWS].note" in text
    assert graph.current_state_id == 0
    assert not graph.has_uncommitted_changes()
    assert len(graph.nodes) == len(graph.edges) == 0
    assert graph.add_node() == 0
    assert graph.add_edge(0, 0) == 0


def test_schema_checks_label_changes_and_property_changes_together():
    graph = Graph()
    node = graph.add_node(name=42)
    head = graph.commit("untyped", "test")
    update = GraphBatch()
    update.set_node_attr(node, "labels", ["Entity"])
    update.set_node_attr(node, "name", "Ada")
    graph.apply_batch(update, head, "label and value", "test", schema=SCHEMA)
    assert graph.validate_schema(SCHEMA) == []


def test_untouched_invalid_entities_block_unrelated_batch():
    graph = Graph()
    node = graph.add_node(labels=["Entity"])
    head = graph.commit("legacy invalid", "test")
    unrelated = GraphBatch()
    unrelated.add_node()
    with pytest.raises(ValueError, match="name"):
        graph.apply_batch(unrelated, head, "unrelated", "test", schema=SCHEMA)
    assert graph.current_state_id == head
    assert not graph.has_uncommitted_changes()
    repair = GraphBatch()
    repair.set_node_attr(node, "name", "fixed")
    graph.apply_batch(repair, head, "repair", "test", schema=SCHEMA)
    assert graph.validate_schema(SCHEMA) == []


@pytest.mark.parametrize("remove_node", [True, False])
def test_removed_entities_and_incident_edges_are_not_validated(remove_node):
    graph = Graph()
    node = graph.add_node(labels=["Entity"])
    other = graph.add_node()
    edge = graph.add_edge(node, other, relationship_type="KNOWS")
    head = graph.commit("legacy invalid", "test")
    repair = GraphBatch()
    if remove_node:
        repair.remove_node(node)
    else:
        repair.set_node_attr(node, "labels", [])
        repair.remove_edge(edge)
    graph.apply_batch(repair, head, "remove invalid", "test", schema=SCHEMA)
    assert len(graph.edges) == 0
    assert graph.validate_schema(SCHEMA) == []


@pytest.mark.parametrize("value,allowed", [(None, True), ("source", True), (42, False)])
def test_nullable_optional_values_are_checked_on_the_final_state(value, allowed):
    graph = Graph()
    result = graph.apply_batch(people_batch(), 0, "initial", "test", schema=SCHEMA)
    update = GraphBatch()
    update.set_edge_attr(0, "note", value)
    if allowed:
        graph.apply_batch(update, result["state_id"], "note", "test", schema=SCHEMA)
        assert graph.get_edge_attr(0, "note") == value
    else:
        with pytest.raises(ValueError, match="note"):
            graph.apply_batch(update, result["state_id"], "note", "test", schema=SCHEMA)
        assert graph.current_state_id == result["state_id"]
        assert not graph.has_uncommitted_changes()


def test_schema_is_per_call_and_legacy_or_flexible_paths_stay_explicit():
    graph = Graph()
    graph.apply_batch(people_batch(), 0, "initial", "test", schema=SCHEMA)
    # Passing a schema does not install a graph-wide write policy.
    flexible = GraphBatch()
    flexible.add_node(labels=["Entity"])
    graph.apply_batch(flexible, graph.current_state_id, "flexible", "test")
    assert graph.validate_schema(SCHEMA)
    graph.set_node_attr(0, "name", "ordinary update")
    published = graph.current_state_id
    blocked = people_batch()
    with pytest.raises(ValueError, match="[Ss]chema"):
        graph.apply_batch(blocked, graph.current_state_id, "blocked", "test", schema=SCHEMA)
    assert graph.get_node_attr(0, "name") == "ordinary update"
    assert graph.current_state_id == published
    assert not graph.has_uncommitted_changes()


def test_stale_state_or_invalid_schema_cannot_allocate():
    graph = Graph()
    first = graph.apply_batch(people_batch(), 0, "initial", "test", schema=SCHEMA)
    with pytest.raises(ValueError, match="Stale"):
        graph.apply_batch(people_batch(), 0, "stale", "test", schema=SCHEMA)
    with pytest.raises(ValueError, match="schema"):
        graph.apply_batch(people_batch(), first["state_id"], "bad schema", "test", schema={"unknown": {}})
    assert graph.current_state_id == first["state_id"]
    assert not graph.has_uncommitted_changes()
    assert graph.add_node() == 2
