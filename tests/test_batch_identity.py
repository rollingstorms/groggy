import pytest
from groggy import Graph, GraphBatch


def test_mixed_batch_assigns_nodes_edges_and_identity_properties_atomically():
    graph = Graph()
    batch = GraphBatch(effective_at=10)
    source = batch.add_node(labels=["Entity"])
    target = batch.add_node()
    edge = batch.add_edge(source, target, relationship_type="KNOWS")
    batch.set_node_external_id(source, "ada", effective_at=5)
    batch.set_node_external_id(target, "lin")
    batch.set_edge_external_id(edge, "fact")
    schema = {"node_labels": {"Entity": {"external_id": {"type": "text"}}}}
    result = graph.apply_batch(batch, 0, "mixed", "test", schema=schema)
    assert graph.node_by_external_id("ada") == 0
    assert graph.edge_by_external_id("fact") == 0
    assert len(graph.commit_history()) == 1
    values = graph.publication_metadata(result["state_id"])["change_times"]
    identity = next(row for row in values if row["kind"] == "node_attr" and row["entity_id"] == 0 and row["name"] == "external_id")
    assert identity["effective_at"] == 5


def test_identity_collision_and_rejected_policy_preserve_all_reservations():
    graph = Graph()
    batch = GraphBatch()
    node = batch.add_node()
    edge = batch.add_edge(node, node)
    batch.set_node_external_id(node, "same")
    batch.set_edge_external_id(edge, "same")
    with pytest.raises(ValueError, match="duplicate external ID"):
        graph.apply_batch(batch, 0, "bad", "test")
    assert graph.allocated_entity_for_external_id("same") is None
    assert graph.add_node() == 0


def test_writer_published_upserts_are_idempotent_and_detect_other_writers():
    graph = Graph()
    writer = graph.writer()
    node = {"external_id": "ada", "labels": ["Person", "Entity"], "name": "Ada"}
    assert writer.upsert_nodes([node]) == [0]
    first = writer.state_id
    assert writer.upsert_nodes([node]) == [0]
    assert writer.state_id == first
    edge = {"external_id": "fact", "source": "ada", "target": "ada"}
    assert writer.upsert_edges([edge]) == [0]
    second = writer.state_id
    assert writer.upsert_edges([edge]) == [0]
    assert writer.state_id == second
    assert len(graph.commit_history()) == 2
    other = graph.writer()
    other.add_node()
    with pytest.raises(ValueError, match="Stale"):
        writer.upsert_nodes([node])


def test_writer_policy_rejects_upsert_without_consuming_names_and_rename_retains_old_name():
    graph = Graph()
    schema = {"node_labels": {"Entity": {"name": {"type": "text"}}}}
    writer = graph.writer(schema=schema)
    with pytest.raises(ValueError, match="name"):
        writer.upsert_nodes([{"external_id": "ada", "labels": ["Entity"]}])
    assert graph.allocated_entity_for_external_id("ada") is None
    assert writer.upsert_nodes([{"external_id": "ada", "labels": ["Entity"], "name": "Ada"}]) == [0]
    writer.set_node_external_id(0, "renamed")
    assert graph.node_by_external_id("renamed") == 0
    assert graph.node_by_external_id("ada") is None
    assert graph.allocated_entity_for_external_id("ada") == ("node", 0)
    state = writer.state_id
    with pytest.raises(ValueError, match="reserved"):
        writer.upsert_nodes([{"external_id": "ada", "name": "new"}])
    assert writer.state_id == state
