import pytest

from groggy import Graph, GraphBatch


def test_upserts_keep_ids_and_are_clean_when_repeated():
    graph = Graph()
    graph.add_nodes(3)
    rows = [
        {"external_id": "alice", "score": 1, "group_id": "g1"},
        {"external_id": "bob", "score": 2},
    ]
    assert graph.upsert_nodes(rows) == [3, 4]
    graph.commit("initial", "test")
    assert graph.upsert_nodes(list(reversed(rows))) == [4, 3]
    assert not graph.has_uncommitted_changes()
    assert graph.node_by_external_id("alice") == 3
    assert graph.node_by_external_id("alice", group_id="g1") == 3
    assert graph.node_by_external_id("alice", group_id="g2") is None
    assert graph.upsert_nodes([{"external_id": "alice", "score": 9}]) == [3]
    assert graph.get_node_attr(3, "group_id") == "g1"
    assert graph.get_node_attr(3, "score") == 9


def test_conflicting_upserts_report_all_rows_before_any_write():
    graph = Graph()
    graph.upsert_nodes([{"external_id": "existing", "score": 1}])
    graph.commit("initial", "test")
    with pytest.raises(ValueError) as error:
        graph.upsert_nodes([
            {"external_id": "existing", "score": 9},
            {"external_id": "new"},
            {"external_id": "new"},
            {"external_id": "", "score": 3},
            {"external_id": "other", "entity_type": "meta"},
        ])
    message = str(error.value)
    assert "2: duplicate" in message
    assert "3: external ID" in message
    assert "4.entity_type" in message
    assert not graph.has_uncommitted_changes()
    assert graph.get_node_attr(0, "score") == 1
    assert graph.node_by_external_id("new") is None
    assert graph.add_node() == 1


def test_conversion_errors_in_later_rows_do_not_allocate():
    graph = Graph()
    with pytest.raises(ValueError) as error:
        graph.upsert_nodes([
            {"external_id": "valid"},
            {"external_id": "bad", "score": object()},
            {"score": object()},
        ])
    assert "1:" in str(error.value) and "2:" in str(error.value)
    assert graph.add_node() == 0
    assert graph.allocated_entity_for_external_id("valid") is None


def test_edges_use_managed_node_ids_and_preserve_parallel_facts():
    graph = Graph(directed=True)
    graph.upsert_nodes([{"external_id": "a"}, {"external_id": "b"}])
    rows = [
        {"external_id": "one", "source": "a", "target": "b", "weight": 1},
        {"external_id": "two", "source": "a", "target": "b", "weight": 2},
    ]
    assert graph.upsert_edges(rows) == [0, 1]
    graph.commit("initial", "test")
    assert graph.upsert_edges(rows) == [0, 1]
    assert not graph.has_uncommitted_changes()
    assert graph.edge_by_external_id("one") == 0
    with pytest.raises(ValueError) as error:
        graph.upsert_edges([
            {"external_id": "new", "source": "a", "target": "b"},
            {"external_id": "one", "source": "b", "target": "a"},
            {"external_id": "a", "source": "a", "target": "b"},
        ])
    assert "1: edge" in str(error.value)
    assert "2: external ID" in str(error.value)
    assert len(graph.edges) == 2
    assert not graph.has_uncommitted_changes()
    assert graph.add_edge(0, 1) == 2


def test_rename_history_and_removal_keep_reserved_identity():
    graph = Graph()
    graph.upsert_nodes([{"external_id": "old"}])
    first = graph.commit("initial", "test")
    graph.create_branch("original")
    graph.set_node_external_id(0, "new")
    graph.commit("rename", "test")
    assert graph.node_by_external_id("old") is None
    assert graph.node_by_external_id("new") == 0
    assert graph.snapshot_at_commit(first).node_by_external_id("old") == 0
    assert graph.allocated_entity_for_external_id("old") == ("node", 0)
    graph.checkout_branch("original")
    assert graph.node_by_external_id("old") == 0
    assert graph.node_by_external_id("new") is None
    with pytest.raises(ValueError, match="reserved"):
        graph.upsert_nodes([{"external_id": "new"}])
    graph.remove_node(0)
    with pytest.raises(ValueError, match="reserved"):
        graph.upsert_nodes([{"external_id": "old"}])
    assert graph.allocated_entity_for_external_id("old") == ("node", 0)


@pytest.mark.parametrize("bulk", [False, True])
def test_ordinary_creation_cannot_write_managed_identity(bulk):
    graph = Graph()
    with pytest.raises(ValueError, match="external_id"):
        if bulk:
            graph.add_nodes([{"name": "valid"}, {"external_id": "bypass"}])
        else:
            graph.add_node(external_id="bypass")
    assert graph.add_node() == 0
    with pytest.raises(ValueError, match="external_id"):
        if bulk:
            graph.add_edges([(0, 0), (0, 0, {"external_id": "bypass"})])
        else:
            graph.add_edge(0, 0, external_id="bypass")
    assert graph.add_edge(0, 0) == 0


@pytest.mark.parametrize("kind", ["node", "edge"])
def test_batch_cannot_bypass_managed_identity_before_allocation(kind):
    graph = Graph()
    batch = GraphBatch()
    if kind == "node":
        batch.add_node(external_id="bypass")
    else:
        node = batch.add_node()
        batch.add_edge(node, node, external_id="bypass")
    with pytest.raises(ValueError, match="external_id"):
        graph.apply_batch(batch, 0, "bad", "test")
    assert graph.add_node() == 0


def test_legacy_uid_creation_rejects_duplicates_and_ambiguity():
    graph = Graph()
    with pytest.raises(ValueError, match="duplicate"):
        graph.add_nodes([{"uid": "a"}, {"uid": "a"}], uid_key="uid")
    assert len(graph.nodes) == 0
    assert graph.add_nodes([{"uid": "a"}], uid_key="uid") == {"a": 0}
    with pytest.raises(ValueError, match="already identifies"):
        graph.add_nodes([{"uid": "a"}], uid_key="uid")
    graph.add_node(uid="a")
    with pytest.raises(ValueError, match="Ambiguous"):
        graph.add_edge("a", "a", uid_key="uid")
    with pytest.raises(ValueError, match="Ambiguous"):
        graph.get_node_mapping("uid")
    assert len(graph.edges) == 0


def test_custom_internal_id_is_rejected_before_allocating():
    graph = Graph()
    with pytest.raises(ValueError, match="Custom internal node IDs"):
        graph.add_node(node_id=7)
    assert graph.add_node() == 0


def test_custom_identity_input_key_is_consumed_into_managed_column():
    graph = Graph()
    assert graph.upsert_nodes([{"uuid": "uuid-a", "name": "Ada"}], uid_key="uuid") == [0]
    assert graph.get_node_attr(0, "external_id") == "uuid-a"
    assert graph.get_node_attr(0, "name") == "Ada"


def test_graph_copy_with_managed_ids_fails_before_allocation():
    source = Graph()
    source.upsert_nodes([{"external_id": "uuid"}])
    destination = Graph()
    with pytest.raises(ValueError, match="explicit upserts"):
        destination.add_graph(source)
    assert destination.add_node() == 0


def test_bad_python_row_does_not_hide_semantic_errors_or_shift_positions():
    graph = Graph()
    with pytest.raises(ValueError) as error:
        graph.upsert_nodes([
            {"external_id": "new"},
            {"external_id": "bad", "value": object()},
            {"external_id": "new"},
            {"external_id": ""},
        ])
    message = str(error.value)
    assert "1:" in message
    assert '2: duplicate external ID "new" (first at 0)' in message
    assert "3: external ID must not be empty" in message
    assert graph.add_node() == 0
