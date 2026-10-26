"""Explicit full captures restore independent graphs; ordinary snapshots remain read pins."""
import pytest
from groggy import DurableGraph, Graph, GraphBatch


def initial(graph):
    batch = GraphBatch(effective_at=10)
    a = batch.add_node(v=[1.25, -0.0])
    b = batch.add_node()
    batch.add_edge(a, b, v=None)
    return graph.apply_batch(batch, 0, "initial", "test")["state_id"]


def test_owned_capture_keeps_all_core_config_history_reservations_and_independent_writes():
    graph = Graph(config={"query_timeout_ms": 123, "worker_threads": 2,
                          "enable_compression": True, "compression_level": 3}, directed=True)
    first = initial(graph)
    graph.create_branch("before")
    graph.writer().set_node_external_id(0, "original")
    graph.writer().set_node_external_id(0, "renamed")
    unused = graph.add_node()
    graph.remove_node(unused)
    graph.writer().unset_node_attr(0, "v", effective_at=20)
    latest = graph.current_state_id
    capture = graph.snapshot_at_commit(first, owned=True)
    metadata = graph.publication_metadata(first)
    config = graph.configuration()
    assert capture.is_owned and capture.configuration() == config
    graph.set_node_attr(0, "v", 99)
    del graph
    copy = capture.to_graph()
    assert copy.current_state_id == first
    assert copy.configuration() == config and copy.is_directed
    assert copy.publication_metadata(first) == metadata
    assert copy.get_node_attr(0, "v") == [1.25, -0.0]
    assert copy.allocated_entity_for_external_id("original") == ("node", 0)
    assert copy.allocated_entity_for_external_id("renamed") == ("node", 0)
    assert copy.node_by_external_id("renamed") is None
    assert copy.snapshot_at_commit(latest).node_attr(0, "v") is None
    assert len(copy.branches()) == 2
    assert copy.add_node() == unused + 1
    assert copy.current_state_id > latest
    other = capture.to_graph()
    assert other.current_state_id == first and len(other.nodes) == 2
    assert not other.has_uncommitted_changes()


def test_default_capture_retains_configuration_and_frozen_sources_without_full_owned_mode():
    graph = Graph(config={"query_timeout_ms": 456}, directed=True)
    initial(graph)
    group = graph.retain_group(graph.read_view_at(10))
    meta = graph.create_meta_node(group)
    snapshot = graph.snapshot_at_commit(graph.current_state_id)
    assert not snapshot.is_owned
    assert snapshot.configuration() == graph.configuration()
    with pytest.raises(ValueError, match="owned snapshot"):
        snapshot.to_graph()
    selected = snapshot.as_subgraph().pin_view()
    assert selected.meta_subgraph(meta).node_attr(0, "v") == [1.25, -0.0]
    del graph
    assert selected.is_directed
    assert selected.meta_subgraph(meta).edge_ids == [0]


def test_owned_capture_after_durable_close_restores_core_and_retained_group_sources(tmp_path):
    path = tmp_path / "graph.ggj"
    with DurableGraph.create(path, directed=True, config={"query_timeout_ms": 789}) as graph:
        first = initial(graph)
        group = graph.retain_group(graph.read_view_at(10))
        meta = graph.create_meta_node(group, effective_at=20)
        state = graph.current_state_id
        graph.writer().remove_node(0, effective_at=30)
        snapshot = graph.snapshot_at_commit(state, owned=True)
    copy = snapshot.to_graph()
    assert copy.configuration()["query_timeout_ms"] == 789
    assert copy.current_state_id == state
    assert copy.read_view_at(20).meta_subgraph(meta).edge_ids == [0]
    assert copy.snapshot_at_commit(first).node_attr(0, "v").value == [1.25, -0.0]
    assert copy.add_node() == meta + 1
    assert snapshot.to_graph().current_state_id == state


def test_owned_timestamp_selection_and_dirty_capture_guards():
    graph = Graph(manual=True)
    graph.add_node(v=None)
    first = graph.commit("initial", "test")
    selected = graph.snapshot_at_timestamp(graph.commit_history()[0].timestamp, owned=True)
    assert selected.is_owned and selected.commit_id == first
    graph.set_node_attr(0, "v", 3)
    with pytest.raises(ValueError, match="uncommitted"):
        graph.snapshot_at_commit(first, owned=True)
    assert selected.to_graph().get_node_attr(0, "v") is None
    assert selected.to_graph().has_node_attribute(0, "v")


@pytest.mark.parametrize("configuration", [{"max_memory_usage": 0}, {"unrecognized": 1},
                                            {"query_timeout_ms": "slow"}])
def test_regular_graph_configuration_rejects_unknown_invalid_and_wrong_types(configuration):
    with pytest.raises(ValueError):
        Graph(config=configuration)


def test_legacy_config_argument_is_honored_and_ambiguous_aliases_reject():
    graph = Graph(_config={"query_timeout_ms": 321})
    assert graph.configuration()["query_timeout_ms"] == 321
    with pytest.raises(ValueError, match="not both"):
        Graph(_config={}, config={})
