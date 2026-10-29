"""Integer sampling produces bounded selected graphs, rather than input clones."""
import pytest
from groggy import Graph


@pytest.mark.parametrize("source", ["graph", "connected", "captured"])
@pytest.mark.parametrize("count", [0, 2, 100])
def test_integer_sampling_caps_nodes_and_induces_only_selected_edges(source, count):
    graph = Graph(directed=True)
    graph.add_nodes(6)
    for node in range(6):
        graph.add_edge(node, node)
    graph.add_edge(0, 1)
    graph.add_edge(0, 1)
    graph.add_edge(1, 0)
    subject = graph if source == "graph" else graph.view() if source == "connected" else graph.read_view()
    head = graph.current_state_id
    result = subject.sample(count)
    assert graph.current_state_id == head
    nodes = set(result.node_ids if source == "captured" else result.node_ids.to_list())
    edges = set(result.edge_ids if source == "captured" else result.edge_ids.to_list())
    assert len(nodes) == min(count, 6)
    assert edges == {edge for edge in graph.edge_ids if set(graph.edge_endpoints(edge)) <= nodes}
    assert graph.node_count() == 6
    assert graph.edge_count() == 9


@pytest.mark.parametrize("source", ["graph", "connected", "captured"])
def test_seeded_node_sampling_is_repeatable_and_rejects_invalid_controls(source):
    graph = Graph()
    graph.add_nodes(12)
    subject = graph if source == "graph" else graph.view() if source == "connected" else graph.read_view()
    first = subject.sample(4, seed=19)
    second = subject.sample(4, seed=19)
    assert sorted(first.node_ids) == sorted(second.node_ids)
    assert len(first.node_ids) == 4
    for invalid in [-1, 2 ** 64, True, 1.5, "3"]:
        with pytest.raises((TypeError, ValueError, OverflowError)):
            subject.sample(4, seed=invalid)
    for invalid in [-1, 1.5, "2", True, 2 ** 128]:
        with pytest.raises((TypeError, ValueError, OverflowError)):
            subject.sample(invalid)


def test_connected_and_frozen_samples_share_scope_and_pin_values_after_owner_change():
    graph = Graph(directed=True)
    graph.add_nodes(8)
    for node in range(6):
        graph.add_edge(node, node)
    selected = graph.induced_subgraph([0, 1, 2, 3, 4, 5])
    frozen = selected.pin_view()
    live = selected.sample(3, seed=31)
    sample = frozen.sample(3, seed=31)
    assert sorted(live.node_ids) == sample.node_ids
    assert sorted(live.edge_ids) == sample.edge_ids
    node = sample.node_ids[0]
    graph.remove_node(node)
    for count in [0, 2, 100]:
        with pytest.raises((ValueError, RuntimeError)):
            selected.sample(count)
    assert sample.node_count() == 3
    assert sample.node_ids[0] == node
    assert sample.write_state_id == frozen.write_state_id


def test_component_and_subgraph_arrays_sample_each_input_and_propagate_invalid_scope():
    graph = Graph()
    graph.add_nodes(6)
    graph.add_edge(0, 1)
    graph.add_edge(2, 3)
    graph.add_edge(4, 5)
    components = graph.view().connected_components()
    samples = components.sample(1)
    assert len(samples) == 3
    assert all(item.node_count() == 1 and item.edge_count() == 0 for item in samples)
    assert all(item.node_count() == 0 for item in samples.sample(0))
    graph.remove_node(0)
    with pytest.raises((ValueError, RuntimeError)):
        components.sample(0)


def test_sample_preserves_temporal_clocks_without_current_write_authority():
    from groggy import GraphBatch

    graph = Graph()
    batch = GraphBatch(effective_at=10)
    for _ in range(4):
        batch.add_node(score=42)
    graph.apply_batch(batch, 0, "initial", "test")
    source = graph.read_view_at(15)
    sample = source.sample(2, seed=9)
    assert sample.read_time == source.read_time
    assert sample.write_state_id is None
    graph.set_node_attr(sample.node_ids[0], "score", 99)
    assert all(sample.node_attr(node, "score") == 42 for node in sample.node_ids)
    assert sample.nodes_table().base_table().source_views()[0].read_time == source.read_time


def test_sample_keeps_checked_write_provenance_but_original_values_remain_frozen():
    from groggy import GraphBatch

    graph = Graph()
    graph.add_nodes(6)
    source = graph.read_view().sample(2, seed=13)
    batch = GraphBatch(effective_at=20)
    for node in source.node_ids:
        batch.set_node_attr(node, "score", 7)
    result = graph.apply_view_batch(source, batch, "sample update", "test")
    refreshed = result["view"]
    assert refreshed.node_ids == source.node_ids
    assert all(source.node_attr(node, "score") is None for node in source.node_ids)
    assert all(refreshed.node_attr(node, "score") == 7 for node in source.node_ids)
    assert sorted(node for node in graph.node_ids if graph.get_node_attr(node, "score") == 7) == source.node_ids


def test_sample_keeps_retained_meta_source_catalog():
    graph = Graph()
    graph.add_nodes(2)
    group = graph.retain_group(graph.read_view().select_nodes([0]), kind="source")
    meta = graph.create_meta_node(group)
    root = graph.read_view().select_nodes([0, meta])
    sample = next(root.sample(1, seed=seed) for seed in range(32) if meta in root.sample(1, seed=seed).node_ids)
    assert sample.meta_subgraph(meta).node_ids == [0]
    graph.remove_node(0)
    assert sample.meta_subgraph(meta).node_ids == [0]


def test_integer_samples_match_pipeline_masks_and_pipeline_seeds_are_explicit():
    import groggy

    graph = Graph(directed=True)
    graph.add_nodes(10)
    for node in range(10):
        graph.add_edge(node, (node + 1) % 10)
    builder = groggy.builder("sample_parity")
    mask = builder.sample_nodes(count=3, seed=17)
    builder.emit_subgraphs(mask, mode="unified")
    sampler = builder.build_sampler()
    piped = graph.view().sample(sampler)[0]
    direct = graph.read_view().sample(3, seed=17)
    assert sorted(piped.node_ids) == direct.node_ids
    assert sorted(piped.edge_ids) == direct.edge_ids
    with pytest.raises(TypeError, match="specification"):
        graph.view().sample(sampler, seed=17)
