"""Preset options and explicit meta targets resolve in Rust before publication."""
import pytest
from groggy import Graph


def fixture(directed=False):
    graph = Graph(directed=directed)
    graph.add_nodes([{}, {}, {}])
    graph.add_edges([(0, 2, {"weight": 2.0}), (1, 2, {"weight": 4.0})])
    return graph


@pytest.mark.parametrize("preset,label,count", [("social_network", "community", True), ("org_hierarchy", "department", False)])
def test_presets_keep_semantic_labels_and_edge_defaults(preset, label, count):
    graph = fixture()
    before = graph.current_state_id
    frozen = graph.read_view()
    meta = graph.induced_subgraph([0, 1]).collapse(preset=preset)
    assert graph.current_state_id == before + 1
    assert graph.get_node_attr(meta.id, "entity_type") == "meta"
    assert graph.node_labels(meta.id) == [label]
    edge = next(e for e in graph.edge_ids if meta.id in graph.edge_endpoints(e))
    assert graph.get_edge_attr(edge, "weight") == (3.0 if preset == "social_network" else 6.0)
    assert graph.get_edge_attr(edge, "edge_count") == (2 if count else None)
    assert graph.snapshot_at_commit(before + 1).node_attr(meta.id, "labels").value == [label]
    assert frozen.node_attr(0, "labels") is None
    assert set(meta.expand().node_ids) == {0, 1}


def test_explicit_overrides_win_even_when_they_equal_old_python_defaults():
    graph = fixture()
    meta = graph.induced_subgraph([0, 1]).collapse(preset="org_hierarchy", include_edge_count=True, entity_type="meta", edge_strategy="keep_external")
    assert graph.node_labels(meta.id) == []
    edges = [e for e in graph.edge_ids if meta.id in graph.edge_endpoints(e)]
    assert len(edges) == 2
    assert all(graph.get_edge_attr(e, "edge_count") == 1 for e in edges)


@pytest.mark.parametrize("marker", [False, True])
def test_custom_kind_is_a_label_with_canonical_meta_identity(marker):
    graph = fixture()
    meta = graph.induced_subgraph([0, 1]).collapse(entity_type="Research", mark_entity_type=marker)
    assert graph.get_node_attr(meta.id, "entity_type") == "meta"
    assert graph.node_labels(meta.id) == ["Research"]
    assert set(meta.expand().node_ids) == {0, 1}


def with_meta(directed=False):
    graph = fixture(directed)
    outside = graph.induced_subgraph([2]).collapse(edge_strategy="drop_all")
    graph.add_edge(0, outside.id)
    if directed:
        graph.add_edge(outside.id, 1)
    return graph, outside.id


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("selected", [False, True])
def test_explicit_meta_selector_keeps_regular_boundaries_and_respects_orientation(directed, selected):
    graph, outside = with_meta(directed)
    before = graph.current_state_id
    meta = graph.induced_subgraph([0, 1]).collapse(meta_targets=[outside] if selected else [], node_strategy="collapse")
    assert graph.current_state_id == before + 1
    endpoints = [graph.edge_endpoints(e) for e in graph.edge_ids]
    assert (meta.id, 2) in endpoints
    assert ((meta.id, outside) in endpoints) == selected
    if directed:
        assert ((outside, meta.id) in endpoints) == selected


@pytest.mark.parametrize("kind", ["blank", "duplicate", "unknown", "regular", "internal"])
def test_invalid_semantics_or_meta_targets_preserve_ids_values_and_version(kind):
    graph, outside = with_meta()
    before = graph.current_state_id
    nodes, edges = set(graph.node_ids), set(graph.edge_ids)
    kwargs = {"blank": {"entity_type": " "}, "duplicate": {"meta_targets": [outside, outside]},
              "unknown": {"meta_targets": [999]}, "regular": {"meta_targets": [2]}, "internal": {"meta_targets": [0]}}[kind]
    with pytest.raises(RuntimeError):
        graph.induced_subgraph([0, 1]).collapse(node_strategy="collapse", **kwargs)
    assert graph.current_state_id == before
    assert set(graph.node_ids) == nodes and set(graph.edge_ids) == edges
    assert graph.add_node() == max(nodes) + 1
    assert not graph.has_uncommitted_changes()


@pytest.mark.parametrize("remove", [False, True])
def test_contract_all_routes_external_pairs_and_keeps_meta_summary_isolated(remove):
    graph = Graph(directed=True)
    graph.add_nodes(4)
    for left, right, capacity, flow in [(0, 1, 8, 9), (1, 2, 5, 7), (2, 3, 6, 3)]:
        graph.add_edge(left, right, capacity=capacity, flow=flow)
    before = graph.current_state_id
    meta = graph.induced_subgraph([1, 2]).collapse(preset="flow_network", node_strategy="collapse" if remove else "extract")
    assert graph.current_state_id == before + 1
    assert graph.node_labels(meta.id) == ["junction"]
    assert graph.get_node_attr(meta.id, "entity_type") == "meta"
    assert all(meta.id not in graph.edge_endpoints(edge) for edge in graph.edge_ids)
    routed = [e for e in graph.edge_ids if graph.edge_endpoints(e) == (0, 3)]
    assert len(routed) == 1
    assert graph.get_edge_attr(routed[0], "capacity") == 5
    assert graph.get_edge_attr(routed[0], "flow") == 19
    assert graph.get_edge_attr(routed[0], "edge_count") is None
    assert set(meta.expand().node_ids) == {1, 2}
    assert (1 not in graph.node_ids) == remove
    assert graph.snapshot_at_commit(before + 1).edge_attr(routed[0], "capacity").value == 5


@pytest.mark.parametrize("directed", [False, True])
def test_contraction_cycles_parallel_edges_and_existing_direct_facts_are_finite(directed):
    graph = Graph(directed=directed)
    graph.add_nodes(4)
    for left, right, weight in [(0, 1, 2), (0, 1, 3), (1, 2, 5), (2, 1, 7), (2, 3, 11)]:
        graph.add_edge(left, right, weight=weight)
    original = graph.add_edge(0, 3, weight=99)
    before = graph.current_state_id
    meta = graph.induced_subgraph([1, 2]).collapse(edge_strategy="contract_all", edge_aggs={"weight": "sum"}, node_strategy="collapse")
    assert graph.current_state_id == before + 1
    pairs = [graph.edge_endpoints(e) for e in graph.edge_ids]
    assert pairs == [(0, 3), (0, 3)]
    assert graph.get_edge_attr(original, "weight") == 99
    derived = next(e for e in graph.edge_ids if e != original)
    assert graph.get_edge_attr(derived, "weight") == 28
    assert graph.get_edge_attr(derived, "edge_count") == 5
    assert meta.id not in (0, 3)


def test_disconnected_selected_components_do_not_create_a_clique():
    graph = Graph()
    graph.add_nodes(6)
    graph.add_edges([(0, 1), (1, 2), (3, 4), (4, 5)])
    graph.induced_subgraph([1, 4]).collapse(edge_strategy="contract_all", node_strategy="collapse")
    assert sorted(graph.edge_endpoints(e) for e in graph.edge_ids) == [(0, 2), (3, 5)]


def test_contraction_failure_does_not_allocate_or_remove_sources():
    graph = Graph(directed=True)
    graph.add_nodes(3)
    graph.add_edge(0, 1, capacity="bad")
    graph.add_edge(1, 2, capacity="bad")
    before = graph.current_state_id
    with pytest.raises(RuntimeError):
        graph.induced_subgraph([1]).collapse(edge_strategy="contract_all", edge_aggs={"capacity": "min"}, node_strategy="collapse")
    assert graph.current_state_id == before
    assert set(graph.node_ids) == {0, 1, 2}
    assert set(graph.edge_ids) == {0, 1}
    assert graph.add_node() == 3


def test_component_array_forwards_omitted_preset_options_without_overwriting():
    graph = Graph()
    graph.add_nodes(4)
    graph.add_edges([(0, 1), (2, 3)])
    before = graph.current_state_id
    metas = graph.connected_components().collapse(preset="org_hierarchy")
    assert len(metas) == 2
    assert graph.current_state_id == before + 2
    assert all(graph.node_labels(meta.id) == ["department"] for meta in metas)
