"""Existing edge identities can be revised through local atomic step programs."""
import pytest
from groggy import Graph, GraphBatch
from groggy.builder import step_pipeline


def stage(kind, **params):
    return {"id": kind, "params": params}


def fixture(directed=False):
    graph = Graph(directed=directed)
    batch = GraphBatch(effective_at=0)
    nodes = [batch.add_node() for _ in range(4)]
    batch.add_edge(nodes[0], nodes[1], w=7)
    batch.add_edge(nodes[0], nodes[1])
    batch.add_edge(nodes[1], nodes[1])
    graph.apply_batch(batch, 0, "fixture", "test")
    return graph


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("reverse_order", [False, True])
def test_revised_edge_survives_cascade_with_captured_adjacency_and_clock(directed, reverse_order):
    graph = fixture(directed)
    head = graph.current_state_id
    before = graph.read_view()
    effects = [stage("graph.set_edge_endpoints", revisions=[[0, 2, 3]], effective_at=10),
               stage("graph.remove_nodes", ids=[1], effective_at=20)]
    if reverse_order:
        effects.reverse()
    _, profile = graph.apply(step_pipeline("rescue edge", effects + [
        stage("graph.capture_staged", output="pending"),
        stage("graph.edges_table", source="pending", output="edges"),
        stage("table.emit", source="edges", name="edges"),
        stage("graph.emit", source="pending", name="pending"),
    ], atomic=True), return_profile=True)
    view = profile["outputs"]["pending"]
    assert graph.current_state_id == head + 1
    assert sorted(view.edge_ids) == [0]
    assert view.edge_endpoints(0) == (2, 3)
    assert list(view.neighbors(2)) == [3]
    assert list(view.neighbors(3)) == ([] if directed else [2])
    assert view.edge_attr(0, "w") == graph.get_edge_attr(0, "w") == 7
    assert profile["outputs"]["edges"].source_entity_ids == [0]
    assert before.edge_endpoints(0) == (0, 1)
    assert graph.read_view_at(9).edge_endpoints(0) == (0, 1)
    assert graph.read_view_at(10).edge_endpoints(0) == (2, 3)
    metadata = graph.publication_metadata(graph.current_state_id)
    times = {(row["kind"], row["entity_id"]): row["effective_at"] for row in metadata["change_times"]}
    assert times[("edge_endpoints", 0)] == 10
    assert times[("edge_removed", 1)] == times[("edge_removed", 2)] == 20


@pytest.mark.parametrize("rows", [
    [[True, 0, 1]], [[-1, 0, 1]], [[0, 1.5, 2]], [[0, 1]], [[0, 1, 2, 3]], [0, 1, 2],
])
def test_invalid_rows_reject_before_any_property_publication(rows):
    graph = fixture()
    head = graph.current_state_id
    program = [stage("core.init_nodes", target="v", value=7),
               stage("core.attach_node_attr", source="v", attr="prefix"),
               stage("graph.set_edge_endpoints", revisions=rows)]
    with pytest.raises((ValueError, RuntimeError)):
        graph.apply(step_pipeline("invalid", program, atomic=True))
    assert graph.current_state_id == head
    assert graph.get_node_attr(0, "prefix") is None


@pytest.mark.parametrize("suffix", [
    [stage("graph.set_edge_endpoints", revisions=[[0, 2, 3], [0, 3, 2]])],
    [stage("graph.set_edge_endpoints", revisions=[[0, 2, 3]]), stage("graph.set_edge_endpoints", revisions=[[0, 3, 2]])],
    [stage("graph.set_edge_endpoints", revisions=[[0, 2, 3]]), stage("graph.remove_edges", ids=[0])],
    [stage("graph.set_edge_endpoints", revisions=[[0, 2, 3]]), stage("graph.remove_nodes", ids=[3])],
    [stage("graph.set_edge_endpoints", revisions=[[0, 99, 3]])],
    [stage("graph.set_edge_endpoints", revisions=[[0, 2, 3]], effective_at=30), stage("graph.remove_nodes", ids=[1], effective_at=20)],
])
def test_invalid_final_plans_roll_back_properties_endpoints_and_outputs(suffix):
    graph = fixture()
    head = graph.current_state_id
    program = [stage("core.init_nodes", target="v", value=7),
               stage("core.attach_node_attr", source="v", attr="prefix")] + suffix
    with pytest.raises(RuntimeError):
        graph.apply(step_pipeline("conflict", program, atomic=True))
    assert graph.current_state_id == head
    assert graph.edge_endpoints(0) == (0, 1)
    assert sorted(graph.edge_ids) == [0, 1, 2]
    assert graph.get_node_attr(0, "prefix") is None
    assert not graph.has_uncommitted_changes()


def test_endpoint_stage_preserves_self_loop_parallel_identity_and_empty_noop():
    graph = fixture()
    head = graph.current_state_id
    graph.apply(step_pipeline("empty", [stage("graph.set_edge_endpoints", revisions=[])], atomic=True))
    assert graph.current_state_id == head
    graph.apply(step_pipeline("loops", [stage("graph.set_edge_endpoints", revisions=[[0, 2, 2], [1, 2, 2]])], atomic=True))
    assert graph.edge_endpoints(0) == graph.edge_endpoints(1) == (2, 2)
    assert graph.get_edge_attr(0, "w") == 7
    assert sorted(graph.edge_ids) == [0, 1, 2]
