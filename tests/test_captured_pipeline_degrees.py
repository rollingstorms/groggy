"""Structural computation from explicit captured topology after staged effects."""
import pytest
from groggy import Graph
from groggy.builder import step_pipeline


def stage(kind, **params):
    return {"id": kind, "params": params}


@pytest.mark.parametrize("directed", [False, True])
def test_historical_degree_table_uses_sparse_captured_ids_not_owner_liveness(directed):
    graph = Graph(directed=directed)
    graph.add_nodes(7)
    graph.remove_nodes([1, 3, 4])
    graph.add_edges([(0, 2), (0, 2), (2, 2), (5, 0)])
    graph.set_node_attr(0, "degree", 999)
    saved = graph.current_state_id
    graph.remove_node(2)
    head = graph.current_state_id
    _, profile = graph.apply(step_pipeline("historical degrees", [
        stage("temporal.snapshot_at", state_id=saved, output="saved"),
        stage("graph.from_snapshot", source="saved", output="old"),
        stage("graph.degrees_table", source="old", output="metrics"),
        stage("table.emit", source="metrics", name="metrics"),
    ], atomic=True), return_profile=True)
    table = profile["outputs"]["metrics"]
    assert table.columns == ["node_id", "degree", "in_degree", "out_degree"]
    assert table.source_entity_ids == [0, 2, 5, 6]
    assert table.column("degree").to_list() == [3, 3, 1, 0]
    assert table.column("in_degree").to_list() == ([1, 3, 0, 0] if directed else [3, 3, 1, 0])
    assert table.column("out_degree").to_list() == ([2, 1, 1, 0] if directed else [3, 3, 1, 0])
    assert graph.current_state_id == head
    assert graph.get_node_attr(0, "degree") == 999
    table.set_column("degree", [0, 0, 0, 0])
    assert table.source_views()[0].node_attr(0, "degree") == 999


@pytest.mark.parametrize("directed", [False, True])
def test_degree_tables_before_and_after_endpoint_revision_and_removal_use_each_capture(directed):
    graph = Graph(directed=directed)
    graph.add_nodes(4)
    graph.add_edges([(0, 1), (0, 1), (1, 1)])
    head = graph.current_state_id
    _, profile = graph.apply(step_pipeline("recompute pending degrees", [
        stage("graph.capture_staged", output="before"),
        stage("graph.set_edge_endpoints", revisions=[[0, 2, 3]]),
        stage("graph.remove_nodes", ids=[1]),
        stage("graph.capture_staged", output="after"),
        stage("graph.degrees_table", source="before", output="before_metrics"),
        stage("graph.degrees_table", source="after", output="after_metrics"),
        stage("table.emit", source="before_metrics", name="before"),
        stage("table.emit", source="after_metrics", name="after"),
    ], atomic=True), return_profile=True)
    before, after = profile["outputs"]["before"], profile["outputs"]["after"]
    assert before.source_entity_ids == [0, 1, 2, 3]
    assert before.column("degree").to_list() == [2, 3, 0, 0]
    assert after.source_entity_ids == [0, 2, 3]
    assert after.column("degree").to_list() == [0, 1, 1]
    assert after.column("in_degree").to_list() == ([0, 0, 1] if directed else [0, 1, 1])
    assert after.column("out_degree").to_list() == ([0, 1, 0] if directed else [0, 1, 1])
    assert graph.current_state_id == head + 1


def test_degree_table_respects_exact_selected_edge_instances_and_empty_shape():
    graph = Graph()
    graph.add_nodes(3)
    graph.add_edges([(0, 1), (0, 1), (1, 1), (1, 2)])
    state = graph.current_state_id
    _, profile = graph.apply(step_pipeline("selected degree", [
        stage("temporal.snapshot_at", state_id=state, output="saved"),
        stage("graph.from_snapshot", source="saved", output="all"),
        stage("graph.select_edges", source="all", ids=[0, 2], output="exact"),
        stage("graph.degrees_table", source="exact", output="degrees"),
        stage("table.emit", source="degrees", name="degrees"),
        stage("graph.select_nodes", source="all", ids=[], output="empty"),
        stage("graph.degrees_table", source="empty", output="empty_degrees"),
        stage("table.emit", source="empty_degrees", name="empty"),
    ], atomic=True), return_profile=True)
    assert profile["outputs"]["degrees"].source_entity_ids == [0, 1]
    assert profile["outputs"]["degrees"].column("degree").to_list() == [1, 2]
    assert profile["outputs"]["empty"].shape == (0, 4)
    assert graph.current_state_id == state


def test_late_degree_type_error_discards_staged_effects():
    graph = Graph()
    graph.add_nodes(2)
    state = graph.current_state_id
    handle = step_pipeline("wrong degree source", [
        stage("graph.capture_staged", output="before"),
        stage("graph.nodes_table", source="before", output="table"),
        stage("graph.remove_nodes", ids=[1]),
        stage("graph.degrees_table", source="table", output="bad"),
    ], atomic=True)
    with pytest.raises(RuntimeError, match="graph view"):
        graph.apply(handle)
    assert graph.current_state_id == state
    assert sorted(graph.node_ids) == [0, 1]


@pytest.mark.parametrize("local_atomic", [False, True])
@pytest.mark.parametrize("outer_atomic", [False, True])
@pytest.mark.parametrize("persist", [False, True])
def test_explicit_degree_inputs_work_in_existing_read_execution_modes(local_atomic, outer_atomic, persist):
    from groggy import Pipeline

    graph = Graph()
    graph.add_nodes(3)
    graph.add_edge(0, 1)
    state = graph.current_state_id
    handle = step_pipeline("explicit read", [
        stage("temporal.snapshot_at", state_id=state, output="saved"),
        stage("graph.from_snapshot", source="saved", output="input"),
        stage("graph.degrees_table", source="input", output="degrees"),
        stage("table.emit", source="degrees", name="degrees"),
    ], atomic=local_atomic)
    _, profile = Pipeline([handle])(graph.view(), atomic=outer_atomic, persist=persist, return_profile=True)
    assert profile["outputs"]["degrees"].column("degree").to_list() == [1, 1, 0]
    assert graph.current_state_id == state
