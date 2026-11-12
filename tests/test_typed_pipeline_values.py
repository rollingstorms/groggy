"""Native typed stages retain historical sources through Python result wrappers."""
import pytest
from groggy import Graph, Pipeline
from groggy.builder import step_pipeline


def stage(kind, **params):
    return {"id": kind, "params": params}


def capture(state):
    return [
        stage("temporal.snapshot_at", state_id=state, output="saved"),
        stage("alias", source="saved", target="snapshot_alias"),
        stage("graph.from_snapshot", source="snapshot_alias", output="graph"),
    ]


def fixture():
    graph = Graph(directed=True)
    graph.add_nodes([{"x": 9007199254740993}, {}, {"x": None}, {}, {"x": 1.0000000001}, {}])
    graph.remove_nodes([1, 3])
    graph.add_edges([(0, 2), (0, 2, {"w": 2.0000000001}), (2, 4), (4, 4)])
    return graph, graph.current_state_id


def test_native_typed_graph_table_pipeline_keeps_history_identity_rows_and_owned_outputs():
    graph, state = fixture()
    graph.remove_node(2)
    graph.set_node_attr(0, "x", -1)
    head = graph.current_state_id
    steps = capture(state) + [
        stage("graph.select_nodes", source="graph", ids=[5, 4, 0, 2, 2], output="subset"),
        stage("alias", source="subset", target="graph_alias"),
        stage("graph.nodes_table", source="graph_alias", output="nodes"),
        stage("alias", source="nodes", target="table_alias"),
        stage("table.emit", source="table_alias", name="all_nodes"),
        stage("table.select_columns", source="nodes", columns=["x"], output="nodes"),
        stage("table.select_rows", source="nodes", rows=[3, 0, 0], output="rows"),
        stage("table.emit", source="rows", name="rows"),
        stage("graph.select_edges", source="graph", ids=[1, 3], output="edges_graph"),
        stage("graph.edges_table", source="edges_graph", output="edges"),
        stage("table.emit", source="edges", name="edges"),
        stage("graph.emit", source="subset", name="graph"),
    ]
    handle = step_pipeline("historical projections", steps, atomic=True)
    # Native programs retain their submitted value/control order, not the
    # caller's later edits or legacy alias-elision policy.
    steps[0]["params"]["state_id"] = 999999
    _, profile = graph.apply(handle, persist=False, return_profile=True)
    out = profile["outputs"]
    assert graph.current_state_id == head
    assert sorted(out["graph"].node_ids) == [0, 2, 4, 5]
    assert sorted(out["graph"].edge_ids) == [0, 1, 2, 3]
    assert out["graph"].node_attr(0, "x") == 9007199254740993
    assert out["all_nodes"].source_entity_ids == [0, 2, 4, 5]
    assert out["all_nodes"].column("x").to_list() == [9007199254740993, None, 1.0000000001, None]
    assert out["rows"].source_entity_ids == [5, 0, 0]
    assert out["rows"].column("x").to_list() == [None, 9007199254740993, 9007199254740993]
    assert out["edges"].source_entity_ids == [1, 3]
    assert out["edges"].column("w").to_list() == [2.0000000001, None]
    out["all_nodes"].set_column("x", [1, 2, 3, 4])
    assert out["graph"].node_attr(0, "x") == 9007199254740993
    assert out["rows"].column("x").to_list() == [None, 9007199254740993, 9007199254740993]
    del graph
    assert out["graph"].edge_endpoints(1) == (0, 2)
    assert sorted(out["all_nodes"].source_views()[0].node_ids) == [0, 2, 4, 5]


@pytest.mark.parametrize("outer", [False, True])
@pytest.mark.parametrize("persist", [False, True])
def test_explicit_snapshot_values_exclude_inflight_writes_in_both_atomic_coordinators(outer, persist):
    graph, state = fixture()
    steps = [
        stage("core.init_nodes", target="values", value=7),
        stage("core.attach_node_attr", source="values", attr="x"),
    ] + capture(state) + [stage("graph.emit", source="graph", name="captured")]
    handle = step_pipeline("staged properties and explicit history", steps, atomic=True)
    _, profile = Pipeline([handle])(graph.view(), atomic=outer, persist=persist, return_profile=True)
    assert graph.current_state_id == state + int(persist)
    assert profile["outputs"]["captured"].node_attr(0, "x") == 9007199254740993
    if persist:
        assert graph.get_node_attr(0, "x") == 7


@pytest.mark.parametrize("bad", [
    stage("graph.nodes_table", source="saved", output="wrong_type"),
    stage("graph.select_nodes", source="graph", ids=[999], output="outside"),
    stage("graph.select_edges", source="graph", ids=[999], output="outside"),
    stage("table.select_rows", source="nodes", rows=[999], output="outside"),
    stage("table.select_columns", source="nodes", columns=["missing"], output="outside"),
])
def test_late_stage_errors_discard_staged_effects_without_partial_graph_publication(bad):
    graph, state = fixture()
    handle = step_pipeline("late failure", capture(state) + [
        stage("graph.nodes_table", source="graph", output="nodes"),
        stage("core.init_nodes", target="values", value=7),
        stage("core.attach_node_attr", source="values", attr="prefix"),
        stage("graph.emit", source="graph", name="partial"), bad,
    ], atomic=True)
    with pytest.raises(RuntimeError):
        graph.apply(handle)
    assert graph.current_state_id == state
    assert graph.get_node_attr(0, "prefix") is None
    assert not graph.has_uncommitted_changes()


@pytest.mark.parametrize("bad", [
    stage("graph.select_nodes", source="graph", ids=[-1], output="bad"),
    stage("graph.select_nodes", source="graph", ids=[True], output="bad"),
    stage("graph.select_edges", source="graph", ids=[1.5], output="bad"),
    stage("table.select_rows", source="nodes", rows=["0"], output="bad"),
    stage("table.select_columns", source="nodes", columns=[1], output="bad"),
    stage("graph.emit", source="graph", name="out", live=True),
])
def test_typed_controls_reject_before_prefix_writes_even_on_empty_graph(bad):
    graph = Graph()
    handle = step_pipeline("invalid control", [
        stage("core.init_nodes", target="values", value=7),
        stage("core.attach_node_attr", source="values", attr="prefix"), bad,
    ], atomic=True)
    with pytest.raises((RuntimeError, ValueError)):
        graph.apply(handle)
    assert graph.current_state_id == 0


def test_empty_selections_tables_and_columns_are_real_values_with_source_provenance():
    graph, state = fixture()
    handle = step_pipeline("empty values", capture(state) + [
        stage("graph.select_nodes", source="graph", ids=[], output="empty"),
        stage("graph.nodes_table", source="empty", output="nodes"),
        stage("table.select_columns", source="nodes", columns=[], output="none"),
        stage("table.emit", source="none", name="table"),
        stage("graph.emit", source="empty", name="graph"),
    ], atomic=True)
    _, profile = graph.apply(handle, return_profile=True)
    assert len(profile["outputs"]["table"]) == 0
    assert list(profile["outputs"]["graph"].node_ids) == []
    assert graph.current_state_id == state
