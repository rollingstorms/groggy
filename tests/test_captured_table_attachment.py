"""Explicit table columns attach through retained row identity in one atomic run."""
import pytest
from groggy import Graph, GraphBatch, Pipeline
from groggy.builder import step_pipeline


def stage(kind, **params):
    return {"id": kind, "params": params}


def builder(*steps, atomic=True):
    return step_pipeline("captured table attachment", list(steps), atomic=atomic)


def fixture(directed=False):
    graph = Graph(directed=directed)
    initial = GraphBatch(effective_at=0)
    a = initial.add_node(score=1.0000000000000002)
    b = initial.add_node(score=None)
    c = initial.add_node(score=3.0)
    initial.add_edge(a, b, weight=1.0)
    initial.add_edge(b, c, weight=2.0)
    initial.add_edge(a, b, weight=3.0)
    graph.writer().apply_batch(initial)
    return graph


def creation():
    return [
        stage("graph.add_nodes", count=1, output="new", effective_at=10, attributes={"score": 4.5}),
        stage("graph.add_edges", edges=[[2, {"added": 0}], [{"added": 0}, 0]],
              output="edges", effective_at=20, attributes={"weight": 2.5}),
    ]


def captured_table(nodes=True):
    return [stage("graph.capture_staged", output="pending"),
            stage("graph.nodes_table" if nodes else "graph.edges_table", source="pending", output="table")]


@pytest.mark.parametrize("outer", [False, True])
@pytest.mark.parametrize("nodes", [False, True])
def test_property_tables_attach_reordered_rows_without_structural_columns(outer, nodes):
    graph = fixture(True)
    old = graph.read_view()
    head = graph.current_state_id
    source = "score" if nodes else "weight"
    rows = [3, 0]
    steps = captured_table(nodes) + [
        stage("table.select_rows", source="table", rows=rows, output="reordered"),
        stage("table.select_columns", source="reordered", columns=[source], output="properties"),
        stage("table.attach", source="properties", columns={source: "copied"}, effective_at=30),
        stage("table.emit", source="properties", name="original"),
    ]
    if outer:
        Pipeline([builder(*creation()), builder(*steps)])(graph.view(), atomic=True)
    else:
        graph.apply(builder(*(creation() + steps)))
    get = graph.get_node_attr if nodes else graph.get_edge_attr
    assert get(3, "copied") == (4.5 if nodes else 2.5)
    assert get(0, "copied") == (1.0000000000000002 if nodes else 1.0)
    assert get(1, "copied") is None
    assert graph.current_state_id == head + 1
    assert old.node_attr(0, "copied") is None
    assert (graph.read_view_at(29).node_attr(3, "copied") if nodes else graph.read_view_at(29).edge_attr(3, "copied")) is None
    assert (graph.read_view_at(30).node_attr(3, "copied") if nodes else graph.read_view_at(30).edge_attr(3, "copied")) == (4.5 if nodes else 2.5)


@pytest.mark.parametrize("algorithm,options,column", [
    ("bfs", {"start": 0}, "value"), ("dfs", {"start": 0}, "value"),
    ("dijkstra", {"start": 0}, "value"), ("astar", {"start": 0, "goal": 3}, "cost"),
    ("pagerank", {}, "value"), ("betweenness", {}, "value"), ("closeness", {}, "value"),
    ("components", {}, "value"), ("louvain", {}, "value"), ("leiden", {"seed": 4}, "value"),
    ("infomap", {"seed": 4, "num_trials": 1}, "value"),
    ("girvan_newman", {}, "value"), ("lpa", {"seed": 4}, "value"),
])
def test_all_captured_kernel_tables_can_attach_explicit_result_columns(algorithm, options, column):
    graph = fixture()
    _, profile = graph.apply(builder(*(creation() + [
        stage("graph.capture_staged", output="pending"),
        stage("graph.algorithm_table", source="pending", output="metrics", algorithm=algorithm, **options),
        stage("table.attach", source="metrics", columns={column: "metric"}, effective_at=30),
        stage("table.emit", source="metrics", name="metrics"),
    ])), return_profile=True)
    table = profile["outputs"]["metrics"]
    assert 3 in table.source_entity_ids
    for node, expected in zip(table.source_entity_ids, table.column(column).to_list()):
        assert graph.get_node_attr(node, "metric") == expected
    assert table.source_views()[0].write_state_id is None
    assert table.source_views()[0].node_attr(0, "metric") is None


@pytest.mark.parametrize("failure", ["duplicate_rows", "removed_target", "reserved_target", "reserved_source", "duplicate_target", "kind", "missing_column", "invalid_topology_clock", "historical_source"])
def test_attachment_rejections_preserve_topology_history_and_allocation(failure):
    graph = fixture()
    head = graph.current_state_id
    steps = creation() + captured_table() + [
        stage("core.init_nodes", target="prefix", value=7),
        stage("core.attach_node_attr", source="prefix", attr="prefix"),
    ]
    columns = {"score": "copied"}
    source = "table"
    at = 30
    if failure == "duplicate_rows":
        steps.append(stage("table.select_rows", source="table", rows=[0, 0], output="bad"))
        source = "bad"
    elif failure == "removed_target":
        # Node 1 existed before the run; removal of newly created IDs stays guarded.
        steps = creation() + captured_table() + [stage("graph.remove_nodes", ids=[1])]
    elif failure == "reserved_target":
        columns = {"score": "entity_type"}
    elif failure == "reserved_source":
        columns = {"node_id": "copied"}
    elif failure == "duplicate_target":
        steps.append(stage("graph.degrees_table", source="pending", output="degrees"))
        source = "degrees"
        columns = {"degree": "same", "out_degree": "same"}
    elif failure == "kind":
        columns = {"score": "labels"}
    elif failure == "missing_column":
        columns = {"missing": "copied"}
    elif failure == "invalid_topology_clock":
        steps[1]["params"]["effective_at"] = 5
    else:
        historical = [stage("temporal.snapshot_at", state_id=head, output="old"),
                      stage("graph.from_snapshot", source="old", output="old_view"),
                      stage("graph.nodes_table", source="old_view", output="historical")]
        steps = historical + steps
        source = "historical"
    steps.append(stage("table.attach", source=source, columns=columns, effective_at=at))
    with pytest.raises(RuntimeError, match="issued by this atomic run" if failure == "historical_source" else None):
        graph.apply(builder(*steps))
    assert graph.current_state_id == head
    assert sorted(graph.node_ids) == [0, 1, 2]
    assert graph.get_node_attr(0, "prefix") is None
    assert graph.get_node_attr(0, "copied") is None
    assert graph.add_node() == 3
    assert graph.add_edge(0, 2) == 3


def test_earlier_capture_can_write_surviving_rows_without_recomputing_values():
    graph = fixture()
    graph.apply(builder(
        stage("graph.capture_staged", output="before"),
        stage("graph.degrees_table", source="before", output="metrics"),
        stage("table.select_rows", source="metrics", rows=[0, 1], output="survivors"),
        stage("graph.remove_nodes", ids=[2]),
        stage("table.attach", source="survivors", columns={"degree": "prior_degree"}),
    ))
    assert graph.get_node_attr(1, "prior_degree") == 3
    assert graph.read_view().degree(1) == 2


@pytest.mark.parametrize("later_default", [False, True])
def test_last_staged_assignment_controls_explicit_property_time(later_default):
    graph = fixture()
    steps = captured_table() + [
        stage("table.attach", source="table", columns={"score": "copied"}, effective_at=30),
    ]
    if later_default:
        steps += [stage("core.init_nodes", target="replacement", value=9),
                  stage("core.attach_node_attr", source="replacement", attr="copied")]
    else:
        steps.append(stage("table.attach", source="table", columns={"score": "copied"}, effective_at=40))
    graph.apply(builder(*steps))
    metadata = graph.publication_metadata(graph.current_state_id)
    times = {(row["entity_id"], row["name"]): row["effective_at"]
             for row in metadata["change_times"] if row["kind"] == "node_attr"}
    for node in [0, 1, 2]:
        assert times[(node, "copied")] == (metadata["recorded_at"] if later_default else 40)
    assert graph.get_node_attr(0, "copied") == (9 if later_default else 1.0000000000000002)


def test_empty_column_selection_validates_identity_and_publishes_nothing():
    graph = fixture()
    head = graph.current_state_id
    graph.apply(builder(*(captured_table() + [stage("table.attach", source="table", columns=[])])))
    assert graph.current_state_id == head


@pytest.mark.parametrize("atomic,persist", [(False, True), (True, False)])
def test_atomic_persistence_requirements_reject_before_any_prefix(atomic, persist):
    graph = fixture()
    head = graph.current_state_id
    handle = builder(stage("core.init_nodes", target="prefix", value=9),
                     stage("core.attach_node_attr", source="prefix", attr="prefix"),
                     stage("table.attach", source="missing", columns=[]), atomic=atomic)
    with pytest.raises(RuntimeError, match="atomic=True|persist=True"):
        graph.apply(handle, persist=persist)
    assert graph.current_state_id == head
    assert graph.get_node_attr(0, "prefix") is None


def test_backdated_property_outside_entity_lifetime_keeps_canonical_time_semantics():
    graph = fixture()
    graph.apply(builder(*(creation() + captured_table() + [
        stage("table.attach", source="table", columns={"score": "copied"}, effective_at=5),
    ])))
    # Accepted-state values and effective-time reconstruction are distinct reads.
    assert graph.get_node_attr(3, "copied") == 4.5
    assert 3 not in graph.read_view_at(5).node_ids
    assert graph.read_view_at(30).node_attr(3, "copied") is None
    assert graph.read_view_at(30).node_attr(0, "copied") == 1.0000000000000002
