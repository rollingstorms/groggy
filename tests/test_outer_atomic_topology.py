"""Whole-pipeline topology, captured reads and one failure-safe acceptance."""

import pytest

from groggy import Graph, GraphBatch, Pipeline
from groggy.builder import step_pipeline


def stage(kind, **params):
    return {"id": kind, "params": params}


def builder(*steps):
    return step_pipeline("outer topology", list(steps), atomic=True)


def fixture(directed=False, manual=False):
    graph = Graph(directed=directed, manual=manual)
    batch = GraphBatch(effective_at=0)
    nodes = [batch.add_node() for _ in range(7)]
    for source, target in [(0, 2), (2, 4), (2, 4), (4, 4), (5, 6)]:
        batch.add_edge(nodes[source], nodes[target])
    graph.writer().apply_batch(batch)
    removals = GraphBatch(effective_at=1)
    removals.remove_node(1)
    removals.remove_node(3)
    graph.writer().apply_batch(removals)
    if manual:
        graph.commit("sparse IDs", "test")
    return graph


def create(at=10):
    return builder(stage("graph.add_nodes", count=1, output="created",
                         attributes={"score": 1.0000000000000002}, effective_at=at))


def connect(at=20):
    return builder(
        stage("graph.add_nodes", count=1, output="more", effective_at=12),
        stage("graph.add_edges", output="edges", effective_at=at, edges=[
            [0, {"added": 0}], [{"added": 0}, {"added": 1}],
            [{"added": 0}, {"added": 1}], [{"added": 1}, {"added": 1}],
        ]),
        stage("graph.capture_staged", output="pending"),
        stage("graph.emit", source="pending", name="before_removal"),
    )


def capture(algorithm="bfs", **options):
    return builder(
        stage("graph.capture_staged", output="pending"),
        stage("graph.algorithm_table", source="pending", output="metrics",
              algorithm=algorithm, **options),
        stage("graph.emit", source="pending", name="final"),
        stage("table.emit", source="metrics", name="metrics"),
    )


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("manual", [False, True])
def test_multiple_builders_publish_one_exact_selection_with_history(
    directed, manual, tmp_path,
):
    graph = fixture(directed, manual)
    prior = graph.read_view()
    head = graph.current_state_id
    selected = graph.induced_subgraph([0, 2, 4])
    pipe = Pipeline([
        create(), connect(),
        builder(stage("graph.remove_nodes", ids=[2], effective_at=40),
                stage("graph.capture_staged", output="removed"),
                stage("graph.emit", source="removed", name="after_removal")),
        builder(stage("graph.set_edge_endpoints", revisions=[[0, 0, 4]],
                      effective_at=30)),
        capture(start=0),
    ])
    result, profile = pipe(selected, atomic=True, return_profile=True)
    assert graph.current_state_id == head + 1
    assert sorted(result.node_ids) == [0, 4, 7, 8]
    assert sorted(result.edge_ids) == [0, 3, 5, 6, 7, 8]
    assert graph.edge_endpoints(4) == (5, 6)
    outputs = profile["outputs"]
    before, after = outputs["before_removal"], outputs["final"]
    assert 0 not in outputs["after_removal"].edge_ids
    assert before.node_ids == [0, 2, 4, 7, 8]
    assert before.edge_endpoints(0) == prior.edge_endpoints(0) == (0, 2)
    assert after.edge_endpoints(0) == (0, 4)
    assert before.write_state_id is after.write_state_id is None
    assert after.node_attr(7, "score") == 1.0000000000000002
    table = outputs["metrics"]
    assert dict(zip(table.source_entity_ids, table.column("value").to_list())) == {
        0: 0, 4: 1, 7: 1, 8: 2,
    }
    assert table.source_views()[0].write_state_id is None
    assert graph.read_view_at(11).node_ids == [0, 2, 4, 5, 6, 7]
    assert graph.read_view_at(25).edge_endpoints(0) == (0, 2)
    assert graph.read_view_at(35).edge_endpoints(0) == (0, 4)
    assert 2 not in graph.read_view_at(45).node_ids
    path = tmp_path / "outer.archive"
    graph.save_archive(path)
    recovered = Graph.load_archive(path)
    assert recovered.current_state_id == head + 1
    assert recovered.read_view_at(25).edge_endpoints(0) == (0, 2)
    assert recovered.edge_endpoints(0) == (0, 4)
    graph.writer().set_node_attr(7, "score", -1)
    assert after.node_attr(7, "score") == 1.0000000000000002
    assert before.edge_endpoints(0) == (0, 2)


@pytest.mark.parametrize("algorithm, options", [
    ("bfs", {"start": 0}), ("dfs", {"start": 0}),
    ("dijkstra", {"start": 0}), ("astar", {"start": 0, "goal": 8}),
    ("pagerank", {}), ("betweenness", {}), ("closeness", {}),
    ("components", {}), ("louvain", {}), ("leiden", {"seed": 4}),
    ("infomap", {"seed": 4, "num_trials": 1}),
    ("girvan_newman", {}), ("lpa", {"seed": 4}),
])
def test_captured_kernels_read_earlier_outer_algorithm_topology(algorithm, options):
    graph = fixture()
    head = graph.current_state_id
    result, profile = Pipeline([create(), connect(), capture(algorithm, **options)])(
        graph.induced_subgraph([0, 2, 4]), atomic=True, return_profile=True,
    )
    assert graph.current_state_id == head + 1
    assert sorted(result.node_ids) == [0, 2, 4, 7, 8]
    rows = set(profile["outputs"]["metrics"].source_entity_ids)
    assert {7, 8} <= rows <= {0, 2, 4, 7, 8}


@pytest.mark.parametrize("failure", ["target", "timeline", "duplicate", "property"])
def test_late_failures_preserve_values_history_and_unallocated_ids(failure):
    graph = fixture()
    head = graph.current_state_id
    programs = [
        create(), connect(at=5 if failure == "timeline" else 20), capture(start=0),
    ]
    if failure == "target":
        programs.append(builder(stage("graph.remove_edges", ids=[999])))
    elif failure == "duplicate":
        programs += [builder(stage("graph.remove_edges", ids=[0]))] * 2
    elif failure == "property":
        # This older calculation reads owner topology and must reject at preflight.
        programs.append(builder(stage("drot.smooth_strokes")))
    with pytest.raises(RuntimeError):
        Pipeline(programs)(graph.view(), atomic=True)
    assert graph.current_state_id == head
    assert sorted(graph.node_ids) == [0, 2, 4, 5, 6]
    assert graph.edge_endpoints(0) == (0, 2)
    assert not graph.has_uncommitted_changes()
    assert graph.add_node() == 7
    assert graph.add_edge(0, 4) == 5


def test_persistence_preflight_and_empty_topology_leave_state_unchanged():
    graph = fixture()
    head = graph.current_state_id
    with pytest.raises(RuntimeError, match="persist=True"):
        Pipeline([create()])(graph.view(), atomic=True, persist=False)
    _, profile = Pipeline([
        builder(stage("graph.add_nodes", count=0, output="none")),
        builder(stage("graph.remove_edges", ids=[])),
        builder(stage("graph.capture_staged", output="empty"),
                stage("graph.emit", source="empty", name="empty")),
    ])(graph.view(), atomic=True, return_profile=True)
    assert profile["outputs"]["empty"].node_ids == [0, 2, 4, 5, 6]
    assert graph.current_state_id == head
    assert graph.add_node() == 7


def test_property_prefix_and_explicit_immutable_reads_retain_one_acceptance():
    graph = fixture()
    head = graph.current_state_id
    prefix = builder(stage("core.init_nodes", target="v", value=7),
                     stage("core.attach_node_attr", source="v", attr="prefix"))
    _, profile = Pipeline([prefix, create(), capture(start=0)])(
        graph.view(), atomic=True, return_profile=True,
    )
    assert graph.current_state_id == head + 1
    assert profile["outputs"]["final"].node_attr(0, "prefix") == 7
    assert graph.get_node_attr(7, "prefix") is None


def test_default_outer_execution_keeps_per_builder_publication():
    graph = fixture()
    head = graph.current_state_id
    result = Pipeline([create(), create(at=12)])(graph.view())
    assert graph.current_state_id == head + 2
    assert sorted(result.node_ids) == [0, 2, 4, 5, 6, 7, 8]
