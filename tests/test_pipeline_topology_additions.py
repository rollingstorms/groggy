"""One accepted creation plan, with immutable provisional reads and real kernels."""
import pytest
from groggy import Graph, GraphBatch, DurableGraph
from groggy.builder import step_pipeline


def stage(kind, **params):
    return {"id": kind, "params": params}


def creation():
    return [
        stage("graph.add_nodes", count=3, output="new_nodes", attributes={"score": 1.0000000000000002}, effective_at=10),
        stage("graph.add_edges", edges=[[0, {"added": 0}], [{"added": 0}, {"added": 1}],
                                        [{"added": 0}, {"added": 1}], [{"added": 1}, {"added": 1}]],
              output="new_edges", attributes={"weight": 2.5}, effective_at=20),
    ]


def run(graph, stages):
    return graph.apply(step_pipeline("staged creation", stages, atomic=True), return_profile=True)


@pytest.mark.parametrize("directed", [False, True])
def test_created_ids_are_visible_to_pending_reads_and_published_once(directed, tmp_path):
    graph = Graph(directed=directed)
    initial = GraphBatch(effective_at=0)
    initial.add_node()
    graph.writer().apply_batch(initial)
    before = graph.read_view()
    head = graph.current_state_id
    result, profile = run(graph, [stage("graph.capture_staged", output="before")] + creation() + [
        stage("graph.capture_staged", output="after"),
        stage("graph.degrees_table", source="after", output="degrees"),
        stage("table.emit", source="degrees", name="degrees"),
        stage("graph.emit", source="before", name="before"),
        stage("graph.emit", source="after", name="after"),
    ])
    pending = profile["outputs"]["after"]
    assert graph.current_state_id == head + 1
    assert sorted(result.nodes.ids) == [0, 1, 2, 3]
    assert pending.node_ids == [0, 1, 2, 3]
    assert pending.edge_ids == [0, 1, 2, 3]
    assert pending.node_attr(1, "entity_type") == "base"
    assert pending.node_attr(1, "score") == 1.0000000000000002
    assert pending.edge_attr(2, "weight") == 2.5
    assert pending.write_state_id is None
    assert before.node_ids == profile["outputs"]["before"].node_ids == [0]
    assert graph.read_view_at(9).node_ids == [0]
    assert graph.read_view_at(15).node_ids == [0, 1, 2, 3]
    assert graph.read_view_at(15).edge_ids == []
    assert graph.read_view_at(25).edge_ids == [0, 1, 2, 3]
    assert profile["outputs"]["degrees"].column("degree").to_list() == [1, 3, 3, 0]
    path = tmp_path / "graph.archive"
    graph.save_archive(path)
    reopened = Graph.load_archive(path)
    assert reopened.read_view_at(15).node_attr(1, "score") == 1.0000000000000002
    assert reopened.read_view_at(25).edge_endpoints(2) == (1, 2)


@pytest.mark.parametrize("algorithm, options", [
    ("bfs", {"start": 0}), ("dfs", {"start": 0}),
    ("dijkstra", {"start": 0, "weight_attr": "weight"}),
    ("astar", {"start": 0, "goal": 2, "weight_attr": "weight"}),
    ("pagerank", {}), ("betweenness", {}), ("closeness", {}),
    ("components", {}), ("louvain", {}), ("leiden", {"seed": 3}),
    ("infomap", {"seed": 3, "num_trials": 1}), ("girvan_newman", {}), ("lpa", {"seed": 3}),
])
def test_classic_kernels_accept_new_pending_topology_without_publishing_results(algorithm, options):
    graph = Graph()
    initial = GraphBatch(effective_at=0)
    initial.add_node()
    graph.writer().apply_batch(initial)
    head = graph.current_state_id
    _, profile = run(graph, creation() + [
        stage("graph.capture_staged", output="pending"),
        stage("graph.algorithm_table", source="pending", output="metrics", algorithm=algorithm, **options),
        stage("table.emit", source="metrics", name="metrics"),
    ])
    table = profile["outputs"]["metrics"]
    rows = table.source_entity_ids
    assert rows
    assert set(rows) <= {0, 1, 2, 3}
    assert graph.current_state_id == head + 1
    assert graph.get_node_attr(0, "value") is None
    assert table.source_views()[0].write_state_id is None
    values = dict(zip(rows, table.column("value").to_list()))
    if algorithm == "bfs":
        assert values == {0: 0, 1: 1, 2: 2}
    elif algorithm == "dijkstra":
        assert values[0] == 0 and values[1] == 2.5 and values[2] == 5.0
    elif algorithm == "astar":
        assert rows == [0, 1, 2]
        assert table.column("cost").to_list() == [5.0, 5.0, 5.0]
    elif algorithm == "components":
        assert values[0] == values[1] == values[2] != values[3]
    elif algorithm == "pagerank":
        assert sum(values.values()) == pytest.approx(1.0)
        assert "converged" in table.columns


@pytest.mark.parametrize("suffix", [
    [stage("graph.add_edges", edges=[[0, {"added": 99}]], output="bad")],
    [stage("graph.add_edges", edges=[[0, 99]], output="bad")],
    [stage("graph.add_edges", edges=[[0, {"added": 0}]], output="bad", effective_at=5)],
    [stage("graph.capture_staged", output="pending"), stage("graph.algorithm_table", source="pending", output="bad", algorithm="bfs", start=99)],
    [stage("graph.capture_staged", output="pending"), stage("graph.algorithm_table", source="pending", output="bad", algorithm="pagerank", damping=1.0)],
])
def test_late_errors_leave_graph_history_properties_and_id_allocation_unchanged(suffix):
    graph = Graph()
    graph.add_node()
    head = graph.current_state_id
    prefix = [stage("core.init_nodes", target="v", value=7),
              stage("core.attach_node_attr", source="v", attr="prefix"),
              stage("graph.add_nodes", count=2, output="nodes", effective_at=10)]
    with pytest.raises((ValueError, RuntimeError)):
        run(graph, prefix + suffix)
    assert graph.current_state_id == head
    assert graph.node_count() == 1 and graph.edge_count() == 0
    assert graph.get_node_attr(0, "prefix") is None
    assert graph.add_node() == 1


@pytest.mark.parametrize("count", [True, -1, 1.5, "3"])
def test_invalid_creation_count_rejects(count):
    graph = Graph()
    with pytest.raises((ValueError, RuntimeError)):
        run(graph, [stage("graph.add_nodes", count=count, output="nodes")])
    assert graph.node_count() == 0


def test_empty_creation_is_noop_and_empty_scope_can_create_a_graph():
    graph = Graph()
    head = graph.current_state_id
    run(graph, [stage("graph.add_nodes", count=0, output="nodes"),
                stage("graph.add_edges", edges=[], output="edges")])
    assert graph.current_state_id == head
    _, profile = run(graph, [stage("graph.add_nodes", count=1, output="nodes"),
                            stage("graph.capture_staged", output="pending"),
                            stage("graph.algorithm_table", source="pending", output="metrics", algorithm="bfs", start=0),
                            stage("table.emit", source="metrics", name="metrics")])
    assert graph.node_count() == 1
    assert profile["outputs"]["metrics"].column("value").to_list() == [0]


@pytest.mark.parametrize("attributes", [
    {"external_id": "unmanaged"}, {"labels": ["duplicate", "duplicate"]},
    {"relationship_type": "edge-only"},
])
def test_creation_property_policy_rejects_without_allocating_ids(attributes):
    graph = Graph()
    with pytest.raises((ValueError, RuntimeError)):
        run(graph, [stage("graph.add_nodes", count=2, output="nodes", attributes=attributes)])
    assert graph.node_count() == 0
    assert graph.add_node() == 0


def test_provisional_algorithm_output_cannot_masquerade_as_accepted_writeback_source():
    graph = Graph()
    graph.add_nodes(2)
    graph.add_edge(0, 1)
    _, profile = run(graph, [stage("graph.capture_staged", output="pending"),
                            stage("graph.algorithm_table", source="pending", output="values", algorithm="bfs", start=0),
                            stage("table.emit", source="values", name="values")])
    table = profile["outputs"]["values"]
    table["value"] = [10, 20]
    head = graph.current_state_id
    with pytest.raises((ValueError, RuntimeError)):
        table.update_source_graph(graph, ["value"])
    assert graph.current_state_id == head
    assert graph.get_node_attr(0, "value") is None


def test_multiple_creation_stages_use_global_added_positions_and_freeze_earlier_captures():
    graph = Graph()
    _, profile = run(graph, [
        stage("graph.add_nodes", count=1, output="first", attributes={"name": "first"}, effective_at=10),
        stage("graph.capture_staged", output="first_capture"),
        stage("graph.add_nodes", count=1, output="second", attributes={"name": "second"}, effective_at=20),
        stage("graph.add_edges", edges=[[{"added": 0}, {"added": 1}]], output="edges", effective_at=30),
        stage("graph.capture_staged", output="second_capture"),
        stage("graph.emit", source="first_capture", name="first"),
        stage("graph.emit", source="second_capture", name="second"),
    ])
    first, second = profile["outputs"]["first"], profile["outputs"]["second"]
    assert first.node_ids == [0] and first.edge_ids == []
    assert first.node_attr(0, "name") == "first"
    assert second.node_ids == [0, 1] and second.edge_endpoints(0) == (0, 1)
    assert second.node_attr(1, "name") == "second"
    assert graph.read_view_at(15).node_ids == [0]
    assert graph.read_view_at(25).node_ids == [0, 1]
    assert graph.read_view_at(35).edge_endpoints(0) == (0, 1)
