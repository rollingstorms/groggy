"""One checked publication through the existing Python builder and native kernels."""

import sys
import threading

import pytest

from groggy import DurableGraph, Graph, GraphBatch, Pipeline
from groggy.builder import step_pipeline


def step(kind, **params):
    return {"id": kind, "params": params}


def pipeline(steps):
    return Pipeline([step_pipeline("durable program", steps, atomic=True)])


def program():
    return [
        step("graph.capture_staged", output="before"),
        step(
            "graph.add_nodes",
            count=2,
            output="new",
            attributes={"x": 1.0000000000000002},
            effective_at=10,
        ),
        step(
            "graph.add_edges",
            edges=[
                [0, {"added": 0}],
                [{"added": 0}, {"added": 1}],
                [{"added": 0}, {"added": 1}],
                [{"added": 1}, {"added": 1}],
            ],
            output="edges",
            effective_at=20,
        ),
        step("core.init_nodes_with_index", target="index"),
        step("core.attach_node_attr", source="index", attr="ordinal", effective_at=25),
        step("graph.capture_staged", output="pending"),
        step(
            "graph.algorithm_table",
            source="pending",
            algorithm="bfs",
            start=0,
            output="distance",
        ),
        step("table.emit", source="distance", name="distance"),
        step("graph.emit", source="before", name="before"),
        step("graph.emit", source="pending", name="pending"),
    ]


def test_durable_pipeline_reuses_builder_preserves_exact_outputs_and_recovers(tmp_path):
    graph = Graph(directed=True)
    batch = GraphBatch(effective_at=0)
    batch.add_node()
    graph.writer().apply_batch(batch)
    path = tmp_path / "graph.journal"
    store = DurableGraph.create_from_graph(path, graph)
    retained = store.read_view()
    head = store.current_state_id
    p = pipeline(program())
    accepted, profile = p.run_durable(store, return_profile=True)
    assert profile["atomic"] and profile["persist_results"]
    assert p.last_profile() is profile
    assert accepted.write_state_id == head + 1 == store.current_state_id
    assert accepted.node_ids == [0, 1, 2]
    assert accepted.edge_ids == [0, 1, 2, 3]
    assert accepted.node_attr(1, "x") == 1.0000000000000002
    assert accepted.node_attr(2, "ordinal") == 2
    assert retained.node_ids == [0]
    assert profile["outputs"]["before"].node_ids == [0]
    pending = profile["outputs"]["pending"]
    assert pending.write_state_id is None
    assert profile["outputs"]["distance"].column("value").to_list() == [0, 1, 2]
    store.close()
    assert pending.node_ids == accepted.node_ids
    reopened = DurableGraph.open(path)
    assert reopened.current_state_id == head + 1
    assert reopened.read_view_at(9).node_ids == [0]
    assert reopened.read_view_at(15).node_ids == [0, 1, 2]
    assert reopened.read_view_at(15).edge_ids == []
    assert reopened.read_view_at(21).edge_endpoints(2) == (1, 2)
    assert reopened.read_view_at(24).node_attr(1, "ordinal") is None
    assert reopened.read_view_at(25).node_attr(1, "ordinal") == 1


def test_late_error_and_unsupported_suffix_consume_no_ids_or_journal_bytes(tmp_path):
    path = tmp_path / "graph.journal"
    store = DurableGraph.create(path)
    original = path.read_bytes()
    with pytest.raises(RuntimeError, match="no detached native kernel"):
        pipeline(
            [
                step("graph.add_nodes", count=1, output="new"),
                step(
                    "core.weighted_degree", target="unsupported", weight_attr="weight"
                ),
            ]
        ).run_durable(store)
    assert path.read_bytes() == original
    with pytest.raises(RuntimeError):
        pipeline(
            [
                step("graph.add_nodes", count=1, output="new"),
                step("graph.capture_staged", output="pending"),
                step(
                    "graph.algorithm_table",
                    source="pending",
                    algorithm="bfs",
                    start=999,
                    output="bad",
                ),
            ]
        ).run_durable(store)
    assert path.read_bytes() == original
    result = pipeline([step("graph.add_nodes", count=1, output="new")]).run_durable(
        store
    )
    assert result.node_ids == [0]


def test_selected_inputs_stale_foreign_and_closed_storage_are_checked(tmp_path):
    graph = Graph()
    graph.add_nodes(3)
    path = tmp_path / "graph.journal"
    store = DurableGraph.create_from_graph(path, graph)
    source = store.read_view().select_nodes([1])
    p = pipeline(
        [
            step("core.init_nodes", target="value", value=7),
            step("core.attach_node_attr", source="value", attr="marked"),
        ]
    )
    result = p.run_durable(store, source)
    assert result.node_ids == [1]
    assert store.read_view().node_attr(0, "marked") is None
    assert store.read_view().node_attr(1, "marked") == 7
    original = path.read_bytes()
    with pytest.raises(RuntimeError, match="[Ss]tale"):
        p.run_durable(store, source)
    with pytest.raises(RuntimeError, match="different graph"):
        p.run_durable(store, graph.read_view())
    assert path.read_bytes() == original
    store.close()
    with pytest.raises(ValueError, match="closed"):
        p.run_durable(store)


def test_detached_preparation_releases_gil_and_returns_only_immutable_outputs(tmp_path):
    # A Python peer must progress while native initialization/materialization is
    # running. All graph writes/acceptance remain on the creator thread.
    path = tmp_path / "graph.journal"
    store = DurableGraph.create(path)
    ready, begin, progress, done = (threading.Event() for _ in range(4))

    def peer():
        ready.set()
        begin.wait()
        progress.set()
        done.wait()

    p = pipeline(
        [
            step("graph.add_nodes", count=40_000, output="nodes"),
            step("core.init_nodes_with_index", target="index"),
            step("core.attach_node_attr", source="index", attr="ordinal"),
            step("graph.capture_staged", output="pending"),
            step("graph.emit", source="pending", name="pending"),
        ]
    )
    p._ensure_built()
    interval = sys.getswitchinterval()
    thread = threading.Thread(target=peer)
    thread.start()
    try:
        assert ready.wait(timeout=5)
        sys.setswitchinterval(10.0)
        begin.set()
        result, profile = p.run_durable(store, return_profile=True)
        progressed = progress.is_set()
    finally:
        sys.setswitchinterval(interval)
        # Release the peer even if readiness/setup fails before native entry.
        begin.set()
        done.set()
        thread.join(timeout=5)
    assert progressed, "Python peer did not progress during native preparation"
    assert not thread.is_alive()
    assert len(result.node_ids) == 40_000
    assert profile["outputs"]["pending"].write_state_id is None


def test_read_only_durable_computation_publishes_no_version_or_journal_record(tmp_path):
    graph = Graph()
    graph.add_nodes(2)
    graph.add_edge(0, 1)
    path = tmp_path / "graph.journal"
    store = DurableGraph.create_from_graph(path, graph)
    head = store.current_state_id
    original = path.read_bytes()
    p = pipeline(
        [
            step("graph.capture_staged", output="pending"),
            step(
                "graph.algorithm_table",
                source="pending",
                algorithm="bfs",
                start=0,
                output="answer",
            ),
            step("table.emit", source="answer", name="answer"),
        ]
    )
    for _ in range(2):
        result, profile = p.run_durable(store, return_profile=True)
        assert result.write_state_id == head == store.current_state_id
        assert profile["outputs"]["answer"].column("value").to_list() == [0, 1]
        assert path.read_bytes() == original


@pytest.mark.parametrize("directed", [False, True])
def test_durable_computed_columns_endpoint_revision_and_external_cascade(
    tmp_path, directed
):
    graph = Graph(directed=directed)
    batch = GraphBatch(effective_at=0)
    nodes = [batch.add_node() for _ in range(4)]
    batch.add_edge(nodes[0], nodes[1], weight=2.5)
    batch.add_edge(nodes[1], nodes[3])  # Outside the selected view below.
    batch.add_edge(nodes[2], nodes[2])
    graph.writer().apply_batch(batch)
    path = tmp_path / "writeback.journal"
    store = DurableGraph.create_from_graph(path, graph)
    selected = store.read_view().select_nodes([0, 1, 2])
    head = store.current_state_id
    result, profile = pipeline(
        [
            step("graph.add_nodes", count=1, output="new", effective_at=10),
            step(
                "graph.set_edge_endpoints",
                revisions=[[0, 0, {"added": 0}]],
                effective_at=20,
            ),
            step("graph.remove_edges", ids=[2], effective_at=15),
            step("graph.remove_nodes", ids=[1], effective_at=20),
            step("graph.capture_staged", output="pending"),
            step(
                "graph.algorithm_table",
                source="pending",
                algorithm="bfs",
                start=0,
                output="answer",
            ),
            step("table.select_rows", source="answer", rows=[1, 0], output="reordered"),
            step(
                "table.attach",
                source="reordered",
                columns={"value": "hops"},
                effective_at=30,
            ),
            step("graph.edges_table", source="pending", output="edges"),
            step(
                "table.attach",
                source="edges",
                columns={"weight": "copied_weight"},
                effective_at=35,
            ),
            step("graph.emit", source="pending", name="pending"),
        ]
    ).run_durable(store, selected, return_profile=True)
    assert result.node_ids == [0, 2, 4]
    assert result.edge_ids == [0]
    assert result.edge_endpoints(0) == (0, 4)
    assert result.node_attr(4, "hops") == 1
    assert result.edge_attr(0, "copied_weight") == 2.5
    assert store.current_state_id == head + 1
    assert store.read_view().node_ids == [0, 2, 3, 4]
    assert store.read_view().edge_ids == [0]
    assert selected.edge_endpoints(0) == (0, 1)
    assert profile["outputs"]["pending"].node_attr(4, "hops") is None
    store.close()
    reopened = DurableGraph.open(path)
    assert reopened.read_view_at(19).edge_endpoints(0) == (0, 1)
    assert reopened.read_view_at(20).edge_endpoints(0) == (0, 4)
    assert reopened.read_view_at(29).node_attr(4, "hops") is None
    assert reopened.read_view_at(30).node_attr(4, "hops") == 1
    assert reopened.read_view_at(34).edge_attr(0, "copied_weight") is None
    assert reopened.read_view_at(35).edge_attr(0, "copied_weight") == 2.5


@pytest.mark.parametrize(
    "suffix",
    [
        [
            step("graph.remove_nodes", ids=[0]),
            step("table.attach", source="table", columns={"score": "copied"}),
        ],
        [step("graph.remove_nodes", ids=[2])],
        [step("graph.set_edge_endpoints", revisions=[[0, 0, 1], [0, 1, 0]])],
    ],
)
def test_durable_late_writeback_and_topology_errors_preserve_entire_graph(
    tmp_path, suffix
):
    graph = Graph()
    graph.add_nodes([{"score": 7}, {"score": 9}])
    graph.add_edge(0, 1)
    path = tmp_path / "reject.journal"
    store = DurableGraph.create_from_graph(path, graph)
    original = path.read_bytes()
    head = store.current_state_id
    with pytest.raises(RuntimeError):
        pipeline(
            [
                step("graph.add_nodes", count=1, output="new"),
                step("graph.capture_staged", output="capture"),
                step("graph.nodes_table", source="capture", output="table"),
            ]
            + suffix
        ).run_durable(store)
    assert store.current_state_id == head
    assert path.read_bytes() == original
    result = pipeline([step("graph.add_nodes", count=1, output="new")]).run_durable(
        store
    )
    assert result.node_ids == [0, 1, 2]


@pytest.mark.parametrize(
    "algorithm,options,column",
    [
        ("bfs", {"start": 0}, "value"),
        ("dfs", {"start": 0}, "value"),
        ("dijkstra", {"start": 0}, "value"),
        ("astar", {"start": 0, "goal": 2}, "cost"),
        ("pagerank", {}, "value"),
        ("betweenness", {}, "value"),
        ("closeness", {}, "value"),
        ("components", {}, "value"),
        ("louvain", {}, "value"),
        ("leiden", {"seed": 7}, "value"),
        ("infomap", {"seed": 7, "num_trials": 1}, "value"),
        ("girvan_newman", {}, "value"),
        ("lpa", {"seed": 7}, "value"),
    ],
)
def test_every_captured_kernel_can_publish_row_bound_results_durably(
    tmp_path, algorithm, options, column
):
    graph = Graph()
    batch = GraphBatch(effective_at=0)
    a, b = batch.add_node(), batch.add_node()
    batch.add_edge(a, b)
    graph.writer().apply_batch(batch)
    path = tmp_path / "compute.journal"
    store = DurableGraph.create_from_graph(path, graph)
    result, profile = pipeline(
        [
            step("graph.add_nodes", count=1, output="new", effective_at=10),
            step(
                "graph.add_edges",
                edges=[[1, {"added": 0}]],
                output="edge",
                effective_at=20,
            ),
            step("graph.capture_staged", output="pending"),
            step(
                "graph.algorithm_table",
                source="pending",
                output="metric",
                algorithm=algorithm,
                **options,
            ),
            step(
                "table.attach",
                source="metric",
                columns={column: "computed"},
                effective_at=30,
            ),
            step("table.emit", source="metric", name="metric"),
        ]
    ).run_durable(store, return_profile=True)
    table = profile["outputs"]["metric"]
    expected = dict(zip(table.source_entity_ids, table.column(column).to_list()))
    assert 2 in expected
    for node, value in expected.items():
        assert result.node_attr(node, "computed") == value
        assert store.read_view_at(29).node_attr(node, "computed") is None
        assert table.source_views()[0].node_attr(node, "computed") is None
    store.close()
    reopened = DurableGraph.open(path)
    for node, value in expected.items():
        assert reopened.read_view_at(30).node_attr(node, "computed") == value


def test_durable_pipeline_checks_additional_schema_without_replacing_installed_policy(
    tmp_path,
):
    path = tmp_path / "schema.journal"
    store = DurableGraph.create(path)
    schema = {"node_labels": {"Entity": {"x": {"type": "text", "required": False}}}}
    original = path.read_bytes()
    with pytest.raises(RuntimeError):
        pipeline(
            [
                step(
                    "graph.add_nodes",
                    count=1,
                    output="new",
                    attributes={"labels": ["Entity"], "x": 1},
                )
            ]
        ).run_durable(store, schema=schema)
    assert path.read_bytes() == original
    result = pipeline(
        [
            step(
                "graph.add_nodes",
                count=1,
                output="new",
                attributes={"labels": ["Entity"], "x": "text"},
            )
        ]
    ).run_durable(store, schema=schema)
    assert result.node_ids == [0]
    original = path.read_bytes()
    with pytest.raises(RuntimeError):
        pipeline([]).run_durable(
            store,
            schema={
                "node_labels": {"Entity": {"x": {"type": "int", "required": False}}}
            },
        )
    assert path.read_bytes() == original
    store.close()
    reopened = DurableGraph.open(path)
    assert reopened.read_view().node_attr(0, "x") == "text"


def test_nested_ordinary_loops_refresh_pending_values_and_recover(tmp_path):
    path = tmp_path / "nested-loop.journal"
    graph = Graph(directed=True)
    initial = GraphBatch(effective_at=0)
    for _ in range(3):
        initial.add_node()
    graph.writer().apply_batch(initial)
    store = DurableGraph.create_from_graph(path, graph)
    original = store.read_view().select_nodes([0, 1])
    head = store.current_state_id
    increment = [
        step("graph.capture_staged", output="before"),
        step("core.load_node_attr", target="x", attr="x"),
        step("core.init_nodes", target="one", value=1),
        step("core.add", left="x", right="one", target="next"),
        step("core.attach_node_attr", source="next", attr="x", effective_at=20),
    ]
    p = pipeline(
        [
            step("core.init_nodes", target="zero", value=0),
            step("core.attach_node_attr", source="zero", attr="x", effective_at=10),
            step(
                "iter.loop",
                iterations=2,
                body=[
                    step(
                        "graph.add_nodes",
                        count=1,
                        output="new",
                        attributes={"x": 0},
                        effective_at=10,
                    ),
                    step("iter.loop", iterations=3, body=increment),
                ],
            ),
            step("graph.emit", source="before", name="before"),
        ]
    )
    result, profile = p.run_durable(store, original, return_profile=True)
    assert result.node_ids == [0, 1, 3, 4]
    assert [result.node_attr(i, "x") for i in result.node_ids] == [6, 6, 6, 3]
    assert store.current_state_id == head + 1
    assert store.read_view().node_attr(2, "x") is None
    before = profile["outputs"]["before"]
    assert [before.node_attr(i, "x") for i in before.node_ids] == [5, 5, 5, 2]
    assert before.write_state_id is None
    assert original.node_ids == [0, 1] and original.node_attr(0, "x") is None
    store.close()
    reopened = DurableGraph.open(path)
    assert reopened.current_state_id == head + 1
    # A batch retains the final write per key, at its final effective time.
    assert reopened.read_view_at(19).node_attr(0, "x") is None
    assert reopened.read_view_at(20).node_attr(0, "x") == 6
    assert before.node_attr(4, "x") == 2
    reopened.close()


@pytest.mark.parametrize("iterations", [0, 2])
def test_nested_ordinary_loop_rejects_unported_body_before_prefix(tmp_path, iterations):
    path = tmp_path / "unported-loop.journal"
    store = DurableGraph.create(path)
    original = path.read_bytes()
    p = pipeline(
        [
            step("graph.add_nodes", count=1, output="new"),
            step(
                "iter.loop",
                iterations=iterations,
                body=[
                    step(
                        "iter.loop",
                        iterations=0,
                        body=[
                            step(
                                "core.weighted_degree",
                                target="bad",
                                weight_attr="weight",
                            )
                        ],
                    )
                ],
            ),
        ]
    )
    with pytest.raises(RuntimeError, match="no detached native kernel"):
        p.run_durable(store)
    assert path.read_bytes() == original
    assert store.read_view().node_ids == []
    store.close()
