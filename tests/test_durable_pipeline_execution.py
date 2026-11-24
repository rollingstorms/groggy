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
                step("core.node_degree", target="unsupported"),
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

    thread = threading.Thread(target=peer)
    thread.start()
    assert ready.wait(timeout=5)
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
    try:
        sys.setswitchinterval(10.0)
        begin.set()
        result, profile = p.run_durable(store, return_profile=True)
        progressed = progress.is_set()
    finally:
        sys.setswitchinterval(interval)
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
