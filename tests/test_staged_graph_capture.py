"""Pending working-state captures are explicit immutable values with no write pin."""
from concurrent.futures import ThreadPoolExecutor

import pytest

from groggy import Graph
from groggy.builder import step_pipeline


def stage(kind, **params):
    return {"id": kind, "params": params}


def emit(source):
    return stage("graph.emit", source=source, name=source)


def test_captures_freeze_each_pending_phase_and_tables_see_surviving_staged_properties():
    graph = Graph(directed=True)
    graph.add_nodes([{}, {}, {}, {"keep": 1}, {}])
    graph.add_edges([(0, 1), (1, 2), (1, 2), (2, 2), (3, 4), (1, 4)])
    head = graph.current_state_id
    program = [
        stage("graph.capture_staged", output="before"),
        stage("core.filter_nodes_by_attr", attr="keep", target="survivors",
              predicate={"type": "eq", "value": {"type": "Int", "value": 1}}),
        stage("core.attach_node_attr", source="survivors", attr="marked"),
        stage("graph.capture_staged", output="props"),
        stage("graph.remove_edges", ids=[4]),
        stage("graph.capture_staged", output="edges_removed"),
        stage("graph.remove_nodes", ids=[1]),
        stage("graph.capture_staged", output="after"),
        stage("alias", source="after", target="alias"),
        stage("graph.nodes_table", source="alias", output="nodes"),
        stage("graph.edges_table", source="after", output="edges"),
        stage("table.select_columns", source="nodes", columns=["marked"], output="marks"),
        stage("table.emit", source="marks", name="marks"),
        stage("table.emit", source="edges", name="edges"),
    ] + [emit(name) for name in ["before", "props", "edges_removed", "after"]]
    result, profile = graph.apply(step_pipeline("pending phases", program, atomic=True), return_profile=True)
    out = profile["outputs"]
    assert graph.current_state_id == head + 1
    assert sorted(out["before"].node_ids) == list(range(5))
    assert sorted(out["before"].edge_ids) == list(range(6))
    assert out["before"].node_attr(3, "marked") is None
    assert out["props"].node_attr(3, "marked") == 1
    assert sorted(out["props"].edge_ids) == list(range(6))
    assert sorted(out["edges_removed"].edge_ids) == [0, 1, 2, 3, 5]
    assert sorted(out["after"].node_ids) == [0, 2, 3, 4]
    assert sorted(out["after"].edge_ids) == [3]
    assert out["marks"].source_entity_ids == [0, 2, 3, 4]
    assert out["marks"].column("marked").to_list() == [None, None, 1, None]
    assert out["edges"].source_entity_ids == [3]
    assert sorted(result.node_ids) == [0, 2, 3, 4]
    assert all(out[name].write_state_id is None for name in ["before", "props", "edges_removed", "after"])
    graph.remove_node(2)
    del graph, result
    with ThreadPoolExecutor(max_workers=1) as workers:
        assert workers.submit(lambda: out["after"].edge_endpoints(3)).result() == (2, 2)
    out["marks"].set_column("marked", [9, 9, 9, 9])
    assert out["after"].node_attr(3, "marked") == 1


@pytest.mark.parametrize("value", [None, 9007199254740993, 1.0000000001, True, "hello"])
def test_staged_values_preserve_types_and_earlier_captures_exclude_later_property_writes(value):
    graph = Graph()
    graph.add_nodes(2)
    graph.add_edge(0, 1)
    head = graph.current_state_id
    handle = step_pipeline("typed pending values", [
        stage("graph.capture_staged", output="before"),
        stage("core.init_nodes", target="values", value=value),
        stage("core.attach_node_attr", source="values", attr="x"),
        stage("graph.capture_staged", output="properties"),
        stage("graph.remove_edges", ids=[0]),
        stage("graph.capture_staged", output="after"),
        stage("graph.nodes_table", source="after", output="nodes"),
        stage("table.select_columns", source="nodes", columns=["x"], output="values_table"),
        stage("table.emit", source="values_table", name="values_table"),
        emit("before"), emit("properties"), emit("after"),
    ], atomic=True)
    _, profile = graph.apply(handle, return_profile=True)
    out = profile["outputs"]
    assert out["before"].node_attr(0, "x") is None
    for name in ["properties", "after"]:
        assert out[name].node_attr(0, "x") == value
        assert type(out[name].node_attr(0, "x")) is type(value)
    assert sorted(out["properties"].edge_ids) == [0]
    assert sorted(out["after"].edge_ids) == []
    assert out["values_table"].column("x").to_list() == [value, value]
    assert graph.current_state_id == head + 1


@pytest.mark.parametrize("late", [
    stage("graph.select_nodes", source="pending", ids=[1], output="bad"),
    stage("table.select_rows", source="nodes", rows=[99], output="bad"),
    stage("table.select_columns", source="nodes", columns=["missing"], output="bad"),
])
def test_late_read_failures_discard_pending_topology(late):
    graph = Graph()
    graph.add_nodes(3)
    head = graph.current_state_id
    handle = step_pipeline("late projection error", [
        stage("graph.remove_nodes", ids=[1]),
        stage("graph.capture_staged", output="pending"),
        emit("pending"),
        stage("graph.nodes_table", source="pending", output="nodes"), late,
    ], atomic=True)
    with pytest.raises(RuntimeError):
        graph.apply(handle)
    assert graph.current_state_id == head
    assert sorted(graph.node_ids) == [0, 1, 2]
    assert not graph.has_uncommitted_changes()


def test_capture_only_publishes_nothing_and_empty_selected_graph_keeps_table_shapes():
    for count in [0, 3]:
        graph = Graph()
        graph.add_nodes(count)
        head = graph.current_state_id
        _, profile = graph.apply(step_pipeline("read pending", [
            stage("graph.capture_staged", output="pending"),
            stage("graph.nodes_table", source="pending", output="nodes"),
            stage("table.emit", source="nodes", name="nodes"), emit("pending"),
        ], atomic=True), return_profile=True)
        assert graph.current_state_id == head
        assert len(profile["outputs"]["pending"].node_ids) == count
        assert profile["outputs"]["nodes"].shape[0] == count


def test_new_capture_controls_do_not_enable_unsupported_execution_modes_or_reads():
    graph = Graph()
    graph.add_node()
    head = graph.current_state_id
    capture = stage("graph.capture_staged", output="pending")
    cases = [
        (step_pipeline("non atomic", [capture]), {}),
        (step_pipeline("transient", [capture], atomic=True), {"persist": False}),
        (step_pipeline("late property read", [stage("graph.remove_nodes", ids=[]), capture,
             stage("core.init_nodes", target="old", value=1)], atomic=True), {}),
        (step_pipeline("unknown control", [stage("graph.capture_staged", output="x", live=True)], atomic=True), {}),
    ]
    for handle, kwargs in cases:
        with pytest.raises((RuntimeError, ValueError)):
            graph.apply(handle, **kwargs)
    assert graph.current_state_id == head


def test_staged_capture_retains_meta_group_sources_after_owner_changes():
    graph = Graph()
    a, b, outside = graph.add_nodes([{"x": 2}, {"x": 4}, {}])
    internal = graph.add_edge(a, b)
    graph.add_edge(b, outside)
    meta = graph.induced_subgraph([a, b]).collapse(node_strategy="extract")
    meta_id = meta.id
    _, profile = graph.apply(step_pipeline("retain meta", [
        stage("graph.capture_staged", output="pending"), emit("pending"),
    ], atomic=True), return_profile=True)
    view = profile["outputs"]["pending"]
    graph.remove_node(a)
    del graph, meta
    group = view.meta_subgraph(meta_id)
    assert sorted(group.node_ids) == [a, b]
    assert sorted(group.edge_ids) == [internal]
    assert group.node_attr(a, "x") == 2
    with ThreadPoolExecutor(max_workers=1) as workers:
        assert workers.submit(lambda: view.meta_subgraph(meta_id).node_attr(a, "x")).result() == 2
