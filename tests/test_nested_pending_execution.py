"""Nested compute frames share pending reads without retaining stale loop pins."""
import pytest
from groggy import Graph, GraphBatch, Pipeline
from groggy.builder import step_pipeline


def stage(kind, **params):
    return {"id": kind, "params": params}


def builder(*steps, atomic=True):
    return step_pipeline("nested pending computation", list(steps), atomic=atomic)


def loop(body, iterations=3, **params):
    return stage("iter.loop", body=body, iterations=iterations, **params)


def fixture(directed=False):
    graph = Graph(directed=directed)
    initial = GraphBatch(effective_at=0)
    nodes = [initial.add_node() for _ in range(5)]
    for source, target in [(0, 2), (2, 4), (2, 4), (4, 4)]:
        initial.add_edge(nodes[source], nodes[target])
    graph.writer().apply_batch(initial)
    graph.remove_nodes([1, 3])
    return graph


def topology():
    return [stage("graph.add_nodes", count=1, output="new", effective_at=10),
            stage("graph.add_edges", edges=[[0, {"added": 0}], [{"added": 0}, 4]], output="edges", effective_at=20),
            stage("graph.set_edge_endpoints", revisions=[[0, 4, 0]], effective_at=25),
            stage("graph.remove_nodes", ids=[2], effective_at=30)]


def increment():
    return [stage("core.load_node_attr", attr="counter", target="counter", default=0),
            stage("core.init_scalar", target="one", value=1),
            stage("core.add", left="counter", right="one", target="next"),
            stage("core.attach_node_attr", source="next", attr="counter"),
            stage("core.load_node_attr", attr="counter", target="reload"),
            stage("core.attach_node_attr", source="reload", attr="mirror")]


@pytest.mark.parametrize("outer", [False, True])
@pytest.mark.parametrize("depth,expected", [(1, 3), (2, 6)])
def test_loop_children_reload_staged_properties_on_every_iteration(outer, depth, expected):
    graph = fixture()
    head = graph.current_state_id
    body = increment()
    operation = loop(body) if depth == 1 else loop([loop(body, 3)], 2)
    if outer:
        Pipeline([builder(*topology()), builder(operation)])(graph.view(), atomic=True)
    else:
        graph.apply(builder(*(topology() + [operation])))
    assert sorted(graph.node_ids) == [0, 4, 5]
    assert graph.current_state_id == head + 1
    for node in [0, 4, 5]:
        assert graph.get_node_attr(node, "counter") == expected
        assert graph.get_node_attr(node, "mirror") == expected


@pytest.mark.parametrize("alias_kind", ["graph", "table", "snapshot"])
def test_typed_loop_bindings_preserve_graph_table_and_snapshot_values(alias_kind):
    graph = fixture()
    head = graph.current_state_id
    if alias_kind == "snapshot":
        prefix = [stage("temporal.snapshot_at", state_id=head, output="seed")] + topology()
        body = [stage("graph.from_snapshot", source="local", output="old"),
                stage("graph.emit", source="old", name="result")]
    else:
        prefix = topology() + [stage("graph.capture_staged", output="seed_graph")]
        if alias_kind == "graph":
            prefix.append(stage("alias", source="seed_graph", target="seed"))
            body = [stage("graph.degrees_table", source="local", output="metrics"),
                    stage("table.attach", source="metrics", columns={"degree": "degree"}, effective_at=40)]
        else:
            prefix.append(stage("graph.degrees_table", source="seed_graph", output="seed"))
            body = [stage("table.attach", source="local", columns={"degree": "degree"}, effective_at=40)]
    _, profile = graph.apply(builder(*(prefix + [loop(body, 2, loop_vars=[["seed", "local"]])])), return_profile=True)
    if alias_kind == "snapshot":
        assert graph.current_state_id == head + 1
        assert sorted(profile["outputs"]["result"].node_ids) == [0, 2, 4]
    else:
        assert graph.current_state_id == head + 1
        assert [graph.get_node_attr(n, "degree") for n in [0, 4, 5]] == [2, 3, 2]
        assert graph.read_view_at(39).node_attr(5, "degree") is None
        assert graph.read_view_at(40).node_attr(5, "degree") == 2


@pytest.mark.parametrize("ordered,expected", [(False, [5, 4, 4]), (True, [5, 9, 9])])
@pytest.mark.parametrize("nested", [False, True])
def test_message_passing_uses_pending_endpoints_for_both_visibility_modes(ordered, expected, nested):
    graph = fixture(True)
    block = stage("core.execution_block", mode="message_pass", target="values",
                  options={"direction": "out", "ordered": ordered, "include_self": False},
                  body={"nodes": [{"id": "sum", "domain": "core", "op_type": "neighbor_agg",
                                   "inputs": ["values"], "output": "result", "metadata": {"agg": "sum"}}]})
    graph.apply(builder(*(topology() + [
        stage("core.load_node_attr", attr="node_id", target="values"),
        loop([block], 1) if nested else block,
        stage("core.attach_node_attr", source="values", attr="answer"),
    ])))
    assert [graph.get_node_attr(n, "answer") for n in [0, 4, 5]] == expected


@pytest.mark.parametrize("iterations", [0, 2])
@pytest.mark.parametrize("unsupported", ["topology", "sampling", "streaming", "batch_plan"])
def test_unsupported_nested_effects_reject_before_any_prefix(iterations, unsupported):
    graph = fixture()
    head = graph.current_state_id
    if unsupported == "topology":
        body = [stage("graph.add_nodes", count=1, output="extra")]
    elif unsupported == "sampling":
        body = [stage("sample.iterate_nodes", target="sample")]
    elif unsupported == "streaming":
        body = [stage("core.execution_block", mode="streaming", target="values", options={}, body={"nodes": []})]
    else:
        body = []
    operation = loop(body, iterations, **({"batch_plan": {}} if unsupported == "batch_plan" else {}))
    with pytest.raises(RuntimeError):
        graph.apply(builder(*(topology() + increment() + [operation])))
    assert graph.current_state_id == head
    assert graph.get_node_attr(0, "counter") is None
    assert graph.add_node() == 5


def test_late_error_inside_nested_loop_discards_every_prior_staged_effect():
    graph = fixture()
    head = graph.current_state_id
    body = increment() + [stage("graph.capture_staged", output="captured"),
                          stage("graph.emit", source="captured", name="captured"),
                          stage("graph.select_nodes", source="captured", ids=[999], output="bad")]
    with pytest.raises(RuntimeError):
        graph.apply(builder(*(topology() + [loop([loop(body, 2)], 2)])))
    assert graph.current_state_id == head
    assert sorted(graph.node_ids) == [0, 2, 4]
    assert graph.get_node_attr(0, "counter") is None
    assert graph.add_node() == 5


def test_nested_capture_and_attachment_require_persistence_even_without_topology():
    graph = fixture()
    head = graph.current_state_id
    body = [stage("graph.capture_staged", output="pending"),
            stage("graph.degrees_table", source="pending", output="table"),
            stage("table.attach", source="table", columns={"degree": "degree"})]
    with pytest.raises(RuntimeError, match="persist=True"):
        graph.apply(builder(loop(body)), persist=False)
    with pytest.raises(RuntimeError, match="atomic=True"):
        graph.apply(builder(loop(body), atomic=False))
    assert graph.current_state_id == head
