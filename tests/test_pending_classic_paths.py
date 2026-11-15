"""Classic path handles share pending selected topology and one outer acceptance."""
import pytest
from groggy import Graph, Pipeline
from groggy.algorithms.base import RustAlgorithmHandle
from tests.test_pending_core_compute import attach, fixture, program, step, topology


def handle(kind, **params):
    return RustAlgorithmHandle("pathfinding." + kind,
        {"start_attr": "start", "output_attr": "result", **({"goal_attr": "goal"} if kind == "astar" else {}), **params})


def prepare(graph, provisional=False):
    if not provisional:
        graph.set_node_attr(0, "start", False)
    graph.set_node_attr(4, "goal", None)
    operations = topology()
    if provisional:
        operations[0]["params"]["attributes"] = {"start": True}
    operations += [step("core.init_edges", target="weights", value=2), attach("weights", "weight", True)]
    return operations


def expected(kind, directed, provisional):
    if kind == "astar":
        path = [7, 4] if provisional else ([0, 7, 4] if directed else [0, 4])
        return {node: order for order, node in enumerate(path)}
    if kind == "dfs":
        order = ([7, 4, 0] if directed else [7, 0, 4]) if provisional else ([0, 7, 4] if directed else [0, 4, 7])
        return {node: i for i, node in enumerate(order)}
    distances = ({7: 0, 4: 1, 0: 2} if directed else {7: 0, 0: 1, 4: 1}) if provisional else ({0: 0, 7: 1, 4: 2} if directed else {0: 0, 4: 1, 7: 1})
    return {node: value * (2 if kind == "dijkstra" else 1) for node, value in distances.items()}


@pytest.mark.parametrize("kind", ["bfs", "dfs", "dijkstra", "astar"])
@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("provisional", [False, True])
def test_classic_paths_read_pending_ids_endpoints_weights_and_feed_later_builders(kind, directed, provisional):
    graph = fixture(directed)
    operations = prepare(graph, provisional)
    head = graph.current_state_id
    prior = graph.read_view()
    suffix = program(step("core.load_node_attr", attr="result", default=-1, target="loaded"), attach("loaded", "copied"))
    path = handle(kind, **({"weight_attr": "weight"} if kind in ["dijkstra", "astar"] else {}))
    result = Pipeline([program(*operations), path, suffix])(graph.induced_subgraph([0, 2, 4]), atomic=True)
    assert sorted(result.node_ids) == [0, 4, 7]
    assert sorted(result.edge_ids) == [0, 3, 5, 6]
    values = expected(kind, directed, provisional)
    assert {node: graph.get_node_attr(node, "result") for node in values} == values
    for node in [0, 4, 7]:
        assert graph.get_node_attr(node, "copied") == values.get(node, -1)
        if node not in values:
            assert graph.get_node_attr(node, "result") is None
    assert graph.get_node_attr(5, "result") is None
    assert graph.current_state_id == head + 1
    assert prior.edge_endpoints(0) == (0, 2)
    assert prior.node_attr(0, "result") is None


@pytest.mark.parametrize("kind", ["bfs", "dfs", "dijkstra", "astar"])
def test_classic_path_failure_after_computation_discards_topology_values_and_ids(kind):
    graph = fixture(True)
    operations = prepare(graph)
    head = graph.current_state_id
    bad = program(step("core.init_nodes", target="bad", value=3), attach("bad", "entity_type"))
    with pytest.raises(RuntimeError):
        Pipeline([program(*operations), handle(kind), bad])(graph.induced_subgraph([0, 2, 4]), atomic=True)
    assert graph.current_state_id == head
    assert graph.edge_endpoints(0) == (0, 2)
    assert graph.get_node_attr(0, "result") is None
    assert graph.add_node() == 7


@pytest.mark.parametrize("kind,params,failure", [
    ("dijkstra", {"weight_attr": "weight"}, "negative_weight"),
    ("astar", {"weight_attr": "weight", "heuristic_attr": "heuristic"}, "negative_weight"),
    ("astar", {"weight_attr": "weight", "heuristic_attr": "heuristic"}, "bad_heuristic"),
])
def test_pending_weight_and_heuristic_validation_is_not_bypassed(kind, params, failure):
    graph = fixture(True)
    operations = prepare(graph)
    operations += [step("core.init_edges", target="bad", value=-1), attach("bad", "weight", True)] if failure == "negative_weight" else [step("core.init_nodes", target="bad", value=-1), attach("bad", "heuristic")]
    head = graph.current_state_id
    with pytest.raises(RuntimeError):
        Pipeline([program(*operations), handle(kind, **params)])(graph.induced_subgraph([0, 2, 4]), atomic=True)
    assert graph.current_state_id == head
    assert graph.edge_endpoints(0) == (0, 2)
    assert graph.get_node_attr(0, "result") is None
    assert graph.add_node() == 7


def test_multiple_classic_paths_refresh_staged_marker_and_result_columns():
    graph = fixture(True)
    operations = prepare(graph)
    head = graph.current_state_id
    pipe = Pipeline([program(*operations), handle("bfs"),
        program(step("core.load_node_attr", attr="result", target="distances"), attach("distances", "bfs_copy")),
        handle("dijkstra", weight_attr="weight"), handle("astar", weight_attr="weight"),
        program(step("core.load_node_attr", attr="result", target="final"), attach("final", "final_copy"))])
    pipe(graph.induced_subgraph([0, 2, 4]), atomic=True)
    assert [graph.get_node_attr(n, "bfs_copy") for n in [0, 7, 4]] == [0, 1, 2]
    assert [graph.get_node_attr(n, "final_copy") for n in [0, 7, 4]] == [0, 1, 2]
    assert graph.current_state_id == head + 1
