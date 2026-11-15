"""Selected structural/neighbor/path kernels read pending IDs and endpoints."""
import json
import statistics

import pytest
from groggy import Graph, Pipeline
from groggy.builder import step_pipeline


def stage(kind, **params):
    return {"id": kind, "params": params}


def builder(*steps):
    return step_pipeline("pending selected reads", list(steps), atomic=True)


def fixture(directed=False):
    graph = Graph(directed=directed)
    graph.add_nodes(7)
    graph.remove_nodes([1, 3])
    graph.add_edges([(0, 2), (2, 4), (2, 4), (4, 4), (5, 6)])
    return graph


def topology():
    return [
        stage("graph.add_nodes", count=1, output="created"),
        stage("graph.add_edges", edges=[[0, {"added": 0}], [{"added": 0}, 4]],
              output="edges", attributes={"weight": 2.5}),
        stage("graph.set_edge_endpoints", revisions=[[0, 4, 0]]),
        stage("graph.remove_nodes", ids=[2]),
    ]


def attach(source, attr="answer"):
    return stage("core.attach_node_attr", source=source, attr=attr)


def run(graph, operations, outer=False):
    selected = graph.induced_subgraph([0, 2, 4])
    if outer:
        return Pipeline([builder(*topology()), builder(*operations)])(
            selected, atomic=True, return_profile=True)
    return selected.apply(builder(*(topology() + operations)), return_profile=True)


@pytest.mark.parametrize("outer", [False, True])
@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("kernel,params", [
    ("core.node_degree", {}),
    ("core.weighted_degree", {"weight_attr": "weight"}),
    ("core.edge_weight_sum", {"weight_attr": "weight"}),
])
def test_structural_rows_use_pending_selection_and_preserve_incidence_rules(kernel, params, directed, outer):
    graph = fixture(directed)
    before = graph.read_view()
    head = graph.current_state_id
    result, _ = run(graph, [stage(kernel, target="result", **params), attach("result")], outer)
    assert sorted(result.node_ids) == [0, 4, 7]
    assert sorted(result.edge_ids) == [0, 3, 5, 6]
    expected = ([1, 2, 1] if directed else [2, 3, 2]) if kernel == "core.node_degree" else [3.5, 5.5, 5.0]
    assert [graph.get_node_attr(n, "answer") for n in [0, 4, 7]] == expected
    assert graph.get_node_attr(5, "answer") is None
    assert graph.current_state_id == head + 1
    assert before.edge_endpoints(0) == (0, 2)
    assert before.node_attr(0, "answer") is None


@pytest.mark.parametrize("kernel,params,expected", [
    ("core.triangle_count", {}, [1, 1, 1]),
    ("core.k_core_mark", {"k": 2}, [1, 1, 1]),
    ("core.k_core_mark", {"k": 3}, [0, 0, 0]),
])
def test_undirected_structural_kernels_include_added_cycle_and_keep_loop_policy(kernel, params, expected):
    graph = fixture()
    run(graph, [stage(kernel, target="result", **params), attach("result")])
    assert [graph.get_node_attr(n, "answer") for n in [0, 4, 7]] == expected


@pytest.mark.parametrize("direction,expected", [
    ("in", [4, 11, 0]),
    ("out", [7, 4, 4]),
    ("undirected", [11, 11, 4]),
])
@pytest.mark.parametrize("directed", [False, True])
def test_neighbor_aggregation_honors_explicit_direction_on_pending_endpoints(direction, expected, directed):
    graph = fixture(directed)
    run(graph, [
        stage("core.load_node_attr", attr="node_id", target="ids"),
        stage("core.neighbor_agg", source="ids", target="result", direction=direction, agg="sum"),
        attach("result"),
    ])
    assert [graph.get_node_attr(n, "answer") for n in [0, 4, 7]] == expected


@pytest.mark.parametrize("directed", [False, True])
def test_neighbor_collection_and_expression_use_the_same_selected_source(directed):
    graph = fixture(directed)
    expr = {"type": "call", "func": "sum", "args": [
        {"type": "call", "func": "neighbor_values", "args": [{"type": "var", "name": "ids"}]}
    ]}
    run(graph, [
        stage("core.load_node_attr", attr="node_id", target="ids"),
        stage("core.collect_neighbor_values", source="ids", target="neighbors", include_self=False),
        attach("neighbors", "neighbors"),
        stage("core.map_nodes", source="ids", target="mapped", expr=expr, async_update=False),
        stage("core.update_in_place", source="mapped", target="ids", output="updated"),
        attach("updated", "neighbor_sum"),
        stage("core.init_nodes", target="labels", value=9),
        stage("core.neighbor_mode_update", target="labels", output="labels_out", include_self=True),
        attach("labels_out", "label"),
    ])
    rows = [graph.get_node_attr(n, "neighbors") for n in [0, 4, 7]]
    rows = [json.loads(row) if isinstance(row, str) else row for row in rows]
    assert [sorted(row) for row in rows] == ([[7], [0, 4], [4]] if directed else [[4, 7], [0, 4, 7], [0, 4]])
    assert [graph.get_node_attr(n, "neighbor_sum") for n in [0, 4, 7]] == [sum(row) for row in rows]
    assert [graph.get_node_attr(n, "label") for n in [0, 4, 7]] == [9, 9, 9]


@pytest.mark.parametrize("kernel,params,expected", [
    ("core.reduce_nodes", {"reducer": "sum"}, 11),
    ("core.std", {}, pytest.approx(statistics.stdev([0, 4, 7]), rel=1e-14)),
    ("core.median", {}, 4),
    ("core.quantile", {"q": 0.5}, 4),
    ("core.mode", {}, 0),
    ("core.entropy", {}, pytest.approx(1.584962500721156)),
])
def test_explicit_variable_reductions_compose_after_topology(kernel, params, expected):
    graph = fixture()
    run(graph, [
        stage("core.load_node_attr", attr="node_id", target="ids"),
        stage(kernel, source="ids", target="value", **params),
        stage("core.broadcast_scalar", scalar="value", reference="ids", target="result"),
        attach("result"),
    ])
    assert graph.get_node_attr(7, "answer") == expected


@pytest.mark.parametrize("directed", [False, True])
def test_paths_and_staged_weights_include_provisional_nodes(directed):
    graph = fixture(directed)
    run(graph, [
        stage("core.init_edges", target="weights", value=2.0),
        stage("core.attach_edge_attr", source="weights", attr="weight"),
        stage("core.init_scalar", target="start", value=0),
        stage("core.shortest_path_map", source="start", output="distances", weight_attr="weight"),
        attach("distances"),
    ])
    assert [graph.get_node_attr(n, "answer") for n in [0, 4, 7]] == ([0, 4, 2] if directed else [0, 2, 2])


@pytest.mark.parametrize("kernel", ["core.k_shortest_paths", "core.random_walk"])
def test_serialized_paths_use_provisional_ids(kernel):
    graph = fixture(True)
    operation = stage(kernel, source="start", target="goal", output="paths", k=3, weight_attr="weight") if kernel == "core.k_shortest_paths" else stage(kernel, start_nodes="start", output="paths", length=2, seed=4)
    run(graph, [
        stage("core.init_scalar", target="start", value=0),
        stage("core.init_scalar", target="goal", value=4),
        operation,
        stage("core.init_nodes", target="nodes", value=0),
        stage("core.broadcast_scalar", scalar="paths", reference="nodes", target="result"),
        attach("result"),
    ])
    value = json.loads(graph.get_node_attr(7, "answer"))
    if kernel == "core.k_shortest_paths":
        assert value == [[[0, 7, 4], 5.0]]
    else:
        assert value == [[0, 7, 4]]


@pytest.mark.parametrize("failure", ["directed_structure", "negative_weight", "stale_source", "missing_variable"])
def test_late_reader_errors_discard_topology_properties_outputs_and_ids(failure):
    graph = fixture(True)
    head = graph.current_state_id
    operations = [stage("core.init_nodes", target="values", value=9), attach("values", "prefix")]
    if failure == "directed_structure":
        operations.append(stage("core.triangle_count", target="bad"))
    elif failure == "negative_weight":
        operations += [stage("core.init_edges", target="negative", value=-1.0),
                       stage("core.attach_edge_attr", source="negative", attr="weight"),
                       stage("core.init_scalar", target="start", value=0),
                       stage("core.shortest_path_map", source="start", output="bad", weight_attr="weight")]
    elif failure == "stale_source":
        operations += [stage("core.init_scalar", target="start", value=2),
                       stage("core.shortest_path_map", source="start", output="bad")]
    else:
        operations.append(stage("core.neighbor_agg", source="missing", target="bad", agg="sum"))
    with pytest.raises(RuntimeError):
        run(graph, operations, outer=True)
    assert graph.current_state_id == head
    assert graph.edge_endpoints(0) == (0, 2)
    assert graph.get_node_attr(0, "prefix") is None
    assert graph.add_node() == 7


def test_histogram_remains_an_owned_output_after_pending_topology():
    graph = fixture()
    head = graph.current_state_id
    _, profile = run(graph, [
        stage("core.load_node_attr", attr="node_id", target="ids"),
        stage("core.histogram", source="ids", target="counts", bins=3),
    ])
    assert profile["outputs"]["histograms.counts"] == [1, 1, 1]
    assert graph.current_state_id == head + 1
