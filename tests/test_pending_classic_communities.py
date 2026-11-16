"""Community registry handles read pending selection and retain their core semantics."""
import pytest

from groggy import Pipeline
from groggy.algorithms.base import RustAlgorithmHandle

from tests.test_pending_core_compute import attach, fixture, program, step, topology


CASES = [
    ("lpa", {"seed": 7}),
    ("lpa", {"seed": 7, "seed_attr": "seed_label"}),
    ("infomap", {"seed": 7, "num_trials": 2}),
    ("connected_components", {"mode": "undirected"}),
    ("connected_components", {"mode": "weak"}),
    ("connected_components", {"mode": "strong"}),
    ("louvain", {}),
    ("leiden", {}),
    ("girvan_newman", {"num_levels": 1}),
    ("girvan_newman", {"num_levels": 1, "weight_attr": "length"}),
]
UNDIRECTED = {"louvain", "leiden", "girvan_newman"}
KINDS = ["lpa", "infomap", "connected_components", *sorted(UNDIRECTED)]


def handle(kind, params):
    return RustAlgorithmHandle(
        "community." + kind, {"output_attr": "result", **params}
    )


def operations(parallel=False):
    edits = topology()
    if parallel:
        edits[1]["params"]["edges"].append([0, {"added": 0}])
    return edits + [
        step("core.init_nodes", target="seeds", value=42),
        attach("seeds", "seed_label"),
        step("core.init_edges", target="lengths", value=2),
        attach("lengths", "length", True),
    ]


@pytest.mark.parametrize("kind,params,directed", [
    (kind, params, directed)
    for kind, params in CASES
    for directed in ([False] if kind in UNDIRECTED else [False, True])
])
@pytest.mark.parametrize("parallel", [False, True])
def test_pending_community_matches_accepted_selected_computation(
    kind, params, directed, parallel
):
    graph = fixture(directed)
    reference = fixture(directed)
    selected = Pipeline([program(*operations(parallel))])(
        reference.induced_subgraph([0, 2, 4]), atomic=True
    )
    Pipeline([handle(kind, params)])(selected)
    old = graph.read_view()
    head = graph.current_state_id
    suffix = program(
        step("core.load_node_attr", attr="result", target="labels"),
        attach("labels", "copied"),
    )
    result = Pipeline([program(*operations(parallel)), handle(kind, params), suffix])(
        graph.induced_subgraph([0, 2, 4]), atomic=True
    )
    assert sorted(result.node_ids) == [0, 4, 7]
    expected_edges = [0, 3, 5, 6, 7] if parallel else [0, 3, 5, 6]
    assert sorted(result.edge_ids) == expected_edges
    for node in [0, 4, 7]:
        expected = reference.get_node_attr(node, "result")
        assert graph.get_node_attr(node, "result") == expected
        assert graph.get_node_attr(node, "copied") == expected
        if "seed_attr" in params:
            assert expected == 42
    assert graph.get_node_attr(5, "result") is None
    assert old.edge_endpoints(0) == (0, 2)
    assert old.node_attr(0, "result") is None
    assert graph.current_state_id == head + 1


@pytest.mark.parametrize("kind", KINDS)
def test_late_community_failure_releases_pending_ids_and_results(kind):
    graph = fixture(kind not in UNDIRECTED)
    head = graph.current_state_id
    invalid = program(
        step("core.init_nodes", target="bad", value=3),
        attach("bad", "entity_type"),
    )
    with pytest.raises(RuntimeError):
        Pipeline([program(*operations()), handle(kind, {}), invalid])(
            graph.induced_subgraph([0, 2, 4]), atomic=True
        )
    assert graph.current_state_id == head
    assert graph.edge_endpoints(0) == (0, 2)
    assert graph.get_node_attr(0, "result") is None
    assert graph.add_node() == 7


@pytest.mark.parametrize("kind", KINDS)
def test_empty_pending_community_selection_publishes_only_topology(kind):
    graph = fixture()
    head = graph.current_state_id
    result = Pipeline([
        program(step("graph.remove_nodes", ids=[0, 2, 4])), handle(kind, {})
    ])(graph.induced_subgraph([0, 2, 4]), atomic=True)
    assert list(result.node_ids) == []
    assert list(result.edge_ids) == []
    assert sorted(graph.node_ids) == [5, 6]
    assert graph.current_state_id == head + 1


@pytest.mark.parametrize("mode", ["undirected", "weak", "strong"])
def test_pending_component_modes_distinguish_one_way_routes(mode):
    graph = fixture(True)
    edits = topology()
    # 0 -> 7 -> 4, 0 -> 4, and the loop at 4: no return routes.
    edits[2]["params"]["revisions"] = [[0, 0, 4]]
    Pipeline([program(*edits), handle("connected_components", {"mode": mode})])(
        graph.induced_subgraph([0, 2, 4]), atomic=True
    )
    labels = [graph.get_node_attr(node, "result") for node in [0, 4, 7]]
    assert len(set(labels)) == (3 if mode == "strong" else 1)


@pytest.mark.parametrize("kind", sorted(UNDIRECTED))
def test_pending_community_rejects_unsupported_directed_graph_without_mutation(kind):
    graph = fixture(True)
    head = graph.current_state_id
    with pytest.raises(RuntimeError, match="undirected"):
        Pipeline([program(*operations()), handle(kind, {})])(
            graph.induced_subgraph([0, 2, 4]), atomic=True
        )
    assert graph.current_state_id == head
    assert graph.edge_endpoints(0) == (0, 2)
    assert graph.add_node() == 7


@pytest.mark.parametrize("length", [-1, 0])
def test_pending_girvan_newman_rejects_nonpositive_staged_lengths(length):
    graph = fixture()
    head = graph.current_state_id
    edits = operations() + [
        step("core.init_edges", target="bad", value=length),
        attach("bad", "length", True),
    ]
    with pytest.raises(RuntimeError):
        Pipeline([
            program(*edits), handle("girvan_newman", {"weight_attr": "length"})
        ])(graph.induced_subgraph([0, 2, 4]), atomic=True)
    assert graph.current_state_id == head
    assert graph.add_node() == 7
