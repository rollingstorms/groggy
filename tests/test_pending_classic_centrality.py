"""Classic centrality handles compute against the pending selected graph."""
import pytest

from groggy import Pipeline
from groggy.algorithms.base import RustAlgorithmHandle

from tests.test_pending_core_compute import attach, fixture, program, step, topology


CASES = [
    ("pagerank", {}),
    ("pagerank", {"personalization_attr": "mass"}),
    ("closeness", {}),
    ("closeness", {"direction": "incoming"}),
    ("closeness", {"weight_attr": "weight", "harmonic": False}),
    ("closeness", {"weight_attr": "weight", "direction": "incoming"}),
    ("betweenness", {"normalized": False}),
    ("betweenness", {"weight_attr": "weight"}),
]


def handle(kind, params):
    return RustAlgorithmHandle(
        "centrality." + kind, {"output_attr": "result", **params}
    )


def operations(parallel=False):
    edits = topology()
    if parallel:
        edits[1]["params"]["edges"].append([0, {"added": 0}])
    return edits + [
        step("core.init_edges", target="weights", value=2),
        attach("weights", "weight", True),
        step("core.init_nodes", target="mass", value=3),
        attach("mass", "mass"),
    ]


@pytest.mark.parametrize("kind,params", CASES)
@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("parallel", [False, True])
def test_pending_centrality_matches_separately_accepted_selected_graph(
    kind, params, directed, parallel
):
    graph = fixture(directed)
    reference = fixture(directed)
    # Independent execution boundaries exercise the existing accepted-state path.
    selected_reference = Pipeline([program(*operations(parallel))])(
        reference.induced_subgraph([0, 2, 4]), atomic=True
    )
    Pipeline([handle(kind, params)])(selected_reference)
    head = graph.current_state_id
    old = graph.read_view()
    suffix = program(
        step("core.load_node_attr", attr="result", target="scores"),
        attach("scores", "copied"),
    )
    result = Pipeline([program(*operations(parallel)), handle(kind, params), suffix])(
        graph.induced_subgraph([0, 2, 4]), atomic=True
    )
    assert sorted(result.node_ids) == [0, 4, 7]
    expected_edges = [0, 3, 5, 6, 7] if parallel else [0, 3, 5, 6]
    assert sorted(result.edge_ids) == expected_edges
    for node in [0, 4, 7]:
        expected = reference.get_node_attr(node, "result")
        assert graph.get_node_attr(node, "result") == pytest.approx(expected)
        assert graph.get_node_attr(node, "copied") == pytest.approx(expected)
    assert graph.current_state_id == head + 1
    assert graph.get_node_attr(5, "result") is None
    assert old.edge_endpoints(0) == (0, 2)
    assert old.node_attr(0, "result") is None


@pytest.mark.parametrize("kind", ["pagerank", "closeness", "betweenness"])
def test_late_centrality_failure_discards_pending_graph_and_reservations(kind):
    graph = fixture(True)
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


@pytest.mark.parametrize("kind,params,attribute", [
    ("pagerank", {"personalization_attr": "mass"}, "mass"),
    ("closeness", {"weight_attr": "weight"}, "weight"),
    ("betweenness", {"weight_attr": "weight"}, "weight"),
])
def test_invalid_staged_centrality_values_reject_without_publication(kind, params, attribute):
    graph = fixture(True)
    head = graph.current_state_id
    edges = attribute == "weight"
    invalid = [
        step("core.init_edges" if edges else "core.init_nodes", target="bad", value=-2),
        attach("bad", attribute, edges),
    ]
    with pytest.raises(RuntimeError):
        Pipeline([program(*operations(), *invalid), handle(kind, params)])(
            graph.induced_subgraph([0, 2, 4]), atomic=True
        )
    assert graph.current_state_id == head
    assert graph.edge_endpoints(0) == (0, 2)
    assert graph.add_node() == 7


@pytest.mark.parametrize("kind", ["pagerank", "closeness", "betweenness"])
def test_pending_centrality_empty_selection_preserves_topology_publication(kind):
    graph = fixture()
    head = graph.current_state_id
    result = Pipeline([
        program(step("graph.remove_nodes", ids=[0, 2, 4])), handle(kind, {})
    ])(graph.induced_subgraph([0, 2, 4]), atomic=True)
    assert list(result.node_ids) == []
    assert list(result.edge_ids) == []
    assert graph.current_state_id == head + 1
    assert sorted(graph.node_ids) == [5, 6]


def test_later_centrality_refreshes_the_prior_staged_result_as_personalization():
    graph = fixture(True)
    reference = fixture(True)
    selected = Pipeline([program(*operations())])(
        reference.induced_subgraph([0, 2, 4]), atomic=True
    )
    first = handle("pagerank", {})
    second = handle("pagerank", {"personalization_attr": "result"})
    Pipeline([first])(selected)
    Pipeline([second])(reference.induced_subgraph([0, 4, 7]))
    head = graph.current_state_id
    Pipeline([program(*operations()), first, second])(
        graph.induced_subgraph([0, 2, 4]), atomic=True
    )
    for node in [0, 4, 7]:
        assert graph.get_node_attr(node, "result") == pytest.approx(
            reference.get_node_attr(node, "result")
        )
    assert graph.current_state_id == head + 1
