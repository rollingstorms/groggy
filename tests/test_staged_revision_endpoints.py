"""Existing edge identity can be rewired to explicit batch-global new nodes."""
import pytest

from groggy import Graph, GraphBatch, Pipeline
from groggy.builder import step_pipeline


def step(kind, **params):
    return {"id": kind, "params": params}


def program(steps):
    return step_pipeline("staged revision endpoints", steps, atomic=True)


def fixture(directed):
    graph = Graph(directed=directed)
    batch = GraphBatch(effective_at=0)
    for _ in range(4):
        batch.add_node()
    batch.add_edge(0, 1, weight=1.0000000001)
    batch.add_edge(1, 3)
    batch.add_edge(0, 1)
    batch.add_edge(1, 1)
    graph.writer().apply_batch(batch)
    graph.remove_node(2)
    return graph


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("outer", [False, True])
@pytest.mark.parametrize("remove_first", [False, True])
def test_explicit_new_endpoint_refs_preserve_identity_capture_clocks_and_cascade_rescue(
    directed, outer, remove_first
):
    graph = fixture(directed)
    head = graph.current_state_id
    before = graph.read_view()
    source = graph.induced_subgraph([0, 1])
    creation = [
        step("graph.add_nodes", count=1, output="first", effective_at=10),
        step("graph.add_nodes", count=1, output="second", effective_at=15),
        step("graph.add_edges", edges=[[0, {"added": 1}]], output="new_edge", effective_at=20),
    ]
    revision = step(
        "graph.set_edge_endpoints",
        revisions=[[0, {"added": 0}, {"added": 1}]],
        effective_at=20,
    )
    removal = step("graph.remove_nodes", ids=[1], effective_at=30)
    edits = ([removal, revision] if remove_first else [revision, removal]) + [
        step("graph.capture_staged", output="pending"),
        step("graph.edges_table", source="pending", output="edge_table"),
        step("table.emit", source="edge_table", name="edge_table"),
        step("graph.emit", source="pending", name="pending"),
    ]
    if outer:
        result, profile = Pipeline([program(creation), program(edits)])(
            source, atomic=True, return_profile=True
        )
    else:
        result, profile = Pipeline([program(creation + edits)])(source, return_profile=True)
    assert graph.current_state_id == head + 1
    assert sorted(result.node_ids) == [0, 4, 5]
    assert sorted(result.edge_ids) == [0, 4]
    captured = profile["outputs"]["pending"]
    assert captured.edge_endpoints(0) == (4, 5)
    assert captured.edge_endpoints(4) == (0, 5)
    assert captured.edge_attr(0, "weight") == 1.0000000001
    assert profile["outputs"]["edge_table"].source_entity_ids == [0, 4]
    assert before.edge_endpoints(0) == (0, 1)
    assert graph.read_view_at(19).edge_endpoints(0) == (0, 1)
    assert graph.read_view_at(20).edge_endpoints(0) == (4, 5)
    assert 1 not in graph.read_view().node_ids
    assert 1 not in graph.read_view().edge_ids
    assert 3 in graph.read_view().node_ids
    assert graph.add_node() == 6
    assert captured.node_ids == [0, 4, 5]


@pytest.mark.parametrize("failure", [
    "out_of_range", "no_prior_creation", "foreign_node", "foreign_edge",
    "before_new_lifetime", "duplicate_revision", "numeric_provisional",
])
@pytest.mark.parametrize("outer", [False, True])
def test_invalid_new_endpoint_references_roll_back_the_entire_run(failure, outer):
    graph = fixture(True)
    head = graph.current_state_id
    source = graph.induced_subgraph([0, 1])
    creation = [step("graph.add_nodes", count=1, output="new", effective_at=10)]
    row = [0, 0, {"added": 0}]
    at = 20
    if failure == "out_of_range":
        row[2] = {"added": 1}
    elif failure == "foreign_node":
        row[2] = 3
    elif failure == "foreign_edge":
        row[0] = 1
    elif failure == "before_new_lifetime":
        at = 5
    elif failure == "numeric_provisional":
        row[2] = 4
    revisions = [row, row] if failure == "duplicate_revision" else [row]
    edits = [step("graph.set_edge_endpoints", revisions=revisions, effective_at=at)]
    if failure == "no_prior_creation":
        creation, edits = edits, creation
    with pytest.raises((ValueError, RuntimeError)):
        if outer:
            Pipeline([program(creation), program(edits)])(source, atomic=True)
        else:
            Pipeline([program(creation + edits)])(source)
    assert graph.current_state_id == head
    assert graph.read_view().edge_endpoints(0) == (0, 1)
    assert sorted(graph.node_ids) == [0, 1, 3]
    assert graph.add_node() == 4
    assert graph.add_edge(0, 3) == 4


@pytest.mark.parametrize("reference", [
    {"added": True}, {"added": -1}, {"added": 0.5}, {"added": 0, "extra": 1},
    {"added": "0"}, {}, {"node": 0}, True, -1, 0.5,
])
def test_malformed_new_node_references_reject_even_in_zero_iteration_loops(reference):
    graph = fixture(False)
    head = graph.current_state_id
    body = [step("graph.set_edge_endpoints", revisions=[[0, 0, reference]])]
    with pytest.raises((ValueError, RuntimeError)):
        graph.apply(program([
            step("graph.add_nodes", count=1, output="prefix"),
            step("iter.loop", iterations=0, body=body),
        ]))
    assert graph.current_state_id == head
    assert graph.add_node() == 4
