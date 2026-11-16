"""Root topology loops stage sequential effects without per-iteration publication."""
import pytest

from groggy import Graph, GraphBatch, Pipeline
from groggy.algorithms.base import RustAlgorithmHandle

from tests.test_pending_core_compute import attach, fixture, program, step
from tests.test_pending_subgraph_frames import each, increment


def loop(body, iterations):
    return step("iter.loop", body=body, iterations=iterations)


def body():
    return [
        step("graph.add_nodes", count=1, output="new"),
        step("graph.add_edges", edges=[[0, {"added": 0}]], output="new_edges"),
        *increment(),
        step("graph.capture_staged", output="pending"),
        step("graph.emit", source="pending", name="last"),
    ]


@pytest.mark.parametrize("outer", [False, True])
@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("iterations", [0, 1, 3])
def test_root_creation_loops_refresh_membership_properties_and_capture(
    outer, directed, nested, iterations
):
    graph = fixture(directed)
    head = graph.current_state_id
    before = graph.read_view()
    count = iterations * (2 if nested else 1)
    operation = loop(body(), iterations)
    if nested:
        operation = loop([operation], 2)
    selected = graph.induced_subgraph([0, 2, 4])
    if outer:
        result, profile = Pipeline([program(operation)])(
            selected, atomic=True, return_profile=True
        )
    else:
        result, profile = selected.apply(program(operation), return_profile=True)
    expected_nodes = [0, 2, 4, *range(7, 7 + count)]
    assert sorted(result.node_ids) == expected_nodes
    assert graph.current_state_id == head + (1 if count else 0)
    for node in expected_nodes:
        expected = count if node < 7 else count + 7 - node
        assert graph.get_node_attr(node, "counter") == (expected if count else None)
        assert graph.get_node_attr(node, "seen") == (expected if count else None)
    for edge in range(5, 5 + count):
        assert graph.edge_endpoints(edge) == (0, 7)
    assert graph.get_node_attr(5, "counter") is None
    assert before.node_attr(0, "counter") is None
    assert 7 not in before.node_ids
    if count:
        assert sorted(profile["outputs"]["last"].node_ids) == expected_nodes
    else:
        assert "last" not in profile["outputs"]


def test_added_references_are_global_batch_positions_even_with_a_creation_prefix():
    graph = fixture(True)
    Pipeline([
        program(step("graph.add_nodes", count=1, output="prefix")),
        program(loop(body(), 3)),
        RustAlgorithmHandle("centrality.pagerank", {"output_attr": "rank"}),
    ])(graph.induced_subgraph([0, 2, 4]), atomic=True)
    for edge in [5, 6, 7]:
        assert graph.edge_endpoints(edge) == (0, 7)
    assert [graph.get_node_attr(node, "counter") for node in [7, 8, 9, 10]] == [3, 3, 2, 1]
    assert all(graph.get_node_attr(node, "rank") is not None for node in [0, 2, 4, 7, 8, 9, 10])


@pytest.mark.parametrize("failure", ["duplicate_remove", "duplicate_revision", "bad_property", "unknown_source"])
def test_root_loop_failure_discards_prior_topology_properties_outputs_and_ids(failure):
    graph = fixture()
    head = graph.current_state_id
    edits = body()
    if failure == "duplicate_remove":
        edits.append(step("graph.remove_nodes", ids=[2]))
    elif failure == "duplicate_revision":
        edits.append(step("graph.set_edge_endpoints", revisions=[[0, 4, 0]]))
    elif failure == "bad_property":
        edits.extend([step("core.init_nodes", value=3, target="bad"), attach("bad", "entity_type")])
    else:
        edits.append(step("core.load_node_attr", attr="missing", target="bad"))
        edits.append(attach("not_a_variable", "bad"))
    with pytest.raises(RuntimeError):
        graph.induced_subgraph([0, 2, 4]).apply(program(loop(edits, 2)))
    assert graph.current_state_id == head
    assert graph.edge_endpoints(0) == (0, 2)
    assert graph.get_node_attr(0, "counter") is None
    assert graph.add_node() == 7


@pytest.mark.parametrize("iterations", [0, 2])
def test_fixed_child_frames_reject_topology_loops_before_prefix(iterations):
    graph = fixture()
    head = graph.current_state_id
    with pytest.raises(RuntimeError):
        graph.induced_subgraph([0, 2, 4]).apply(program(
            step("graph.add_nodes", count=1, output="prefix"),
            step("sample.iterate_nodes", target="seeds"),
            each([loop(body(), iterations)]),
        ))
    assert graph.current_state_id == head
    assert graph.add_node() == 7


@pytest.mark.parametrize("iterations", [0, 2])
def test_topology_loops_require_atomic_persistent_execution_even_when_empty(iterations):
    graph = fixture()
    operation = loop(body(), iterations)
    with pytest.raises(RuntimeError, match="persist=True"):
        graph.apply(program(operation), persist=False)
    from groggy.builder import step_pipeline
    with pytest.raises(RuntimeError, match="atomic=True"):
        graph.apply(step_pipeline("non-atomic topology loop", [operation], atomic=False))


def test_root_loop_uses_one_explicit_property_time_and_survives_archive(tmp_path):
    graph = Graph()
    initial = GraphBatch(effective_at=0)
    initial.add_node()
    graph.writer().apply_batch(initial)
    head = graph.current_state_id
    edits = [
        step("graph.add_nodes", count=1, output="new", effective_at=10),
        step("core.load_node_attr", attr="counter", default=0, target="old"),
        step("core.init_scalar", value=1, target="one"),
        step("core.add", left="old", right="one", target="next"),
        attach("next", "counter"),
        step("graph.capture_staged", output="pending"),
        step("graph.nodes_table", source="pending", output="rows"),
        step("table.attach", source="rows", columns={"counter": "counter"}, effective_at=40),
    ]
    graph.apply(program(loop(edits, 3)))
    assert graph.current_state_id == head + 1
    assert [graph.get_node_attr(n, "counter") for n in [0, 1, 2, 3]] == [3, 3, 2, 1]
    assert graph.read_view_at(39).node_attr(3, "counter") is None
    assert graph.read_view_at(40).node_attr(3, "counter") == 1
    path = str(tmp_path / "root-loop.groggy")
    graph.save_archive(path)
    restored = Graph.load_archive(path)
    assert restored.current_state_id == graph.current_state_id
    assert [restored.get_node_attr(n, "counter") for n in [0, 1, 2, 3]] == [3, 3, 2, 1]


def test_new_nodes_enter_each_property_only_child_frame_on_later_iterations():
    graph = fixture()
    edits = [
        step("graph.add_nodes", count=1, output="new"),
        step("sample.iterate_nodes", target="seeds"),
        each(increment()),
    ]
    graph.induced_subgraph([0, 2, 4]).apply(program(loop(edits, 3)))
    assert [graph.get_node_attr(n, "counter") for n in [0, 2, 4, 7, 8, 9]] == [3, 3, 3, 3, 2, 1]
    assert graph.get_node_attr(5, "counter") is None


@pytest.mark.parametrize("edges", [False, True])
def test_each_iteration_can_remove_a_different_currently_selected_entity(edges):
    graph = fixture()
    head = graph.current_state_id
    if edges:
        edits = [step("core.sample_edges", count=1, seed=43, target="smallest")]
    else:
        edits = [
            step("core.load_node_attr", attr="node_id", target="ids"),
            step("core.top_k", source="ids", k=1, order="asc", target="smallest"),
        ]
    edits.append(step("graph.remove_edges" if edges else "graph.remove_nodes", source="smallest"))
    result = graph.induced_subgraph([0, 2, 4]).apply(program(loop(edits, 2)))
    assert sorted(result.node_ids) == ([0, 2, 4] if edges else [4])
    if edges:
        assert len(result.edge_ids) == 2
        assert set(result.edge_ids) <= {0, 1, 2, 3}
    else:
        assert sorted(result.edge_ids) == [3]
    assert graph.current_state_id == head + 1
    assert graph.edge_endpoints(4) == (5, 6)
