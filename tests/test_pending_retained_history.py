"""Retained snapshot reads stay accepted-history inputs inside pending programs."""
import pytest
from groggy import Graph, Pipeline
from tests.test_pending_core_compute import attach, fixture, program, step, topology


def snapshots(before, after):
    return [step("temporal.snapshot_at", state_id=before, output="before"),
            step("temporal.snapshot_at", state_id=after, output="after")]


def diffs():
    return [step("temporal.diff_nodes", before="before", after="after", output_prefix="diff"),
            step("temporal.diff_edges", before="before", after="after", output_prefix="diff")]


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("outer", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_retained_reads_ignore_pending_topology_and_values(directed, outer, nested):
    graph = fixture(directed)
    before = graph.current_state_id
    graph.set_node_attr(0, "marker", 10)
    graph.set_edge_attr(0, "marker", 20)
    head = graph.current_state_id
    old_nodes = sorted(graph.node_ids)
    selected = graph.induced_subgraph([0, 2, 4])
    pending = topology() + [step("core.init_nodes", target="values", value=99), attach("values", "marker"),
                            step("core.init_edges", target="weights", value=99), attach("weights", "marker", True)]
    operations = snapshots(before, head) + diffs() + [
        attach("diff_nodes_modified", "recorded_change"), attach("diff_edges_modified", "recorded_change", True),
        attach("diff_nodes_added", "recorded_addition"),
        step("graph.from_snapshot", source="after", output="history"),
        step("graph.emit", source="history", name="history"),
        step("graph.capture_staged", output="working"),
        step("graph.emit", source="working", name="working"),
    ]
    if nested:
        operations = [step("iter.loop", iterations=2, body=operations)]
    if outer:
        _, profile = Pipeline([program(*pending), program(*operations)])(selected, atomic=True, return_profile=True)
    else:
        _, profile = selected.apply(program(*(pending + operations)), return_profile=True)
    history = profile["outputs"]["history"]
    working = profile["outputs"]["working"]
    assert sorted(history.node_ids) == old_nodes
    assert history.edge_endpoints(0) == (0, 2)
    assert history.node_attr(0, "marker") == 10
    assert history.edge_attr(0, "marker") == 20
    assert sorted(working.node_ids) == [0, 4, 7]
    assert working.edge_endpoints(0) == (4, 0)
    assert working.node_attr(0, "marker") == 99
    assert working.edge_attr(0, "marker") == 99
    assert graph.get_node_attr(0, "recorded_change") == 1
    assert graph.get_edge_attr(0, "recorded_change") == 1
    assert graph.get_node_attr(7, "recorded_addition") is None
    assert graph.current_state_id == head + 1
    graph.set_node_attr(0, "marker", 100)
    assert history.node_attr(0, "marker") == 10
    assert working.node_attr(0, "marker") == 99


@pytest.mark.parametrize("outer", [False, True])
@pytest.mark.parametrize("failure", ["future_state", "missing_snapshot", "removed_target", "outside_target"])
def test_late_retained_history_errors_discard_pending_ids_values_and_endpoints(outer, failure):
    graph = fixture(True)
    before = graph.current_state_id
    graph.set_node_attr(5 if failure == "outside_target" else 2, "marker", 7)
    head = graph.current_state_id
    pending = topology() + [step("core.init_nodes", target="prefix", value=9), attach("prefix", "prefix")]
    if failure == "future_state":
        operations = snapshots(before, head + 1)
    else:
        operations = snapshots(before, head) + [step("temporal.diff_nodes", before="missing" if failure == "missing_snapshot" else "before", after="after", output_prefix="diff")]
        if failure in ["removed_target", "outside_target"]:
            operations.append(attach("diff_nodes_modified", "invalid"))
    with pytest.raises(RuntimeError):
        if outer:
            Pipeline([program(*pending), program(*operations)])(graph.induced_subgraph([0, 2, 4]), atomic=True)
        else:
            graph.induced_subgraph([0, 2, 4]).apply(program(*(pending + operations)))
    assert graph.current_state_id == head
    assert graph.edge_endpoints(0) == (0, 2)
    assert graph.get_node_attr(0, "prefix") is None
    assert graph.has_node(2)
    assert graph.add_node() == 7


def test_pending_program_can_read_sibling_state_ids_without_switching_branch(tmp_path):
    graph = Graph()
    graph.add_nodes(2)
    base = graph.current_state_id
    graph.create_branch("side")
    graph.checkout_branch("side")
    graph.set_node_attr(0, "score", 7)
    sibling = graph.current_state_id
    graph.checkout_branch("main")
    assert graph.current_state_id == base
    _, profile = graph.apply(program(
        step("graph.add_nodes", count=1, output="new"),
        *snapshots(base, sibling), *diffs(),
        attach("diff_nodes_modified", "from_sibling"),
        step("graph.from_snapshot", source="after", output="retained"),
        step("graph.emit", source="retained", name="sibling"),
    ), return_profile=True)
    assert [branch.name for branch in graph.branches() if branch.is_current] == ["main"]
    assert graph.get_node_attr(0, "score") is None
    assert graph.get_node_attr(0, "from_sibling") == 1
    assert profile["outputs"]["sibling"].node_attr(0, "score") == 7
    assert sorted(profile["outputs"]["sibling"].node_ids) == [0, 1]
    path = str(tmp_path / "pending-retained.groggy")
    graph.save_archive(path)
    recovered = Graph.load_archive(path)
    assert [branch.name for branch in recovered.branches() if branch.is_current] == ["main"]
    assert recovered.get_node_attr(0, "from_sibling") == 1
    assert recovered.snapshot_at_commit(sibling).node_attr(0, "score").value == 7
    assert sorted(recovered.node_ids) == [0, 1, 2]
