"""Terminal graph removals compose with checked property work in one publication."""
import pytest

from groggy import Graph, GraphBatch
from groggy.builder import step_pipeline


def stage(kind, **params):
    return {"id": kind, "params": params}


def fixture(directed=False):
    graph = Graph(directed=directed)
    batch = GraphBatch(effective_at=0)
    nodes = [batch.add_node() for _ in range(5)]
    batch.set_node_attr(nodes[3], "keep", 1)
    for source, target in [(0, 1), (1, 2), (1, 2), (2, 2), (3, 4), (1, 4)]:
        batch.add_edge(nodes[source], nodes[target])
    graph.apply_batch(batch, 0, "fixture", "test")
    return graph


def prefix():
    return [
        stage("core.filter_nodes_by_attr", attr="keep", target="survivors",
              predicate={"type": "eq", "value": {"type": "Int", "value": 1}}),
        stage("core.attach_node_attr", source="survivors", attr="marked"),
    ]


@pytest.mark.parametrize("directed", [False, True])
def test_mixed_removals_publish_once_preserve_history_and_return_surviving_exact_selection(directed):
    graph = fixture(directed)
    before = graph.read_view()
    head = graph.current_state_id
    handle = step_pipeline("mark and remove", prefix() + [
        stage("graph.remove_edges", ids=[4], effective_at=30),
        stage("graph.remove_nodes", ids=[1], effective_at=20),
    ], atomic=True)
    result = graph.apply(handle)
    assert graph.current_state_id == head + 1
    assert not graph.has_uncommitted_changes()
    assert sorted(result.node_ids) == [0, 2, 3, 4]
    assert sorted(result.edge_ids) == [3]
    assert graph.get_node_attr(3, "marked") == 1
    assert sorted(before.node_ids) == list(range(5))
    assert sorted(before.edge_ids) == list(range(6))
    assert before.node_attr(3, "marked") is None
    metadata = graph.publication_metadata(graph.current_state_id)
    times = {(row["kind"], row["entity_id"], row["name"]): row["effective_at"]
             for row in metadata["change_times"]}
    assert times[("node_removed", 1, None)] == 20
    assert times[("edge_removed", 4, None)] == 30
    assert all(times[("edge_removed", edge, None)] == 20 for edge in [0, 1, 2, 5])
    assert times[("node_attr", 3, "marked")] == metadata["recorded_at"]
    # The returned selection is itself frozen by the Python facade.
    graph.remove_node(2)
    assert sorted(result.node_ids) == [0, 2, 3, 4]
    assert sorted(result.edge_ids) == [3]


@pytest.mark.parametrize("suffix", [
    [stage("graph.remove_nodes", ids=[1]), stage("graph.remove_edges", ids=[999])],
    [stage("graph.remove_nodes", ids=[1]), stage("graph.remove_nodes", ids=[1])],
    [stage("graph.remove_nodes", ids=[1, 1])],
    [stage("graph.remove_nodes", ids=[3])],  # Property write to a removed target.
    [stage("graph.remove_edges", source="survivors")],  # Wrong domain.
    [stage("graph.remove_nodes", source="missing")],
])
def test_late_removal_errors_roll_back_all_properties_and_topology(suffix):
    graph = fixture()
    head = graph.current_state_id
    handle = step_pipeline("late failure", prefix() + suffix, atomic=True)
    with pytest.raises(RuntimeError):
        graph.apply(handle)
    assert graph.current_state_id == head
    assert graph.get_node_attr(3, "marked") is None
    assert sorted(graph.node_ids) == list(range(5))
    assert sorted(graph.edge_ids) == list(range(6))
    assert not graph.has_uncommitted_changes()


def test_map_keys_drive_removal_and_empty_suffix_publishes_nothing():
    graph = fixture()
    head = graph.current_state_id
    empty = step_pipeline("empty", [stage("graph.remove_nodes", ids=[]),
                                   stage("graph.remove_edges", ids=[])], atomic=True)
    graph.apply(empty)
    assert graph.current_state_id == head
    handle = step_pipeline("filtered deletion", prefix()[:1] + [
        stage("graph.remove_nodes", source="survivors")], atomic=True)
    graph.apply(handle)
    assert sorted(graph.node_ids) == [0, 1, 2, 4]
    assert sorted(graph.edge_ids) == [0, 1, 2, 3, 5]


@pytest.mark.parametrize("bad", [
    stage("graph.remove_nodes", ids=[True]),
    stage("graph.remove_nodes", ids=[-1]),
    stage("graph.remove_nodes", ids=[1.5]),
    stage("graph.remove_nodes", ids=[1], source="both"),
    stage("graph.remove_nodes", ids=[], effective_at=True),
    stage("graph.remove_nodes", ids=[], effective_at=1.5),
    stage("graph.remove_nodes", ids=[], ignored=True),
    stage("graph.remove_nodes", source=""),
])
def test_bad_controls_reject_before_prefix_work(bad):
    graph = fixture()
    head = graph.current_state_id
    with pytest.raises((RuntimeError, ValueError)):
        graph.apply(step_pipeline("bad", prefix() + [bad], atomic=True))
    assert graph.current_state_id == head
    assert graph.get_node_attr(3, "marked") is None


def test_order_scope_and_persistence_guards_reject_before_effects():
    graph = fixture()
    head = graph.current_state_id
    remove = stage("graph.remove_nodes", ids=[1])
    for handle, kwargs in [
        (step_pipeline("non atomic", prefix() + [remove]), {}),
        (step_pipeline("late read", [remove, stage("sample.iterate_nodes", target="late")], atomic=True), {}),
        (step_pipeline("transient", prefix() + [remove], atomic=True), {"persist": False}),
    ]:
        with pytest.raises(RuntimeError):
            graph.apply(handle, **kwargs)
    narrow = graph.view().filter_nodes("node_id == 0")
    with pytest.raises(RuntimeError, match="selection"):
        narrow.apply(step_pipeline("outside", [stage("graph.remove_nodes", ids=[2])], atomic=True))
    assert graph.current_state_id == head
    assert graph.get_node_attr(3, "marked") is None


def test_explicit_edge_removal_clock_overrides_implicit_node_cascade():
    graph = fixture()
    graph.apply(step_pipeline("explicit cascade", [
        stage("graph.remove_edges", ids=[1], effective_at=15),
        stage("graph.remove_nodes", ids=[1], effective_at=20),
    ], atomic=True))
    times = {(row["kind"], row["entity_id"]): row["effective_at"]
             for row in graph.publication_metadata(graph.current_state_id)["change_times"]}
    assert times[("edge_removed", 1)] == 15
    assert times[("edge_removed", 2)] == 20


def test_invalid_effective_removal_and_stale_sources_reject_without_publication():
    graph = fixture()
    head = graph.current_state_id
    with pytest.raises(RuntimeError):
        graph.apply(step_pipeline("invalid time", prefix() + [
            stage("graph.remove_edges", ids=[1], effective_at=25),
            stage("graph.remove_nodes", ids=[1], effective_at=20)], atomic=True))
    assert graph.current_state_id == head
    assert graph.get_node_attr(3, "marked") is None
    old = graph.view()
    graph.set_node_attr(0, "changed", 1)
    head = graph.current_state_id
    with pytest.raises((RuntimeError, ValueError)):
        old.apply(step_pipeline("stale", [stage("graph.remove_nodes", ids=[1])], atomic=True))
    assert graph.current_state_id == head
    assert sorted(graph.node_ids) == list(range(5))


def test_following_regular_pipeline_algorithm_receives_surviving_selection():
    from groggy import Pipeline

    graph = fixture()
    head = graph.current_state_id
    remove = step_pipeline("remove", [stage("graph.remove_nodes", ids=[1])], atomic=True)
    attach = step_pipeline("attach to survivors", [
        stage("core.init_nodes", target="values", value=7),
        stage("core.attach_node_attr", source="values", attr="after"),
    ], atomic=True)
    result = Pipeline([remove, attach])(graph.view())
    assert graph.current_state_id == head + 2
    assert sorted(result.node_ids) == [0, 2, 3, 4]
    for node in [0, 2, 3, 4]:
        assert graph.get_node_attr(node, "after") == 7
