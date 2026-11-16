"""Direct bulk attachment clocks use the canonical atomic property boundary."""
import pytest

from groggy import Graph, GraphBatch, Pipeline
from groggy.builder import step_pipeline

from tests.test_pending_core_compute import program, step
from tests.test_pending_subgraph_frames import each


def fixture():
    graph = Graph()
    initial = GraphBatch(effective_at=0)
    nodes = [initial.add_node() for _ in range(3)]
    initial.add_edge(nodes[0], nodes[1])
    initial.add_edge(nodes[1], nodes[2])
    graph.writer().apply_batch(initial)
    return graph


def assignment(kind, value=9, at=40):
    nodes = kind != "edges"
    load = (step("core.community_seed", strategy="singleton", target="values")
            if kind == "column" else
            step("core.init_nodes" if nodes else "core.init_edges", target="values", value=value))
    params = {"source": "values", "attr": "timed"}
    if at is not None:
        params["effective_at"] = at
    return [load, step("core.attach_node_attr" if nodes else "core.attach_edge_attr", **params)]


@pytest.mark.parametrize("kind", ["nodes", "column", "edges"])
@pytest.mark.parametrize("pending", [False, True])
@pytest.mark.parametrize("outer", [False, True])
def test_direct_attachment_clocks_cover_maps_columns_and_provisional_rows(
    kind, pending, outer, tmp_path
):
    graph = fixture()
    head = graph.current_state_id
    edits = []
    if pending:
        edits += [
            step("graph.add_nodes", count=1, output="new", effective_at=10),
            step("graph.add_edges", edges=[[0, {"added": 0}]], output="edge", effective_at=20),
        ]
    edits += assignment(kind)
    if outer:
        Pipeline([program(*edits)])(graph.view(), atomic=True)
    else:
        graph.apply(program(*edits))
    assert graph.current_state_id == head + 1
    ids = list(graph.edge_ids if kind == "edges" else graph.node_ids)
    for entity in ids:
        expected = entity if kind == "column" else 9
        read = graph.read_view_at(39)
        assert (read.edge_attr(entity, "timed") if kind == "edges" else read.node_attr(entity, "timed")) is None
        read = graph.read_view_at(40)
        assert (read.edge_attr(entity, "timed") if kind == "edges" else read.node_attr(entity, "timed")) == expected
    metadata = graph.publication_metadata(graph.current_state_id)
    assert {row["effective_at"] for row in metadata["change_times"] if row.get("name") == "timed"} == {40}
    path = str(tmp_path / "timed.groggy")
    graph.save_archive(path)
    restored = Graph.load_archive(path)
    for entity in ids:
        assert (restored.read_view_at(39).edge_attr(entity, "timed") if kind == "edges" else restored.read_view_at(39).node_attr(entity, "timed")) is None


@pytest.mark.parametrize("kind", ["nodes", "column", "edges"])
@pytest.mark.parametrize("later_default", [False, True])
def test_last_attachment_replaces_both_pending_value_and_time(kind, later_default):
    graph = fixture()
    graph.apply(program(*assignment(kind), *assignment(kind, value=12, at=None if later_default else 80)))
    metadata = graph.publication_metadata(graph.current_state_id)
    times = {row["effective_at"] for row in metadata["change_times"] if row.get("name") == "timed"}
    assert times == {metadata["recorded_at"] if later_default else 80}
    read = graph.read_view_at(40)
    assert (read.edge_attr(0, "timed") if kind == "edges" else read.node_attr(0, "timed")) is None


@pytest.mark.parametrize("kind", ["nodes", "edges"])
def test_later_accepted_attachment_wins_an_exact_time_tie_without_changing_old_view(kind):
    graph = fixture()
    graph.apply(program(*assignment(kind)))
    before = graph.read_view_at(40)
    graph.apply(program(*assignment(kind, value=12)))
    after = graph.read_view_at(40)
    assert (before.edge_attr(0, "timed") if kind == "edges" else before.node_attr(0, "timed")) == 9
    assert (after.edge_attr(0, "timed") if kind == "edges" else after.node_attr(0, "timed")) == 12


@pytest.mark.parametrize("kind", ["nodes", "edges"])
@pytest.mark.parametrize("params", [
    {"effective_at": True}, {"effective_at": 1.5}, {"effective_at": "40"},
    {"effective_at": None}, {"effective_at": 2 ** 64}, {"effective_time": 40},
])
@pytest.mark.parametrize("iterations", [0, 2])
def test_bad_attachment_parameters_reject_during_preflight_before_prefix(kind, params, iterations):
    graph = fixture()
    head = graph.current_state_id
    bad = step("core.attach_node_attr" if kind == "nodes" else "core.attach_edge_attr",
               source="values", attr="timed", **params)
    with pytest.raises(RuntimeError):
        graph.apply(program(
            step("graph.add_nodes", count=1, output="new"),
            step("iter.loop", iterations=iterations, body=[bad]),
        ))
    assert graph.current_state_id == head
    assert graph.add_node() == 3


@pytest.mark.parametrize("kind", ["nodes", "edges"])
def test_late_schema_failure_discards_entire_timed_batch(kind):
    graph = fixture()
    head = graph.current_state_id
    edits = [step("graph.add_nodes", count=1, output="new", effective_at=10)]
    edits += assignment(kind, at=40)
    edits += [step("core.init_nodes", value=3, target="bad"),
              step("core.attach_node_attr", source="bad", attr="entity_type")]
    with pytest.raises(RuntimeError):
        graph.apply(program(*edits))
    assert graph.current_state_id == head
    assert graph.get_node_attr(0, "timed") is None
    assert graph.get_edge_attr(0, "timed") is None
    assert graph.add_node() == 3


@pytest.mark.parametrize("kind", ["nodes", "edges"])
def test_timed_history_diff_cannot_escape_a_fixed_child(kind):
    graph = fixture()
    before = graph.current_state_id
    if kind == "nodes":
        graph.writer().set_node_attr(2, "value", 7, effective_at=10)
    else:
        graph.writer().set_edge_attr(1, "value", 7, effective_at=10)
    head = graph.current_state_id
    edits = [
        step("temporal.snapshot_at", state_id=before, output="before"),
        step("temporal.snapshot_at", state_id=head, output="after"),
        step("temporal.diff_" + kind, before="before", after="after", output_prefix="diff"),
        step("core.attach_node_attr" if kind == "nodes" else "core.attach_edge_attr",
             source="diff_" + kind + "_modified", attr="timed", effective_at=40),
    ]
    with pytest.raises(RuntimeError, match="outside the fixed execution frame"):
        graph.apply(program(step("sample.iterate_" + kind, target="seeds"), each(edits)))
    assert graph.current_state_id == head


@pytest.mark.parametrize("kind", ["nodes", "edges"])
def test_explicit_attachment_time_requires_atomic_persistent_execution(kind):
    graph = fixture()
    head = graph.current_state_id
    with pytest.raises(RuntimeError, match="persist=True"):
        graph.apply(program(*assignment(kind)), persist=False)
    with pytest.raises(RuntimeError, match="atomic=True"):
        graph.apply(step_pipeline("non-atomic timed attachment", assignment(kind), atomic=False))
    assert graph.current_state_id == head


@pytest.mark.parametrize("kind", ["nodes", "column", "edges"])
def test_timed_attachments_accept_exact_child_rows(kind):
    graph = fixture()
    source_kind = "edges" if kind == "edges" else "nodes"
    graph.apply(program(
        step("sample.iterate_" + source_kind, target="seeds"),
        each(assignment(kind)),
    ))
    read = graph.read_view_at(40)
    ids = graph.edge_ids if kind == "edges" else graph.node_ids
    for entity in ids:
        expected = entity if kind == "column" else 9
        assert (read.edge_attr(entity, "timed") if kind == "edges" else read.node_attr(entity, "timed")) == expected


@pytest.mark.parametrize("kind", ["nodes", "edges"])
def test_outside_lifetime_backfills_do_not_carry_into_later_activation(kind):
    graph = fixture()
    graph.apply(program(
        step("graph.add_nodes", count=1, output="new", effective_at=10),
        step("graph.add_edges", edges=[[0, {"added": 0}]], output="edge", effective_at=20),
        *assignment(kind, at=-1),
    ))
    if kind == "nodes":
        assert 3 not in graph.read_view_at(9).node_ids
        assert graph.read_view_at(10).node_attr(3, "timed") is None
        assert graph.get_node_attr(3, "timed") == 9
    else:
        assert 2 not in graph.read_view_at(19).edge_ids
        assert graph.read_view_at(20).edge_attr(2, "timed") is None
        assert graph.get_edge_attr(2, "timed") == 9
    metadata = graph.publication_metadata(graph.current_state_id)
    assert {row["effective_at"] for row in metadata["change_times"] if row.get("name") == "timed"} == {-1}
