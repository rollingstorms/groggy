"""The legacy edge-weight transform has checked memory/durable read parity."""

import pytest

from groggy import DurableGraph, Graph, GraphBatch, Pipeline
from groggy.builder import step_pipeline


def step(kind, **params):
    return {"id": kind, "params": params}


def fixture(path, directed=True):
    graph = Graph(directed=directed)
    batch = GraphBatch(effective_at=0)
    for _ in range(3):
        batch.add_node()
    for pair, attrs in zip(
        [(0, 1), (0, 1), (1, 1), (1, 2), (2, 0), (2, 2)],
        [
            {"weight": 3},
            {"weight": -0.5},
            {},
            {"weight": None},
            {"weight": "text"},
            {"weight": True},
        ],
    ):
        batch.add_edge(*pair, **attrs)
    graph.writer().apply_batch(batch)
    return DurableGraph.create_from_graph(path, graph)


def pipeline(steps):
    return Pipeline([step_pipeline("edge weights", steps, atomic=True)])


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("factor", [0.0, -2.0, 0.5])
def test_edge_weight_transform_preserves_instances_defaults_and_passthrough(
    tmp_path, directed, factor
):
    path = tmp_path / "weights.journal"
    store = fixture(path, directed)
    head = store.current_state_id
    result = pipeline(
        [
            step("core.edge_weight_scale", attr="weight", factor=factor, target="out"),
            step("core.attach_edge_attr", source="out", attr="scaled", effective_at=20),
        ]
    ).run_durable(store)
    expected = [3 * factor, -0.5 * factor, factor, None, "text", True]
    assert result.edge_ids == list(range(6))
    assert [result.edge_attr(i, "scaled") for i in range(6)] == expected
    assert result.edge_attr(5, "scaled") is True
    assert store.current_state_id == head + 1
    store.close()
    reopened = DurableGraph.open(path)
    assert reopened.read_view_at(19).edge_attr(0, "scaled") is None
    assert reopened.read_view_at(20).edge_attr(0, "scaled") == expected[0]
    reopened.close()


@pytest.mark.parametrize(
    "factor", [None, "bad", True, float("nan"), float("inf"), -float("inf")]
)
def test_explicit_malformed_factors_reject_without_defaulting_silently(
    tmp_path, factor
):
    path = tmp_path / "invalid.journal"
    store = fixture(path)
    before = path.read_bytes()
    with pytest.raises(Exception):
        pipeline(
            [step("core.edge_weight_scale", attr="weight", factor=factor, target="bad")]
        ).run_durable(store)
    assert path.read_bytes() == before
    store.close()


def test_overflow_rejects_after_staged_edits_without_publication(tmp_path):
    path = tmp_path / "overflow.journal"
    store = fixture(path)
    head, before = store.current_state_id, path.read_bytes()
    p = pipeline(
        [
            step("graph.add_nodes", count=1, output="new"),
            step("core.init_edges", value=5, target="prior"),
            step("core.attach_edge_attr", source="prior", attr="tentative"),
            step("core.edge_weight_scale", attr="weight", factor=1e308, target="bad"),
        ]
    )
    with pytest.raises(Exception, match="finite"):
        p.run_durable(store)
    assert store.current_state_id == head
    assert path.read_bytes() == before
    assert store.read_view().node_ids == [0, 1, 2]
    assert store.read_view().edge_attr(0, "tentative") is None
    store.close()


@pytest.mark.parametrize("attach", [False, True])
def test_omitted_factor_keeps_unit_scaling_and_unattached_reads_do_not_publish(
    tmp_path, attach
):
    path = tmp_path / "unit.journal"
    store = fixture(path)
    head = store.current_state_id
    steps = [step("core.edge_weight_scale", attr="weight", target="out")]
    if attach:
        steps.append(step("core.attach_edge_attr", source="out", attr="scaled"))
    result = pipeline(steps).run_durable(store)
    assert store.current_state_id == head + attach
    if attach:
        assert result.edge_attr(0, "scaled") == 3.0
        assert result.edge_attr(2, "scaled") == 1.0
    store.close()
