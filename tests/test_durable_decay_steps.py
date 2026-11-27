"""Decay math preserves owned node/column/edge layouts on durable programs."""

import pytest

from groggy import DurableGraph, Graph, GraphBatch, Pipeline
from groggy.builder import step_pipeline


def step(kind, **params):
    return {"id": kind, "params": params}


def fixture(path):
    graph = Graph(directed=True)
    batch = GraphBatch(effective_at=0)
    for _ in range(3):
        batch.add_node()
    for pair in [(0, 1), (0, 1), (1, 1), (1, 2)]:
        batch.add_edge(*pair)
    graph.writer().apply_batch(batch)
    return DurableGraph.create_from_graph(path, graph)


def pipeline(steps):
    return Pipeline([step_pipeline("owned decay", steps, atomic=True)])


@pytest.mark.parametrize("layout", ["nodes", "column", "edges"])
@pytest.mark.parametrize("elapsed,factor", [(0.0, 1.0), (5e-324, 0.5), (1e308, 0.0)])
def test_subnormal_half_life_keeps_layout_aliases_and_explicit_attachment_clock(
    tmp_path, layout, elapsed, factor
):
    path = tmp_path / "decay.journal"
    store = fixture(path)
    head = store.current_state_id
    load = (
        step("core.community_seed", strategy="singleton", target="values")
        if layout == "column"
        else step(
            "core.init_edges" if layout == "edges" else "core.init_nodes",
            value=8,
            target="values",
        )
    )
    attach = "core.attach_edge_attr" if layout == "edges" else "core.attach_node_attr"
    result = pipeline(
        [
            load,
            step("alias", source="values", target="saved"),
            step(
                "temporal.decay",
                attr="values",
                output="values",
                half_life=5e-324,
                time_delta=elapsed,
            ),
            step(attach, source="values", attr="result", effective_at=20),
            step(attach, source="saved", attr="original", effective_at=20),
        ]
    ).run_durable(store)
    ids = result.edge_ids if layout == "edges" else result.node_ids
    get = result.edge_attr if layout == "edges" else result.node_attr
    expected = [float(i) if layout == "column" else 8.0 for i in ids]
    assert [get(i, "result") for i in ids] == [v * factor for v in expected]
    assert [get(i, "original") for i in ids] == expected
    assert store.current_state_id == head + 1
    store.close()
    reopened = DurableGraph.open(path)
    old, new = reopened.read_view_at(19), reopened.read_view_at(20)
    old_get = old.edge_attr if layout == "edges" else old.node_attr
    new_get = new.edge_attr if layout == "edges" else new.node_attr
    assert old_get(0, "result") is None
    assert [new_get(i, "result") for i in ids] == [v * factor for v in expected]
    reopened.close()


@pytest.mark.parametrize("edges", [False, True])
@pytest.mark.parametrize("value", [None, True, "bad"])
def test_invalid_decay_source_rolls_back_staged_properties_and_ids(
    tmp_path, edges, value
):
    path = tmp_path / "invalid.journal"
    store = fixture(path)
    head, before = store.current_state_id, path.read_bytes()
    p = pipeline(
        [
            step("graph.add_nodes", count=1, output="new"),
            step("core.init_nodes", value=8, target="prior"),
            step("core.attach_node_attr", source="prior", attr="tentative"),
            step(
                "core.init_edges" if edges else "core.init_nodes",
                value=value,
                target="values",
            ),
            step(
                "temporal.decay",
                attr="values",
                output="result",
                half_life=2,
                time_delta=0,
            ),
        ]
    )
    with pytest.raises(Exception, match="finite numeric"):
        p.run_durable(store)
    assert store.current_state_id == head
    assert path.read_bytes() == before
    assert store.read_view().node_ids == [0, 1, 2]
    store.close()


@pytest.mark.parametrize("edges", [False, True])
def test_stale_decay_maps_cannot_inspect_removed_rows_even_at_zero_elapsed(
    tmp_path, edges
):
    path = tmp_path / "stale.journal"
    store = fixture(path)
    before = path.read_bytes()
    p = pipeline(
        [
            step(
                "core.init_edges" if edges else "core.init_nodes", value=8, target="old"
            ),
            step("graph.remove_edges" if edges else "graph.remove_nodes", ids=[0]),
            step(
                "temporal.decay", attr="old", output="result", half_life=2, time_delta=0
            ),
        ]
    )
    with pytest.raises(Exception, match="outside"):
        p.run_durable(store)
    assert path.read_bytes() == before
    store.close()


def test_unattached_decay_is_a_read_computation(tmp_path):
    path = tmp_path / "read.journal"
    store = fixture(path)
    head, before = store.current_state_id, path.read_bytes()
    pipeline(
        [
            step("core.init_nodes", value=8, target="values"),
            step(
                "temporal.decay",
                attr="values",
                output="result",
                half_life=2,
                time_delta=2,
            ),
        ]
    ).run_durable(store)
    assert store.current_state_id == head
    assert path.read_bytes() == before
    store.close()
