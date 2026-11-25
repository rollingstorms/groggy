"""Seeded mask sampling is selected, owned and explicitly attached."""

import pytest

from groggy import DurableGraph, Graph, GraphBatch, Pipeline
from groggy.builder import step_pipeline


def step(kind, **params):
    return {"id": kind, "params": params}


def fixture(path):
    graph = Graph()
    batch = GraphBatch(effective_at=0)
    for _ in range(5):
        batch.add_node()
    for source, target in [(0, 1), (0, 1), (1, 1), (1, 2), (3, 4)]:
        batch.add_edge(source, target)
    graph.writer().apply_batch(batch)
    return DurableGraph.create_from_graph(path, graph)


def pipeline(steps):
    return Pipeline([step_pipeline("seeded sampling", steps, atomic=True)])


@pytest.mark.parametrize("entity", ["nodes", "edges"])
@pytest.mark.parametrize(
    "spec,size",
    [
        ({"count": 0}, 0),
        ({"count": 2}, 2),
        ({"fraction": 0.5}, 3),
        ({"fraction": 1.0}, 5),
    ],
)
def test_seeded_sample_and_reservoir_masks_preserve_instances(
    tmp_path, entity, spec, size
):
    path = tmp_path / "sample.journal"
    store = fixture(path)
    head = store.current_state_id
    attach = "core.attach_node_attr" if entity == "nodes" else "core.attach_edge_attr"
    p = pipeline(
        [
            step("core.sample_" + entity, target="sample", seed=42, **spec),
            step(
                "core.reservoir_sample",
                source="sample",
                target="reservoir",
                entity_type=entity,
                k=2,
                seed=42,
            ),
            step(attach, source="sample", attr="sample", effective_at=20),
            step(attach, source="reservoir", attr="reservoir", effective_at=20),
        ]
    )
    result = p.run_durable(store)
    get = result.node_attr if entity == "nodes" else result.edge_attr
    ids = result.node_ids if entity == "nodes" else result.edge_ids
    sampled = [i for i in ids if get(i, "sample") == 1]
    reservoir = [i for i in ids if get(i, "reservoir") == 1]
    assert len(ids) == 5
    assert len(sampled) == size
    assert len(reservoir) == min(2, size)
    assert set(reservoir) <= set(sampled)
    assert store.current_state_id == head + (size != 0)
    # Fixed seed and unchanged population reproduce the same masks.
    repeated = p.run_durable(store)
    repeated_get = repeated.node_attr if entity == "nodes" else repeated.edge_attr
    assert [i for i in ids if repeated_get(i, "sample") == 1] == sampled
    store.close()
    reopened = DurableGraph.open(path)
    before, after = reopened.read_view_at(19), reopened.read_view_at(20)
    old_get = before.node_attr if entity == "nodes" else before.edge_attr
    new_get = after.node_attr if entity == "nodes" else after.edge_attr
    for i in sampled:
        assert old_get(i, "sample") is None
        assert new_get(i, "sample") == 1
    reopened.close()


@pytest.mark.parametrize("entity", ["nodes", "edges"])
def test_oversized_sample_rejects_after_pending_creation_without_journal_change(
    tmp_path, entity
):
    path = tmp_path / "reject.journal"
    store = fixture(path)
    before = path.read_bytes()
    head = store.current_state_id
    p = pipeline(
        [
            step("graph.add_nodes", count=1, output="new"),
            step("core.sample_" + entity, count=99, target="bad", seed=1),
        ]
    )
    with pytest.raises(Exception, match="Cannot sample"):
        p.run_durable(store)
    assert store.current_state_id == head
    assert path.read_bytes() == before
    assert store.read_view().node_ids == [0, 1, 2, 3, 4]
    store.close()


def test_unattached_sampling_keeps_the_retained_state_unchanged(tmp_path):
    path = tmp_path / "read.journal"
    store = fixture(path)
    head, before = store.current_state_id, path.read_bytes()
    result = pipeline(
        [
            step("core.sample_nodes", fraction=0.5, target="nodes", seed=42),
            step("core.sample_edges", fraction=0.5, target="edges", seed=42),
            step("core.reservoir_sample", source="nodes", target="one", k=1, seed=1),
        ]
    ).run_durable(store)
    assert result.node_ids == [0, 1, 2, 3, 4]
    assert store.current_state_id == head
    assert path.read_bytes() == before
    store.close()
