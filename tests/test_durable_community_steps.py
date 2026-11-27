"""Community helper columns share selected durable reads and retain their kinds."""

import pytest

from groggy import DurableGraph, Graph, GraphBatch, Pipeline
from groggy.builder import step_pipeline


def step(kind, **params):
    return {"id": kind, "params": params}


def fixture(path, directed=False):
    graph = Graph(directed=directed)
    batch = GraphBatch(effective_at=0)
    for _ in range(4):
        batch.add_node()
    for a, b in [(0, 1), (0, 1), (1, 1), (1, 2), (2, 0)]:
        batch.add_edge(a, b)
    graph.writer().apply_batch(batch)
    return DurableGraph.create_from_graph(path, graph)


def pipeline(steps):
    return Pipeline([step_pipeline("community helpers", steps, atomic=True)])


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("strategy", ["singleton", "degree", "random"])
def test_seed_gain_and_propagation_preserve_columns_instances_and_effective_time(
    tmp_path, directed, strategy
):
    path = tmp_path / "community.journal"
    store = fixture(path, directed)
    head = store.current_state_id
    p = pipeline(
        [
            step(
                "core.community_seed", strategy=strategy, k=3, seed=42, target="labels"
            ),
            step("core.modularity_gain", partition="labels", target="gain"),
            step("core.label_propagate_step", labels="labels", target="next"),
            step(
                "core.attach_node_attr", source="labels", attr="seed", effective_at=20
            ),
            step("core.attach_node_attr", source="gain", attr="gain", effective_at=20),
            step("core.attach_node_attr", source="next", attr="next", effective_at=20),
        ]
    )
    result = p.run_durable(store)
    seeds = [result.node_attr(n, "seed") for n in range(4)]
    assert all(isinstance(v, int) for v in seeds)
    if strategy == "singleton":
        assert seeds == list(range(4))
    if strategy == "random":
        assert all(0 <= v < 3 for v in seeds)
    assert all(result.node_attr(n, "gain") >= 0 for n in range(4))
    assert result.node_attr(3, "next") == result.node_attr(3, "seed")
    assert store.current_state_id == head + 1
    repeated = p.run_durable(store)
    assert [repeated.node_attr(n, "seed") for n in range(4)] == seeds
    store.close()
    reopened = DurableGraph.open(path)
    assert reopened.read_view_at(19).node_attr(0, "seed") is None
    assert [reopened.read_view_at(20).node_attr(n, "seed") for n in range(4)] == seeds
    reopened.close()


@pytest.mark.parametrize("value", [True, "not an integer"])
def test_modularity_requires_integer_partition_and_rolls_back_the_operation(
    tmp_path, value
):
    path = tmp_path / "reject.journal"
    store = fixture(path)
    before = path.read_bytes()
    with pytest.raises(Exception, match="integers"):
        pipeline(
            [
                step("graph.add_nodes", count=1, output="new"),
                step("core.init_nodes", value=value, target="bad"),
                step("core.attach_node_attr", source="bad", attr="tentative"),
                step("core.modularity_gain", partition="bad", target="gain"),
            ]
        ).run_durable(store)
    assert path.read_bytes() == before
    assert store.read_view().node_ids == [0, 1, 2, 3]
    store.close()


@pytest.mark.parametrize(
    "operation", ["core.label_propagate_step", "core.modularity_gain"]
)
def test_old_community_columns_reject_removed_membership(tmp_path, operation):
    path = tmp_path / "stale.journal"
    store = fixture(path)
    before = path.read_bytes()
    with pytest.raises(Exception, match="selected node IDs"):
        pipeline(
            [
                step("core.community_seed", strategy="singleton", target="labels"),
                step("graph.remove_nodes", ids=[1]),
                step(
                    operation,
                    target="out",
                    **{
                        (
                            "labels"
                            if operation == "core.label_propagate_step"
                            else "partition"
                        ): "labels"
                    },
                ),
            ]
        ).run_durable(store)
    assert path.read_bytes() == before
    store.close()


def test_unattached_community_outputs_do_not_create_history(tmp_path):
    path = tmp_path / "read.journal"
    store = fixture(path)
    head, before = store.current_state_id, path.read_bytes()
    pipeline(
        [
            step("core.community_seed", strategy="singleton", target="labels"),
            step("core.modularity_gain", partition="labels", target="gain"),
            step("core.label_propagate_step", labels="labels", target="next"),
        ]
    ).run_durable(store)
    assert store.current_state_id == head
    assert path.read_bytes() == before
    store.close()
