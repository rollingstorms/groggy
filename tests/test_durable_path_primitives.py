"""Portable path primitives retain selected geometry, checked weights and RNG."""

import json

import pytest

from groggy import DurableGraph, Graph, GraphBatch, Pipeline
from groggy.builder import step_pipeline


def step(kind, **params):
    return {"id": kind, "params": params}


def pipeline(steps):
    return Pipeline([step_pipeline("portable path primitives", steps, atomic=True)])


def fixture(path, directed=True):
    graph = Graph(directed=directed)
    batch = GraphBatch(effective_at=0)
    for _ in range(4):
        batch.add_node()
    for a, b, weight in [
        (0, 1, 2.0),
        (0, 1, 4.0),
        (1, 1, 0.0),
        (1, 2, 1.0),
        (0, 2, 10.0),
    ]:
        batch.add_edge(a, b, w=weight)
    graph.writer().apply_batch(batch)
    return DurableGraph.create_from_graph(path, graph)


def primitives(weighted):
    options = {"weight_attr": "w"} if weighted else {}
    return [
        step("core.shortest_path_map", source="start", output="distances", **options),
        step(
            "core.k_shortest_paths",
            source="start",
            target="end",
            k=10,
            output="paths",
            **options,
        ),
        step(
            "core.random_walk",
            start_nodes="start",
            length=6,
            restart_prob=0.2,
            seed=19,
            output="walk",
            **options,
        ),
    ]


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_pending_paths_and_seeded_walks_keep_selection_results_types_and_recovery(
    tmp_path, directed, weighted
):
    path = tmp_path / "paths.journal"
    store = fixture(path, directed)
    source = store.read_view().select_nodes([0, 1, 2])
    result = pipeline(
        [
            step("graph.add_nodes", count=1, output="new", effective_at=10),
            step(
                "graph.add_edges",
                edges=[[2, {"added": 0}]],
                attributes={"w": 2.0},
                output="edge",
                effective_at=10,
            ),
            step(
                "graph.add_edges",
                edges=[[0, {"added": 0}]],
                attributes={"w": 8.0},
                output="direct",
                effective_at=10,
            ),
            step("graph.remove_edges", ids=[1], effective_at=10),
            step("graph.set_edge_endpoints", revisions=[[4, 2, 0]], effective_at=10),
            step("core.init_scalar", target="start", value="0"),
            step("core.init_scalar", target="end", value=4),
            step("iter.loop", iterations=2, body=primitives(weighted)),
            step(
                "core.attach_node_attr",
                source="distances",
                attr="distance",
                effective_at=20,
            ),
            step(
                "core.broadcast_scalar",
                scalar="paths",
                reference="distances",
                target="path_rows",
            ),
            step(
                "core.broadcast_scalar",
                scalar="walk",
                reference="distances",
                target="walk_rows",
            ),
            step(
                "core.attach_node_attr",
                source="path_rows",
                attr="paths",
                effective_at=20,
            ),
            step(
                "core.attach_node_attr",
                source="walk_rows",
                attr="walk",
                effective_at=20,
            ),
        ]
    ).run_durable(store, source)
    expected = [0.0, 2.0, 3.0, 5.0] if weighted else [0, 1, 2 if directed else 1, 1]
    assert [result.node_attr(n, "distance") for n in [0, 1, 2, 4]] == expected
    assert all(
        isinstance(result.node_attr(n, "distance"), float if weighted else int)
        for n in [0, 1, 2, 4]
    )
    paths = json.loads(result.node_attr(0, "paths"))
    expected_paths = (
        [[[0, 1, 2, 4], 5.0], [[0, 4], 8.0]]
        if weighted
        else [[[0, 4], 1.0], [[0, 1, 2, 4], 3.0]]
    )
    if not directed:
        expected_paths.append([[0, 2, 4], 12.0 if weighted else 2.0])
        expected_paths.sort(key=lambda row: (row[1], row[0]))
    assert paths == expected_paths
    walk = json.loads(result.node_attr(0, "walk"))[0]
    allowed = {(0, 1), (1, 1), (1, 2), (2, 0), (2, 4), (0, 4)}
    assert walk[0] == 0 and len(walk) <= 7
    assert all(
        b == 0 or (a, b) in allowed or (not directed and (b, a) in allowed)
        for a, b in zip(walk, walk[1:])
    )
    assert 3 not in result.node_ids
    assert source.node_attr(0, "distance") is None
    store.close()
    recovered = DurableGraph.open(path)
    assert recovered.read_view_at(19).node_attr(0, "distance") is None
    assert (
        json.loads(recovered.read_view_at(20).node_attr(4, "paths")) == expected_paths
    )
    recovered.close()


@pytest.mark.parametrize(
    "op", ["core.shortest_path_map", "core.k_shortest_paths", "core.random_walk"]
)
@pytest.mark.parametrize("value", [-1.0, True, None, "bad"])
def test_bad_pending_weights_reject_even_zero_path_counts_or_walk_lengths(
    tmp_path, op, value
):
    path = tmp_path / "bad-weight.journal"
    store = fixture(path)
    before, head = path.read_bytes(), store.current_state_id
    params = {"source": "start", "output": "out", "weight_attr": "w"}
    if op == "core.k_shortest_paths":
        params.update(target="start", k=0)
    if op == "core.random_walk":
        params = {
            "start_nodes": "start",
            "output": "out",
            "weight_attr": "w",
            "length": 0,
            "seed": 19,
        }
    with pytest.raises(Exception, match="weights"):
        pipeline(
            [
                step("graph.add_nodes", count=1, output="new"),
                step("core.init_edges", value=value, target="bad"),
                step("core.attach_edge_attr", source="bad", attr="w"),
                step("core.init_scalar", target="start", value=0),
                step(op, **params),
            ]
        ).run_durable(store)
    assert store.current_state_id == head
    assert path.read_bytes() == before
    store.close()


@pytest.mark.parametrize(
    "op", ["core.shortest_path_map", "core.k_shortest_paths", "core.random_walk"]
)
def test_fixed_foreach_path_source_cannot_escape_its_child_even_for_trivial_work(
    tmp_path, op
):
    path = tmp_path / "scope.journal"
    store = fixture(path)
    before, head = path.read_bytes(), store.current_state_id
    params = {"source": "start", "output": "out"}
    if op == "core.k_shortest_paths":
        params.update(target="start", k=0)
    if op == "core.random_walk":
        params = {"start_nodes": "start", "output": "out", "length": 0, "seed": 19}
    with pytest.raises(Exception, match="outside"):
        pipeline(
            [
                step("sample.iterate_nodes", target="seeds"),
                step(
                    "sample.for_each",
                    source="seeds",
                    target="out",
                    body=[
                        step("core.init_nodes", value=9, target="prefix"),
                        step("core.attach_node_attr", source="prefix", attr="prefix"),
                        step("core.init_scalar", target="start", value=2),
                        step(op, **params),
                    ],
                ),
            ]
        ).run_durable(store)
    assert path.read_bytes() == before
    assert store.current_state_id == head
    assert store.read_view().node_attr(0, "prefix") is None
    store.close()


def test_unattached_path_computation_and_repeated_seeded_walks_do_not_publish(tmp_path):
    path = tmp_path / "read.journal"
    store = fixture(path)
    before, head = path.read_bytes(), store.current_state_id
    program = pipeline(
        [
            step("core.init_scalar", target="start", value=0),
            step("core.init_scalar", target="end", value=2),
            *primitives(True),
        ]
    )
    program.run_durable(store)
    program.run_durable(store)
    assert path.read_bytes() == before
    assert store.current_state_id == head
    store.close()
