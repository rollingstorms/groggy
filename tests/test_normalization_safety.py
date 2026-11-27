"""Ordinary normalization steps reject unsafe arithmetic atomically."""

import pytest

from groggy import DurableGraph, Graph, GraphBatch, Pipeline
from groggy.builder import step_pipeline


def step(kind, **params):
    return {"id": kind, "params": params}


def fixture(path, values):
    graph = Graph(directed=True)
    batch = GraphBatch(effective_at=0)
    for value in values:
        batch.add_node(score=value)
    for node in range(len(values)):
        batch.add_edge(node, node, score=values[node])
    graph.writer().apply_batch(batch)
    return DurableGraph.create_from_graph(path, graph)


def pipeline(steps):
    return Pipeline([step_pipeline("checked normalization", steps, atomic=True)])


@pytest.mark.parametrize("entity", ["nodes", "edges"])
@pytest.mark.parametrize(
    "operation,params,values",
    [
        ("core.normalize_values", {"epsilon": -1.0}, [0.0, 0.0]),
        ("core.standardize", {"epsilon": -1.0}, [1.0, 2.0]),
        ("core.clip", {"min": 2.0, "max": 1.0}, [1.0, 2.0]),
        ("core.normalize_values", {"method": "sum"}, [1e308, 1e308]),
        ("core.normalize_values", {"method": "minmax"}, [-1e308, 1e308]),
        ("core.standardize", {}, [-1e308, 1e308]),
    ],
)
def test_invalid_normalization_rolls_back_pending_writes_and_allocations(
    tmp_path, entity, operation, params, values
):
    path = tmp_path / "reject.journal"
    store = fixture(path, values)
    before, head = path.read_bytes(), store.current_state_id
    with pytest.raises(Exception, match="epsilon|clip range|finite"):
        p = pipeline(
            [
                step(
                    (
                        "core.load_node_attr"
                        if entity == "nodes"
                        else "core.load_edge_attr"
                    ),
                    attr="score",
                    target="input",
                ),
                step("core.init_nodes", target="prior", value=99),
                step(
                    "core.attach_node_attr",
                    source="prior",
                    attr="tentative",
                    effective_at=10,
                ),
                step("graph.add_nodes", count=1, output="new"),
                step(operation, source="input", target="output", **params),
            ]
        )
        p.run_durable(store)
    assert store.current_state_id == head
    assert path.read_bytes() == before
    assert store.read_view().node_ids == list(range(len(values)))
    assert store.read_view().node_attr(0, "tentative") is None
    store.close()


@pytest.mark.parametrize("entity", ["nodes", "edges"])
@pytest.mark.parametrize(
    "operation,params,values,expected",
    [
        (
            "core.normalize_values",
            {"method": "sum"},
            [2.0, 4.0],
            [1.0 / 3.0, 2.0 / 3.0],
        ),
        ("core.normalize_values", {"method": "max"}, [-2.0, -4.0], [1.0, 2.0]),
        ("core.normalize_values", {"method": "minmax"}, [2.0, 4.0], [0.0, 1.0]),
        ("core.standardize", {}, [2.0, 4.0], [-1.0, 1.0]),
        ("core.clip", {"min": 2.5, "max": 3.5}, [2.0, 4.0], [2.5, 3.5]),
        ("core.clip", {"min": 3.0}, [2.0, 4.0], [3.0, 4.0]),
        ("core.clip", {"max": 3.0}, [2.0, 4.0], [2.0, 3.0]),
        ("core.clip", {}, [2.0, 4.0], [2.0, 4.0]),
    ],
)
def test_valid_normalization_preserves_map_kind_and_recovery(
    tmp_path, entity, operation, params, values, expected
):
    path = tmp_path / "valid.journal"
    store = fixture(path, values)
    p = pipeline(
        [
            step(
                "core.load_node_attr" if entity == "nodes" else "core.load_edge_attr",
                attr="score",
                target="input",
            ),
            step(operation, source="input", target="output", **params),
            step(
                (
                    "core.attach_node_attr"
                    if entity == "nodes"
                    else "core.attach_edge_attr"
                ),
                source="output",
                attr="result",
                effective_at=20,
            ),
        ]
    )
    result = p.run_durable(store)
    get = result.node_attr if entity == "nodes" else result.edge_attr
    assert [get(i, "result") for i in range(2)] == pytest.approx(expected)
    store.close()
    reopened = DurableGraph.open(path)
    before, after = reopened.read_view_at(19), reopened.read_view_at(20)
    old_get = before.node_attr if entity == "nodes" else before.edge_attr
    new_get = after.node_attr if entity == "nodes" else after.edge_attr
    assert old_get(0, "result") is None
    assert [new_get(i, "result") for i in range(2)] == pytest.approx(expected)
    reopened.close()


MALFORMED_CONTROLS = (
    [("core.normalize_values", {"epsilon": value}) for value in ["bad", True, None, []]]
    + [
        ("core.normalize_node_values", {"epsilon": None}),
        ("core.standardize", {"epsilon": False}),
    ]
    + [
        (kind, {"method": value})
        for kind in ["core.normalize_values", "core.normalize_node_values"]
        for value in [None, True, 42]
    ]
    + [
        ("core.clip", {name: "bad"})
        for name in ["min", "max", "min_value", "max_value"]
    ]
    + [
        ("core.clip", {"min_value": "bad", "min": 0}),
        ("core.clip", {"min_value": 0, "min": "bad"}),
        ("core.normalize_values", {"epsilon": -1.0}),
        ("core.normalize_node_values", {"epsilon": -1.0}),
        ("core.standardize", {"epsilon": -1.0}),
        ("core.clip", {"min": 2, "max": 1}),
    ]
)


@pytest.mark.parametrize("kind,params", MALFORMED_CONTROLS)
@pytest.mark.parametrize("iterations", [0, 1])
def test_explicit_normalization_controls_reject_before_atomic_prefix_even_in_zero_loops(
    tmp_path, kind, params, iterations
):
    path = tmp_path / "malformed-controls.journal"
    store = fixture(path, [2.0, 4.0])
    before, head = path.read_bytes(), store.current_state_id
    with pytest.raises(Exception, match="epsilon|method|min|max|clip range"):
        pipeline(
            [
                step("core.init_nodes", target="prior", value=99),
                step("core.attach_node_attr", source="prior", attr="tentative"),
                step("graph.add_nodes", count=1, output="new"),
                step("core.load_node_attr", attr="score", default=2.0, target="input"),
                step(
                    "iter.loop",
                    iterations=iterations,
                    body=[step(kind, source="input", target="output", **params)],
                ),
            ]
        ).run_durable(store)
    assert store.current_state_id == head
    assert path.read_bytes() == before
    assert store.read_view().node_ids == [0, 1]
    assert store.read_view().node_attr(0, "tentative") is None
    store.close()


@pytest.mark.parametrize("method", ["min", "mean"])
def test_valid_integer_epsilon_and_normalization_alias_keep_the_shared_contract(
    tmp_path, method
):
    store = fixture(tmp_path / "valid-control.journal", [2.0, 4.0])
    result = pipeline(
        [
            step("core.load_node_attr", attr="score", target="input"),
            step(
                "core.normalize_node_values",
                source="input",
                target="out",
                method=method,
                epsilon=0,
            ),
            step("core.attach_node_attr", source="out", attr="result"),
        ]
    ).run_durable(store)
    assert [result.node_attr(n, "result") for n in [0, 1]] == pytest.approx(
        [1.0, 2.0] if method == "min" else [2.0 / 3.0, 4.0 / 3.0]
    )
    store.close()


def test_valid_clip_aliases_keep_primary_precedence(tmp_path):
    store = fixture(tmp_path / "clip-aliases.journal", [2.0, 4.0])
    result = pipeline(
        [
            step("core.load_node_attr", attr="score", target="input"),
            step(
                "core.clip",
                source="input",
                target="out",
                min_value=1,
                max_value=3,
                min=-100,
                max=100,
            ),
            step("core.attach_node_attr", source="out", attr="result"),
        ]
    ).run_durable(store)
    assert [result.node_attr(n, "result") for n in [0, 1]] == [2.0, 3.0]
    store.close()
