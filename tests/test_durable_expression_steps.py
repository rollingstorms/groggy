"""Public expression builders share one evaluator over immutable read inputs."""

import pytest
from groggy import DurableGraph, Graph, GraphBatch, Pipeline
from groggy.builder import AlgorithmBuilder, step_pipeline


def fixture(path, directed=False, scores=(1, 1, 1), edges=((0, 1), (1, 2), (2, 0))):
    graph = Graph(directed=directed)
    batch = GraphBatch(effective_at=0)
    for value in scores:
        batch.add_node(score=value)
    for source, target in edges:
        batch.add_edge(source, target)
    graph.writer().apply_batch(batch)
    return DurableGraph.create_from_graph(path, graph)


@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_public_neighbor_expression_preserves_default_direction_distinct_neighbors_and_async(
    tmp_path, optimize, directed, asynchronous
):
    store = fixture(
        tmp_path / "neighbor.journal",
        directed=directed,
        edges=((0, 1), (1, 2), (2, 0), (0, 1), (0, 0)),
    )
    builder = AlgorithmBuilder("durable expression neighbors")
    values = builder.load_attr("score")
    result = builder.map_nodes(
        "sum(values[neighbors(node)])",
        inputs={"values": values},
        async_update=asynchronous,
    )
    builder.attach_as("answer", result)
    head = store.current_state_id
    accepted = Pipeline([builder.build(atomic=True, optimize=optimize)]).run_durable(
        store
    )
    expected = (
        ([2, 1, 2] if asynchronous else [2, 1, 1])
        if directed
        else ([3, 4, 7] if asynchronous else [3, 2, 2])
    )
    assert [accepted.node_attr(n, "answer") for n in accepted.node_ids] == expected
    assert store.current_state_id == head + 1
    store.close()


@pytest.mark.parametrize("optimize", [False, True])
def test_public_structured_iteration_composes_expressions_arithmetic_and_aliases(
    tmp_path, optimize
):
    store = fixture(tmp_path / "iteration.journal")
    builder = AlgorithmBuilder("durable structured expressions")
    ranks = builder.init_nodes(1.0)
    with builder.iterate(2):
        sums = builder.map_nodes("sum(ranks[neighbors(node)])", inputs={"ranks": ranks})
        scaled = builder.core.mul(sums, 0.85)
        ranks = builder.var("ranks", builder.core.add(scaled, 0.15))
    builder.attach_as("rank", ranks)
    accepted = Pipeline([builder.build(atomic=True, optimize=optimize)]).run_durable(
        store
    )
    assert [accepted.node_attr(n, "rank") for n in accepted.node_ids] == pytest.approx(
        [3.295] * 3
    )
    store.close()


def step(kind, **params):
    return {"id": kind, "params": params}


@pytest.mark.parametrize("asynchronous", [False, True])
def test_expression_attributes_read_pending_properties_and_keep_short_circuit_rules(
    tmp_path, asynchronous
):
    store = fixture(tmp_path / "attrs.journal", scores=(1, 2, 3), edges=())
    expression = {
        "type": "if",
        "condition": {"type": "const", "value": False},
        "then_expr": {
            "type": "binary_op",
            "op": "div",
            "left": {"type": "const", "value": 1},
            "right": {"type": "const", "value": 0},
        },
        "else_expr": {
            "type": "binary_op",
            "op": "add",
            "left": {"type": "var", "name": "value"},
            "right": {"type": "attr", "name": "score"},
        },
    }
    handle = step_pipeline(
        "pending expression",
        [
            step("core.load_node_attr", attr="score", target="values"),
            step("core.init_nodes", target="replacement", value=2**53 + 1),
            step(
                "core.attach_node_attr",
                source="replacement",
                attr="score",
                effective_at=10,
            ),
            step(
                "core.map_nodes",
                source="values",
                target="answer",
                expr=expression,
                async_update=asynchronous,
            ),
            step(
                "core.attach_node_attr", source="answer", attr="answer", effective_at=20
            ),
        ],
        atomic=True,
    )
    result = Pipeline([handle]).run_durable(store)
    assert [result.node_attr(n, "answer") for n in result.node_ids] == [
        2**53 + 2,
        2**53 + 3,
        2**53 + 4,
    ]
    store.close()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_expression_late_failure_discards_source_updates_and_publication(
    tmp_path, asynchronous
):
    path = tmp_path / "reject.journal"
    store = fixture(path, scores=(1, 2, 3), edges=())
    before = path.read_bytes()
    expression = {
        "type": "binary_op",
        "op": "div",
        "left": {"type": "const", "value": 1},
        "right": {
            "type": "binary_op",
            "op": "sub",
            "left": {"type": "var", "name": "value"},
            "right": {"type": "const", "value": 3},
        },
    }
    handle = step_pipeline(
        "bad expression",
        [
            step("graph.add_nodes", count=1, output="new", attributes={"score": 4}),
            step("core.load_node_attr", attr="score", target="values"),
            step(
                "core.map_nodes",
                source="values",
                target="values",
                expr=expression,
                async_update=asynchronous,
            ),
        ],
        atomic=True,
    )
    with pytest.raises(RuntimeError, match="zero"):
        Pipeline([handle]).run_durable(store)
    assert path.read_bytes() == before
    assert store.read_view().node_ids == [0, 1, 2]
    assert store.read_view().node_attr(0, "score") == 1
    store.close()
