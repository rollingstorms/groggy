"""Message-passing bodies use the same selected native kernel on durable graphs."""

import pytest
from groggy import DurableGraph, Graph, GraphBatch, Pipeline
from groggy.builder import AlgorithmBuilder, step_pipeline


def step(kind, **params):
    return {"id": kind, "params": params}


def fixture(path, scores, edges):
    graph = Graph(directed=True)
    batch = GraphBatch(effective_at=0)
    for score in scores:
        batch.add_node(score=score)
    for source, target in edges:
        batch.add_edge(source, target)
    graph.writer().apply_batch(batch)
    return DurableGraph.create_from_graph(path, graph)


@pytest.mark.parametrize("direction", ["in", "out", "undirected"])
@pytest.mark.parametrize("ordered", [False, True])
@pytest.mark.parametrize("include_self", [False, True])
def test_block_matches_independent_two_round_incidence_updates(
    tmp_path, direction, ordered, include_self
):
    edges = [(0, 1), (0, 1), (1, 2), (2, 2)]
    path = tmp_path / "block.journal"
    store = fixture(path, [1, 2, 3], edges)
    expected = [1, 2, 3]
    for _ in range(2):
        before, after = expected[:], []
        for node in range(3):
            neighbors = []
            for source, target in edges:
                if direction in ("out", "undirected") and source == node:
                    neighbors.append(target)
                if direction in ("in", "undirected") and target == node:
                    if direction != "undirected" or source != target:
                        neighbors.append(source)
            if include_self and node not in neighbors:
                neighbors.append(node)
            source_values = expected if ordered else before
            value = sum(source_values[other] for other in sorted(neighbors))
            if ordered:
                expected[node] = value
            else:
                after.append(value)
        if not ordered:
            expected = after
    handle = step_pipeline(
        "message block",
        [
            step("core.load_node_attr", attr="score", target="values"),
            step(
                "iter.loop",
                iterations=2,
                body=[
                    step(
                        "core.execution_block",
                        mode="message_pass",
                        target="values",
                        options={
                            "ordered": ordered,
                            "include_self": include_self,
                            "direction": direction,
                        },
                        body={
                            "nodes": [
                                {
                                    "id": "sum",
                                    "domain": "core",
                                    "op_type": "neighbor_agg",
                                    "inputs": ["values"],
                                    "output": "sum",
                                    "metadata": {"agg": "sum"},
                                }
                            ]
                        },
                    )
                ],
            ),
            step(
                "core.attach_node_attr", source="values", attr="answer", effective_at=30
            ),
        ],
        atomic=True,
    )
    head = store.current_state_id
    result = Pipeline([handle]).run_durable(store)
    assert [result.node_attr(n, "answer") for n in result.node_ids] == expected
    assert store.current_state_id == head + 1
    store.close()
    reopened = DurableGraph.open(path)
    assert reopened.read_view_at(29).node_attr(0, "answer") is None
    assert reopened.read_view_at(30).node_attr(0, "answer") == expected[0]
    reopened.close()


@pytest.mark.parametrize("ordered,expected", [(True, [0, 0, 0]), (False, [0, 1, 2])])
def test_public_message_builder_runs_durably_with_existing_visibility(
    tmp_path, ordered, expected
):
    store = fixture(tmp_path / "public.journal", [1, 2, 3], [(0, 1), (1, 2)])
    builder = AlgorithmBuilder("durable message visibility")
    values = builder.load_attr("score")
    with builder.message_pass(
        target=values, ordered=ordered, include_self=False, direction="in"
    ) as messages:
        messages.apply(builder.graph_ops.neighbor_agg(values, "sum"))
    builder.attach_as("answer", values)
    result = Pipeline([builder.build(atomic=True, optimize=False)]).run_durable(store)
    assert [result.node_attr(n, "answer") for n in result.node_ids] == expected
    store.close()


@pytest.mark.parametrize(
    "operator,expected",
    [
        ("eq", [-10, 10, -10]),
        ("ne", [10, -10, 10]),
        ("lt", [10, -10, -10]),
        ("le", [10, 10, -10]),
        ("gt", [-10, -10, 10]),
        ("ge", [-10, 10, 10]),
    ],
)
def test_public_conditional_builder_preserves_checked_block_operations(
    tmp_path, operator, expected
):
    store = fixture(tmp_path / "conditional.journal", [-1, 0, 2], [])
    builder = AlgorithmBuilder("durable conditional")
    values = builder.load_attr("score")
    with builder.message_pass(target=values) as messages:
        mask = builder.core.compare(values, operator, 0)
        messages.apply(builder.core.where(mask, 10, -10))
    builder.attach_as("answer", values)
    result = Pipeline([builder.build(atomic=True, optimize=False)]).run_durable(store)
    assert [result.node_attr(n, "answer") for n in result.node_ids] == expected
    store.close()


def test_public_mode_builder_preserves_neighbor_collection_and_exact_labels(tmp_path):
    labels = [2**60 + 1, 2**60, 2**60]
    store = fixture(tmp_path / "labels.journal", labels, [(0, 1), (0, 2)])
    builder = AlgorithmBuilder("durable labels")
    values = builder.load_attr("score")
    with builder.message_pass(
        target=values, include_self=True, direction="out"
    ) as messages:
        messages.apply(builder.core.mode(messages.pull(values), tie_break="lowest"))
    builder.attach_as("answer", values)
    result = Pipeline([builder.build(atomic=True, optimize=False)]).run_durable(store)
    assert [result.node_attr(n, "answer") for n in result.node_ids] == [2**60] * 3
    store.close()


@pytest.mark.parametrize(
    "mode,options,nodes",
    [
        ("streaming", {}, []),
        ("message_pass", {"direction": "invalid"}, []),
        (
            "message_pass",
            {},
            [
                {
                    "id": "bad",
                    "domain": "core",
                    "op_type": "unsupported",
                    "inputs": [],
                    "output": "bad",
                }
            ],
        ),
    ],
)
def test_unsupported_block_preflight_rejects_before_prefix_even_at_zero(
    tmp_path, mode, options, nodes
):
    path = tmp_path / "reject.journal"
    store = fixture(path, [1], [])
    before = path.read_bytes()
    handle = step_pipeline(
        "unsupported",
        [
            step("graph.add_nodes", count=1, output="new"),
            step(
                "iter.loop",
                iterations=0,
                body=[
                    step(
                        "core.execution_block",
                        mode=mode,
                        target="values",
                        options=options,
                        body={"nodes": nodes},
                    )
                ],
            ),
        ],
        atomic=True,
    )
    with pytest.raises(RuntimeError, match="no detached native kernel|cannot run after staged topology"):
        Pipeline([handle]).run_durable(store)
    assert path.read_bytes() == before
    assert store.read_view().node_ids == [0]
    store.close()
