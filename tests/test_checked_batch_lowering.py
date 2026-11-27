"""Checked legacy-plan lowering through the public Python durable facade."""

import pytest

from groggy import DurableGraph, Graph, GraphBatch, Pipeline
from groggy.builder import step_pipeline


def step(kind, **params):
    return {"id": kind, "params": params}


def loop(instructions, slots=3, iterations=1, carried=()):
    return step(
        "iter.loop",
        body=[],
        batch_plan={
            "instructions": instructions,
            "slot_count": slots,
            "carried_slots": list(carried),
        },
        iterations=iterations,
    )


def pipeline(steps):
    return Pipeline([step_pipeline("checked numeric plan", steps, atomic=True)])


def run(store, steps):
    return pipeline(steps).run_durable(store)


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("divisor", [1e-20, -1e-20, 1e-300])
def test_checked_plan_reads_pending_changes_and_keeps_tiny_nonzero_divisors(
    tmp_path, directed, divisor
):
    graph = Graph(directed=directed)
    initial = GraphBatch(effective_at=0)
    initial.add_node()
    graph.writer().apply_batch(initial)
    path = tmp_path / "numeric.journal"
    store = DurableGraph.create_from_graph(path, graph)
    retained = store.read_view()
    head = store.current_state_id
    result = run(
        store,
        [
            step("core.init_nodes", target="prefix", value=divisor),
            step("core.attach_node_attr", source="prefix", attr="x", effective_at=10),
            loop(
                [
                    {
                        "type": "load_node_attr",
                        "dst": 0,
                        "attr_name": "x",
                        "default": 0.0,
                    },
                    {"type": "load_scalar", "dst": 1, "value": divisor},
                    {"type": "div", "dst": 0, "lhs": 0, "rhs": 1},
                    {"type": "store_node_prop", "src": 0, "var_name": "answer"},
                ],
                iterations=2,
            ),
            step(
                "core.attach_node_attr", source="answer", attr="answer", effective_at=20
            ),
        ],
    )
    assert result.node_attr(0, "answer") == 1.0
    assert result.write_state_id == head + 1 == store.current_state_id
    assert retained.node_attr(0, "x") is None
    store.close()
    reopened = DurableGraph.open(path)
    assert reopened.read_view_at(15).node_attr(0, "x") == divisor
    assert reopened.read_view_at(15).node_attr(0, "answer") is None
    assert reopened.read_view_at(20).node_attr(0, "answer") == 1.0


@pytest.mark.parametrize("iterations", [0, 1])
@pytest.mark.parametrize(
    "plan",
    [
        {},
        {
            "instructions": [{"type": "load_scalar", "dst": 2, "value": 1.0}],
            "slot_count": 1,
            "carried_slots": [],
        },
        {"instructions": [], "slot_count": 2, "carried_slots": [[0, 1], [1, 1]]},
        {
            "instructions": [
                {"type": "clip", "dst": 0, "src": 0, "min": 2.0, "max": 1.0}
            ],
            "slot_count": 1,
            "carried_slots": [],
        },
        {
            "instructions": [
                {
                    "type": "load_edge_attr",
                    "dst": 0,
                    "attr_name": "weight",
                    "default": 0.0,
                }
            ],
            "slot_count": 1,
            "carried_slots": [],
        },
    ],
)
def test_bad_or_unported_plan_is_rejected_before_an_atomic_prefix(
    tmp_path, iterations, plan
):
    graph = Graph()
    graph.add_node()
    path = tmp_path / "rejected.journal"
    store = DurableGraph.create_from_graph(path, graph)
    before = path.read_bytes()
    head = store.current_state_id
    with pytest.raises(RuntimeError):
        run(
            store,
            [
                step("core.init_nodes", target="prefix", value=9),
                step("core.attach_node_attr", source="prefix", attr="prefix"),
                step("iter.loop", body=[], batch_plan=plan, iterations=iterations),
            ],
        )
    assert store.current_state_id == head
    assert path.read_bytes() == before
    assert store.read_view().node_attr(0, "prefix") is None


@pytest.mark.parametrize("divisor", [0.0, -0.0])
def test_zero_division_rolls_back_prefix_and_does_not_publish(tmp_path, divisor):
    graph = Graph()
    graph.add_node()
    path = tmp_path / "failed.journal"
    store = DurableGraph.create_from_graph(path, graph)
    before = path.read_bytes()
    with pytest.raises(RuntimeError, match="division by zero"):
        run(
            store,
            [
                step("core.init_nodes", target="prefix", value=9),
                step("core.attach_node_attr", source="prefix", attr="prefix"),
                loop(
                    [
                        {"type": "init_nodes", "dst": 0, "value": 1.0},
                        {"type": "load_scalar", "dst": 1, "value": divisor},
                        {"type": "div", "dst": 0, "lhs": 0, "rhs": 1},
                    ]
                ),
            ],
        )
    assert path.read_bytes() == before
    assert store.read_view().node_attr(0, "prefix") is None


def test_register_names_and_large_integer_values_are_preserved(tmp_path):
    graph = Graph()
    graph.add_node()
    store = DurableGraph.create_from_graph(tmp_path / "exact.journal", graph)
    exact = 9_007_199_254_740_993
    result = run(
        store,
        [
            step("core.init_nodes", target="slot/0", value=exact),
            step("core.init_nodes", target="low", value=exact - 1),
            loop(
                [
                    {"type": "load_node_prop", "dst": 0, "var_name": "slot/0"},
                    {"type": "load_node_prop", "dst": 1, "var_name": "low"},
                    {"type": "compare", "dst": 2, "lhs": 0, "rhs": 1, "op": "gt"},
                    {"type": "store_node_prop", "src": 2, "var_name": "comparison"},
                    {"type": "abs", "dst": 0, "src": 0},
                    {"type": "store_node_prop", "src": 0, "var_name": "answer"},
                ]
            ),
            step("core.attach_node_attr", source="answer", attr="answer"),
            step("core.attach_node_attr", source="comparison", attr="comparison"),
            step("core.attach_node_attr", source="slot/0", attr="unchanged"),
        ],
    )
    assert result.node_attr(0, "answer") == exact
    assert isinstance(result.node_attr(0, "answer"), int)
    assert result.node_attr(0, "unchanged") == exact
    assert result.node_attr(0, "comparison") == 1.0


@pytest.mark.parametrize("iterations", [0, -1])
def test_empty_loop_validation_has_no_surprising_publication(tmp_path, iterations):
    graph = Graph()
    graph.add_node()
    store = DurableGraph.create_from_graph(tmp_path / "empty.journal", graph)
    head = store.current_state_id
    if iterations < 0:
        with pytest.raises((ValueError, RuntimeError), match="non-negative"):
            run(store, [loop([], slots=0, iterations=iterations)])
    else:
        result = run(store, [loop([], slots=0, iterations=iterations)])
        assert result.node_ids == [0]
        assert result.write_state_id == head
    assert store.current_state_id == head


def test_public_algorithm_builder_uses_shared_typed_math_without_optimization(tmp_path):
    from groggy import AlgorithmBuilder

    graph = Graph()
    graph.add_node()
    store = DurableGraph.create_from_graph(tmp_path / "builder.journal", graph)
    exact = 9_007_199_254_740_993
    builder = AlgorithmBuilder("typed durable arithmetic")
    values = builder.init_nodes(default=exact)
    builder.attach_as("answer", values * 2 + 1)
    handle = builder.build(atomic=True, optimize=False)
    result = Pipeline([handle]).run_durable(store)
    assert result.node_attr(0, "answer") == exact * 2 + 1
    assert isinstance(result.node_attr(0, "answer"), int)


def test_list_intermediates_use_the_same_native_mode_kernel(tmp_path):
    graph = Graph()
    graph.add_node(choices=[5, 5, 7])
    store = DurableGraph.create_from_graph(tmp_path / "lists.journal", graph)
    result = run(
        store,
        [
            loop(
                [
                    {
                        "type": "load_node_attr",
                        "dst": 0,
                        "attr_name": "choices",
                        "default": 0.0,
                    },
                    {"type": "mode_list", "dst": 1, "src": 0, "tie_break": "lowest"},
                    {"type": "store_node_prop", "src": 1, "var_name": "answer"},
                ],
                slots=2,
            ),
            step("core.attach_node_attr", source="answer", attr="answer"),
        ],
    )
    assert result.node_attr(0, "answer") == 5
    assert isinstance(result.node_attr(0, "answer"), int)


@pytest.mark.parametrize(
    "direction,expected",
    [("in", [5.0, 9.0]), ("out", [10.0, 7.0]), ("undirected", [15.0, 11.0])],
)
@pytest.mark.parametrize("directed", [False, True])
def test_neighbor_plans_keep_direction_multiplicity_and_pending_values(
    tmp_path, direction, expected, directed
):
    graph = Graph(directed=directed)
    batch = GraphBatch(effective_at=0)
    batch.add_node()
    batch.add_node()
    batch.add_edge(0, 1)
    batch.add_edge(0, 1)
    batch.add_edge(1, 1)
    batch.add_edge(1, 0)
    graph.writer().apply_batch(batch)
    graph.writer().set_node_attr(0, "x", 2, effective_at=0)
    graph.writer().set_node_attr(1, "x", 5, effective_at=0)
    path = tmp_path / "neighbors.journal"
    store = DurableGraph.create_from_graph(path, graph)
    before = store.read_view()
    head = store.current_state_id
    instructions = [
        {"type": "load_node_attr", "dst": 0, "attr_name": "x", "default": 0.0},
        {
            "type": "neighbor_aggregate",
            "src": 0,
            "dst": 0,
            "operation": "sum",
            "direction": direction,
        },
        {"type": "store_node_prop", "src": 0, "var_name": "answer"},
    ]
    p = pipeline(
        [
            {
                "id": "iter.loop",
                "params": {
                    "iterations": 2,
                    "body": [],
                    "batch_plan": {
                        "instructions": instructions,
                        "slot_count": 1,
                        "carried_slots": [],
                    },
                },
            },
            {
                "id": "core.attach_node_attr",
                "params": {"source": "answer", "attr": "answer", "effective_at": 20},
            },
        ]
    )
    result = p.run_durable(store)
    assert result.write_state_id == head + 1
    assert [result.node_attr(n, "answer") for n in range(2)] == expected
    assert before.node_attr(0, "answer") is None
    store.close()
    recovered = DurableGraph.open(path)
    assert [recovered.read_view().node_attr(n, "answer") for n in range(2)] == expected
    assert recovered.read_view_at(19).node_attr(0, "answer") is None
    recovered.close()


@pytest.mark.parametrize(
    "tie,expected",
    [
        ("lowest", 9007199254740992),
        ("highest", 9007199254740993),
        ("first", 9007199254740993),
    ],
)
def test_neighbor_mode_ties_keep_exact_integer_labels(tmp_path, tie, expected):
    graph = Graph(directed=True)
    graph.add_node(label=9007199254740993)
    graph.add_node(label=9007199254740992)
    graph.add_edge(0, 0)
    graph.add_edge(0, 1)
    graph.add_edge(1, 0)
    path = tmp_path / "labels.journal"
    store = DurableGraph.create_from_graph(path, graph)
    instructions = [
        {"type": "load_node_attr", "dst": 0, "attr_name": "label", "default": 0.0},
        {
            "type": "neighbor_mode",
            "src": 0,
            "dst": 0,
            "tie_break": tie,
            "direction": "out",
        },
        {"type": "store_node_prop", "src": 0, "var_name": "answer"},
    ]
    result = pipeline(
        [
            {
                "id": "iter.loop",
                "params": {
                    "iterations": 1,
                    "body": [],
                    "batch_plan": {
                        "instructions": instructions,
                        "slot_count": 1,
                        "carried_slots": [],
                    },
                },
            },
            {
                "id": "core.attach_node_attr",
                "params": {"source": "answer", "attr": "answer"},
            },
        ]
    ).run_durable(store)
    assert result.node_attr(0, "answer") == expected
    assert isinstance(result.node_attr(0, "answer"), int)
    store.close()


@pytest.mark.parametrize("kind", ["fused_madd", "fused_axpy"])
def test_python_fused_plan_spellings_and_overflow_rollback(tmp_path, kind):
    graph = Graph()
    graph.add_node()
    path = tmp_path / "fused.journal"
    store = DurableGraph.create_from_graph(path, graph)
    original = path.read_bytes()
    op = {"type": kind, "dst": 0}
    if kind == "fused_madd":
        op.update(a=0, b=1, c=2)
    else:
        op.update(alpha=1, x=0, y=2)
    instructions = [
        {"type": "init_nodes", "dst": 0, "value": 1e308},
        {"type": "init_nodes", "dst": 1, "value": 1e308},
        {"type": "init_nodes", "dst": 2, "value": 0.0},
        op,
    ]
    p = pipeline(
        [
            {"id": "graph.add_nodes", "params": {"count": 1, "output": "new"}},
            {
                "id": "iter.loop",
                "params": {
                    "iterations": 1,
                    "body": [],
                    "batch_plan": {
                        "instructions": instructions,
                        "slot_count": 3,
                        "carried_slots": [],
                    },
                },
            },
        ]
    )
    with pytest.raises(RuntimeError, match="not finite"):
        p.run_durable(store)
    assert path.read_bytes() == original
    assert store.read_view().node_ids == [0]
    store.close()


@pytest.mark.parametrize(
    "expression,expected", [("value * 2", 15.0), ("2", 3.0), ("value", 5.0)]
)
def test_compiled_map_expression_binds_its_named_register_for_following_steps(
    tmp_path, expression, expected
):
    from groggy.builder.ir.batch import compile_loop_to_batch_plan

    body = [
        {
            "type": "map_nodes",
            "fn": expression,
            "inputs": {"value": "values"},
            "output": "mapped",
        },
        {"type": "core.constant", "value": 1.0, "output": "one"},
        {"type": "core.add", "a": "mapped", "b": "one", "output": "next"},
        {"type": "alias", "source": "next", "target": "values"},
    ]
    compiled = compile_loop_to_batch_plan(body)
    assert compiled is not None
    graph = Graph()
    graph.add_node()
    store = DurableGraph.create_from_graph(tmp_path / "expressions.journal", graph)
    result = run(
        store,
        [
            step("core.init_nodes", target="values", value=3.0),
            step("iter.loop", body=[], batch_plan=compiled, iterations=2),
            step("core.attach_node_attr", source="values", attr="answer"),
        ],
    )
    assert result.node_attr(0, "answer") == expected
    store.close()


@pytest.mark.parametrize("method", ["min", "mean"])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
@pytest.mark.parametrize("layout", ["plan", "nodes", "edges"])
def test_signed_minimum_and_mean_match_independent_oracle_and_recover(
    tmp_path, method, sign, layout
):
    path = tmp_path / "signed-normalize.journal"
    graph = Graph(directed=True)
    graph.add_nodes(2)
    graph.add_edge(0, 1)
    graph.add_edge(1, 1)
    setter = (
        graph.writer().set_edge_attr
        if layout == "edges"
        else graph.writer().set_node_attr
    )
    setter(0, "input", sign * 2.0, effective_at=0)
    setter(1, "input", sign * 4.0, effective_at=0)
    store = DurableGraph.create_from_graph(path, graph)
    load = "core.load_edge_attr" if layout == "edges" else "core.load_node_attr"
    attach = "core.attach_edge_attr" if layout == "edges" else "core.attach_node_attr"
    steps = [step(load, attr="input", target="input")]
    divisor = sign * (3.0 if method == "mean" else (2.0 if sign > 0 else 4.0))
    expected = [sign * 2.0 / divisor, sign * 4.0 / divisor]
    if layout == "plan":
        steps.append(
            loop(
                [
                    {"type": "load_node_prop", "dst": 0, "var_name": "input"},
                    {
                        "type": "normalize",
                        "dst": 0,
                        "src": 0,
                        "method": method,
                        "epsilon": 0.0,
                    },
                    {"type": "store_node_prop", "src": 0, "var_name": "answer"},
                ],
                slots=1,
                iterations=2,
            )
        )
    else:
        steps.append(
            step(
                "core.normalize_values",
                source="input",
                target="answer",
                method=method,
                epsilon=0.0,
            )
        )
    steps += [step(attach, source="answer", attr="answer", effective_at=20)]
    retained = store.read_view()
    result = run(store, steps)
    getter = result.edge_attr if layout == "edges" else result.node_attr
    assert [getter(i, "answer") for i in [0, 1]] == pytest.approx(expected)
    assert retained.node_attr(0, "answer") is None
    store.close()
    reopened = DurableGraph.open(path)
    prior, recovered = reopened.read_view_at(19), reopened.read_view_at(20)
    assert (prior.edge_attr if layout == "edges" else prior.node_attr)(
        0, "answer"
    ) is None
    getter = recovered.edge_attr if layout == "edges" else recovered.node_attr
    assert [getter(i, "answer") for i in [0, 1]] == pytest.approx(expected)
    reopened.close()


@pytest.mark.parametrize("method", ["min", "mean"])
@pytest.mark.parametrize("value,epsilon", [(0.0, 0.0), (1e-30, 1e-20)])
def test_checked_minimum_mean_rejection_rolls_back_pending_writes(
    tmp_path, method, value, epsilon
):
    path = tmp_path / "zero-normalize.journal"
    graph = Graph()
    graph.add_node()
    store = DurableGraph.create_from_graph(path, graph)
    before, head = path.read_bytes(), store.current_state_id
    with pytest.raises(RuntimeError, match="epsilon"):
        run(
            store,
            [
                step("core.init_nodes", target="prefix", value=7),
                step("core.attach_node_attr", source="prefix", attr="prefix"),
                loop(
                    [
                        {"type": "init_nodes", "dst": 0, "value": value},
                        {
                            "type": "normalize",
                            "dst": 0,
                            "src": 0,
                            "method": method,
                            "epsilon": epsilon,
                        },
                    ],
                    slots=1,
                ),
            ],
        )
    assert path.read_bytes() == before
    assert store.current_state_id == head
    assert store.read_view().node_attr(0, "prefix") is None
    store.close()
