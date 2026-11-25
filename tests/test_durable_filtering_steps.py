"""Selected filter masks and rank maps share the durable native read boundary."""

import pytest
from groggy import DurableGraph, Graph, GraphBatch, Pipeline
from groggy.builder import step_pipeline


def step(kind, **params):
    return {"id": kind, "params": params}


def fixture(path, values):
    graph = Graph()
    batch = GraphBatch(effective_at=0)
    for value in values:
        batch.add_node(score=value)
    graph.writer().apply_batch(batch)
    return DurableGraph.create_from_graph(path, graph)


def pipeline(steps):
    return Pipeline([step_pipeline("filters and ranks", steps, atomic=True)])


@pytest.mark.parametrize(
    "order,ordered", [("asc", [3, 0, 1, 2]), ("desc", [0, 1, 3, 2])]
)
@pytest.mark.parametrize("k", [0, 1, 10])
def test_sort_and_topk_return_stable_ranks_with_nulls_last(tmp_path, order, ordered, k):
    path = tmp_path / "ranks.journal"
    store = fixture(path, [3, 3, None, 1])
    head = store.current_state_id
    p = pipeline(
        [
            step("core.load_node_attr", attr="score", target="values"),
            step("core.sort_nodes_by_attr", attr="score", order=order, target="rank"),
            step("core.top_k", source="values", order=order, k=k, target="top"),
            step("core.attach_node_attr", source="rank", attr="rank", effective_at=20),
            step("core.attach_node_attr", source="top", attr="top", effective_at=20),
        ]
    )
    result = p.run_durable(store)
    for rank, node in enumerate(ordered):
        assert result.node_attr(node, "rank") == rank
        assert result.node_attr(node, "top") == (rank if rank < k else None)
    assert store.current_state_id == head + 1
    store.close()
    reopened = DurableGraph.open(path)
    assert reopened.read_view_at(19).node_attr(0, "rank") is None
    assert reopened.read_view_at(20).node_attr(ordered[0], "rank") == 0
    reopened.close()


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("eq", [1]),
        ("ne", [0, 2, 3]),
        ("lt", [0]),
        ("le", [0, 1]),
        ("gt", [2]),
        ("ge", [1, 2]),
    ],
)
def test_typed_comparison_filter_marks_rows_without_removing_entities(
    tmp_path, kind, expected
):
    store = fixture(tmp_path / "predicate.journal", [1, 2, 3, None])
    p = pipeline(
        [
            step(
                "core.filter_nodes_by_attr",
                attr="score",
                target="mask",
                predicate={"type": kind, "value": {"type": "Int", "value": 2}},
            ),
            step("core.attach_node_attr", source="mask", attr="selected"),
        ]
    )
    result = p.run_durable(store)
    assert result.node_ids == [0, 1, 2, 3]
    assert [
        n for n in result.node_ids if result.node_attr(n, "selected") == 1
    ] == expected
    store.close()


@pytest.mark.parametrize(
    "values,predicate,expected",
    [
        ([1, 2, 3], {"type": "range", "min": 1.5, "max": 2.5}, [1]),
        (["red", "blue", "dark red"], {"type": "contains", "substring": "red"}, [0, 2]),
        ([1, 2, 3], {"type": "gt", "value": {"type": "Int", "value": 100}}, []),
        (
            [2**53, 2**53 + 1],
            {"type": "gt", "value": {"type": "Float", "value": float(2**53)}},
            [1],
        ),
    ],
)
def test_range_contains_empty_and_exact_mixed_numeric_filters(
    tmp_path, values, predicate, expected
):
    path = tmp_path / "domains.journal"
    store = fixture(path, values)
    head = store.current_state_id
    p = pipeline(
        [
            step(
                "core.filter_nodes_by_attr",
                attr="score",
                target="mask",
                predicate=predicate,
            ),
            step("core.attach_node_attr", source="mask", attr="selected"),
        ]
    )
    result = p.run_durable(store)
    assert [
        n for n in result.node_ids if result.node_attr(n, "selected") == 1
    ] == expected
    assert store.current_state_id == head + bool(expected)
    store.close()


def test_edge_filter_reads_pending_weights_and_keeps_selected_instances(tmp_path):
    path = tmp_path / "edges.journal"
    graph = Graph(directed=True)
    batch = GraphBatch(effective_at=0)
    for _ in range(3):
        batch.add_node()
    for source, target in [(0, 1), (0, 1), (1, 2), (2, 2)]:
        batch.add_edge(source, target, weight=0)
    graph.writer().apply_batch(batch)
    store = DurableGraph.create_from_graph(path, graph)
    selected = store.read_view().select_edges([0, 3])
    p = pipeline(
        [
            step("core.init_edges", target="weights", value=2**53 + 1),
            step(
                "core.attach_edge_attr",
                source="weights",
                attr="weight",
                effective_at=10,
            ),
            step(
                "core.filter_edges_by_attr",
                attr="weight",
                target="mask",
                predicate={
                    "type": "gt",
                    "value": {"type": "Float", "value": float(2**53)},
                },
            ),
            step(
                "core.attach_edge_attr", source="mask", attr="selected", effective_at=20
            ),
        ]
    )
    result = p.run_durable(store, selected)
    assert result.edge_ids == [0, 3]
    assert [result.edge_attr(e, "selected") for e in result.edge_ids] == [1, 1]
    assert store.read_view().edge_attr(1, "selected") is None
    store.close()


@pytest.mark.parametrize(
    "operation",
    [
        step("core.sort_nodes_by_attr", attr="score", target="bad"),
        step("core.top_k", source="values", k=0, target="bad"),
    ],
)
def test_invalid_sort_domain_rejects_before_even_zero_limit_can_hide_bad_input(
    tmp_path, operation
):
    path = tmp_path / "reject.journal"
    store = fixture(path, [1, "bad"])
    before = path.read_bytes()
    p = pipeline(
        [
            step("graph.add_nodes", count=1, output="new", attributes={"score": 0}),
            step("core.load_node_attr", attr="score", target="values"),
            operation,
        ]
    )
    with pytest.raises(RuntimeError, match="cannot mix"):
        p.run_durable(store)
    assert path.read_bytes() == before
    assert store.read_view().node_ids == [0, 1]
    store.close()
