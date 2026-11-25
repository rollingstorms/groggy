"""Owned aggregation values share memory/durable kernels and publication rules."""

import statistics

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
    if values:
        graph.writer().apply_batch(batch)
    return DurableGraph.create_from_graph(path, graph)


def pipeline(steps):
    return Pipeline([step_pipeline("aggregation", steps, atomic=True)])


CASES = [
    ("reduce_nodes", {"reducer": "sum"}, 10),
    ("reduce_nodes", {"reducer": "mean"}, 2.5),
    ("reduce_nodes", {"reducer": "min"}, 1),
    ("reduce_nodes", {"reducer": "max"}, 4),
    ("std", {}, statistics.stdev([1, 2, 3, 4])),
    ("median", {}, 2.5),
    ("quantile", {"q": 0.25}, 1.75),
    ("mode", {}, 1),
    ("entropy", {}, 2),
    ("histogram", {"bins": 2}, [2, 2]),
]


@pytest.mark.parametrize("kind,options,expected", CASES)
def test_aggregation_nested_body_selected_rows_recovery(
    tmp_path, kind, options, expected
):
    path = tmp_path / "aggregation.journal"
    store = fixture(path, [1, 999, 2, 3, 4])
    selected = store.read_view().select_nodes([0, 2, 3, 4])
    head = store.current_state_id
    p = pipeline(
        [
            step("core.load_node_attr", attr="score", target="values"),
            step(
                "iter.loop",
                iterations=2,
                body=[
                    step("core." + kind, source="values", target="stat", **options),
                    step(
                        "core.broadcast_scalar",
                        scalar="stat",
                        reference="values",
                        target="answer",
                    ),
                ],
            ),
            step(
                "core.attach_node_attr", source="answer", attr="answer", effective_at=30
            ),
        ]
    )
    result, profile = p.run_durable(store, selected, return_profile=True)
    for node in result.node_ids:
        actual = result.node_attr(node, "answer")
        assert (
            actual == expected
            if isinstance(expected, list)
            else actual == pytest.approx(expected)
        )
    assert store.read_view().node_attr(1, "answer") is None
    assert store.current_state_id == head + 1
    if kind == "histogram":
        counts = profile["outputs"]["histograms.stat"]
        assert counts == expected
        counts.clear()
        assert result.node_attr(0, "answer") == expected
    store.close()
    reopened = DurableGraph.open(path)
    assert reopened.read_view_at(29).node_attr(0, "answer") is None
    recovered = reopened.read_view_at(30).node_attr(0, "answer")
    assert (
        recovered == expected
        if isinstance(expected, list)
        else recovered == pytest.approx(expected)
    )
    reopened.close()


@pytest.mark.parametrize("kind,options,_", CASES)
def test_empty_owned_statistics_and_histograms_create_no_history(
    tmp_path, kind, options, _
):
    path = tmp_path / "empty.journal"
    store = fixture(path, [])
    before = path.read_bytes()
    p = pipeline(
        [
            step("core.load_node_attr", attr="score", target="values"),
            step("core." + kind, source="values", target="stat", **options),
        ]
    )
    result, profile = p.run_durable(store, return_profile=True)
    assert result.node_ids == []
    assert path.read_bytes() == before
    if kind == "histogram":
        assert profile["outputs"]["histograms.stat"] == [0, 0]
    store.close()


@pytest.mark.parametrize(
    "kind,values,expected",
    [
        ("std", [2**60, 2**60 + 1, 2**60 + 2], 1),
        ("median", [1e308, 1e308], 1e308),
        ("mode", [2**60 + 1, 2**60, 2**60, 2**60 + 1], 2**60 + 1),
        ("mode", ["red", "blue", "blue", "red"], "red"),
        ("entropy", ["red", "blue", "blue", "red"], 1),
    ],
)
def test_aggregation_preserves_precision_and_category_ties(
    tmp_path, kind, values, expected
):
    store = fixture(tmp_path / "precision.journal", values)
    p = pipeline(
        [
            step("core.load_node_attr", attr="score", target="values"),
            step("core." + kind, source="values", target="stat"),
            step(
                "core.broadcast_scalar",
                scalar="stat",
                reference="values",
                target="answer",
            ),
            step("core.attach_node_attr", source="answer", attr="answer"),
        ]
    )
    result = p.run_durable(store)
    actual = result.node_attr(0, "answer")
    assert actual == expected
    store.close()


@pytest.mark.parametrize(
    "kind,options",
    [
        ("reduce_nodes", {"reducer": "sum"}),
        ("std", {}),
        ("median", {}),
        ("quantile", {"q": 0.25}),
        ("histogram", {"bins": 2}),
    ],
)
def test_bad_numeric_input_discards_prefix_histogram_and_allocations(
    tmp_path, kind, options
):
    path = tmp_path / "reject.journal"
    store = fixture(path, [1, "bad"])
    before = path.read_bytes()
    p = pipeline(
        [
            step("graph.add_nodes", count=1, output="new"),
            step("core.init_nodes", target="good", value=1),
            step("core.histogram", source="good", target="discard", bins=2),
            step("core.load_node_attr", attr="score", target="values", default=1),
            step("core." + kind, source="values", target="stat", **options),
        ]
    )
    with pytest.raises(RuntimeError, match="finite numeric"):
        p.run_durable(store)
    assert path.read_bytes() == before
    assert store.read_view().node_ids == [0, 1]
    store.close()


def test_loaded_aggregation_input_stays_frozen_while_new_load_sees_pending_write(
    tmp_path,
):
    store = fixture(tmp_path / "pending.journal", [1, 2, 3, 4])
    head = store.current_state_id
    p = pipeline(
        [
            step("core.load_node_attr", attr="score", target="original"),
            step("core.init_nodes", target="replacement", value=9),
            step(
                "core.attach_node_attr",
                source="replacement",
                attr="score",
                effective_at=10,
            ),
            step("core.median", source="original", target="old_median"),
            step(
                "core.broadcast_scalar",
                scalar="old_median",
                reference="original",
                target="old",
            ),
            step(
                "core.attach_node_attr",
                source="old",
                attr="old_median",
                effective_at=20,
            ),
            step("core.load_node_attr", attr="score", target="updated"),
            step("core.median", source="updated", target="new_median"),
            step(
                "core.broadcast_scalar",
                scalar="new_median",
                reference="updated",
                target="new",
            ),
            step(
                "core.attach_node_attr",
                source="new",
                attr="new_median",
                effective_at=30,
            ),
        ]
    )
    result = p.run_durable(store)
    assert result.node_attr(0, "old_median") == 2.5
    assert result.node_attr(0, "new_median") == 9
    assert store.current_state_id == head + 1
    store.close()
