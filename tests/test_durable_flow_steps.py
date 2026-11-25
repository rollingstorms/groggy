"""Exact flow maps, immutable aliases and explicit property publication."""

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
    for source, target in [(0, 1), (0, 1), (1, 2), (2, 2)]:
        batch.add_edge(source, target)
    graph.writer().apply_batch(batch)
    return DurableGraph.create_from_graph(path, graph)


def pipeline(steps):
    return Pipeline([step_pipeline("flow and aliases", steps, atomic=True)])


@pytest.mark.parametrize("map_delta", [False, True])
@pytest.mark.parametrize(
    "initial,delta,capacity,expected,residual",
    [
        (2**60 + 1, 1, 2**60 + 10, 2**60 + 3, 7),
        (3.5, 1.25, 10, 6, 4),
    ],
)
def test_flow_and_residual_preserve_sparse_selection_saved_alias_and_clocks(
    tmp_path, map_delta, initial, delta, capacity, expected, residual
):
    path = tmp_path / "flow.journal"
    store = fixture(path)
    selected = store.read_view().select_edges([0, 3])
    head = store.current_state_id
    p = pipeline(
        [
            step("core.init_edges", target="flow", value=initial),
            step("alias", source="flow", target="saved"),
            step("core.init_edges", target="capacity", value=capacity),
            step(
                "core.init_edges" if map_delta else "core.init_scalar",
                target="delta",
                value=delta,
            ),
            step(
                "iter.loop",
                iterations=2,
                body=[
                    step("core.flow_update", flow="flow", delta="delta", target="flow")
                ],
            ),
            step(
                "core.residual_capacity",
                capacity="capacity",
                flow="flow",
                target="residual",
            ),
            step("core.attach_edge_attr", source="flow", attr="flow", effective_at=20),
            step(
                "core.attach_edge_attr", source="saved", attr="saved", effective_at=20
            ),
            step(
                "core.attach_edge_attr",
                source="residual",
                attr="residual",
                effective_at=30,
            ),
        ]
    )
    result = p.run_durable(store, selected)
    assert result.edge_ids == [0, 3]
    for edge in result.edge_ids:
        assert result.edge_attr(edge, "flow") == expected
        assert result.edge_attr(edge, "saved") == initial
        assert result.edge_attr(edge, "residual") == residual
    assert store.read_view().edge_attr(1, "flow") is None
    assert store.current_state_id == head + 1
    store.close()
    reopened = DurableGraph.open(path)
    assert reopened.read_view_at(19).edge_attr(0, "flow") is None
    assert reopened.read_view_at(30).edge_attr(0, "residual") == residual
    reopened.close()


@pytest.mark.parametrize("value", [7, "category", None, [1, 2, 3]])
def test_alias_preserves_owned_value_when_source_is_rebound(tmp_path, value):
    store = fixture(tmp_path / "alias.journal")
    p = pipeline(
        [
            step("core.init_nodes", target="original", value=value),
            step("alias", source="original", target="copy"),
            step("core.init_nodes", target="original", value=9),
            step("core.attach_node_attr", source="copy", attr="copy"),
            step("core.attach_node_attr", source="original", attr="original"),
        ]
    )
    result = p.run_durable(store)
    assert result.node_attr(0, "copy") == value
    assert result.node_attr(0, "original") == 9
    store.close()


def test_aliases_of_graph_and_table_keep_immutable_pin_after_owner_closes(tmp_path):
    store = fixture(tmp_path / "pins.journal")
    p = pipeline(
        [
            step("graph.capture_staged", output="original"),
            step("graph.nodes_table", source="original", output="table"),
            step("alias", source="original", target="view_copy"),
            step("alias", source="table", target="table_copy"),
            step("graph.add_nodes", count=1, output="new"),
            step("graph.emit", source="view_copy", name="view"),
            step("table.emit", source="table_copy", name="table"),
        ]
    )
    result, profile = p.run_durable(store, return_profile=True)
    assert result.node_ids == [0, 1, 2, 3]
    store.close()
    assert profile["outputs"]["view"].node_ids == [0, 1, 2]
    assert profile["outputs"]["table"].shape[0] == 3


@pytest.mark.parametrize(
    "suffix",
    [
        step("alias", source="missing", target="bad"),
        step("core.flow_update", flow="large", delta="one", target="bad"),
        step(
            "core.residual_capacity", capacity="negative", flow="one_map", target="bad"
        ),
    ],
)
def test_flow_failure_preserves_journal_ids_and_source(tmp_path, suffix):
    path = tmp_path / "reject.journal"
    store = fixture(path)
    before = path.read_bytes()
    p = pipeline(
        [
            step("graph.add_nodes", count=1, output="new"),
            step("core.init_edges", target="large", value=2**63 - 1),
            step("core.init_edges", target="negative", value=-(2**63)),
            step("core.init_edges", target="one_map", value=1),
            step("core.init_scalar", target="one", value=1),
            suffix,
        ]
    )
    with pytest.raises(RuntimeError):
        p.run_durable(store)
    assert path.read_bytes() == before
    assert store.read_view().node_ids == [0, 1, 2]
    store.close()
