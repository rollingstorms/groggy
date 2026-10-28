"""Property timelines preserve exact values, effective clocks and active lifetimes."""
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
import math
import struct

import pytest
from groggy import DurableGraph, Graph, GraphBatch


def create(graph):
    batch = GraphBatch(effective_at=10)
    a, b = batch.add_node(x=1), batch.add_node()
    batch.add_edge(a, b, x=1)
    return graph.writer().apply_batch(batch)["state_id"]


def assign(graph, at, value):
    batch = GraphBatch(effective_at=at)
    batch.set_node_attr(0, "x", value)
    batch.set_edge_attr(0, "x", value)
    return graph.writer().apply_batch(batch)["state_id"]


def row(start, end, value=None, present=True):
    return dict(start=start, end=end, present=present, value=value)


@pytest.mark.parametrize("restored_at", [40, 50])
def test_property_intervals_match_independent_rows_and_point_reads(restored_at):
    graph = Graph()
    first = create(graph)
    assign(graph, 30, 3)
    assign(graph, 20, 2)
    assign(graph, 20, None)
    batch = GraphBatch(effective_at=25)
    batch.unset_node_attr(0, "x")
    batch.unset_edge_attr(0, "x")
    graph.writer().apply_batch(batch)
    graph.writer().remove_nodes([0], effective_at=40)
    old = graph.build_effective_index()
    graph.restore_state(first, mode="tracked", effective_at=restored_at,
                        expected_state=graph.current_state_id)
    index = graph.build_effective_index()
    expected = [row(10, 20, 1), row(20, 25), row(25, 30, present=False)]
    expected += ([row(30, 40, 3), row(40, None, 1)] if restored_at == 40
                 else [row(30, 40, 3), row(50, None, 1)])
    for getter in [index.node_property_intervals, index.edge_property_intervals]:
        assert getter(0, "x") == expected
        assert getter(999, "x") == []
        for at in [9, 10, 19, 20, 24, 25, 29, 30, 39, 40, 49, 50, 2**63 - 1]:
            view = index.read_view_at(at)
            actual = [r for r in expected if r["start"] <= at
                      and (r["end"] is None or at < r["end"])]
            assert bool(actual) == (0 in view.node_ids)
            if actual:
                assert actual[0]["value"] == view.node_attr(0, "x")
    assert old.node_property_intervals(0, "x")[-1] == row(30, 40, 3)
    cutoff = graph.publication_metadata(first)["recorded_at"]
    assert index.node_property_intervals(0, "x", recorded_by=cutoff) == [row(10, None, 1)]
    assert index.edge_property_intervals(0, "x", recorded_by=-(2**63)) == []
    expected[0]["value"] = 999
    assert index.node_property_intervals(0, "x")[0]["value"] == 1


@pytest.mark.parametrize("value", [
    True, 2**53 + 1, 1.25, b"binary", [1, 2, 3], [1.25, 2.5],
    {"nested": [1, None]}, date(2024, 1, 1),
    datetime(2024, 1, 1, tzinfo=timezone.utc), timedelta(microseconds=23),
])
def test_property_intervals_preserve_typed_values_and_owned_outputs(value):
    graph = Graph()
    create(graph)
    assign(graph, 20, value)
    index = graph.build_effective_index()
    for getter in [index.node_property_intervals, index.edge_property_intervals]:
        result = getter(0, "x")
        assert result[-1] == row(20, None, value)
        assert type(result[-1]["value"]) is type(value)
        if isinstance(value, list):
            result[-1]["value"].clear()
            assert getter(0, "x")[-1]["value"] == value


def test_property_intervals_preserve_signed_zero_and_nan_payloads():
    graph = Graph()
    create(graph)
    values = [0., -0., struct.unpack("d", struct.pack("Q", 0x7ff8000000000001))[0],
              struct.unpack("d", struct.pack("Q", 0x7ff8000000000002))[0]]
    for at, value in enumerate(values, 20):
        assign(graph, at, value)
    rows = graph.build_effective_index().node_property_intervals(0, "x")
    assert len(rows) == 5
    assert [struct.pack("d", r["value"]) for r in rows[1:]] == [struct.pack("d", v) for v in values]
    assert math.isnan(rows[-1]["value"])


def test_property_intervals_skip_inactive_assignments_and_handle_extreme_bounds():
    graph = Graph()
    batch = GraphBatch(effective_at=10)
    a = batch.add_node()
    batch.set_node_attr(a, "x", 7, effective_at=-(2**63))
    graph.writer().apply_batch(batch)
    index = graph.build_effective_index()
    assert index.node_property_intervals(0, "x") == [row(10, None, present=False)]
    batch = GraphBatch(effective_at=2**63 - 1)
    batch.set_node_attr(0, "x", 8)
    graph.writer().apply_batch(batch)
    assert graph.build_effective_index().node_property_intervals(0, "x") == [
        row(10, 2**63 - 1, present=False), row(2**63 - 1, None, 8)]
    assert index.node_property_intervals(0, "unknown") == [row(10, None, present=False)]


def test_property_intervals_retain_durable_and_legacy_captures(tmp_path):
    path = tmp_path / "intervals.ggj"
    with DurableGraph.create(path) as graph:
        create(graph)
        index = graph.build_effective_index()
        assign(graph, 20, [3, 4])
        current = graph.build_effective_index()
    with DurableGraph.open(path) as graph:
        reopened = graph.build_effective_index()
        assert reopened.edge_property_intervals(0, "x") == current.edge_property_intervals(0, "x")
    assert index.node_property_intervals(0, "x") == [row(10, None, 1)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: current.node_property_intervals(0, "x"), range(8)))
    assert all(r == results[0] for r in results)
    graph = Graph(manual=True)
    graph.add_node(x=9)
    state = graph.commit("legacy", "test")
    early = graph.build_effective_index(legacy_times={state: (10, 100)})
    late = graph.build_effective_index(legacy_times={state: (20, 200)})
    assert early.node_property_intervals(0, "x") == [row(10, None, 9)]
    assert late.node_property_intervals(0, "x") == [row(20, None, 9)]
    assert late.node_property_intervals(0, "x", recorded_by=199) == []


@pytest.mark.parametrize("bad", [True, -1, 1.5, "0", 2**100])
def test_property_interval_ids_reject_invalid_types_and_ranges(bad):
    index = Graph().build_effective_index()
    for getter in [index.node_property_intervals, index.edge_property_intervals]:
        with pytest.raises((TypeError, OverflowError)):
            getter(bad, "x")
        with pytest.raises(TypeError):
            getter(0, None)


@pytest.mark.parametrize("bad", [True, 1.5, "0", 2**100])
def test_property_interval_recorded_cutoffs_are_explicit_strict_clocks(bad):
    graph = Graph()
    create(graph)
    index, head = graph.build_effective_index(), graph.current_state_id
    for getter in [index.node_property_intervals, index.edge_property_intervals]:
        with pytest.raises((TypeError, OverflowError)):
            getter(0, "x", recorded_by=bad)
    assert graph.current_state_id == head
