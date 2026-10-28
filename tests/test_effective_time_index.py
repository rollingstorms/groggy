"""Explicit effective-time indexes retain branch/head, events and historical reads."""
from concurrent.futures import ThreadPoolExecutor

import pytest
from groggy import DurableGraph, EffectiveTimeIndex, Graph, GraphBatch


def create(graph):
    batch = GraphBatch(effective_at=10)
    a, b = batch.add_node(x=1), batch.add_node()
    batch.add_edge(a, b, weight=2)
    return graph.apply_batch(batch, graph.current_state_id, "initial", "test")["state_id"]


def update(graph, at, value):
    batch = GraphBatch(effective_at=at)
    batch.set_node_attr(0, "x", value)
    return graph.apply_batch(batch, graph.current_state_id, "update", "test")["state_id"]


def test_indexed_windows_and_reads_match_canonical_metadata_and_time_views():
    graph = Graph()
    first = create(graph)
    states = [first, update(graph, 30, 3), update(graph, 20, 2), update(graph, 20, None)]
    batch = GraphBatch(effective_at=25)
    batch.unset_node_attr(0, "x")
    states.append(graph.writer().apply_batch(batch)["state_id"])
    graph.writer().set_edge_endpoints(0, 1, 0, effective_at=24)
    states.append(graph.current_state_id)
    index = graph.build_effective_index()
    assert isinstance(index, EffectiveTimeIndex)
    assert index.branch == "main" and index.captured_head == graph.current_state_id
    expected = []
    for state in states:
        metadata = graph.publication_metadata(state)
        expected.extend(dict(row, state_id=state, recorded_at=metadata["recorded_at"])
                        for row in metadata["change_times"])
    assert index.event_count == len(expected)
    for start, end in [(-(2**63), 2**63 - 1), (20, 20), (20, 25), (26, 29)]:
        rows = index.events_between(start, end)
        reference = [row for row in expected if start <= row["effective_at"] <= end]
        key = lambda row: (row["effective_at"], row["state_id"], row["kind"], row["entity_id"], row["name"] or "")
        assert sorted(rows, key=key) == sorted(reference, key=key)
        assert index.node_events(0, start, end) == [row for row in rows if row["kind"].startswith("node_") and row["entity_id"] == 0]
        assert index.edge_events(0, start, end) == [row for row in rows if row["kind"].startswith("edge_") and row["entity_id"] == 0]
    for at in [9, 10, 20, 24, 25, 30]:
        actual, reference = index.read_view_at(at), graph.read_view_at(at)
        assert actual.node_ids == reference.node_ids and actual.edge_ids == reference.edge_ids
        assert actual.read_time == reference.read_time and actual.write_state_id is None
        if at >= 10:
            assert actual.node_attr(0, "x") == reference.node_attr(0, "x")
    cutoff = graph.publication_metadata(first)["recorded_at"]
    assert index.read_view_at(25, recorded_by=cutoff).node_attr(0, "x") == 1
    assert all(row["recorded_at"] <= cutoff for row in index.events_between(0, 100, recorded_by=cutoff))
    assert index.node_events(999, 0, 100) == []
    rows = index.events_between(20, 20)
    rows.clear()
    assert index.events_between(20, 20)


@pytest.mark.parametrize("restored_at", [40, 50])
def test_activation_intervals_preserve_inactive_ids_and_resolve_equal_time_changes(restored_at):
    graph = Graph()
    first = create(graph)
    graph.writer().remove_edges([0], effective_at=35)
    graph.writer().remove_nodes([0], effective_at=40)
    old = graph.build_effective_index()
    graph.restore_state(first, mode="tracked", effective_at=restored_at, expected_state=graph.current_state_id)
    index = graph.build_effective_index()
    expected = [(10, None)] if restored_at == 40 else [(10, 40), (50, None)]
    assert index.node_activation_intervals(0) == expected
    assert index.edge_activation_intervals(0) == [(10, 35), (restored_at, None)]
    assert old.node_activation_intervals(0) == [(10, 40)]
    assert index.node_activation_intervals(0, recorded_by=-(2**63)) == []
    assert index.edge_activation_intervals(999) == []
    assert old.node_events(0, 0, 100)[-1]["kind"] == "node_removed"


def test_durable_capture_survives_close_reopen_branch_changes_and_threads(tmp_path):
    path = tmp_path / "effective-index.ggj"
    with DurableGraph.create(path) as graph:
        first = create(graph)
        head, before = graph.current_state_id, path.read_bytes()
        index = graph.build_effective_index()
        assert graph.current_state_id == head and path.read_bytes() == before
        update(graph, 20, 2)
        new = graph.build_effective_index()
    assert index.read_view_at(25).node_attr(0, "x") == 1
    with DurableGraph.open(path) as graph:
        assert graph.build_effective_index().read_view_at(25).node_attr(0, "x") == 2
        assert new.captured_head == graph.current_state_id
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(lambda _: new.read_view_at(25).node_attr(0, "x"), range(12))) == [2] * 12
    assert index.copy().captured_head == first
    graph = Graph()
    create(graph)
    graph.create_branch("sibling")
    update(graph, 20, 2)
    main = graph.build_effective_index()
    graph.checkout_branch("sibling")
    assert graph.build_effective_index().read_view_at(25).node_attr(0, "x") == 1
    assert main.read_view_at(25).node_attr(0, "x") == 2


def test_frozen_group_sources_and_explicit_legacy_clocks_survive_index_ownership():
    source = Graph()
    source.add_node(x=7)
    graph = Graph()
    group = graph.retain_group(source.read_view())
    meta = graph.create_meta_node(group, effective_at=10)
    index = graph.build_effective_index()
    source.set_node_attr(0, "x", 9)
    del source, graph
    assert index.read_view_at(15).meta_subgraph(meta).node_attr(0, "x") == 7
    legacy = Graph(manual=True)
    legacy.add_node(x=1)
    with pytest.raises(ValueError, match="clean|publish"):
        legacy.build_effective_index()
    state = legacy.commit("legacy", "test")
    with pytest.raises(ValueError, match="precise|legacy"):
        legacy.build_effective_index()
    adapted = legacy.build_effective_index(legacy_times={state: (10, 100)})
    assert adapted.read_view_at(10).node_attr(0, "x") == 1
    assert adapted.read_view_at(10, recorded_by=99).node_ids == []
    assert adapted.events_between(10, 10)[0]["recorded_at"] == 100
    later = legacy.build_effective_index(legacy_times={state: (20, 200)})
    assert later.read_view_at(15).node_ids == []
    assert adapted.read_view_at(15).node_ids == [0]
    assert later.events_between(20, 20)[0]["recorded_at"] == 200
    empty = Graph().build_effective_index()
    assert empty.event_count == 0 and empty.events_between(0, 0) == []


@pytest.mark.parametrize("bad", [True, 1.5, "2", 2**100])
def test_effective_index_strict_controls_reject_without_publishing(bad):
    graph = Graph()
    create(graph)
    index, head = graph.build_effective_index(), graph.current_state_id
    for call in [lambda: index.read_view_at(bad), lambda: index.read_view_at(20, recorded_by=bad),
                 lambda: index.events_between(bad, 20), lambda: index.events_between(10, bad),
                 lambda: index.node_events(0, 10, 20, recorded_by=bad),
                 lambda: index.edge_activation_intervals(0, recorded_by=bad)]:
        with pytest.raises((TypeError, OverflowError)):
            call()
    assert graph.current_state_id == head


@pytest.mark.parametrize("bad", [True, -1, 1.5, "2", 2**100])
def test_effective_index_entity_ids_are_strict_and_reversed_windows_reject(bad):
    index = Graph().build_effective_index()
    for call in [lambda: index.node_events(bad, 0, 1), lambda: index.edge_events(bad, 0, 1),
                 lambda: index.node_activation_intervals(bad), lambda: index.edge_activation_intervals(bad)]:
        with pytest.raises((TypeError, OverflowError)):
            call()
    with pytest.raises(ValueError, match="start"):
        index.events_between(1, 0)
    with pytest.raises(ValueError, match="start"):
        index.node_events(999, 1, 0)
