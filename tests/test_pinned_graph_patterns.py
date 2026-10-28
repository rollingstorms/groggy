"""Pattern bindings preserve exact edge identities and captured source state."""
from concurrent.futures import ThreadPoolExecutor

import pytest
from groggy import DurableGraph, Graph, GraphBatch, GraphPattern, PatternMatches


def fixture():
    graph = Graph(directed=True)
    batch = GraphBatch(effective_at=10)
    nodes = [batch.add_node(kind=kind) for kind in ["person", "person", "place", "isolate"]]
    for source, target, kind in [(0, 1, "knows"), (0, 1, "knows"),
                                 (1, 2, "visits"), (2, 0, "returns"), (0, 0, "self")]:
        batch.add_edge(nodes[source], nodes[target], kind=kind)
    graph.apply_batch(batch, 0, "fixture", "test")
    return graph


def path():
    return GraphPattern(nodes=[{"name": "a", "filter": 'kind == "person"'},
                               {"name": "b"}, {"name": "c"}],
                        edges=[{"name": "ab", "source": "a", "target": "b",
                                "filter": 'kind == "knows"'},
                               {"name": "bc", "source": "b", "target": "c",
                                "filter": 'kind == "visits"'}])


def test_named_filtered_parallel_matches_and_exact_bound_graphs():
    graph = fixture()
    index = graph.build_index(node_properties=["kind"], edge_properties=["kind"])
    result = index.match_pattern(path())
    assert isinstance(result, PatternMatches)
    expected = [{"nodes": {"a": 0, "b": 1, "c": 2}, "edges": {"ab": e, "bc": 2}}
                for e in [0, 1]]
    assert result.bindings == expected
    assert result.node_variables == ["a", "b", "c"]
    assert result.edge_variables == ["ab", "bc"]
    assert result.copy().bindings == expected
    assert path().copy().node_variables == ["a", "b", "c"]
    assert not result.truncated
    child = result.match_view(1)
    assert child.node_ids == [0, 1, 2]
    assert child.edge_ids == [1, 2]  # no unrelated parallel, cycle or loop edges
    assert child.write_state_id == index.view.write_state_id
    detached = result.bindings
    detached[0]["nodes"]["a"] = 99
    assert result.bindings == expected
    graph.set_node_attr(0, "kind", "changed")
    assert child.node_attr(0, "kind") == "person"
    assert result.source_view.node_attr(0, "kind") == "person"
    del graph, index
    assert child.edge_endpoints(1) == (0, 1)


def test_pages_search_budget_and_strict_integer_controls():
    index = fixture().build_index(node_properties=["kind"], edge_properties=["kind"])
    full = index.match_pattern(path())
    first = index.match_pattern(path(), limit=1)
    assert first.bindings == full.bindings[:1]
    assert first.truncated
    assert index.match_pattern(path(), offset=1, limit=1).bindings == full.bindings[1:]
    assert not index.match_pattern(path(), offset=1, limit=1).truncated
    assert index.match_pattern(path(), limit=0).truncated
    assert index.match_pattern(path(), offset=2**64 - 1).bindings == []
    assert index.match_pattern(path(), max_expansions=full.expansions).bindings == full.bindings
    for budget in [0, full.expansions - 1]:
        with pytest.raises(ValueError, match="budget"):
            index.match_pattern(path(), max_expansions=budget)
    for bad in [True, -1, 1.5, "2", 2**100]:
        for option in ["offset", "limit", "max_expansions"]:
            with pytest.raises((TypeError, OverflowError)):
                index.match_pattern(path(), **{option: bad})
        with pytest.raises((TypeError, OverflowError)):
            full.match_view(bad)
    with pytest.raises(ValueError, match="bounds"):
        full.match_view(len(full))


def test_reuse_direction_cycles_and_isolated_single_node_patterns():
    index = fixture().build_index()
    repeated = GraphPattern(nodes=[{"name": "a"}, {"name": "b"}],
                            edges=[{"name": e, "source": "a", "target": "b"}
                                   for e in ["x", "y"]])
    rows = index.match_pattern(repeated).bindings
    assert len(rows) == 7  # 4 parallel combinations + visits + returns + self
    assert len(index.match_pattern(repeated, distinct_edges=True)) == 2
    assert len(index.match_pattern(repeated, distinct_nodes=True)) == 6
    assert len(index.match_pattern(repeated, distinct_nodes=True, distinct_edges=True)) == 2
    incoming = GraphPattern(nodes=[{"name": "a"}, {"name": "b"}],
                            edges=[{"name": "e", "source": "a", "target": "b",
                                    "direction": "incoming"}])
    assert {tuple(r["nodes"].values()) for r in index.match_pattern(incoming).bindings} == {
        (1, 0), (2, 1), (0, 2), (0, 0)}
    single = index.match_pattern(GraphPattern(nodes=[{"name": "only"}]))
    assert len(single) == 4
    assert single.match_view(3).node_ids == [3]
    assert single.match_view(3).edge_ids == []


@pytest.mark.parametrize("nodes,edges", [([], None), ([{"name": "a"}, {"name": "a"}], None),
    ([{"name": " "}], None), ([{"name": "a", "typo": 1}], None),
    ([{"filter": None}], None), ([{"name": 1}], None),
    ([{"name": "a"}], [{"name": "e", "source": "a", "target": "missing"}]),
    ([{"name": "a"}], [{"name": "a", "source": "a", "target": "a"}]),
    ([{"name": "a"}], [{"name": "e", "source": "a", "target": "a", "direction": "sideways"}])])
def test_invalid_pattern_shapes_reject_at_construction(nodes, edges):
    with pytest.raises((TypeError, ValueError)):
        GraphPattern(nodes=nodes, edges=edges)


def test_missing_indexes_require_explicit_scan_even_behind_empty_filters():
    graph = fixture()
    index = graph.build_index(topology=False)
    with pytest.raises(ValueError, match="topology"):
        index.match_pattern(path())
    assert len(index.match_pattern(path(), allow_scan=True)) == 2
    assert not index.has_topology
    empty = GraphPattern(nodes=[{"name": "a", "filter": 'kind == "missing"'}])
    with pytest.raises(ValueError, match="index"):
        index.match_pattern(empty, limit=0)
    assert len(index.match_pattern(empty, allow_scan=True)) == 0


def test_owner_frontdoors_capture_selected_subgraphs_and_durable_state(tmp_path):
    graph = fixture()
    subgraph = graph.filter_nodes('kind == "person"')
    index = subgraph.build_index(node_properties=["kind"], edge_properties=["kind"])
    assert index.view.node_ids == [0, 1]
    assert len(index.match_pattern(path())) == 0
    journal = tmp_path / "patterns.ggj"
    with DurableGraph.create(journal) as owner:
        batch = GraphBatch(effective_at=10)
        batch.add_node(kind="old")
        owner.apply_batch(batch, 0, "first", "test")
        head, before = owner.current_state_id, journal.read_bytes()
        captured = owner.build_index(node_properties=["kind"])
        timed = owner.read_view_at(15).build_index(node_properties=["kind"])
        pattern = GraphPattern(nodes=[{"name": "n", "filter": 'kind == "old"'}])
        rows = timed.match_pattern(pattern)
        assert rows.match_view(0).read_time == timed.view.read_time
        assert rows.match_view(0).write_state_id is None
        assert owner.current_state_id == head
        assert journal.read_bytes() == before
        owner.writer().set_node_attr(0, "kind", "new")
    assert len(captured.match_pattern(pattern)) == 1
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(lambda _: rows.match_view(0).node_attr(0, "kind"), range(8))) == ["old"] * 8
