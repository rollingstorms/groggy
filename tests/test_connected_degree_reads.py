"""Degree and neighbor compatibility reads share canonical scope and liveness."""
import groggy as gr
import pytest


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("method", ["degree", "in_degree", "out_degree"])
@pytest.mark.parametrize("full_graph", [False, True])
def test_degree_rows_match_captured_domain_without_skipping_ids(directed, method, full_graph):
    graph = gr.Graph(directed=directed)
    graph.add_nodes(5)
    for source, target in [(0, 1), (1, 0), (0, 1), (1, 1), (2, 1), (1, 3)]:
        graph.add_edge(source, target)
    selected = graph.subgraph_from_edges([0, 1, 3])
    reference = graph.read_view() if full_graph else selected.pin_view()
    read = getattr(selected, method)
    expected = getattr(reference, method)
    before = graph.current_state_id
    ids = [1, 0, 1]
    assert read(ids, full_graph=full_graph).to_list() == [expected(node) for node in ids]
    assert read(full_graph=full_graph).to_list() == [expected(node) for node in [0, 1]]
    assert read(1, full_graph=full_graph) == expected(1)
    assert read([], full_graph=full_graph).to_list() == []
    for bad in [2, [1, 2, 0], [99]]:
        with pytest.raises((ValueError, RuntimeError, KeyError)):
            read(bad, full_graph=full_graph)
    assert graph.current_state_id == before


@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("removal", ["node", "edge", "rewire"])
def test_default_neighbor_and_degree_reads_retain_the_captured_domain(warm, removal):
    graph = gr.Graph()
    graph.add_nodes(3)
    graph.add_edge(0, 1)
    selected = graph.induced_subgraph([0, 1])
    frozen = selected.pin_view()
    if warm:
        assert selected.neighbors(0).to_list() == [1]
        assert selected.degree(0) == 1
        assert selected.in_degree().to_list() == [1, 1]
    if removal == "node":
        graph.remove_node(1)
    elif removal == "edge":
        graph.remove_edge(0)
    else:
        graph.set_edge_endpoints(0, 0, 2)
    before = graph.current_state_id
    for node in [0, 1]:
        assert selected.neighbors(node).to_list() == frozen.neighbors(node)
    for method in ["degree", "in_degree", "out_degree"]:
        for full in [False, True]:
            for ids in [None, 0, 1, [1, 0], []]:
                values = getattr(selected, method)(ids, full_graph=full)
                if isinstance(ids, int):
                    assert values == 1
                else:
                    assert values.to_list() == ([1, 1] if ids is None or ids else [])
        assert getattr(frozen, method)(1) == 1
    assert frozen.neighbors(0) == [1]
    assert graph.current_state_id == before


def test_degree_calls_handle_empty_and_reject_invalid_input_types():
    graph = gr.Graph()
    selected = graph.induced_subgraph([])
    for method in ["degree", "in_degree", "out_degree"]:
        read = getattr(selected, method)
        assert read().to_list() == []
        assert read([], full_graph=True).to_list() == []
        for bad in ["node", {}, object()]:
            with pytest.raises((TypeError, ValueError)):
                read(bad)
