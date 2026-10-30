"""Structural summaries use exact simple undirected scope and edge-instance density."""
from concurrent.futures import ThreadPoolExecutor

import groggy as gr
import pytest


def fixture(directed=False):
    graph = gr.Graph(directed=directed)
    graph.add_nodes(4)
    for source, target in [(0, 1), (1, 2), (0, 2), (0, 1), (0, 0)]:
        graph.add_edge(source, target)
    return graph


@pytest.mark.parametrize("surface", ["graph", "connected", "captured"])
def test_clustering_and_transitivity_ignore_loop_and_parallel_multiplicity(surface):
    graph = fixture()
    source = {"graph": graph, "connected": graph.induced_subgraph([0, 1, 2, 3]),
              "captured": graph.read_view()}[surface]
    before = graph.current_state_id
    assert source.clustering_coefficient() == pytest.approx(0.75)
    assert source.clustering_coefficient(node_id=0) == pytest.approx(1)
    assert source.clustering_coefficient(node_id=3) == 0
    assert source.transitivity() == pytest.approx(1)
    assert source.density() == pytest.approx(5 / 6)
    assert graph.current_state_id == before


def test_excluded_triangle_edge_does_not_leak_from_the_owner():
    graph = fixture()
    selected = graph.subgraph_from_edges([0, 1, 3, 4])
    for source in [selected, selected.pin_view()]:
        assert source.clustering_coefficient() == 0
        assert source.clustering_coefficient(0) == 0
        assert source.transitivity() == 0
    assert graph.clustering_coefficient(0) == 1


def test_captured_coefficients_are_ordered_owned_rows_and_outlive_owner_changes():
    graph = fixture()
    source = graph.read_view()
    expected = [(0, 1.0), (1, 1.0), (2, 1.0), (3, 0.0)]
    assert source.clustering_coefficients() == expected
    graph.remove_node(1)
    del graph
    with ThreadPoolExecutor(max_workers=2) as workers:
        assert list(workers.map(lambda _: source.clustering_coefficients(), range(4))) == [expected] * 4
    assert source.transitivity() == 1


@pytest.mark.parametrize("surface", ["graph", "connected", "captured"])
def test_directed_density_uses_directed_denominator_and_undefined_clustering_rejects(surface):
    graph = gr.Graph(directed=True)
    graph.add_nodes(2)
    graph.add_edge(0, 1)
    source = {"graph": graph, "connected": graph.induced_subgraph([0, 1]),
              "captured": graph.read_view()}[surface]
    assert source.density() == 0.5
    with pytest.raises((ValueError, RuntimeError), match="undirected"):
        source.clustering_coefficient()
    with pytest.raises((ValueError, RuntimeError), match="undirected"):
        source.transitivity()


@pytest.mark.parametrize("count", [0, 1])
@pytest.mark.parametrize("surface", ["graph", "connected", "captured"])
def test_empty_and_one_node_summaries_are_zero(count, surface):
    graph = gr.Graph()
    graph.add_nodes(count)
    source = {"graph": graph, "connected": graph.induced_subgraph(list(range(count))),
              "captured": graph.read_view()}[surface]
    assert source.clustering_coefficient() == source.transitivity() == source.density() == 0
    with pytest.raises((ValueError, RuntimeError)):
        source.clustering_coefficient(99)


@pytest.mark.parametrize("node", [True, -1, 1.5, "0", 2**100])
@pytest.mark.parametrize("surface", ["graph", "connected", "captured"])
def test_local_node_ids_reject_noninteger_negative_boolean_and_overflow(node, surface):
    graph = fixture()
    source = {"graph": graph, "connected": graph.induced_subgraph([0, 1, 2]),
              "captured": graph.read_view()}[surface]
    with pytest.raises((ValueError, RuntimeError, TypeError, OverflowError)):
        source.clustering_coefficient(node)


def test_older_structural_reads_stay_fixed_while_fresh_selections_see_changes():
    graph = fixture()
    selected = graph.induced_subgraph([0, 1, 2, 3])
    assert selected.clustering_coefficient() == 0.75
    graph.set_edge_endpoints(2, 2, 3)
    assert selected.clustering_coefficient() == 0.75
    assert graph.induced_subgraph([0, 1, 2, 3]).clustering_coefficient() == 0
    before = selected.transitivity()
    graph.set_node_attr(0, "arbitrary", "new")
    assert selected.transitivity() == before
