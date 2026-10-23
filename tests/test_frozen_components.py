"""Components share frozen state and never publish algorithm properties."""
from concurrent.futures import ThreadPoolExecutor

import groggy as gr
import pytest


def test_components_keep_frozen_properties_and_exact_selection():
    graph = gr.Graph(directed=True)
    a, b, c, isolate = graph.add_nodes(4)
    first = graph.add_edge(a, b)
    back = graph.add_edge(b, a)
    onward = graph.add_edge(b, c)
    graph.add_edge(c, a)  # Excluded return path.
    parallel = graph.add_edge(a, b)
    graph.set_node_attr(a, "score", 7)
    view = graph.read_view().select_nodes([a, b, c, isolate])
    selected = graph.read_view().select_edges([first, back, onward, parallel])
    graph.set_node_attr(a, "score", 99)
    del graph
    groups = selected.connected_components("strong")
    assert [g.node_ids for g in groups] == [[a, b], [c]]
    assert groups[0].edge_ids == [first, back, parallel]
    assert groups[0].node_attr(a, "score") == 7
    assert len(selected.connected_components()) == 1
    assert len(selected.connected_components("weak")) == 1
    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(lambda _: view.connected_components("strong"), range(4)))
    assert [[g.node_ids for g in result] for result in results] == [[[a, b, c], [isolate]]] * 4
    with pytest.raises(ValueError, match="Invalid component mode"):
        selected.connected_components("unknown")


def test_empty_and_single_node_components():
    graph = gr.Graph()
    assert graph.read_view().connected_components() == []
    node = graph.add_node()
    for mode in ["undirected", "weak", "strong"]:
        result = graph.read_view().connected_components(mode)
        assert len(result) == 1
        assert result[0].node_ids == [node]
        assert result[0].edge_ids == []
        assert "component" not in graph.read_view().node_attribute_keys(node)
