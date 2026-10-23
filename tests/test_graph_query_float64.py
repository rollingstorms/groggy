"""Graph literals retain floating width and precision in live/frozen filters."""
import pytest
from groggy import AttrValue, Graph


def test_live_and_captured_graph_queries_preserve_f64_literals_and_f32_identity():
    graph = Graph(directed=True)
    nodes = graph.add_nodes(4)
    for node, value in zip(nodes, [AttrValue.float64(1), AttrValue.float64(1 + 2**-40),
                                   AttrValue.float64(1e300), 1.0]):
        graph.set_node_attr(node, "x", value)
        graph.add_edge(node, node, x=value)
    legacy = graph.induced_subgraph(nodes)
    frozen = legacy.pin_view()
    for source in [legacy, frozen]:
        for query, expected in [("x == 1.0", [nodes[0]]), ("x == 1f32", [nodes[3]]),
                                ("x == 1.0000000000009095", [nodes[1]]),
                                ("x >= +1E+300", [nodes[2]])]:
            assert list(source.filter_nodes(query).node_ids) == expected
            assert list(source.filter_edges(query).edge_ids) == expected
    graph.set_node_attr(nodes[1], "x", AttrValue.float64(0))
    assert list(legacy.filter_nodes("x == 1.0000000000009095").node_ids) == []
    assert list(frozen.filter_nodes("x == 1.0000000000009095").node_ids) == [nodes[1]]


@pytest.mark.parametrize("query", ["x == 1e400", "x == 1e40f32", "x > 1e+",
                                    "x == 1f16", "x == 1 trailing", "x == 1)",
                                    "x == 'unterminated"])
def test_invalid_literal_or_trailing_text_never_silently_filters_a_prefix(query):
    graph = Graph()
    node = graph.add_node(x=1)
    for source in [graph.induced_subgraph([node]), graph.read_view()]:
        for method in [source.filter_nodes, source.filter_edges]:
            with pytest.raises((ValueError, RuntimeError)):
                method(query)
