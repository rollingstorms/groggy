"""
Regression tests for node attribute access through Node proxies.
"""

from groggy import Graph
from groggy.builder import AlgorithmBuilder


def test_node_proxy_attr_matches_direct_lookup():
    """Builder result node proxies should expose the same values as direct attribute lookups."""
    graph = Graph(directed=True)
    nodes = [graph.add_node() for _ in range(3)]
    graph.add_edge(nodes[0], nodes[1])
    graph.add_edge(nodes[1], nodes[2])

    result = graph.view().apply(_build_degree_algorithm())
    expected = {
        nodes[0]: 1,
        nodes[1]: 1,
        nodes[2]: 0,
    }

    for node in result.nodes:
        proxy_val = node.deg
        direct_val = result.get_node_attribute(node.id, "deg")
        assert proxy_val == direct_val
        assert direct_val == expected[node.id]


def _build_degree_algorithm():
    builder = AlgorithmBuilder("node_proxy_degree_check")
    base = builder.init_nodes(default=0)
    degrees = builder.node_degrees(base)
    builder.attach_as("deg", degrees)
    return builder.build()
