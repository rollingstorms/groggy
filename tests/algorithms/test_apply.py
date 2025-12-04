"""
Tests for the apply() convenience function and Subgraph.apply() method.
"""

import pytest

from groggy import Graph, algorithms, apply, pipeline


def build_test_graph():
    """Build a small test graph."""
    g = Graph()
    nodes = [g.add_node() for _ in range(10)]

    for i in range(len(nodes) - 1):
        g.add_edge(nodes[i], nodes[i + 1])

    return g, nodes


def test_apply_single_algorithm():
    """Test apply with a single algorithm."""
    g, nodes = build_test_graph()
    sg = g.view()

    # Apply single algorithm
    result = apply(sg, algorithms.centrality.pagerank(max_iter=10, output_attr="pr"))

    assert result is not None
    assert len(result.nodes) == len(nodes)


def test_apply_algorithm_list():
    """Test apply with a list of algorithms."""
    g, nodes = build_test_graph()
    g.nodes.set_attrs({nodes[0]: {"is_start": True}})
    sg = g.view()

    # Apply list of algorithms
    result = apply(
        sg,
        [
            algorithms.centrality.pagerank(max_iter=10, output_attr="pr"),
            algorithms.pathfinding.bfs(start_attr="is_start", output_attr="dist"),
        ],
    )

    assert result is not None
    assert len(result.nodes) == len(nodes)


def test_apply_pipeline_object():
    """Test apply with a Pipeline object."""
    g, nodes = build_test_graph()
    sg = g.view()

    # Create pipeline
    pipe = pipeline([algorithms.centrality.pagerank(max_iter=10, output_attr="pr")])

    # Apply the pipeline
    result = apply(sg, pipe)

    assert result is not None
    assert len(result.nodes) == len(nodes)


def test_apply_invalid_type():
    """Test that apply raises for invalid input."""
    g, nodes = build_test_graph()
    sg = g.view()

    with pytest.raises(TypeError):
        apply(sg, "not an algorithm")


def test_apply_usage_example():
    """
    Example showing the convenience of apply().

    Instead of:
        pipe = pipeline([algo])
        result = pipe(sg)

    You can write:
        result = apply(sg, algo)
    """
    g, nodes = build_test_graph()
    sg = g.view()

    # Short form
    result = apply(sg, algorithms.centrality.pagerank(output_attr="pr"))

    # Verify it worked
    assert result is not None

    # Can access attributes directly
    for node in list(result.nodes)[:3]:
        assert hasattr(node, "pr")
        assert node.pr > 0


def test_apply_multiple_algorithms():
    """Test apply with multiple algorithms in list form."""
    g, nodes = build_test_graph()
    g.nodes.set_attrs({nodes[0]: {"start": True}})
    sg = g.view()

    # Apply multiple algorithms at once
    result = apply(
        sg,
        [
            algorithms.centrality.pagerank(max_iter=20, output_attr="importance"),
            algorithms.pathfinding.bfs(start_attr="start", output_attr="distance"),
            algorithms.community.lpa(max_iter=10, output_attr="community"),
        ],
    )

    assert result is not None

    # Verify all algorithms ran
    node = list(result.nodes)[0]
    assert hasattr(node, "importance")
    assert hasattr(node, "distance")
    assert hasattr(node, "community")


# New tests for Subgraph.apply() method chaining


def test_subgraph_apply_method():
    """Test sg.apply() method for fluent chaining."""
    g, nodes = build_test_graph()
    sg = g.view()

    # Use method chaining
    result = sg.apply(algorithms.centrality.pagerank(max_iter=10, output_attr="pr"))

    assert result is not None
    assert len(result.nodes) == len(nodes)


def test_subgraph_apply_with_table():
    """Test sg.apply().table() chaining."""
    g, nodes = build_test_graph()
    sg = g.view()

    # Apply and get table in one chain
    table = sg.apply(
        algorithms.centrality.pagerank(max_iter=10, output_attr="pr")
    ).table()

    assert table is not None
    assert len(table.nodes) == len(nodes)


def test_subgraph_apply_multiple_algorithms():
    """Test sg.apply([algo1, algo2]) with list."""
    g, nodes = build_test_graph()
    g.nodes.set_attrs({nodes[0]: {"start": True}})
    sg = g.view()

    # Apply multiple algorithms via method
    result = sg.apply(
        [
            algorithms.centrality.pagerank(max_iter=10, output_attr="importance"),
            algorithms.pathfinding.bfs(start_attr="start", output_attr="dist"),
        ]
    )

    assert result is not None
    assert len(result.nodes) == len(nodes)


def test_subgraph_apply_pipeline_object():
    """Test sg.apply(pipeline) with Pipeline object."""
    g, nodes = build_test_graph()
    sg = g.view()

    # Create pipeline
    pipe = pipeline([algorithms.centrality.pagerank(max_iter=10, output_attr="pr")])

    # Apply via method
    result = sg.apply(pipe)

    assert result is not None
    assert len(result.nodes) == len(nodes)


def test_subgraph_apply_chaining_example():
    """
    Example showing fluent method chaining.

    The sg.apply() method enables:
        sg.apply(algo).table()
        sg.apply([algo1, algo2]).viz.draw()
    """
    g, nodes = build_test_graph()
    sg = g.view()

    # Fluent chaining
    result = sg.apply(algorithms.centrality.pagerank(max_iter=10, output_attr="pr"))

    # Can chain with other subgraph methods
    table = result.table()
    assert table is not None

    # Can access node_count, edge_count, etc
    assert result.node_count() == len(nodes)
