"""Test graph constant primitives (graph_node_count, graph_edge_count)."""

import pytest

import groggy as gg
from groggy.builder import AlgorithmBuilder


def test_graph_node_count():
    """Test retrieving node count as a scalar."""
    print("\n=== Test graph_node_count ===")

    # Create a simple graph
    g = gg.Graph()
    nodes = [g.add_node() for _ in range(10)]

    # Build algorithm that gets node count
    builder = AlgorithmBuilder("test_node_count")
    ref = builder.init_nodes(default=0.0)  # Reference for which nodes exist
    n_scalar = builder.graph_node_count()
    n_broadcast = builder.core.broadcast_scalar(n_scalar, ref)
    builder.attach_as("count", n_broadcast)

    algo = builder.build()
    result = g.apply(algo)

    count_value = result.get_node_attribute(nodes[0], "count")
    print(f"Node count: {count_value}")
    assert count_value == 10, f"Expected 10, got {count_value}"
    print("✓ Node count correct")


def test_graph_edge_count():
    """Test retrieving edge count as a scalar."""
    print("\n=== Test graph_edge_count ===")

    # Create graph with edges
    g = gg.Graph()
    nodes = [g.add_node() for _ in range(5)]
    edges = []
    for i in range(4):
        edges.append(g.add_edge(nodes[i], nodes[i + 1]))

    # Build algorithm that gets edge count
    builder = AlgorithmBuilder("test_edge_count")
    ref = builder.init_nodes(default=0.0)
    m_scalar = builder.graph_edge_count()
    m_broadcast = builder.core.broadcast_scalar(m_scalar, ref)
    builder.attach_as("edge_count", m_broadcast)

    algo = builder.build()
    result = g.apply(algo)

    count_value = result.get_node_attribute(nodes[0], "edge_count")
    print(f"Edge count: {count_value}")
    assert count_value == 4, f"Expected 4, got {count_value}"
    print("✓ Edge count correct")


def test_graph_constants_in_arithmetic():
    """Test using graph constants in arithmetic operations."""
    print("\n=== Test graph constants in arithmetic ===")

    # Create graph
    g = gg.Graph()
    nodes = [g.add_node() for _ in range(10)]

    # Build algorithm: compute 1/N (uniform distribution)
    builder = AlgorithmBuilder("uniform_dist")
    ones = builder.init_nodes(default=1.0)
    n = builder.graph_node_count()
    uniform = builder.core.div(ones, n)  # 1/N for each node
    builder.attach_as("uniform", uniform)

    algo = builder.build()
    result = g.apply(algo)

    uniform_value = result.get_node_attribute(nodes[0], "uniform")
    expected = 1.0 / 10.0
    print(f"Uniform value: {uniform_value}, expected: {expected}")
    assert (
        abs(uniform_value - expected) < 1e-6
    ), f"Expected {expected}, got {uniform_value}"
    print("✓ Arithmetic with graph constants works")


@pytest.mark.skip(reason="Variable tracking in iterate() needs fixing")
def test_pagerank_with_node_count():
    """Test a simple PageRank-like algorithm using node count."""
    print("\n=== Test PageRank with node count ===")

    # Create simple graph: 0 -> 1 -> 2 -> 0 (cycle)
    g = gg.Graph(directed=True)
    nodes = [g.add_node() for _ in range(3)]
    g.add_edge(nodes[0], nodes[1])
    g.add_edge(nodes[1], nodes[2])
    g.add_edge(nodes[2], nodes[0])

    # Simple PageRank: uniform teleport
    builder = AlgorithmBuilder("simple_pagerank")

    n = builder.graph_node_count()
    ranks = builder.init_nodes(default=1.0)

    # One iteration: damping * sum(neighbors) + (1-damping)/N
    damping = 0.85
    teleport = 1.0 - damping

    # Teleport term: (1-damping) / N
    # Need to broadcast to map first
    teleport_const = builder.init_nodes(default=teleport)
    teleport_per_node = builder.core.div(teleport_const, n)

    with builder.iterate(1):
        # Just add teleport term (simplified)
        ranks = builder.core.add(ranks, teleport_per_node)

    builder.attach_as("rank", ranks)

    algo = builder.build()
    result = g.apply(algo)

    rank0 = result.get_node_attribute(nodes[0], "rank")
    expected = 1.0 + (1.0 - damping) / 3.0
    print(f"Rank after one iteration: {rank0}, expected: {expected}")
    assert abs(rank0 - expected) < 1e-6, f"Expected {expected}, got {rank0}"
    print("✓ PageRank with node count works")


def test_spec_encoding():
    """Verify the step specs are encoded correctly."""
    print("\n=== Test spec encoding ===")

    builder = AlgorithmBuilder("test_spec")
    ref = builder.init_nodes(default=0.0)
    n = builder.graph_node_count()
    m = builder.graph_edge_count()
    n_map = builder.core.broadcast_scalar(n, ref)
    m_map = builder.core.broadcast_scalar(m, ref)
    builder.attach_as("n", n_map)
    builder.attach_as("m", m_map)

    algo = builder.build(validate=False)  # Skip validation
    spec = algo.to_spec()

    # The spec is wrapped, need to parse it
    import json

    steps_str = str(spec["params"]["steps"])
    steps_obj = json.loads(steps_str)
    steps = steps_obj["steps"]
    print(f"Generated spec has {len(steps)} steps")

    # Check that the steps were encoded
    step_ids = [s["id"] for s in steps]
    assert "core.graph_node_count" in step_ids, "Missing core.graph_node_count"
    assert "core.graph_edge_count" in step_ids, "Missing core.graph_edge_count"
    print("✓ Spec encoding correct")


if __name__ == "__main__":
    test_graph_node_count()
    test_graph_edge_count()
    test_graph_constants_in_arithmetic()
    test_pagerank_with_node_count()
    test_spec_encoding()

    print("\n" + "=" * 50)
    print("✅ All graph constants tests passed!")
    print("=" * 50)
