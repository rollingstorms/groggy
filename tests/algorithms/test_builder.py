"""Tests for the Builder DSL and custom step execution."""

import pytest

from groggy import AlgorithmBuilder, Graph, VarHandle, apply, builder


def test_builder_creation():
    """Test creating a builder."""
    b = builder("my_algorithm")
    assert isinstance(b, AlgorithmBuilder)
    assert b.name == "my_algorithm"


def test_var_handle_creation():
    """Test variable handle creation."""
    b = builder("test")
    var = b._new_var("test")
    assert isinstance(var, VarHandle)
    assert "test" in var.name


def test_init_nodes_step():
    """Test init_nodes step."""
    b = builder("test")
    nodes = b.init_nodes(default=1.0)

    assert isinstance(nodes, VarHandle)
    assert len(b.steps) == 1
    assert b.steps[0]["type"] == "core.init_nodes"
    assert b.steps[0]["default"] == 1.0


def test_node_degrees_step():
    """Test node_degrees step."""
    b = builder("test")
    nodes = b.init_nodes()
    degrees = b.node_degrees(nodes)

    assert isinstance(degrees, VarHandle)
    assert len(b.steps) == 2
    assert b.steps[1]["type"] == "graph.degree"


def test_normalize_step():
    """Test normalize step."""
    b = builder("test")
    values = b.init_nodes()
    normalized = b.normalize(values, method="sum")

    assert isinstance(normalized, VarHandle)
    assert len(b.steps) == 2
    assert b.steps[1]["type"] == "normalize"
    assert b.steps[1]["method"] == "sum"


def test_attach_as_step():
    """Test attach_as step."""
    b = builder("test")
    values = b.init_nodes()
    b.attach_as("my_attr", values)

    assert len(b.steps) == 2
    assert b.steps[1]["type"] == "attach_attr"
    assert b.steps[1]["attr_name"] == "my_attr"


def test_multi_step_composition():
    """Test composing multiple steps."""
    b = builder("degree_centrality")

    # Build a simple degree centrality algorithm
    nodes = b.init_nodes(default=0.0)
    degrees = b.node_degrees(nodes)
    normalized = b.normalize(degrees)
    b.attach_as("degree_centrality", normalized)

    assert len(b.steps) == 4
    assert b.steps[0]["type"] == "core.init_nodes"
    assert b.steps[1]["type"] == "graph.degree"
    assert b.steps[2]["type"] == "normalize"
    assert b.steps[3]["type"] == "attach_attr"


def test_build_returns_algorithm():
    """Test that build() returns a BuiltAlgorithm."""
    b = builder("test")
    nodes = b.init_nodes()
    b.attach_as("value", nodes)

    algo = b.build()
    assert algo is not None
    assert hasattr(algo, "id")
    assert "custom.test" in algo.id


def test_built_algorithm_properties():
    """Test BuiltAlgorithm properties."""
    b = builder("my_algo")
    nodes = b.init_nodes()
    b.attach_as("value", nodes)
    algo = b.build()

    assert algo.id == "custom.my_algo"
    assert "my_algo" in str(algo)
    assert "2 steps" in str(algo)


@pytest.mark.skip(reason="Requires updated apply() implementation for new builder")
def test_built_algorithm_executes_and_sets_attribute():
    """Custom pipeline should run through apply() and attach attributes."""
    b = builder("degree_score")
    nodes = b.init_nodes(default=0.0)
    degrees = b.node_degrees(nodes)
    normalized = b.normalize(degrees, method="max")
    b.attach_as("degree_score", normalized)

    algo = b.build()
    spec = algo.to_spec()

    assert spec["id"] == "builder.step_pipeline"
    assert "steps" in spec["params"]

    g = Graph()
    n1 = g.add_node()
    n2 = g.add_node()
    n3 = g.add_node()
    g.add_edge(n1, n2)
    g.add_edge(n2, n3)

    result = apply(g.view(), algo)

    node_attrs = {node.id: getattr(node, "degree_score") for node in result.nodes}
    assert node_attrs[n1] >= 0.0
    assert node_attrs[n2] >= 0.0
    assert node_attrs[n3] >= 0.0
    assert len(set(node_attrs.values())) > 1


def test_variable_tracking():
    """Test that variables are tracked correctly."""
    b = builder("test")
    v1 = b.init_nodes()
    v2 = b.init_nodes()
    v3 = b.node_degrees(v1)

    # Should have 3 variables
    assert len(b.variables) == 3
    assert v1.name in b.variables
    assert v2.name in b.variables
    assert v3.name in b.variables


def test_builder_fluent_interface():
    """Test that builder methods can be chained conceptually."""
    b = builder("test")

    # While not strictly fluent, should be able to use results
    v1 = b.init_nodes()
    v2 = b.node_degrees(v1)
    v3 = b.normalize(v2)
    b.attach_as("result", v3)

    # Should have a complete sequence
    assert len(b.steps) == 4


def test_different_normalization_methods():
    """Test different normalization methods."""
    for method in ["sum", "max", "minmax"]:
        b = builder(f"norm_{method}")
        values = b.init_nodes()
        normalized = b.normalize(values, method=method)

        assert b.steps[-1]["method"] == method


def test_multiple_attachments():
    """Test attaching multiple attributes."""
    b = builder("multi_attr")
    v1 = b.init_nodes()
    v2 = b.node_degrees(v1)

    b.attach_as("attr1", v1)
    b.attach_as("attr2", v2)

    attach_steps = [s for s in b.steps if s["type"] == "attach_attr"]
    assert len(attach_steps) == 2
    assert attach_steps[0]["attr_name"] == "attr1"
    assert attach_steps[1]["attr_name"] == "attr2"


def test_builder_independence():
    """Test that builders are independent."""
    b1 = builder("algo1")
    b2 = builder("algo2")

    b1.init_nodes()
    b2.init_nodes()
    b2.init_nodes()

    # Each builder should have its own steps
    assert len(b1.steps) == 1
    assert len(b2.steps) == 2


# ==============================================================================
# Documentation/Example Tests
# ==============================================================================


def test_builder_example_degree_centrality():
    """
    Example: Build a simple degree centrality algorithm.

    This demonstrates the intended usage pattern.
    """
    b = builder("degree_centrality")

    # Initialize node values
    nodes = b.init_nodes(default=0.0)

    # Compute degrees
    degrees = b.node_degrees(nodes)

    # Normalize to [0, 1]
    normalized = b.normalize(degrees, method="max")

    # Attach as attribute
    b.attach_as("degree_score", normalized)

    # Build the algorithm
    algo = b.build()

    # Verify structure
    assert len(b.steps) == 4
    assert algo.id == "custom.degree_centrality"

    print("\n✓ Built custom degree centrality algorithm")
    print(f"  Steps: {len(b.steps)}")
    print(f"  ID: {algo.id}")


@pytest.mark.skip(reason="Requires updated apply() implementation for new builder")
def test_builder_example_with_documentation():
    """Example mirroring documentation narrative for custom builders."""
    b = builder("my_centrality")

    # Step 1: Initialize
    nodes = b.init_nodes(default=0.0)

    # Step 2: Compute something interesting
    degrees = b.node_degrees(nodes)

    # Step 3: Normalize
    scores = b.normalize(degrees)

    # Step 4: Output
    b.attach_as("centrality_score", scores)

    # Build
    algo = b.build()
    assert algo is not None

    g = Graph()
    n1 = g.add_node()
    n2 = g.add_node()
    g.add_edge(n1, n2)

    result = apply(g.view(), algo)
    values = {node.id: getattr(node, "centrality_score") for node in result.nodes}
    assert pytest.approx(sum(values.values()), rel=1e-6) == 1.0

    print("\n✓ Builder DSL example completed")
    print("  Attached 'centrality_score' attribute to nodes")
