"""
Tests for builder core operations (Phase 1.1-1.3).
"""


def test_builder_core_namespace_exists():
    """Verify CoreOps namespace is accessible."""
    from groggy.builder import AlgorithmBuilder

    builder = AlgorithmBuilder("test")
    assert hasattr(builder, "core")
    assert builder.core is not None


def test_builder_core_add():
    """Test addition operation."""
    from groggy.builder import AlgorithmBuilder

    builder = AlgorithmBuilder("test_add")
    x = builder.init_nodes(1.0)
    y = builder.init_nodes(2.0)
    result = builder.core.add(x, y)

    assert result is not None
    assert len(builder.steps) == 3  # init, init, add
    assert builder.steps[2]["type"] == "core.add"


def test_builder_core_mul_scalar():
    """Test scalar multiplication."""
    from groggy.builder import AlgorithmBuilder

    builder = AlgorithmBuilder("test_mul")
    x = builder.init_nodes(1.0)
    scaled = builder.core.mul(x, 0.85)

    assert scaled is not None
    assert len(builder.steps) == 3  # init_nodes, core.constant, mul
    assert builder.steps[1]["type"] == "core.constant"
    assert builder.steps[1]["value"] == 0.85
    assert builder.steps[2]["type"] == "core.mul"


def test_builder_core_normalize_sum():
    """Test sum normalization."""
    from groggy.builder import AlgorithmBuilder

    builder = AlgorithmBuilder("test_normalize")
    x = builder.init_nodes(1.0)
    normalized = builder.core.normalize_sum(x)

    assert normalized is not None
    assert len(builder.steps) == 2  # init, normalize
    assert builder.steps[1]["type"] == "normalize_sum"


def test_builder_var_creation():
    """Test var() for variable reassignment."""
    from groggy.builder import AlgorithmBuilder

    builder = AlgorithmBuilder("test_var")
    x = builder.init_nodes(1.0)
    x = builder.var("x", builder.core.mul(x, 2.0))

    assert x.name == "x"
    assert "x" in builder.variables


def test_builder_arithmetic_chain():
    """Test chaining multiple arithmetic operations."""
    from groggy.builder import AlgorithmBuilder

    builder = AlgorithmBuilder("test_chain")
    x = builder.init_nodes(1.0)

    # Simulate: 0.85 * x + 0.15
    scaled = builder.core.mul(x, 0.85)
    result = builder.core.add(scaled, 0.15)

    assert result is not None
    assert (
        len(builder.steps) == 5
    )  # init_nodes, init_scalar(0.85), mul, init_scalar(0.15), add


def test_builder_step_encoding():
    """Test that steps encode to correct Rust format."""
    from groggy.builder import AlgorithmBuilder

    builder = AlgorithmBuilder("test_encode")
    x = builder.init_nodes(1.0)
    y = builder.core.mul(x, 0.85)
    builder.attach_as("result", y)

    algo = builder.build()

    # Check steps were generated correctly
    assert len(builder.steps) == 4  # init_nodes, core.constant, mul, attach

    # Verify that scalar constant was created
    scalar_step = builder.steps[1]
    assert scalar_step["type"] == "core.constant"
    assert scalar_step["value"] == 0.85

    # Verify mul step structure
    mul_step = builder.steps[2]
    assert mul_step["type"] == "core.mul"

    # Verify encoding
    encoded = algo._encode_step(mul_step, {})
    assert encoded["id"] == "core.mul"
    assert "params" in encoded


def test_builder_core_all_operations():
    """Test all core operations are available."""
    from groggy.builder import AlgorithmBuilder

    builder = AlgorithmBuilder("test_all")

    assert hasattr(builder.core, "add")
    assert hasattr(builder.core, "sub")
    assert hasattr(builder.core, "mul")
    assert hasattr(builder.core, "div")
    assert hasattr(builder.core, "normalize_sum")


def test_builder_map_nodes_basic():
    """Test basic map_nodes operation."""
    from groggy.builder import AlgorithmBuilder

    builder = AlgorithmBuilder("test_map")
    ranks = builder.init_nodes(1.0)

    sums = builder.map_nodes("sum(ranks[neighbors(node)])", inputs={"ranks": ranks})

    assert sums is not None
    assert len(builder.steps) == 2  # init, map_nodes
    assert builder.steps[1]["type"] == "map_nodes"
    assert builder.steps[1]["fn"] == "sum(ranks[neighbors(node)])"
    assert "ranks" in builder.steps[1]["inputs"]


def test_builder_map_nodes_with_context():
    """Test map_nodes with multiple input variables."""
    from groggy.builder import AlgorithmBuilder

    builder = AlgorithmBuilder("test_map_multi")
    values = builder.init_nodes(1.0)
    weights = builder.init_nodes(0.5)

    result = builder.map_nodes(
        "sum(values[neighbors(node)] * weights[neighbors(node)])",
        inputs={"values": values, "weights": weights},
    )

    assert result is not None
    assert len(builder.steps[2]["inputs"]) == 2


def test_builder_node_degrees_directed_chain():
    """Node degrees respect directed edges and preserve ordering."""
    from groggy import Graph
    from groggy.builder import AlgorithmBuilder

    graph = Graph(
        directed=True
    )  # Fixed: test name says "directed" but was using undirected
    n0, n1, n2 = graph.add_node(), graph.add_node(), graph.add_node()
    graph.add_edge(n0, n1)
    graph.add_edge(n1, n2)
    sg = graph.view()

    builder = AlgorithmBuilder("degree_chain")
    init = builder.init_nodes(default=0)
    degrees = builder.node_degrees(init)
    builder.attach_as("deg", degrees)

    result = sg.apply(builder.build())
    deg0 = result.get_node_attribute(n0, "deg")
    deg1 = result.get_node_attribute(n1, "deg")
    deg2 = result.get_node_attribute(n2, "deg")

    assert deg0 == 1
    assert deg1 == 1
    assert deg2 == 0


def test_builder_map_nodes_encoding():
    """Test that map_nodes encodes correctly."""
    from groggy.builder import AlgorithmBuilder

    builder = AlgorithmBuilder("test_encode_map")
    ranks = builder.init_nodes(1.0)
    sums = builder.map_nodes("sum(ranks[neighbors(node)])", inputs={"ranks": ranks})

    algo = builder.build()
    map_step = builder.steps[1]
    encoded = algo._encode_step(map_step, {})

    assert encoded is not None
    assert encoded["id"] == "core.map_nodes"
    assert "expr" in encoded["params"]
    # Expression is parsed to JSON, so check it's a dict with the right structure
    assert isinstance(encoded["params"]["expr"], dict)


def test_builder_iterate_basic():
    """Test basic iteration with structured loop optimization."""
    from groggy import Graph
    from groggy.builder import AlgorithmBuilder

    builder = AlgorithmBuilder("test_iterate")
    x = builder.init_nodes(1.0)

    with builder.iterate(3):
        x = builder.var("x", builder.core.mul(x, 2.0))

    # NEW: With batch optimization, loops are structured not unrolled
    assert len(builder.steps) == 2  # init + iter.loop

    # Check structured loop was created
    loop_step = builder.steps[1]
    assert loop_step["type"] == "iter.loop"
    assert loop_step["iterations"] == 3

    # Check batch optimization occurred
    assert "_batch_optimized" in loop_step
    assert loop_step["_batch_optimized"] == True

    # Verify execution correctness (behavior test)
    graph = Graph()
    for _ in range(3):
        graph.add_node()
    builder.attach_as("result", x)
    result = graph.view().apply(builder.build())
    for node in result.nodes:
        # 1.0 * (2^3) = 8.0
        assert node.result == 8.0


def test_builder_iterate_var_persistence():
    """Test that variables persist across iterations with structured loops."""
    from groggy import Graph
    from groggy.builder import AlgorithmBuilder

    builder = AlgorithmBuilder("test_persist")
    value = builder.init_nodes(1.0)

    with builder.iterate(2):
        value = builder.var("value", builder.core.add(value, 1.0))

    # NEW: Structured loop creates init + iter.loop
    steps = builder.steps
    assert len(steps) == 2
    assert steps[1]["type"] == "iter.loop"
    assert steps[1]["iterations"] == 2

    # Verify execution correctness (behavior test)
    graph = Graph()
    for _ in range(3):
        graph.add_node()
    builder.attach_as("result", value)
    result = graph.view().apply(builder.build())
    for node in result.nodes:
        # 1.0 + 1.0 + 1.0 = 3.0 (2 iterations of adding 1.0)
        assert node.result == 3.0


def test_builder_iterate_complex():
    """Test iteration with multiple operations in structured loop."""
    from groggy import Graph
    from groggy.builder import AlgorithmBuilder

    builder = AlgorithmBuilder("test_complex_loop")
    x = builder.init_nodes(1.0)

    with builder.iterate(2):
        scaled = builder.core.mul(x, 0.85)
        x = builder.var("x", builder.core.add(scaled, 0.15))

    # NEW: Structured loop with multiple operations in body
    assert len(builder.steps) == 2  # init + iter.loop
    loop_step = builder.steps[1]
    assert loop_step["type"] == "iter.loop"
    assert loop_step["iterations"] == 2

    # Check loop body has both operations
    body = loop_step["body"]
    body_types = [s["type"] for s in body]
    assert "core.mul" in body_types or "mul" in body_types
    assert "core.add" in body_types or "add" in body_types

    # Verify execution correctness (behavior test)
    graph = Graph()
    for _ in range(3):
        graph.add_node()
    builder.attach_as("result", x)
    result = graph.view().apply(builder.build())
    for node in result.nodes:
        # Iteration 1: 1.0 * 0.85 + 0.15 = 1.0
        # Iteration 2: 1.0 * 0.85 + 0.15 = 1.0
        assert abs(node.result - 1.0) < 0.01


def test_builder_iterate_with_map_nodes():
    """Test iteration with map_nodes in structured loop (PageRank pattern)."""
    from groggy import Graph
    from groggy.builder import AlgorithmBuilder

    builder = AlgorithmBuilder("test_pagerank_structure")
    ranks = builder.init_nodes(1.0)

    with builder.iterate(2):
        sums = builder.map_nodes("sum(ranks[neighbors(node)])", inputs={"ranks": ranks})
        scaled = builder.core.mul(sums, 0.85)
        ranks = builder.var("ranks", builder.core.add(scaled, 0.15))

    # NEW: Structured loop contains map_nodes in body
    assert len(builder.steps) == 2  # init + iter.loop
    loop_step = builder.steps[1]
    assert loop_step["type"] == "iter.loop"

    # map_nodes is NOT batch-compatible, so loop won't be optimized
    # But it should still execute correctly via fallback
    assert loop_step["iterations"] == 2

    # Check body contains map_nodes
    body = loop_step["body"]
    body_types = [s["type"] for s in body]
    assert "map_nodes" in body_types

    # Verify execution works (even without batch optimization)
    graph = Graph()
    for i in range(3):
        graph.add_node()
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    graph.add_edge(2, 0)

    builder.attach_as("ranks", ranks)
    result = graph.view().apply(builder.build())
    # Just verify it executes without error
    for node in result.nodes:
        assert hasattr(node, "ranks")


def test_builder_input():
    """Test input() for referencing input subgraph."""
    from groggy.builder import AlgorithmBuilder, SubgraphHandle

    builder = AlgorithmBuilder("test_input")

    # Get input reference
    sg_input = builder.input("graph")

    assert sg_input is not None
    assert isinstance(sg_input, SubgraphHandle)
    assert sg_input.name == "graph"

    # Should return same reference on subsequent calls
    sg_input2 = builder.input("different_name")
    assert sg_input is sg_input2
    assert sg_input.name == "graph"  # Keeps original name


def test_builder_input_default_name():
    """Test input() with default name."""
    from groggy.builder import AlgorithmBuilder

    builder = AlgorithmBuilder("test_input_default")

    # Get input with default name
    sg_input = builder.input()

    assert sg_input is not None
    assert sg_input.name == "subgraph"


def test_builder_auto_var():
    """Test auto_var() for public unique name generation."""
    from groggy.builder import AlgorithmBuilder

    builder = AlgorithmBuilder("test_auto_var")

    # Create variables with custom prefix
    temp1 = builder.auto_var("temp")
    temp2 = builder.auto_var("temp")
    result = builder.auto_var("result")

    # Should have unique names
    assert temp1.name != temp2.name
    assert "temp" in temp1.name
    assert "temp" in temp2.name
    assert "result" in result.name


def test_builder_load_attr():
    """Test load_attr() for loading node attributes."""
    from groggy.builder import AlgorithmBuilder

    builder = AlgorithmBuilder("test_load_attr")

    # Load attribute with default
    weights = builder.load_attr("weight", default=1.0)

    assert weights is not None
    assert len(builder.steps) == 1
    assert builder.steps[0]["type"] == "load_attr"
    assert builder.steps[0]["attr_name"] == "weight"
    assert builder.steps[0]["default"] == 1.0


def test_builder_load_edge_attr():
    """Test load_edge_attr() for loading edge attributes."""
    from groggy.builder import AlgorithmBuilder

    builder = AlgorithmBuilder("test_load_edge_attr")

    # Load edge attribute
    edge_weights = builder.load_edge_attr("weight", default=1.0)

    assert edge_weights is not None
    assert len(builder.steps) == 1
    assert builder.steps[0]["type"] == "graph.load_edge_attr"
    assert builder.steps[0]["attr_name"] == "weight"
    assert builder.steps[0]["default"] == 1.0


def test_builder_load_attr_encoding():
    """Test that load_attr encodes correctly."""
    from groggy.builder import AlgorithmBuilder

    builder = AlgorithmBuilder("test_encode_load")
    weights = builder.load_attr("weight", default=1.0)

    algo = builder.build()
    load_step = builder.steps[0]
    encoded = algo._encode_step(load_step, {})

    assert encoded is not None
    assert encoded["id"] == "core.load_node_attr"
    assert "params" in encoded
    assert encoded["params"]["attr"] == "weight"
    assert encoded["params"]["default"] == 1.0


def test_builder_load_edge_attr_encoding():
    """Test that load_edge_attr encodes correctly."""
    from groggy.builder import AlgorithmBuilder

    builder = AlgorithmBuilder("test_encode_load_edge")
    edge_weights = builder.load_edge_attr("weight", default=1.0)

    algo = builder.build()
    load_step = builder.steps[0]
    encoded = algo._encode_step(load_step, {})

    assert encoded is not None
    assert encoded["id"] == "core.load_edge_attr"
    assert "params" in encoded
    assert encoded["params"]["attr"] == "weight"
    assert encoded["params"]["default"] == 1.0


def test_builder_load_attr_with_operations():
    """Test using loaded attributes in operations."""
    from groggy.builder import AlgorithmBuilder

    builder = AlgorithmBuilder("test_load_ops")

    # Load attribute and perform operations
    weights = builder.load_attr("weight", default=1.0)
    doubled = builder.core.mul(weights, 2.0)
    builder.attach_as("doubled_weight", doubled)

    algo = builder.build()

    # Should have: load, core.constant, mul, attach
    assert len(builder.steps) == 4
    assert builder.steps[0]["type"] == "load_attr"
    assert builder.steps[1]["type"] == "core.constant"
    assert builder.steps[2]["type"] == "core.mul"
    assert builder.steps[3]["type"] == "attach_attr"


def test_builder_validation_undefined_variable():
    """Test that validation catches undefined variable references."""
    import warnings

    from groggy.builder import AlgorithmBuilder

    builder = AlgorithmBuilder("test_undefined")

    # Create a step that references undefined variable
    builder.steps.append(
        {"type": "core.mul", "left": "undefined_var", "right": 2.0, "output": "result"}
    )

    # Validation should issue a warning about undefined variable
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        algo = builder.build(validate=True)

        # Check that a warning was issued
        assert len(w) >= 1
        # Warning should mention either the validation issue or missing attributes
        warning_messages = [str(warning.message) for warning in w]
        assert any(
            "validation" in msg.lower() or "attribute" in msg.lower()
            for msg in warning_messages
        )


def test_builder_validation_passes():
    """Test that valid pipelines pass validation."""
    from groggy.builder import AlgorithmBuilder

    builder = AlgorithmBuilder("test_valid")
    x = builder.init_nodes(1.0)
    y = builder.core.mul(x, 2.0)
    builder.attach_as("result", y)

    # Should not raise
    algo = builder.build(validate=True)
    assert algo is not None


def test_builder_validation_can_be_disabled():
    """Test that validation can be disabled."""
    from groggy.builder import AlgorithmBuilder

    builder = AlgorithmBuilder("test_no_validate")

    # Create invalid step
    builder.steps.append(
        {"type": "core.mul", "left": "undefined_var", "right": 2.0, "output": "result"}
    )

    # Should not raise when validation is disabled
    algo = builder.build(validate=False)
    assert algo is not None


def test_builder_validation_warnings():
    """Test that validation produces warnings for non-fatal issues."""
    import warnings

    from groggy.builder import AlgorithmBuilder

    builder = AlgorithmBuilder("test_warnings")
    x = builder.init_nodes(1.0)
    y = builder.core.mul(x, 2.0)
    # Note: no attach_as, should warn

    # Should produce warning but not error
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        algo = builder.build(validate=True)

        # Should have at least one warning
        assert len(w) >= 1
        warning_messages = [str(warning.message) for warning in w]
        assert any("attach" in msg.lower() for msg in warning_messages)


def test_builder_validation_map_nodes_undefined_input():
    """Test that validation catches undefined variables in map_nodes inputs."""
    import pytest

    from groggy.builder import AlgorithmBuilder
    from groggy.errors import ValidationError

    builder = AlgorithmBuilder("test_map_undefined")

    # Create map_nodes with undefined input
    builder.steps.append(
        {
            "type": "map_nodes",
            "fn": "sum(ranks[neighbors(node)])",
            "inputs": {"ranks": "undefined_ranks"},
            "output": "result",
        }
    )

    # Should raise ValidationError
    with pytest.raises(ValidationError) as exc_info:
        algo = builder.build(validate=True)

    assert "undefined_ranks" in str(exc_info.value)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])


def test_builder_core_histogram():
    """Test histogram computation."""
    from groggy import Graph
    from groggy.builder import AlgorithmBuilder

    # Create test graph with known value distribution
    g = Graph()
    for i in range(10):
        g.add_node(i, value=float(i))  # Values 0-9

    # Build histogram algorithm
    builder = AlgorithmBuilder("test_histogram")
    values = builder.load_attr("value", default=0.0)
    hist = builder.core.histogram(values, bins=5)
    builder.attach_as("histogram", hist)

    # Execute
    algo = builder.build()
    result = g.apply(algo)

    # Histogram returns a map where keys are bin indices and values are counts
    # Since we have 10 values (0-9) in 5 bins, each bin should have ~2 values
    # The map will only have entries for bin indices 0-4
    hist_data = {}
    for node in result.nodes:
        if hasattr(node, "histogram"):
            # Node ID corresponds to bin index, value is the count
            hist_data[node.id] = node.histogram

    # Should have 5 bins (0-4)
    assert len(hist_data) == 5
    # Each bin should have approximately 2 values
    for bin_idx, count in hist_data.items():
        assert 0 <= bin_idx < 5
        assert count == 2
