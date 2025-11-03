"""
Tests for builder core operations (Phase 1.1-1.3).
"""

def test_builder_core_namespace_exists():
    """Verify CoreOps namespace is accessible."""
    from groggy.builder import AlgorithmBuilder
    
    builder = AlgorithmBuilder("test")
    assert hasattr(builder, 'core')
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
    assert len(builder.steps) == 3  # init_nodes, init_scalar, mul
    assert builder.steps[1]["type"] == "init_scalar"
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
    assert len(builder.steps) == 5  # init_nodes, init_scalar(0.85), mul, init_scalar(0.15), add


def test_builder_step_encoding():
    """Test that steps encode to correct Rust format."""
    from groggy.builder import AlgorithmBuilder
    
    builder = AlgorithmBuilder("test_encode")
    x = builder.init_nodes(1.0)
    y = builder.core.mul(x, 0.85)
    builder.attach_as("result", y)
    
    algo = builder.build()
    
    # Check steps were generated correctly
    assert len(builder.steps) == 4  # init_nodes, init_scalar, mul, attach
    
    # Verify that scalar init was created
    scalar_step = builder.steps[1]
    assert scalar_step["type"] == "init_scalar"
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
    
    assert hasattr(builder.core, 'add')
    assert hasattr(builder.core, 'sub')
    assert hasattr(builder.core, 'mul')
    assert hasattr(builder.core, 'div')
    assert hasattr(builder.core, 'normalize_sum')


def test_builder_map_nodes_basic():
    """Test basic map_nodes operation."""
    from groggy.builder import AlgorithmBuilder
    
    builder = AlgorithmBuilder("test_map")
    ranks = builder.init_nodes(1.0)
    
    sums = builder.map_nodes(
        "sum(ranks[neighbors(node)])",
        inputs={"ranks": ranks}
    )
    
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
        inputs={"values": values, "weights": weights}
    )
    
    assert result is not None
    assert len(builder.steps[2]["inputs"]) == 2


def test_builder_node_degrees_directed_chain():
    """Node degrees respect directed edges and preserve ordering."""
    from groggy import Graph
    from groggy.builder import AlgorithmBuilder

    graph = Graph()
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
    sums = builder.map_nodes(
        "sum(ranks[neighbors(node)])",
        inputs={"ranks": ranks}
    )
    
    algo = builder.build()
    map_step = builder.steps[1]
    encoded = algo._encode_step(map_step, {})
    
    assert encoded is not None
    assert encoded["id"] == "core.map_nodes"
    assert "expr" in encoded["params"]
    # Expression is parsed to JSON, so check it's a dict with the right structure
    assert isinstance(encoded["params"]["expr"], dict)


def test_builder_iterate_basic():
    """Test basic iteration."""
    from groggy.builder import AlgorithmBuilder
    
    builder = AlgorithmBuilder("test_iterate")
    x = builder.init_nodes(1.0)
    
    with builder.iterate(3):
        x = builder.var("x", builder.core.mul(x, 2.0))
    
    # Should have: init + (mul * 3) + alias
    # Actually: init, then 3 iterations of mul, then alias to restore x
    assert len(builder.steps) > 3
    
    # Count mul steps
    mul_steps = [s for s in builder.steps if s["type"] == "core.mul"]
    assert len(mul_steps) == 3


def test_builder_iterate_var_persistence():
    """Test that variables persist across iterations."""
    from groggy.builder import AlgorithmBuilder
    
    builder = AlgorithmBuilder("test_persist")
    value = builder.init_nodes(1.0)
    
    with builder.iterate(2):
        value = builder.var("value", builder.core.add(value, 1.0))
    
    # Should generate: init, add (iter0), add (iter1), alias
    steps = builder.steps
    assert len(steps) >= 3  # At least init + 2 adds


def test_builder_iterate_complex():
    """Test iteration with multiple operations."""
    from groggy.builder import AlgorithmBuilder
    
    builder = AlgorithmBuilder("test_complex_loop")
    x = builder.init_nodes(1.0)
    
    with builder.iterate(2):
        scaled = builder.core.mul(x, 0.85)
        x = builder.var("x", builder.core.add(scaled, 0.15))
    
    # Should unroll to: init, (mul, add) * 2, alias
    mul_steps = [s for s in builder.steps if s["type"] == "core.mul"]
    add_steps = [s for s in builder.steps if s["type"] == "core.add"]
    
    assert len(mul_steps) == 2
    assert len(add_steps) == 2


def test_builder_iterate_with_map_nodes():
    """Test iteration with map_nodes (simulating PageRank structure)."""
    from groggy.builder import AlgorithmBuilder
    
    builder = AlgorithmBuilder("test_pagerank_structure")
    ranks = builder.init_nodes(1.0)
    
    with builder.iterate(2):
        sums = builder.map_nodes(
            "sum(ranks[neighbors(node)])",
            inputs={"ranks": ranks}
        )
        scaled = builder.core.mul(sums, 0.85)
        ranks = builder.var("ranks", builder.core.add(scaled, 0.15))
    
    # Should have map_nodes steps
    map_steps = [s for s in builder.steps if s["type"] == "map_nodes"]
    assert len(map_steps) == 2  # One per iteration


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
    assert builder.steps[0]["type"] == "load_edge_attr"
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
    
    # Should have: load, init_scalar, mul, attach
    assert len(builder.steps) == 4
    assert builder.steps[0]["type"] == "load_attr"
    assert builder.steps[1]["type"] == "init_scalar"
    assert builder.steps[2]["type"] == "core.mul"
    assert builder.steps[3]["type"] == "attach_attr"


def test_builder_validation_undefined_variable():
    """Test that validation catches undefined variable references."""
    from groggy.builder import AlgorithmBuilder
    from groggy.errors import ValidationError
    import pytest
    
    builder = AlgorithmBuilder("test_undefined")
    
    # Create a step that references undefined variable
    builder.steps.append({
        "type": "core.mul",
        "left": "undefined_var",
        "right": 2.0,
        "output": "result"
    })
    
    # Should raise ValidationError
    with pytest.raises(ValidationError) as exc_info:
        algo = builder.build(validate=True)
    
    # Check error message mentions undefined variable
    assert "undefined_var" in str(exc_info.value)
    assert "undefined" in str(exc_info.value).lower()


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
    builder.steps.append({
        "type": "core.mul",
        "left": "undefined_var",
        "right": 2.0,
        "output": "result"
    })
    
    # Should not raise when validation is disabled
    algo = builder.build(validate=False)
    assert algo is not None


def test_builder_validation_warnings():
    """Test that validation produces warnings for non-fatal issues."""
    from groggy.builder import AlgorithmBuilder
    import warnings
    
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
    from groggy.builder import AlgorithmBuilder
    from groggy.errors import ValidationError
    import pytest
    
    builder = AlgorithmBuilder("test_map_undefined")
    
    # Create map_nodes with undefined input
    builder.steps.append({
        "type": "map_nodes",
        "fn": "sum(ranks[neighbors(node)])",
        "inputs": {"ranks": "undefined_ranks"},
        "output": "result"
    })
    
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
        if hasattr(node, 'histogram'):
            # Node ID corresponds to bin index, value is the count
            hist_data[node.id] = node.histogram
    
    # Should have 5 bins (0-4)
    assert len(hist_data) == 5
    # Each bin should have approximately 2 values
    for bin_idx, count in hist_data.items():
        assert 0 <= bin_idx < 5
        assert count == 2
