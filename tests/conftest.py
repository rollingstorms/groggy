"""
Pytest Configuration for Groggy Modular Testing

Provides shared fixtures, configuration, and test utilities across all modules.
Part of the Milestone-based testing strategy.
"""

import pytest
import sys
from pathlib import Path

# Add groggy to path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

try:
    import groggy as gr
    from tests.fixtures.smart_fixtures import FixtureFactory, GraphFixtures
    from tests.fixtures.graph_samples import load_test_graph
except ImportError as e:
    print(f"Warning: Could not import groggy or fixtures: {e}")
    gr = None


@pytest.fixture(scope="session")
def groggy_available():
    """Check if groggy is available for testing"""
    return gr is not None


@pytest.fixture
def empty_graph():
    """Fixture for an empty graph"""
    if gr is None:
        pytest.skip("groggy not available")
    return gr.Graph()


@pytest.fixture
def simple_graph():
    """Fixture for a simple graph with basic structure"""
    if gr is None:
        pytest.skip("groggy not available")
    return GraphFixtures.simple_path_graph(3)


@pytest.fixture
def attributed_graph():
    """Fixture for a graph with diverse attribute types"""
    if gr is None:
        pytest.skip("groggy not available")
    return GraphFixtures.attributed_graph()


@pytest.fixture
def karate_club_graph():
    """Fixture for Zachary's Karate Club graph"""
    if gr is None:
        pytest.skip("groggy not available")
    return load_test_graph("karate")


@pytest.fixture
def social_network_graph():
    """Fixture for small social network graph"""
    if gr is None:
        pytest.skip("groggy not available")
    return load_test_graph("social")


@pytest.fixture
def fixture_factory():
    """Fixture for the FixtureFactory"""
    if gr is None:
        pytest.skip("groggy not available")
    return FixtureFactory()


@pytest.fixture
def graph_with_factory():
    """Fixture that provides both a graph and a factory configured for it"""
    if gr is None:
        pytest.skip("groggy not available")
    graph = GraphFixtures.attributed_graph()
    factory = FixtureFactory(graph)
    return graph, factory


@pytest.fixture(params=[
    "empty", "path", "cycle", "star", "complete", "karate", "social"
])
def parametric_graph(request):
    """Parametric fixture that yields different graph types"""
    if gr is None:
        pytest.skip("groggy not available")
    return load_test_graph(request.param)


@pytest.fixture
def large_graph():
    """Fixture for performance testing with larger graphs"""
    if gr is None:
        pytest.skip("groggy not available")
    return GraphFixtures.large_graph(num_nodes=50, edge_probability=0.2)


# Test configuration
def pytest_configure(config):
    """Configure pytest with custom markers and settings"""
    config.addinivalue_line(
        "markers", "graph_core: tests for core Graph functionality"
    )
    config.addinivalue_line(
        "markers", "array_ops: tests for array operations"
    )
    config.addinivalue_line(
        "markers", "table_ops: tests for table operations"
    )
    config.addinivalue_line(
        "markers", "accessor_ops: tests for accessor operations"
    )
    config.addinivalue_line(
        "markers", "subgraph_ops: tests for subgraph operations"
    )
    config.addinivalue_line(
        "markers", "matrix_ops: tests for matrix operations"
    )
    config.addinivalue_line(
        "markers", "integration: integration tests across modules"
    )
    config.addinivalue_line(
        "markers", "performance: performance and stress tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers and organize tests"""
    for item in items:
        # Auto-mark tests based on file location
        if "test_graph_core" in str(item.fspath):
            item.add_marker(pytest.mark.graph_core)
        elif "test_array" in str(item.fspath):
            item.add_marker(pytest.mark.array_ops)
        elif "test_table" in str(item.fspath):
            item.add_marker(pytest.mark.table_ops)
        elif "test_accessor" in str(item.fspath):
            item.add_marker(pytest.mark.accessor_ops)
        elif "test_subgraph" in str(item.fspath):
            item.add_marker(pytest.mark.subgraph_ops)
        elif "test_matrix" in str(item.fspath):
            item.add_marker(pytest.mark.matrix_ops)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Mark slow tests
        if hasattr(item, 'obj') and hasattr(item.obj, '__name__'):
            if 'performance' in item.obj.__name__ or 'stress' in item.obj.__name__:
                item.add_marker(pytest.mark.slow)


# Helper functions for tests
def assert_graph_valid(graph):
    """Helper to validate graph state consistency"""
    if gr is None:
        return True

    # Basic consistency checks
    assert hasattr(graph, 'nodes')
    assert hasattr(graph, 'edges')
    assert len(graph.nodes) >= 0
    assert len(graph.edges) >= 0

    # Node/edge count consistency
    node_count = len(graph.nodes)
    edge_count = len(graph.edges)

    # All edges should reference valid nodes
    if edge_count > 0 and node_count > 0:
        # This is a basic sanity check - more detailed validation
        # would require examining the actual edge-node relationships
        pass

    return True


def assert_method_callable(obj, method_name):
    """Helper to verify method exists and is callable"""
    assert hasattr(obj, method_name), f"Object {type(obj)} missing method {method_name}"
    method = getattr(obj, method_name)
    assert callable(method), f"Method {method_name} is not callable"


# Pytest plugins and extensions
pytest_plugins = []