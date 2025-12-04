"""
Pytest Configuration for Groggy Modular Testing

Provides shared fixtures, configuration, and test utilities across all modules.
Part of the Milestone-based testing strategy.
"""

import sys
from pathlib import Path

import pytest

# Add groggy to path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

try:
    import groggy as gr
    from tests.fixtures.graph_samples import load_test_graph
    from tests.fixtures.smart_fixtures import FixtureFactory, GraphFixtures
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


@pytest.fixture(
    params=["empty", "path", "cycle", "star", "complete", "karate", "social"]
)
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


# Array-specific fixtures for Milestone 2
@pytest.fixture
def small_num_array():
    """Fixture for a small NumArray (5 elements)"""
    if gr is None:
        pytest.skip("groggy not available")
    graph = gr.Graph()
    for i in range(5):
        graph.add_node(value=i * 2)
    return graph.node_ids


@pytest.fixture
def medium_num_array():
    """Fixture for a medium NumArray (50 elements)"""
    if gr is None:
        pytest.skip("groggy not available")
    graph = gr.Graph()
    for i in range(50):
        graph.add_node(value=i, score=i * 0.1)
    return graph.node_ids


@pytest.fixture
def large_num_array():
    """Fixture for a large NumArray (500 elements)"""
    if gr is None:
        pytest.skip("groggy not available")
    graph = gr.Graph()
    for i in range(500):
        graph.add_node(value=i, score=i * 0.01)
    return graph.node_ids


@pytest.fixture
def small_nodes_array():
    """Fixture for a small NodesArray"""
    if gr is None:
        pytest.skip("groggy not available")
    graph = gr.Graph()
    for i in range(3):
        graph.add_node(label=f"Node{i}", value=i)
    return graph.nodes.array()


@pytest.fixture
def medium_nodes_array():
    """Fixture for a medium NodesArray"""
    if gr is None:
        pytest.skip("groggy not available")
    graph = gr.Graph()
    for i in range(20):
        graph.add_node(label=f"Node{i}", value=i, active=i % 2 == 0)
    return graph.nodes.array()


@pytest.fixture
def small_edges_array():
    """Fixture for a small EdgesArray"""
    if gr is None:
        pytest.skip("groggy not available")
    graph = gr.Graph()
    nodes = []
    for i in range(4):
        node_id = graph.add_node(label=f"Node{i}")
        nodes.append(node_id)

    # Create a simple path
    for i in range(len(nodes) - 1):
        graph.add_edge(nodes[i], nodes[i + 1], weight=i + 1)

    return graph.edges.array()


@pytest.fixture
def medium_edges_array():
    """Fixture for a medium EdgesArray"""
    if gr is None:
        pytest.skip("groggy not available")
    graph = gr.Graph()
    nodes = []
    for i in range(10):
        node_id = graph.add_node(label=f"Node{i}")
        nodes.append(node_id)

    # Create a more complex graph (each node connects to next 2 nodes)
    for i in range(len(nodes)):
        for j in range(1, min(3, len(nodes) - i)):
            if i + j < len(nodes):
                graph.add_edge(nodes[i], nodes[i + j], weight=(i + j) * 0.1)

    return graph.edges.array()


@pytest.fixture(params=["small", "medium"])
def parametric_num_array(request):
    """Parametric fixture for NumArrays of different sizes"""
    if gr is None:
        pytest.skip("groggy not available")

    size_map = {"small": 5, "medium": 50}

    size = size_map[request.param]
    graph = gr.Graph()
    for i in range(size):
        graph.add_node(value=i, score=i * 0.1)

    return graph.node_ids


@pytest.fixture(params=["small", "medium"])
def parametric_nodes_array(request):
    """Parametric fixture for NodesArrays of different sizes"""
    if gr is None:
        pytest.skip("groggy not available")

    size_map = {"small": 3, "medium": 20}

    size = size_map[request.param]
    graph = gr.Graph()
    for i in range(size):
        graph.add_node(label=f"Node{i}", value=i)

    return graph.nodes.array()


@pytest.fixture(params=["small", "medium"])
def parametric_edges_array(request):
    """Parametric fixture for EdgesArrays of different sizes"""
    if gr is None:
        pytest.skip("groggy not available")

    node_counts = {"small": 4, "medium": 10}

    node_count = node_counts[request.param]
    graph = gr.Graph()
    nodes = []
    for i in range(node_count):
        node_id = graph.add_node(label=f"Node{i}")
        nodes.append(node_id)

    # Create edges (path graph for small, more connected for medium)
    if request.param == "small":
        for i in range(len(nodes) - 1):
            graph.add_edge(nodes[i], nodes[i + 1], weight=i + 1)
    else:  # medium
        for i in range(len(nodes)):
            for j in range(1, min(3, len(nodes) - i)):
                if i + j < len(nodes):
                    graph.add_edge(nodes[i], nodes[i + j], weight=(i + j) * 0.1)

    return graph.edges.array()


# Accessor-specific fixtures for Milestone 3
@pytest.fixture
def diverse_graph():
    """Fixture for a graph with diverse node and edge attributes for accessor testing"""
    if gr is None:
        pytest.skip("groggy not available")

    graph = gr.Graph()

    # Add nodes with diverse attributes
    node_ids = []
    for i in range(8):
        node_id = graph.add_node(
            label=f"Node{i}",
            value=i * 2,
            category="A" if i % 2 == 0 else "B",
            active=i % 3 == 0,
            priority="high" if i < 3 else "low",
        )
        node_ids.append(node_id)

    # Add edges with diverse attributes
    edge_ids = []
    for i in range(len(node_ids) - 1):
        edge_id = graph.add_edge(
            node_ids[i],
            node_ids[i + 1],
            strength=i * 0.1,
            relationship="friend" if i % 2 == 0 else "colleague",
            years_known=i + 1,
        )
        edge_ids.append(edge_id)

    # Add a few more complex edges
    if len(node_ids) >= 4:
        graph.add_edge(node_ids[0], node_ids[3], relationship="family", strength=0.9)
        graph.add_edge(node_ids[1], node_ids[4], relationship="friend", strength=0.7)

    return graph


@pytest.fixture
def nodes_accessor(diverse_graph):
    """Fixture for NodesAccessor from diverse graph"""
    return diverse_graph.nodes


@pytest.fixture
def edges_accessor(diverse_graph):
    """Fixture for EdgesAccessor from diverse graph"""
    return diverse_graph.edges


@pytest.fixture
def large_accessor_graph():
    """Fixture for performance testing accessors with larger graphs"""
    if gr is None:
        pytest.skip("groggy not available")

    graph = gr.Graph()
    node_count = 100

    # Add many nodes with attributes
    node_ids = []
    for i in range(node_count):
        node_id = graph.add_node(
            label=f"Node{i}",
            value=i,
            category=f"Cat{i % 5}",  # 5 categories
            priority="high" if i % 10 == 0 else "low",
        )
        node_ids.append(node_id)

    # Add edges (create a connected graph)
    for i in range(node_count - 1):
        graph.add_edge(
            node_ids[i], node_ids[i + 1], weight=i * 0.01, relationship="connection"
        )

    # Add some random additional edges
    import random

    random.seed(42)  # For reproducible tests
    for _ in range(50):
        i, j = random.sample(range(node_count), 2)
        try:
            graph.add_edge(node_ids[i], node_ids[j], relationship="random", weight=0.5)
        except:
            # Edge might already exist
            pass

    return graph


# Table-specific fixtures for Milestone 4
@pytest.fixture
def table_test_graph():
    """Fixture for a graph with good tabular data for table testing"""
    if gr is None:
        pytest.skip("groggy not available")

    graph = gr.Graph()

    # Add nodes with diverse attributes for table operations
    node_ids = []
    for i in range(10):
        node_id = graph.add_node(
            label=f"Node{i}",
            value=i * 10,
            category="A" if i % 3 == 0 else ("B" if i % 3 == 1 else "C"),
            active=i % 2 == 0,
            score=i * 0.1,
            name=f"Name_{i}",
            priority="high" if i < 3 else ("medium" if i < 7 else "low"),
        )
        node_ids.append(node_id)

    # Add edges with diverse attributes for table operations
    edge_ids = []
    for i in range(len(node_ids) - 1):
        edge_id = graph.add_edge(
            node_ids[i],
            node_ids[i + 1],
            weight=i * 0.1,
            relationship=(
                "friend" if i % 3 == 0 else ("colleague" if i % 3 == 1 else "family")
            ),
            years_known=i + 1,
            strength=i * 0.05,
            type="strong" if i % 2 == 0 else "weak",
        )
        edge_ids.append(edge_id)

    # Add some additional complex edges
    if len(node_ids) >= 5:
        graph.add_edge(
            node_ids[0], node_ids[4], relationship="mentor", weight=0.9, years_known=10
        )
        graph.add_edge(
            node_ids[2], node_ids[7], relationship="friend", weight=0.7, years_known=5
        )
        graph.add_edge(
            node_ids[1],
            node_ids[8],
            relationship="colleague",
            weight=0.6,
            years_known=3,
        )

    return graph


@pytest.fixture
def graph_table(table_test_graph):
    """Fixture for GraphTable from table test graph"""
    return table_test_graph.table()


@pytest.fixture
def nodes_table(table_test_graph):
    """Fixture for NodesTable from table test graph"""
    return table_test_graph.nodes.table()


@pytest.fixture
def edges_table(table_test_graph):
    """Fixture for EdgesTable from table test graph"""
    return table_test_graph.edges.table()


@pytest.fixture
def large_table_graph():
    """Fixture for performance testing tables with larger graphs"""
    if gr is None:
        pytest.skip("groggy not available")

    graph = gr.Graph()
    node_count = 200

    # Add many nodes with diverse attributes for table testing
    node_ids = []
    for i in range(node_count):
        node_id = graph.add_node(
            label=f"Node{i}",
            value=i,
            category=f"Cat{i % 8}",  # 8 categories for good grouping
            active=i % 4 == 0,
            score=i * 0.01,
            priority="high" if i % 20 == 0 else ("medium" if i % 10 == 0 else "low"),
        )
        node_ids.append(node_id)

    # Add many edges with attributes
    edge_count = 0
    for i in range(node_count - 1):
        # Create a connected graph
        graph.add_edge(
            node_ids[i],
            node_ids[i + 1],
            weight=i * 0.005,
            relationship="connection",
            strength=i * 0.002,
        )
        edge_count += 1

    # Add some random additional edges for complexity
    import random

    random.seed(42)  # For reproducible tests
    for _ in range(100):
        i, j = random.sample(range(node_count), 2)
        try:
            graph.add_edge(
                node_ids[i],
                node_ids[j],
                relationship=random.choice(["friend", "colleague", "family"]),
                weight=random.uniform(0.1, 1.0),
            )
            edge_count += 1
        except:
            # Edge might already exist
            pass

    return graph


# Test configuration
def pytest_configure(config):
    """Configure pytest with custom markers and settings"""
    config.addinivalue_line("markers", "graph_core: tests for core Graph functionality")
    config.addinivalue_line("markers", "array_ops: tests for array operations")
    config.addinivalue_line("markers", "num_array: tests for NumArray functionality")
    config.addinivalue_line(
        "markers", "nodes_array: tests for NodesArray functionality"
    )
    config.addinivalue_line(
        "markers", "edges_array: tests for EdgesArray functionality"
    )
    config.addinivalue_line(
        "markers", "array_base: tests for shared array functionality"
    )
    config.addinivalue_line("markers", "base_array: tests for BaseArray functionality")
    config.addinivalue_line("markers", "graph_arrays: tests for graph-specific arrays")
    config.addinivalue_line("markers", "table_ops: tests for table operations")
    config.addinivalue_line(
        "markers", "subgraph_array: tests for SubgraphArray functionality"
    )
    config.addinivalue_line(
        "markers", "components: tests for ComponentsArray functionality"
    )
    config.addinivalue_line("markers", "graph_view: tests for GraphView functionality")
    config.addinivalue_line(
        "markers", "graph_matrix: tests for GraphMatrix functionality"
    )
    config.addinivalue_line("markers", "matrix_operations: tests for matrix operations")
    config.addinivalue_line("markers", "algorithms: tests for graph algorithms")
    config.addinivalue_line(
        "markers", "neural: tests for neural network and autograd operations"
    )
    config.addinivalue_line(
        "markers", "table_base: tests for shared table functionality"
    )
    config.addinivalue_line(
        "markers", "graph_table: tests for GraphTable functionality"
    )
    config.addinivalue_line(
        "markers", "nodes_table: tests for NodesTable functionality"
    )
    config.addinivalue_line(
        "markers", "edges_table: tests for EdgesTable functionality"
    )
    config.addinivalue_line(
        "markers", "nodes_edges_tables: tests for table integration"
    )
    config.addinivalue_line("markers", "accessor_ops: tests for accessor operations")
    config.addinivalue_line(
        "markers", "nodes_accessor: tests for NodesAccessor functionality"
    )
    config.addinivalue_line(
        "markers", "edges_accessor: tests for EdgesAccessor functionality"
    )
    config.addinivalue_line("markers", "accessors: tests for accessor integration")
    config.addinivalue_line("markers", "subgraph_ops: tests for subgraph operations")
    config.addinivalue_line("markers", "matrix_ops: tests for matrix operations")
    config.addinivalue_line("markers", "integration: integration tests across modules")
    config.addinivalue_line("markers", "performance: performance and stress tests")
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
        if hasattr(item, "obj") and hasattr(item.obj, "__name__"):
            if "performance" in item.obj.__name__ or "stress" in item.obj.__name__:
                item.add_marker(pytest.mark.slow)


# Helper functions for tests
def assert_graph_valid(graph):
    """Helper to validate graph state consistency"""
    if gr is None:
        return True

    # Basic consistency checks
    assert hasattr(graph, "nodes")
    assert hasattr(graph, "edges")
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
