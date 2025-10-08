"""
Module 8.2: Performance Testing - Milestone 8

Tests performance characteristics of groggy operations.
Validates that operations scale appropriately with data size.

Test Coverage:
- Graph construction performance
- Array operations scaling
- Matrix operations efficiency
- Algorithm complexity
- Memory usage patterns

Success Criteria: Operations complete within acceptable time bounds
"""

import pytest
import sys
import time
from pathlib import Path

# Add path for groggy
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

try:
    import groggy as gr
except ImportError:
    gr = None


@pytest.mark.performance
class TestGraphConstructionPerformance:
    """Test graph construction performance"""

    @pytest.mark.parametrize("size", [100, 500, 1000])
    def test_node_addition_performance(self, size):
        """Test node addition scales linearly"""
        if gr is None:
            pytest.skip("groggy not available")

        graph = gr.Graph()

        start_time = time.time()
        for i in range(size):
            graph.add_node(label=f"Node{i}", value=i)
        elapsed = time.time() - start_time

        # Should complete in reasonable time (< 1 second per 1000 nodes)
        max_time = size / 1000.0
        assert elapsed < max_time, \
            f"Adding {size} nodes took {elapsed:.3f}s, should be < {max_time:.3f}s"

    @pytest.mark.parametrize("size", [100, 500, 1000])
    def test_edge_addition_performance(self, size):
        """Test edge addition scales linearly"""
        if gr is None:
            pytest.skip("groggy not available")

        graph = gr.Graph()

        # Add nodes first
        nodes = [graph.add_node(label=f"Node{i}") for i in range(size)]

        # Add edges (path graph)
        start_time = time.time()
        for i in range(size - 1):
            graph.add_edge(nodes[i], nodes[i + 1])
        elapsed = time.time() - start_time

        # Should complete in reasonable time
        max_time = size / 1000.0
        assert elapsed < max_time, \
            f"Adding {size-1} edges took {elapsed:.3f}s, should be < {max_time:.3f}s"


@pytest.mark.performance
class TestArrayOperationsPerformance:
    """Test array operations performance"""

    @pytest.mark.parametrize("size", [100, 500])
    def test_array_filtering_performance(self, size):
        """Test array filtering performance"""
        if gr is None:
            pytest.skip("groggy not available")

        graph = gr.Graph()
        for i in range(size):
            graph.add_node(label=f"Node{i}", value=i, category=i % 10)

        start_time = time.time()
        filtered = graph.nodes[graph.nodes["category"] == 5]
        elapsed = time.time() - start_time

        # Filtering should be fast
        assert elapsed < 0.5, f"Filtering {size} nodes took {elapsed:.3f}s, should be < 0.5s"

    @pytest.mark.parametrize("size", [100, 500])
    def test_array_grouping_performance(self, size):
        """Test array grouping performance"""
        if gr is None:
            pytest.skip("groggy not available")

        graph = gr.Graph()
        for i in range(size):
            graph.add_node(label=f"Node{i}", group=i % 10)

        start_time = time.time()
        groups = graph.nodes.group_by("group")
        elapsed = time.time() - start_time

        # Grouping should be efficient
        assert elapsed < 1.0, f"Grouping {size} nodes took {elapsed:.3f}s, should be < 1.0s"
        assert len(groups) == 10, "Should have 10 groups"


@pytest.mark.performance
class TestMatrixOperationsPerformance:
    """Test matrix operations performance"""

    @pytest.mark.parametrize("size", [50, 100])
    def test_matrix_creation_performance(self, size):
        """Test matrix creation performance"""
        if gr is None:
            pytest.skip("groggy not available")

        graph = gr.Graph()
        nodes = [graph.add_node(label=f"Node{i}") for i in range(size)]

        # Create dense connections
        for i in range(min(size - 1, 50)):
            graph.add_edge(nodes[i], nodes[i + 1])

        start_time = time.time()
        matrix = graph.to_matrix()
        elapsed = time.time() - start_time

        # Matrix creation should be fast
        assert elapsed < 1.0, f"Creating {size}x{size} matrix took {elapsed:.3f}s"

    @pytest.mark.parametrize("size", [50, 100])
    def test_matrix_operations_performance(self, size):
        """Test matrix operation chaining performance"""
        if gr is None:
            pytest.skip("groggy not available")

        graph = gr.Graph()
        nodes = [graph.add_node(label=f"Node{i}") for i in range(size)]

        for i in range(min(size - 1, 50)):
            graph.add_edge(nodes[i], nodes[i + 1])

        matrix = graph.to_matrix()

        start_time = time.time()
        result = matrix.dense().abs().flatten()
        elapsed = time.time() - start_time

        # Chained operations should be efficient
        assert elapsed < 1.0, f"Matrix operations on {size}x{size} took {elapsed:.3f}s"


@pytest.mark.performance
class TestAlgorithmPerformance:
    """Test graph algorithm performance"""

    @pytest.mark.parametrize("size", [100, 500])
    def test_connected_components_performance(self, size):
        """Test connected components algorithm performance"""
        if gr is None:
            pytest.skip("groggy not available")

        graph = gr.Graph()

        # Create multiple components
        num_components = 5
        nodes_per_component = size // num_components

        for comp in range(num_components):
            comp_nodes = []
            for i in range(nodes_per_component):
                node = graph.add_node(
                    label=f"C{comp}N{i}",
                    component=comp
                )
                comp_nodes.append(node)

            # Connect within component
            for i in range(len(comp_nodes) - 1):
                graph.add_edge(comp_nodes[i], comp_nodes[i + 1])

        start_time = time.time()
        components = graph.connected_components()
        elapsed = time.time() - start_time

        # Component detection should be efficient
        assert elapsed < 1.0, \
            f"Finding components in {size} node graph took {elapsed:.3f}s"
        assert len(components) == num_components, \
            f"Should find {num_components} components"

    @pytest.mark.parametrize("size", [50, 100])
    def test_shortest_path_performance(self, size):
        """Test shortest path algorithm performance"""
        if gr is None:
            pytest.skip("groggy not available")

        graph = gr.Graph()
        nodes = [graph.add_node(label=f"Node{i}") for i in range(size)]

        # Create path graph
        for i in range(size - 1):
            graph.add_edge(nodes[i], nodes[i + 1])

        start_time = time.time()
        path = graph.shortest_path(nodes[0], nodes[-1])
        elapsed = time.time() - start_time

        # Pathfinding should be fast
        assert elapsed < 0.5, \
            f"Finding path in {size} node graph took {elapsed:.3f}s"
        assert path is not None, "Should find path"


@pytest.mark.performance
class TestTableOperationsPerformance:
    """Test table operations performance"""

    @pytest.mark.parametrize("size", [100, 500])
    def test_table_creation_performance(self, size):
        """Test table creation performance"""
        if gr is None:
            pytest.skip("groggy not available")

        graph = gr.Graph()
        for i in range(size):
            graph.add_node(label=f"Node{i}", value=i)

        start_time = time.time()
        table = graph.nodes.table()
        elapsed = time.time() - start_time

        # Table creation should be fast
        assert elapsed < 0.5, \
            f"Creating table from {size} nodes took {elapsed:.3f}s"

    @pytest.mark.parametrize("size", [100, 500])
    def test_table_operations_performance(self, size):
        """Test table operations performance"""
        if gr is None:
            pytest.skip("groggy not available")

        graph = gr.Graph()
        for i in range(size):
            graph.add_node(label=f"Node{i}", value=i)

        table = graph.nodes.table()

        start_time = time.time()
        head = table.head()
        shape = table.shape()
        elapsed = time.time() - start_time

        # Table operations should be fast
        assert elapsed < 0.2, \
            f"Table operations on {size} rows took {elapsed:.3f}s"


@pytest.mark.performance
@pytest.mark.slow
class TestScalabilityLimits:
    """Test scalability with larger datasets"""

    def test_large_graph_construction(self):
        """Test constructing a large graph"""
        if gr is None:
            pytest.skip("groggy not available")

        size = 5000
        graph = gr.Graph()

        start_time = time.time()
        nodes = [graph.add_node(label=f"Node{i}", value=i) for i in range(size)]
        elapsed = time.time() - start_time

        # Should handle 5k nodes efficiently
        assert elapsed < 5.0, \
            f"Creating {size} nodes took {elapsed:.3f}s, should be < 5.0s"

        # Add edges
        start_time = time.time()
        for i in range(0, size - 1, 10):  # Sparse edges
            graph.add_edge(nodes[i], nodes[i + 1])
        elapsed = time.time() - start_time

        assert elapsed < 2.0, \
            f"Adding edges to {size} node graph took {elapsed:.3f}s"

    def test_large_component_analysis(self):
        """Test component analysis on large graph"""
        if gr is None:
            pytest.skip("groggy not available")

        size = 2000
        graph = gr.Graph()

        # Create 10 components
        for comp in range(10):
            comp_nodes = []
            for i in range(size // 10):
                node = graph.add_node(
                    label=f"C{comp}N{i}",
                    component=comp
                )
                comp_nodes.append(node)

            for i in range(len(comp_nodes) - 1):
                graph.add_edge(comp_nodes[i], comp_nodes[i + 1])

        start_time = time.time()
        components = graph.connected_components()
        elapsed = time.time() - start_time

        # Should handle large component analysis
        assert elapsed < 3.0, \
            f"Component analysis on {size} nodes took {elapsed:.3f}s"
        assert len(components) == 10, "Should find 10 components"


if __name__ == "__main__":
    # Allow running this module directly for development
    pytest.main([__file__, "-v"])
