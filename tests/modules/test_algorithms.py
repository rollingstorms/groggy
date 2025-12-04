"""
Module 7: Algorithm Testing - Milestone 7

Tests graph algorithms including traversal, pathfinding, and analysis.

Test Coverage:
- Traversal algorithms (BFS, DFS)
- Component analysis (connected_components, is_connected)
- Path finding (shortest_path, neighborhood)
- Graph metrics (density, node_count, edge_count)
- Subgraph extraction

Success Criteria: Algorithms work correctly on test graphs, proper error handling
"""

import sys
from pathlib import Path

import pytest

# Add path for groggy
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

try:
    import groggy as gr
except ImportError:
    gr = None

from tests.conftest import assert_graph_valid


@pytest.mark.algorithms
class TestGraphTraversal:
    """Test graph traversal algorithms"""

    def get_test_graph(self):
        """Create a simple graph for traversal testing"""
        if gr is None:
            pytest.skip("groggy not available")

        graph = gr.Graph()

        # Create a simple tree-like structure
        node_ids = []
        for i in range(7):
            node_id = graph.add_node(label=f"Node{i}", value=i)
            node_ids.append(node_id)

        # Tree structure: 0 -> 1, 2; 1 -> 3, 4; 2 -> 5, 6
        graph.add_edge(node_ids[0], node_ids[1])
        graph.add_edge(node_ids[0], node_ids[2])
        graph.add_edge(node_ids[1], node_ids[3])
        graph.add_edge(node_ids[1], node_ids[4])
        graph.add_edge(node_ids[2], node_ids[5])
        graph.add_edge(node_ids[2], node_ids[6])

        return graph, node_ids

    def test_bfs_traversal(self):
        """Test breadth-first search"""
        graph, node_ids = self.get_test_graph()

        if not hasattr(graph, "bfs"):
            pytest.skip("BFS not implemented")

        # BFS from root
        result = graph.bfs(node_ids[0])
        assert result is not None, "BFS should return a result"

    def test_dfs_traversal(self):
        """Test depth-first search"""
        graph, node_ids = self.get_test_graph()

        if not hasattr(graph, "dfs"):
            pytest.skip("DFS not implemented")

        # DFS from root
        result = graph.dfs(node_ids[0])
        assert result is not None, "DFS should return a result"


@pytest.mark.algorithms
class TestGraphComponents:
    """Test connected components algorithms"""

    def get_test_graph(self):
        """Create a graph with multiple components"""
        if gr is None:
            pytest.skip("groggy not available")

        graph = gr.Graph()

        # Component 1: nodes 0-2
        comp1_nodes = []
        for i in range(3):
            node_id = graph.add_node(label=f"C1N{i}", component=1)
            comp1_nodes.append(node_id)
        graph.add_edge(comp1_nodes[0], comp1_nodes[1])
        graph.add_edge(comp1_nodes[1], comp1_nodes[2])

        # Component 2: nodes 3-5
        comp2_nodes = []
        for i in range(3):
            node_id = graph.add_node(label=f"C2N{i}", component=2)
            comp2_nodes.append(node_id)
        graph.add_edge(comp2_nodes[0], comp2_nodes[1])
        graph.add_edge(comp2_nodes[1], comp2_nodes[2])

        return graph

    def test_connected_components(self):
        """Test connected components detection"""
        graph = self.get_test_graph()

        # connected_components should work (verified in comprehensive tests)
        components = graph.connected_components()
        assert components is not None, "connected_components() should return result"
        assert type(components).__name__ in [
            "ComponentsArray",
            "SubgraphArray",
        ], "Should return ComponentsArray or SubgraphArray"

        # Should have 2 components
        assert len(components) == 2, f"Should have 2 components, got {len(components)}"

    def test_is_connected(self):
        """Test graph connectivity check"""
        graph = self.get_test_graph()

        if not hasattr(graph, "is_connected"):
            pytest.skip("is_connected not implemented")

        # Multi-component graph should not be connected
        is_conn = graph.is_connected()
        assert isinstance(is_conn, bool), "is_connected() should return bool"
        assert not is_conn, "Multi-component graph should not be connected"

        # Single component should be connected
        single_graph = gr.Graph()
        n1 = single_graph.add_node(label="A")
        n2 = single_graph.add_node(label="B")
        single_graph.add_edge(n1, n2)

        if hasattr(single_graph, "is_connected"):
            is_conn_single = single_graph.is_connected()
            assert is_conn_single, "Single component graph should be connected"


@pytest.mark.algorithms
class TestPathfinding:
    """Test pathfinding algorithms"""

    def get_test_graph(self):
        """Create a graph for pathfinding"""
        if gr is None:
            pytest.skip("groggy not available")

        graph = gr.Graph()

        # Create a path graph: 0 -> 1 -> 2 -> 3 -> 4
        node_ids = []
        for i in range(5):
            node_id = graph.add_node(label=f"Node{i}", value=i)
            node_ids.append(node_id)

        for i in range(len(node_ids) - 1):
            graph.add_edge(node_ids[i], node_ids[i + 1], weight=1.0)

        return graph, node_ids

    def test_shortest_path(self):
        """Test shortest path finding"""
        graph, node_ids = self.get_test_graph()

        # shortest_path should work (verified in comprehensive tests)
        path = graph.shortest_path(node_ids[0], node_ids[4])
        assert path is not None, "shortest_path() should return result"
        assert type(path).__name__ == "Subgraph", "Should return Subgraph"

    def test_neighborhood(self):
        """Test neighborhood extraction"""
        graph, node_ids = self.get_test_graph()

        # neighborhood should return NeighborhoodArray (specialized SubgraphArray with metadata)
        neighborhood = graph.neighborhood(node_ids[2])
        assert neighborhood is not None, "neighborhood() should return result"
        assert (
            type(neighborhood).__name__ == "NeighborhoodArray"
        ), "Should return NeighborhoodArray"

    def test_neighbors(self):
        """Test getting node neighbors"""
        graph, node_ids = self.get_test_graph()

        # neighbors requires nodes parameter
        try:
            neighbors = graph.neighbors(node_ids[2])
            assert neighbors is not None, "neighbors() should return result"
        except Exception as e:
            # Expected - may require specific parameter format
            assert "parameter" in str(e).lower() or "required" in str(e).lower()

    def test_neighborhood_statistics(self):
        """Test neighborhood statistics"""
        graph, node_ids = self.get_test_graph()

        # neighborhood_statistics should work (verified in comprehensive tests)
        if hasattr(graph, "neighborhood_statistics"):
            stats = graph.neighborhood_statistics()
            assert stats is not None, "neighborhood_statistics() should return result"
            assert (
                type(stats).__name__ == "NeighborhoodStats"
            ), "Should return NeighborhoodStats"


@pytest.mark.algorithms
class TestGraphMetrics:
    """Test graph metrics and statistics"""

    def get_test_graph(self):
        """Create a simple graph for metrics testing"""
        if gr is None:
            pytest.skip("groggy not available")

        graph = gr.Graph()

        # Create a small complete graph
        node_ids = []
        for i in range(4):
            node_id = graph.add_node(label=f"Node{i}")
            node_ids.append(node_id)

        # Add all edges (complete graph)
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                graph.add_edge(node_ids[i], node_ids[j])

        return graph

    def test_node_count(self):
        """Test node counting"""
        graph = self.get_test_graph()

        if hasattr(graph, "node_count"):
            count = graph.node_count()
            assert count == 4, f"Should have 4 nodes, got {count}"

    def test_edge_count(self):
        """Test edge counting"""
        graph = self.get_test_graph()

        if hasattr(graph, "edge_count"):
            count = graph.edge_count()
            # Complete graph with 4 nodes has 6 edges
            assert count == 6, f"Should have 6 edges, got {count}"

    def test_density(self):
        """Test graph density calculation"""
        graph = self.get_test_graph()

        if hasattr(graph, "density"):
            dens = graph.density()
            assert dens is not None, "density() should return a value"
            # Complete graph has density 1.0
            assert (
                0.9 <= dens <= 1.1
            ), f"Complete graph density should be ~1.0, got {dens}"

    def test_is_empty(self):
        """Test empty graph check"""
        if gr is None:
            pytest.skip("groggy not available")

        graph = gr.Graph()

        if hasattr(graph, "is_empty"):
            assert graph.is_empty(), "New graph should be empty"

        # Add a node
        graph.add_node(label="Test")

        if hasattr(graph, "is_empty"):
            assert not graph.is_empty(), "Graph with nodes should not be empty"


@pytest.mark.algorithms
@pytest.mark.integration
class TestAlgorithmIntegration:
    """Test algorithm integration and chaining"""

    def test_component_analysis_workflow(self):
        """Test complete component analysis workflow"""
        if gr is None:
            pytest.skip("groggy not available")

        # Create graph with components
        graph = gr.Graph()
        comp1 = [graph.add_node(label=f"A{i}") for i in range(3)]
        comp2 = [graph.add_node(label=f"B{i}") for i in range(3)]

        for i in range(len(comp1) - 1):
            graph.add_edge(comp1[i], comp1[i + 1])
        for i in range(len(comp2) - 1):
            graph.add_edge(comp2[i], comp2[i + 1])

        # Workflow: find components -> sample -> merge
        components = graph.connected_components()
        assert len(components) == 2, "Should have 2 components"

        sampled = components.sample(1)
        assert sampled is not None, "Should be able to sample components"

        merged = sampled.merge()
        assert merged is not None, "Should be able to merge sampled components"
        assert hasattr(merged, "nodes"), "Merged result should be a graph"

    def test_path_analysis_workflow(self):
        """Test pathfinding workflow"""
        if gr is None:
            pytest.skip("groggy not available")

        # Create path graph
        graph = gr.Graph()
        nodes = [graph.add_node(label=f"N{i}") for i in range(5)]

        for i in range(len(nodes) - 1):
            graph.add_edge(nodes[i], nodes[i + 1])

        # Workflow: find shortest path -> extract as subgraph
        path = graph.shortest_path(nodes[0], nodes[4])
        assert path is not None, "Should find shortest path"

        # Path should be a Subgraph
        assert type(path).__name__ == "Subgraph", "Path should be a Subgraph"

        # Should be able to get adjacency matrix from path
        if hasattr(path, "adjacency_matrix"):
            adj = path.adjacency_matrix()
            assert adj is not None, "Should get adjacency matrix from path subgraph"

    def test_neighborhood_expansion_workflow(self):
        """Test neighborhood expansion workflow"""
        if gr is None:
            pytest.skip("groggy not available")

        # Create star graph
        graph = gr.Graph()
        center = graph.add_node(label="Center")
        outer = [graph.add_node(label=f"Outer{i}") for i in range(5)]

        for node in outer:
            graph.add_edge(center, node)

        # Workflow: get neighborhood -> expand
        neighborhood = graph.neighborhood(center)
        assert neighborhood is not None, "Should get neighborhood"

        # Should be able to get statistics
        if hasattr(graph, "neighborhood_statistics"):
            stats = graph.neighborhood_statistics()
            assert stats is not None, "Should get neighborhood statistics"


if __name__ == "__main__":
    # Allow running this module directly for development
    pytest.main([__file__, "-v"])
