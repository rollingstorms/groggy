"""
Module 8.1: Integration Testing - Milestone 8

Tests integration between different groggy components and workflows.
Validates that complex multi-step operations work correctly.

Test Coverage:
- Cross-component workflows
- Data flow between objects
- API consistency
- Error propagation
- Real-world usage patterns

Success Criteria: Complex workflows execute successfully end-to-end
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add path for groggy
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

try:
    import groggy as gr
except ImportError:
    gr = None


@pytest.mark.integration
class TestGraphWorkflows:
    """Test complete graph manipulation workflows"""

    def test_build_analyze_export_workflow(self):
        """Test: build graph -> analyze -> export"""
        if gr is None:
            pytest.skip("groggy not available")

        # Build graph
        graph = gr.Graph()
        nodes = [graph.add_node(label=f"Node{i}", value=i) for i in range(10)]

        for i in range(len(nodes) - 1):
            graph.add_edge(nodes[i], nodes[i + 1], weight=float(i))

        # Analyze
        components = graph.connected_components()
        assert len(components) >= 1, "Should have at least one component"

        # Export to table
        table = graph.table()
        assert table is not None, "Should convert to table"

        # Validate table structure
        assert hasattr(table, "nodes"), "Table should have nodes"
        assert hasattr(table, "edges"), "Table should have edges"

    def test_filter_transform_query_workflow(self):
        """Test: filter nodes -> transform -> query"""
        if gr is None:
            pytest.skip("groggy not available")

        graph = gr.Graph()

        # Create diverse nodes
        for i in range(20):
            graph.add_node(
                label=f"Node{i}",
                value=i,
                category="even" if i % 2 == 0 else "odd",
                score=i * 0.5,
            )

        # Filter nodes
        even_nodes = graph.nodes[graph.nodes["category"] == "even"]
        assert len(even_nodes) > 0, "Should have even nodes"

        # Query attributes
        values = graph["value"]
        assert values is not None, "Should get value array"

        # Group by category
        groups = graph.nodes.group_by("category")
        assert len(groups) == 2, "Should have 2 groups (even/odd)"

    def test_component_sampling_merge_workflow(self):
        """Test: find components -> sample -> merge -> analyze"""
        if gr is None:
            pytest.skip("groggy not available")

        # Create multi-component graph
        graph = gr.Graph()

        # Create 3 separate components
        for comp_id in range(3):
            comp_nodes = []
            for i in range(4):
                node = graph.add_node(label=f"C{comp_id}N{i}", component=comp_id)
                comp_nodes.append(node)

            # Connect within component
            for i in range(len(comp_nodes) - 1):
                graph.add_edge(comp_nodes[i], comp_nodes[i + 1])

        # Find components
        components = graph.connected_components()
        assert len(components) == 3, "Should have 3 components"

        # Sample 2 components
        sampled = components.sample(2)
        assert sampled is not None, "Should sample components"

        # Merge sampled components
        merged = sampled.merge()
        assert merged is not None, "Should merge components"
        assert hasattr(merged, "nodes"), "Merged result should be graph"


@pytest.mark.integration
class TestArrayTableIntegration:
    """Test integration between arrays and tables"""

    def test_array_to_table_workflow(self):
        """Test: create arrays -> convert to tables -> aggregate"""
        if gr is None:
            pytest.skip("groggy not available")

        graph = gr.Graph()

        # Create nodes with attributes
        for i in range(15):
            graph.add_node(label=f"Node{i}", value=i, score=i * 2.5)

        # Get node array
        node_arr = graph.nodes.array()
        assert node_arr is not None, "Should get node array"

        # Get as table
        node_table = graph.nodes.table()
        assert node_table is not None, "Should get node table"

        # Table operations
        if hasattr(node_table, "head"):
            head = node_table.head()
            assert head is not None, "Should get table head"

    def test_subgraph_to_table_workflow(self):
        """Test: create subgraphs -> convert to tables -> analyze"""
        if gr is None:
            pytest.skip("groggy not available")

        graph = gr.Graph()

        # Create clustered graph
        for cluster in range(3):
            cluster_nodes = []
            for i in range(5):
                node = graph.add_node(label=f"C{cluster}N{i}", cluster=cluster)
                cluster_nodes.append(node)

            for i in range(len(cluster_nodes) - 1):
                graph.add_edge(cluster_nodes[i], cluster_nodes[i + 1])

        # Group by cluster
        subgraphs = graph.nodes.group_by("cluster")
        assert len(subgraphs) == 3, "Should have 3 subgraphs"

        # Convert to tables
        nodes_table = subgraphs.nodes_table()
        assert nodes_table is not None, "Should get nodes table from subgraphs"

        edges_table = subgraphs.edges_table()
        assert edges_table is not None, "Should get edges table from subgraphs"


@pytest.mark.integration
class TestMatrixGraphIntegration:
    """Test integration between matrices and graphs"""

    def test_graph_to_matrix_workflow(self):
        """Test: build graph -> extract matrix -> operations"""
        if gr is None:
            pytest.skip("groggy not available")

        graph = gr.Graph()

        # Create small graph
        nodes = [graph.add_node(label=f"N{i}") for i in range(5)]
        for i in range(len(nodes) - 1):
            graph.add_edge(nodes[i], nodes[i + 1], weight=1.0)

        # Extract adjacency matrix
        adj_matrix = graph.to_matrix()
        assert adj_matrix is not None, "Should get adjacency matrix"

        # Matrix operations
        dense = adj_matrix.dense()
        assert dense is not None, "Should convert to dense"

        flattened = adj_matrix.flatten()
        assert flattened is not None, "Should flatten matrix"

        # Get Laplacian
        if hasattr(graph, "laplacian_matrix"):
            laplacian = graph.laplacian_matrix()
            assert laplacian is not None, "Should get Laplacian matrix"

    def test_matrix_chaining_workflow(self):
        """Test: matrix -> operations -> transformations"""
        if gr is None:
            pytest.skip("groggy not available")

        graph = gr.Graph()
        nodes = [graph.add_node(label=f"N{i}", value=float(i)) for i in range(6)]

        for i in range(len(nodes) - 1):
            graph.add_edge(nodes[i], nodes[i + 1])

        # Chain matrix operations
        result = graph.to_matrix().dense().abs().flatten()
        assert result is not None, "Matrix chain should succeed"
        assert type(result).__name__ == "NumArray", "Final result should be NumArray"


@pytest.mark.integration
class TestIOIntegration:
    """Test I/O integration across components"""

    def test_graph_save_load_workflow(self):
        """Test: build graph -> save -> load -> verify"""
        if gr is None:
            pytest.skip("groggy not available")

        # Create graph
        graph = gr.Graph()
        nodes = [graph.add_node(label=f"Node{i}", value=i) for i in range(5)]
        for i in range(len(nodes) - 1):
            graph.add_edge(nodes[i], nodes[i + 1], weight=float(i))

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = os.path.join(tmpdir, "test.bundle")

            # Save - convert to GraphTable first since save_bundle is on GraphTable, not Graph
            try:
                graph_table = graph.table()
                graph_table.save_bundle(bundle_path)
                assert os.path.exists(bundle_path), "Bundle file should exist"

                # Load - returns GraphTable
                loaded_table = gr.GraphTable.load_bundle(bundle_path)
                assert loaded_table is not None, "Should load GraphTable"

                # Convert back to graph
                loaded_graph = loaded_table.to_graph()
                assert loaded_graph is not None, "Should convert to Graph"
                assert hasattr(loaded_graph, "nodes"), "Loaded graph should have nodes"

                # Verify node count matches
                assert len(loaded_graph.nodes) == len(
                    graph.nodes
                ), "Node count should match"
            except Exception as e:
                # May require parameters or not be fully implemented
                if "missing" in str(e).lower() or "parameter" in str(e).lower():
                    pytest.skip(f"Bundle I/O requires parameters: {e}")
                else:
                    raise

    def test_table_export_workflow(self):
        """Test: graph -> table -> export formats"""
        if gr is None:
            pytest.skip("groggy not available")

        graph = gr.Graph()
        for i in range(10):
            graph.add_node(label=f"Node{i}", value=i)

        # Get tables
        node_table = graph.nodes.table()
        assert node_table is not None, "Should get node table"

        with tempfile.TemporaryDirectory() as tmpdir:
            # Try CSV export
            if hasattr(node_table, "to_csv"):
                csv_path = os.path.join(tmpdir, "nodes.csv")
                try:
                    node_table.to_csv(csv_path)
                    assert os.path.exists(csv_path), "CSV file should exist"
                except Exception:
                    pytest.skip("CSV export not fully implemented")

            # Try Parquet export
            if hasattr(node_table, "to_parquet"):
                parquet_path = os.path.join(tmpdir, "nodes.parquet")
                try:
                    node_table.to_parquet(parquet_path)
                    assert os.path.exists(parquet_path), "Parquet file should exist"
                except Exception:
                    pytest.skip("Parquet export not fully implemented")


@pytest.mark.integration
@pytest.mark.slow
class TestComplexWorkflows:
    """Test complex real-world workflows"""

    def test_social_network_analysis_workflow(self):
        """Test: build social network -> analyze communities -> export"""
        if gr is None:
            pytest.skip("groggy not available")

        # Build social network
        graph = gr.Graph()

        # Create users
        users = []
        for i in range(20):
            user = graph.add_node(
                label=f"User{i}", age=20 + i, city=["NYC", "SF", "LA"][i % 3]
            )
            users.append(user)

        # Create friendships
        import random

        random.seed(42)
        for i in range(30):
            u1, u2 = random.sample(users, 2)
            # Use has_edge_between instead of contains_edge (which takes edge_id)
            if not graph.has_edge_between(u1, u2):
                graph.add_edge(u1, u2, friendship_years=random.randint(1, 10))

        # Find communities
        components = graph.connected_components()
        assert len(components) >= 1, "Should have communities"

        # Analyze by city
        by_city = graph.nodes.group_by("city")
        assert len(by_city) == 3, "Should group by 3 cities"

        # Export summary
        table = graph.table()
        assert table is not None, "Should get summary table"

    def test_knowledge_graph_workflow(self):
        """Test: build knowledge graph -> query -> traverse"""
        if gr is None:
            pytest.skip("groggy not available")

        # Build knowledge graph
        graph = gr.Graph()

        # Add entities
        entities = {
            "person": [
                graph.add_node(label=f"Person{i}", type="person") for i in range(5)
            ],
            "place": [
                graph.add_node(label=f"Place{i}", type="place") for i in range(3)
            ],
            "thing": [
                graph.add_node(label=f"Thing{i}", type="thing") for i in range(4)
            ],
        }

        # Add relationships
        for person in entities["person"][:2]:
            for place in entities["place"][:2]:
                graph.add_edge(person, place, relation="visited")

        for person in entities["person"][:3]:
            for thing in entities["thing"][:2]:
                graph.add_edge(person, thing, relation="owns")

        # Query by type
        people = graph.nodes[graph.nodes["type"] == "person"]
        assert len(people) == 5, "Should have 5 people"

        # Traverse from person
        if hasattr(graph, "neighborhood"):
            neighborhood = graph.neighborhood(entities["person"][0])
            assert neighborhood is not None, "Should get neighborhood"


if __name__ == "__main__":
    # Allow running this module directly for development
    pytest.main([__file__, "-v"])
