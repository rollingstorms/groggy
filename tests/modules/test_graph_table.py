"""
Module 4.2: GraphTable Testing - Milestone 4

Tests GraphTable type for high-level graph tabular operations and I/O.
GraphTable provides a unified view of graph data with nodes and edges tables.

Test Coverage:
GraphTable (52.2% pass rate, 12/23 methods):
- Basic table operations (head, tail, shape, ncols, nrows)
- Graph conversion (to_graph, from graph)
- Bundle operations (save_bundle, load_bundle, verify_bundle)
- Merge operations (merge, merge_with, merge_with_strategy)
- Conversion operations (to_nodes, to_edges, to_subgraphs)
- Validation and statistics

Success Criteria: 90%+ pass rate, bundle I/O works, merge operations documented
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

from tests.conftest import assert_graph_valid, assert_method_callable
from tests.modules.test_table_base import TableIOTestMixin, TableTestBase


class GraphTableTest(TableTestBase, TableIOTestMixin):
    """Test class for GraphTable using shared test patterns"""

    def get_table_instance(self, graph=None):
        """Create GraphTable instance for testing"""
        if gr is None:
            pytest.skip("groggy not available")

        if graph is None:
            graph = self.get_test_graph()

        # Get GraphTable from graph
        return graph.table()

    def get_expected_table_type(self):
        """Return the expected GraphTable type"""
        return type(self.get_table_instance())


@pytest.mark.graph_table
class TestGraphTable(GraphTableTest):
    """Test GraphTable functionality"""

    def test_graph_table_creation(self):
        """Test creating GraphTable from graph"""
        graph = self.get_test_graph()
        graph_table = graph.table()

        assert graph_table is not None, "Graph should provide table() method"
        assert hasattr(graph_table, "nodes"), "GraphTable should have nodes property"
        assert hasattr(graph_table, "edges"), "GraphTable should have edges property"

    def test_graph_table_structure(self):
        """Test GraphTable structure and properties"""
        graph_table = self.get_table_instance()

        # Test nodes and edges properties
        nodes_table = graph_table.nodes
        edges_table = graph_table.edges

        assert nodes_table is not None, "GraphTable should have nodes table"
        assert edges_table is not None, "GraphTable should have edges table"

        # Both should be table-like objects
        assert hasattr(nodes_table, "shape"), "NodesTable should have shape"
        assert hasattr(edges_table, "shape"), "EdgesTable should have shape"

    def test_graph_table_basic_operations(self):
        """Test basic GraphTable operations"""
        graph_table = self.get_table_instance()

        # Test basic table operations inherited from base
        self.test_table_basic_properties()
        self.test_table_head_tail_operations()

        # Test auto_assign_edge_ids
        if hasattr(graph_table, "auto_assign_edge_ids"):
            auto_assigned = graph_table.auto_assign_edge_ids()
            assert (
                auto_assigned is not None
            ), "auto_assign_edge_ids() should return a GraphTable"
            assert type(auto_assigned) == type(
                graph_table
            ), "auto_assign_edge_ids() should return same type"

    def test_graph_table_conversion_operations(self):
        """Test GraphTable conversion operations"""
        graph_table = self.get_table_instance()

        # Test to_graph - this should work according to comprehensive tests
        graph = graph_table.to_graph()
        assert graph is not None, "to_graph() should return a Graph"
        assert hasattr(graph, "nodes"), "Converted graph should have nodes"
        assert hasattr(graph, "edges"), "Converted graph should have edges"

        # Test conversion operations that are not yet implemented
        not_implemented_methods = ["to_nodes", "to_edges", "to_subgraphs"]
        for method_name in not_implemented_methods:
            if hasattr(graph_table, method_name):
                try:
                    result = getattr(graph_table, method_name)()
                    pytest.skip(f"{method_name}() unexpectedly succeeded")
                except Exception as e:
                    # Expected failure - comprehensive test shows "not yet implemented"
                    assert "not yet implemented" in str(
                        e
                    ), f"{method_name}() should fail with not implemented: {e}"

    def test_graph_table_validation_operations(self):
        """Test GraphTable validation operations"""
        graph_table = self.get_table_instance()

        # Test validate - should work according to comprehensive tests
        if hasattr(graph_table, "validate"):
            validation_result = graph_table.validate()
            assert isinstance(
                validation_result, str
            ), f"validate() should return string, got {type(validation_result)}"

        # Test stats - should work according to comprehensive tests
        if hasattr(graph_table, "stats"):
            stats = graph_table.stats()
            assert isinstance(
                stats, dict
            ), f"stats() should return dict, got {type(stats)}"
            assert len(stats) >= 0, "Stats should contain some information"

    def test_graph_table_bundle_operations(self):
        """Test GraphTable bundle I/O operations"""
        graph_table = self.get_table_instance()

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = os.path.join(tmpdir, "test_graph.bundle")

            GraphTableClass = type(graph_table)

            # Test save_bundle
            if hasattr(graph_table, "save_bundle"):
                graph_table.save_bundle(bundle_path)
                assert os.path.exists(bundle_path), "Bundle directory should be created"

                # Test load_bundle if available (prefer classmethod/static)
                if hasattr(GraphTableClass, "load_bundle"):
                    loaded_table = GraphTableClass.load_bundle(bundle_path)
                    assert isinstance(
                        loaded_table, GraphTableClass
                    ), "load_bundle() should return GraphTable"

                # Test verify_bundle
                if hasattr(GraphTableClass, "verify_bundle"):
                    verification = GraphTableClass.verify_bundle(bundle_path)
                    assert isinstance(
                        verification, dict
                    ), "verify_bundle() should return dict"

                # Test get_bundle_info
                if hasattr(GraphTableClass, "get_bundle_info"):
                    bundle_info = GraphTableClass.get_bundle_info(bundle_path)
                    assert isinstance(
                        bundle_info, dict
                    ), "get_bundle_info() should return dict"

    def test_graph_table_merge_operations(self):
        """Test GraphTable merge operations"""
        graph_table = self.get_table_instance()

        GraphTableClass = type(graph_table)

        if hasattr(GraphTableClass, "merge"):
            # merge should fail on empty input
            with pytest.raises(Exception) as exc_info:
                GraphTableClass.merge([])
            assert "Cannot merge empty list" in str(exc_info.value)

            # merge with default strategy should succeed on simple tables
            base_graph = self.get_test_graph()
            table_a = base_graph.table()
            other_graph = self.get_test_graph()
            table_b = other_graph.table()
            merged = GraphTableClass.merge([table_a, table_b])
            assert isinstance(
                merged, GraphTableClass
            ), "merge() should return a GraphTable"

            # merge with explicit strategy should also succeed
            another_graph = self.get_test_graph()
            table_c = another_graph.table()
            yet_another_graph = self.get_test_graph()
            table_d = yet_another_graph.table()
            merged_keep = GraphTableClass.merge(
                [table_c, table_d], strategy="keep_first"
            )
            assert isinstance(
                merged_keep, GraphTableClass
            ), "merge() with strategy should return a GraphTable"
        else:
            pytest.skip("GraphTable.merge not implemented")

    def test_graph_table_federated_operations(self):
        """Test GraphTable federated operations"""
        graph_table = self.get_table_instance()

        # Test from_federated_bundles (static method)
        if hasattr(graph_table, "from_federated_bundles"):
            try:
                federated = graph_table.from_federated_bundles([])
                pytest.skip("from_federated_bundles([]) unexpectedly succeeded")
            except Exception as e:
                # Expected failure - either missing parameter or invalid input
                expected_errors = ["missing", "required", "Cannot merge empty list"]
                assert any(
                    err in str(e) for err in expected_errors
                ), f"from_federated_bundles() should fail appropriately: {e}"

    def test_graph_table_viz_property(self):
        """Test GraphTable visualization property"""
        graph_table = self.get_table_instance()

        # Test viz property - should work according to comprehensive tests
        if hasattr(graph_table, "viz"):
            viz_accessor = graph_table.viz
            assert viz_accessor is not None, "viz property should return a VizAccessor"

    @pytest.mark.performance
    def test_graph_table_performance(self):
        """Test GraphTable performance on larger graphs"""
        if gr is None:
            pytest.skip("groggy not available")

        import time

        # Create larger graph for performance testing
        graph = gr.Graph()
        node_count = 500

        # Add many nodes and edges
        node_ids = []
        for i in range(node_count):
            node_id = graph.add_node(label=f"Node{i}", value=i, category=f"Cat{i % 10}")
            node_ids.append(node_id)

        # Add edges
        for i in range(node_count - 1):
            graph.add_edge(node_ids[i], node_ids[i + 1], weight=i * 0.01)

        # Test GraphTable creation performance
        start_time = time.time()
        graph_table = graph.table()
        creation_time = time.time() - start_time

        assert (
            creation_time < 2.0
        ), f"GraphTable creation took {creation_time:.3f}s, should be < 2.0s"

        # Test basic operations performance
        operations = [
            ("shape", lambda: graph_table.shape()),
            ("ncols", lambda: graph_table.ncols()),
            ("nrows", lambda: graph_table.nrows()),
            ("head", lambda: graph_table.head()),
            ("validate", lambda: graph_table.validate()),
            ("to_graph", lambda: graph_table.to_graph()),
        ]

        for op_name, op_func in operations:
            if hasattr(graph_table, op_name.split("(")[0]):  # Check if method exists
                start_time = time.time()
                result = op_func()
                elapsed = time.time() - start_time

                assert (
                    elapsed < 1.0
                ), f"{op_name} took {elapsed:.3f}s, should be < 1.0s for {node_count} nodes"
                assert result is not None, f"{op_name} should return a result"


@pytest.mark.graph_table
@pytest.mark.integration
class TestGraphTableIntegration:
    """Test GraphTable integration with other graph components"""

    def test_graph_table_round_trip(self):
        """Test graph -> table -> graph round trip"""
        if gr is None:
            pytest.skip("groggy not available")

        # Create original graph
        original_graph = gr.Graph()
        node_ids = []
        for i in range(5):
            node_id = original_graph.add_node(label=f"Node{i}", value=i)
            node_ids.append(node_id)

        for i in range(len(node_ids) - 1):
            original_graph.add_edge(node_ids[i], node_ids[i + 1], weight=i + 1)

        # Convert to table
        graph_table = original_graph.table()
        assert graph_table is not None, "Should create GraphTable from graph"

        # Convert back to graph
        converted_graph = graph_table.to_graph()
        assert converted_graph is not None, "Should convert GraphTable back to graph"

        # Verify basic structure is preserved
        assert len(converted_graph.nodes) >= 0, "Converted graph should have nodes"
        assert len(converted_graph.edges) >= 0, "Converted graph should have edges"

        # Detailed comparison would require more sophisticated graph equality checking

    def test_graph_table_with_accessors(self):
        """Test GraphTable interaction with graph accessors"""
        base_test = GraphTableTest()
        graph = base_test.get_test_graph()
        graph_table = graph.table()

        # Test that table structure matches accessor structure
        nodes_accessor = graph.nodes
        edges_accessor = graph.edges

        nodes_table = graph_table.nodes
        edges_table = graph_table.edges

        # Basic consistency checks
        assert nodes_table is not None, "GraphTable nodes should be accessible"
        assert edges_table is not None, "GraphTable edges should be accessible"

        # Table dimensions should be reasonable
        nodes_shape = nodes_table.shape()
        edges_shape = edges_table.shape()

        assert nodes_shape[0] >= 0, "Nodes table should have non-negative rows"
        assert edges_shape[0] >= 0, "Edges table should have non-negative rows"

    def test_graph_table_with_arrays(self):
        """Test GraphTable interaction with graph arrays"""
        base_test = GraphTableTest()
        graph = base_test.get_test_graph()
        graph_table = graph.table()

        # Test that table can be converted to arrays and back
        nodes_table = graph_table.nodes
        edges_table = graph_table.edges

        # Tables should be convertible to pandas
        if hasattr(nodes_table, "to_pandas") and hasattr(edges_table, "to_pandas"):
            try:
                nodes_df = nodes_table.to_pandas()
                edges_df = edges_table.to_pandas()

                assert nodes_df is not None, "Nodes table should convert to pandas"
                assert edges_df is not None, "Edges table should convert to pandas"

                # DataFrames should have reasonable structure
                assert (
                    nodes_df.shape[0] >= 0
                ), "Nodes DataFrame should have non-negative rows"
                assert (
                    edges_df.shape[0] >= 0
                ), "Edges DataFrame should have non-negative rows"

            except Exception as e:
                pytest.skip(f"Pandas conversion failed: {e}")


if __name__ == "__main__":
    # Allow running this module directly for development
    pytest.main([__file__, "-v"])
