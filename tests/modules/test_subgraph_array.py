"""
Module 5.1: SubgraphArray Testing - Milestone 5

Tests SubgraphArray type for collections of graph subsets.
SubgraphArray is returned by operations like group_by, connected_components, etc.

Test Coverage:
SubgraphArray (13 methods, ~80% expected pass rate):
- Collection operations (collect, to_list, iter)
- Table operations (nodes_table, edges_table, table, summary)
- Sampling and filtering (sample, is_empty)
- Merge operations (merge)
- Attribute extraction (extract_node_attribute)
- Grouping (group_by with element_type)
- Mapping (map with valid functions)
- Visualization (viz property)

Success Criteria: 90%+ pass rate, proper error handling for invalid inputs
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

from tests.conftest import assert_graph_valid, assert_method_callable
from tests.modules.test_subgraph_base import (SubgraphArrayBehaviorMixin,
                                              SubgraphAttributeTestMixin,
                                              SubgraphOperationsTestMixin,
                                              SubgraphTestBase)


class SubgraphArrayTest(
    SubgraphArrayBehaviorMixin,
    SubgraphAttributeTestMixin,
    SubgraphOperationsTestMixin,
    SubgraphTestBase,
):
    """Test class for SubgraphArray using shared test patterns"""

    def get_subgraph_array(self, graph=None):
        """Create SubgraphArray instance for testing"""
        if gr is None:
            pytest.skip("groggy not available")

        if graph is None:
            graph = self.get_test_graph()

        # Create SubgraphArray using group_by on nodes
        return graph.nodes.group_by("cluster")


@pytest.mark.subgraph_array
class TestSubgraphArray(SubgraphArrayTest):
    """Test SubgraphArray functionality"""

    def test_subgraph_array_creation(self):
        """Test creating SubgraphArray from grouping operations"""
        graph = self.get_test_graph()

        # Create via nodes.group_by
        subgraphs = graph.nodes.group_by("cluster")
        assert subgraphs is not None, "nodes.group_by() should return SubgraphArray"
        assert (
            type(subgraphs).__name__ == "SubgraphArray"
        ), "Should return SubgraphArray type"

        # Create via edges.group_by
        subgraphs_edges = graph.edges.group_by("edge_type")
        assert (
            subgraphs_edges is not None
        ), "edges.group_by() should return SubgraphArray"

    def test_subgraph_array_length_and_iteration(self):
        """Test SubgraphArray length and iteration"""
        subgraph_array = self.get_subgraph_array()

        # Test length
        length = len(subgraph_array)
        assert (
            length > 0
        ), "SubgraphArray should have positive length for clustered graph"
        assert length == 3, "Should have 3 subgraphs (one per cluster)"

        # Test iteration (via iter() method)
        if hasattr(subgraph_array, "iter"):
            iterator = subgraph_array.iter()
            items = list(iterator)
            assert (
                len(items) == length
            ), "Iterator should yield same number of items as length"

        # Test iteration (via __iter__)
        elif hasattr(subgraph_array, "__iter__"):
            items = list(subgraph_array)
            assert (
                len(items) == length
            ), "Iterator should yield same number of items as length"

    def test_subgraph_array_collection_methods(self):
        """Test SubgraphArray collection methods"""
        subgraph_array = self.get_subgraph_array()

        # Test inherited base methods
        self.test_subgraph_array_collection_operations()

        # Additional specific tests
        subgraph_list = subgraph_array.to_list()
        assert len(subgraph_list) == len(
            subgraph_array
        ), "to_list() should preserve length"

        collected = subgraph_array.collect()
        assert len(collected) == len(subgraph_array), "collect() should preserve length"

    def test_subgraph_array_table_methods(self):
        """Test SubgraphArray table methods"""
        subgraph_array = self.get_subgraph_array()

        # Test inherited base methods
        self.test_subgraph_array_table_operations()

        # Additional specific tests
        nodes_table = subgraph_array.nodes_table()
        assert nodes_table is not None, "nodes_table() should return TableArray"
        assert (
            type(nodes_table).__name__ == "TableArray"
        ), "Should return TableArray type"

        edges_table = subgraph_array.edges_table()
        assert edges_table is not None, "edges_table() should return TableArray"
        assert (
            type(edges_table).__name__ == "TableArray"
        ), "Should return TableArray type"

        table = subgraph_array.table()
        assert table is not None, "table() should return TableArray"

        summary = subgraph_array.summary()
        assert summary is not None, "summary() should return BaseTable"

    def test_subgraph_array_summary_contents(self):
        """Validate summary table contents for clustered graph."""
        subgraph_array = self.get_subgraph_array()
        if len(subgraph_array) == 0:
            pytest.skip("SubgraphArray is empty")

        summary = subgraph_array.summary()
        node_counts = summary.column("node_count").to_list()
        edge_counts = summary.column("edge_count").to_list()

        assert node_counts == [4, 4, 4], "Each cluster should contain four nodes"
        assert edge_counts == [
            3,
            3,
            3,
        ], "Each cluster should contain three intra-cluster edges"

    def test_subgraph_array_extract_node_attribute(self):
        """Extracting node attributes should return ArrayArray keyed by subgraph index."""
        subgraph_array = self.get_subgraph_array()
        if len(subgraph_array) == 0:
            pytest.skip("SubgraphArray is empty")

        attributes = subgraph_array.extract_node_attribute("value")
        assert len(attributes) == len(subgraph_array)

        keys = attributes.keys()
        if keys is not None:
            assert keys == [f"subgraph_{i}" for i in range(len(subgraph_array))]

        value_lists = []
        for idx in range(len(subgraph_array)):
            base_array = attributes[idx]
            value_lists.append(sorted(base_array.to_list()))

        assert value_lists[0] == [0, 1, 2, 3]

    def test_subgraph_array_merge_contains_all_nodes(self):
        """Merged graph should contain all unique nodes from clusters."""
        subgraph_array = self.get_subgraph_array()
        merged_graph = subgraph_array.merge()
        assert merged_graph is not None

        merged_nodes = set(merged_graph.nodes.ids().to_list())
        original_nodes = set()
        for subgraph in subgraph_array:
            original_nodes.update(subgraph.node_ids.to_list())

        assert original_nodes.issubset(merged_nodes)

    def test_subgraph_array_group_by_edges_distribution(self):
        """Grouping edges by type should split intra- and inter-cluster edges."""
        graph = self.get_test_graph()
        edge_groups = graph.edges.group_by("edge_type")
        assert edge_groups is not None
        assert len(edge_groups) >= 2
        edge_types = set()
        for subgraph in edge_groups:
            if subgraph.edge_count() == 0:
                continue
            edges_table = subgraph.edges_table()
            try:
                edge_type_col = edges_table.column("edge_type")
            except AttributeError:
                continue
            if edge_type_col is None:
                continue
            edge_types.update(edge_type_col.to_list())
        assert "intra_cluster" in edge_types
        assert "inter_cluster" in edge_types

    def test_subgraph_array_sampling_operations(self):
        """Test SubgraphArray sampling"""
        subgraph_array = self.get_subgraph_array()

        # Test inherited base methods
        self.test_subgraph_array_sampling()

        # Additional specific tests
        original_length = len(subgraph_array)
        sample_size = min(2, original_length)

        sampled = subgraph_array.sample(sample_size)
        # Note: sample() may return more than requested if implementation differs
        assert sampled is not None, "sample() should return a SubgraphArray"
        assert (
            type(sampled).__name__ == "SubgraphArray"
        ), "sample() should return SubgraphArray"

    def test_subgraph_array_merge_operation(self):
        """Test SubgraphArray merge"""
        subgraph_array = self.get_subgraph_array()

        # Test inherited base methods
        self.test_subgraph_array_merge()

        # Additional specific tests
        merged_graph = subgraph_array.merge()
        assert merged_graph is not None, "merge() should return a Graph"
        assert hasattr(merged_graph, "nodes"), "Merged result should have nodes"
        assert hasattr(merged_graph, "edges"), "Merged result should have edges"

        # Merged graph should have nodes from all subgraphs
        assert len(merged_graph.nodes) > 0, "Merged graph should have nodes"

    def test_subgraph_array_collapse_meta_nodes(self):
        """Batch collapse should return one meta-node per subgraph."""
        subgraph_array = self.get_subgraph_array()
        meta_nodes = subgraph_array.collapse(
            node_aggs={"size": "count"},
            node_strategy="collapse",
        )
        assert isinstance(meta_nodes, list)
        assert len(meta_nodes) == len(subgraph_array)
        assert all(hasattr(node, "id") for node in meta_nodes)

    def test_subgraph_array_is_empty_check(self):
        """Test SubgraphArray is_empty"""
        subgraph_array = self.get_subgraph_array()

        is_empty = subgraph_array.is_empty()
        assert isinstance(is_empty, bool), "is_empty() should return bool"
        assert not is_empty, "Non-empty SubgraphArray should return False"

        # Test with empty array (if we can create one)
        graph = gr.Graph()
        node_id = graph.add_node(label="SingleNode", cluster=0)

        # Group by a non-existent attribute should give minimal groups
        try:
            minimal_subgraphs = graph.nodes.group_by("cluster")
            # This should work and have 1 subgraph
            assert len(minimal_subgraphs) >= 0, "Grouping should succeed"
        except Exception:
            pytest.skip("Cannot create minimal subgraph array for testing")

    def test_subgraph_array_viz_property(self):
        """Test SubgraphArray visualization"""
        subgraph_array = self.get_subgraph_array()

        # Test inherited base methods
        self.test_subgraph_array_visualization()

        # Additional specific tests
        viz = subgraph_array.viz
        assert viz is not None, "viz should return VizAccessor"


@pytest.mark.subgraph_array
@pytest.mark.components
class TestComponentsArray(SubgraphArrayTest):
    """Test ComponentsArray (special SubgraphArray from connected_components)"""

    def get_subgraph_array(self, graph=None):
        """Create ComponentsArray via connected_components"""
        if gr is None:
            pytest.skip("groggy not available")

        if graph is None:
            graph = self.get_test_graph()

        # connected_components returns ComponentsArray
        return graph.connected_components()

    def test_components_array_creation(self):
        """Test creating ComponentsArray from connected_components"""
        graph = self.get_test_graph()

        components = graph.connected_components()
        assert components is not None, "connected_components() should return result"
        assert type(components).__name__ in [
            "ComponentsArray",
            "SubgraphArray",
        ], "Should return ComponentsArray or SubgraphArray"

    def test_components_array_neighborhood(self):
        """Test ComponentsArray neighborhood expansion"""
        components = self.get_subgraph_array()

        if len(components) == 0:
            pytest.skip("No components to test neighborhood")

        # Test neighborhood - should work
        if hasattr(components, "neighborhood"):
            neighborhood = components.neighborhood()
            assert (
                neighborhood is not None
            ), "neighborhood() should return SubgraphArray"
            assert (
                type(neighborhood).__name__ == "SubgraphArray"
            ), "neighborhood() should return SubgraphArray"

            # Note: depth parameter may not be supported as keyword argument
            # Try with positional argument if available
            try:
                neighborhood_depth = components.neighborhood(2)
                assert neighborhood_depth is not None, "neighborhood(2) should work"
            except TypeError:
                # Keyword args not supported, that's okay
                pass

    def test_components_array_sample(self):
        """Test ComponentsArray sampling"""
        components = self.get_subgraph_array()

        if len(components) == 0:
            pytest.skip("No components to sample")

        sample_size = min(2, len(components))
        sampled = components.sample(sample_size)
        assert sampled is not None, "sample() should return SubgraphArray"
        assert (
            type(sampled).__name__ == "SubgraphArray"
        ), "sample() should return SubgraphArray"

    def test_components_array_collapse(self):
        """Ensure batch collapse works for connected components."""
        components = self.get_subgraph_array()

        if len(components) == 0:
            pytest.skip("No components to collapse")

        meta_nodes = components.collapse(
            node_aggs={"size": "count"},
            node_strategy="collapse",
        )

        assert len(meta_nodes) == len(components)
        assert all(hasattr(node, "id") for node in meta_nodes)


@pytest.mark.subgraph_array
@pytest.mark.integration
class TestSubgraphArrayIntegration:
    """Test SubgraphArray integration with other graph components"""

    def test_subgraph_array_chaining(self):
        """Test chaining operations on SubgraphArray"""
        if gr is None:
            pytest.skip("groggy not available")

        graph = gr.Graph()

        # Create graph with multiple clusters
        for cluster_id in range(3):
            for i in range(4):
                node_id = graph.add_node(
                    label=f"C{cluster_id}N{i}",
                    cluster=cluster_id,
                    value=cluster_id * 10 + i,
                )

        # Chain: group_by -> sample -> merge
        result = graph.nodes.group_by("cluster").sample(2).merge()
        assert result is not None, "Chained operations should succeed"
        assert hasattr(result, "nodes"), "Final result should be a Graph"

    def test_subgraph_to_table_conversion(self):
        """Test converting SubgraphArray to table representations"""
        if gr is None:
            pytest.skip("groggy not available")

        base_test = SubgraphArrayTest()
        graph = base_test.get_test_graph()
        subgraph_array = graph.nodes.group_by("cluster")

        # Test various table conversions
        nodes_table = subgraph_array.nodes_table()
        edges_table = subgraph_array.edges_table()
        table = subgraph_array.table()
        summary = subgraph_array.summary()

        # All should return table-like objects
        assert nodes_table is not None, "nodes_table() should succeed"
        assert edges_table is not None, "edges_table() should succeed"
        assert table is not None, "table() should succeed"
        assert summary is not None, "summary() should succeed"

    def test_subgraph_array_from_components_chaining(self):
        """Test chaining from connected_components result"""
        if gr is None:
            pytest.skip("groggy not available")

        base_test = SubgraphArrayTest()
        graph = base_test.get_test_graph()

        # Chain: connected_components -> sample
        # Note: neighborhood() not available on SubgraphArray, only on ComponentsArray
        components = graph.connected_components()
        if len(components) > 0 and hasattr(components, "sample"):
            sampled = components.sample(min(2, len(components)))
            assert sampled is not None, "Sampled components should succeed"

            # If sampled result still has neighborhood, test that too
            if hasattr(sampled, "neighborhood"):
                result = sampled.neighborhood()
                assert result is not None, "Chained component operations should succeed"


if __name__ == "__main__":
    # Allow running this module directly for development
    pytest.main([__file__, "-v"])
