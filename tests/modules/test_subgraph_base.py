"""
Shared test infrastructure for subgraph operations.

Provides base classes and mixins for testing SubgraphArray and GraphView objects.
"""

import sys
from abc import ABC, abstractmethod
from pathlib import Path

import pytest

# Add path for groggy
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

try:
    import groggy as gr
except ImportError:
    gr = None


class SubgraphTestBase(ABC):
    """Base class for subgraph testing with shared patterns"""

    @abstractmethod
    def get_subgraph_array(self, graph=None):
        """Create SubgraphArray instance for testing"""
        pass

    def get_test_graph(self):
        """Create test graph with structure suitable for subgraph operations"""
        if gr is None:
            pytest.skip("groggy not available")

        graph = gr.Graph()

        # Create 3 clusters of nodes
        cluster_nodes = []
        for cluster_id in range(3):
            cluster = []
            for i in range(4):
                node_id = graph.add_node(
                    label=f"C{cluster_id}N{i}",
                    cluster=cluster_id,
                    value=cluster_id * 10 + i,
                )
                cluster.append(node_id)
            cluster_nodes.append(cluster)

        # Connect nodes within each cluster (create components)
        for cluster in cluster_nodes:
            for i in range(len(cluster) - 1):
                graph.add_edge(
                    cluster[i], cluster[i + 1], weight=1.0, edge_type="intra_cluster"
                )

        # Add a few inter-cluster edges
        graph.add_edge(
            cluster_nodes[0][0],
            cluster_nodes[1][0],
            weight=0.5,
            edge_type="inter_cluster",
        )
        graph.add_edge(
            cluster_nodes[1][0],
            cluster_nodes[2][0],
            weight=0.5,
            edge_type="inter_cluster",
        )

        return graph


class SubgraphArrayBehaviorMixin:
    """Mixin containing shared SubgraphArray behavior tests."""

    def get_first_subgraph(self):
        """Utility to fetch the first subgraph if available."""
        subgraph_array = self.get_subgraph_array()

        if not hasattr(subgraph_array, "__len__"):
            pytest.skip("SubgraphArray behavior tests require an array-like object")

        if len(subgraph_array) == 0:
            pytest.skip("SubgraphArray is empty for this scenario")

        first = subgraph_array[0]
        assert type(first).__name__ == "Subgraph", "Expected Subgraph instances"
        return first

    def test_subgraph_array_basic_properties(self):
        """Test basic SubgraphArray properties"""
        subgraph_array = self.get_subgraph_array()

        assert hasattr(subgraph_array, "__len__"), "SubgraphArray should have length"
        assert len(subgraph_array) >= 0, "SubgraphArray length should be non-negative"
        assert hasattr(subgraph_array, "iter") or hasattr(
            subgraph_array, "__iter__"
        ), "SubgraphArray should be iterable"

    def test_subgraph_array_getitem_variants(self):
        """Ensure __getitem__ supports indexing and attribute extraction."""
        subgraph_array = self.get_subgraph_array()

        if len(subgraph_array) == 0:
            pytest.skip("SubgraphArray is empty")

        first = subgraph_array[0]
        last = subgraph_array[-1]
        assert type(first).__name__ == "Subgraph"
        assert type(last).__name__ == "Subgraph"

        if hasattr(subgraph_array, "extract_node_attribute"):
            attributes = subgraph_array["cluster"]
            assert hasattr(
                attributes, "keys"
            ), "Attribute extraction should return ArrayArray"
            assert len(attributes) == len(subgraph_array)

            table_array = subgraph_array[["cluster", "value"]]
            assert hasattr(
                table_array, "__len__"
            ), "Column extraction should return TableArray"
            assert len(table_array) == len(subgraph_array)

    def test_subgraph_array_collection_operations(self):
        """Test SubgraphArray collection operations"""
        subgraph_array = self.get_subgraph_array()

        if hasattr(subgraph_array, "to_list"):
            subgraph_list = subgraph_array.to_list()
            assert isinstance(subgraph_list, list), "to_list() should return a list"

        if hasattr(subgraph_array, "collect"):
            collected = subgraph_array.collect()
            assert isinstance(collected, list), "collect() should return a list"
            assert all(type(item).__name__ == "Subgraph" for item in collected)

    def test_subgraph_array_map_and_summary(self):
        """Validate map() output and summary metrics."""
        subgraph_array = self.get_subgraph_array()

        if len(subgraph_array) == 0:
            pytest.skip("SubgraphArray is empty")

        if hasattr(subgraph_array, "map"):

            def node_counter(subgraph):
                return subgraph.node_count()

            node_counts_array = subgraph_array.map(node_counter)
            assert hasattr(
                node_counts_array, "to_list"
            ), "map() should return BaseArray"
            node_counts = node_counts_array.to_list()
            assert len(node_counts) == len(subgraph_array)

        if hasattr(subgraph_array, "summary"):
            summary_table = subgraph_array.summary()
            assert summary_table is not None, "summary() should return a table"
            node_column = summary_table.column("node_count")
            edge_column = summary_table.column("edge_count")
            density_column = summary_table.column("density")

            assert hasattr(node_column, "to_list")
            assert hasattr(edge_column, "to_list")
            assert hasattr(density_column, "to_list")

            node_values = node_column.to_list()
            edge_values = edge_column.to_list()
            density_values = density_column.to_list()

            assert len(node_values) == len(subgraph_array)
            assert len(edge_values) == len(subgraph_array)
            assert len(density_values) == len(subgraph_array)
            assert all(isinstance(val, (int, float)) for val in density_values)

    def test_subgraph_array_table_operations(self):
        """Test SubgraphArray table operations"""
        subgraph_array = self.get_subgraph_array()

        if hasattr(subgraph_array, "nodes_table"):
            nodes_table = subgraph_array.nodes_table()
            assert nodes_table is not None

        if hasattr(subgraph_array, "edges_table"):
            edges_table = subgraph_array.edges_table()
            assert edges_table is not None

        if hasattr(subgraph_array, "table"):
            table = subgraph_array.table()
            assert table is not None

        if hasattr(subgraph_array, "summary"):
            summary = subgraph_array.summary()
            assert summary is not None

        if len(subgraph_array) > 0 and hasattr(subgraph_array, "nodes_table"):
            nodes_table = subgraph_array.nodes_table()
            edges_table = subgraph_array.edges_table()
            table = subgraph_array.table()

            for tbl in (nodes_table, edges_table, table):
                assert hasattr(tbl, "__len__")
                assert len(tbl) == len(subgraph_array)

    def test_subgraph_array_sampling(self):
        """Test SubgraphArray sampling operations"""
        subgraph_array = self.get_subgraph_array()

        if len(subgraph_array) == 0:
            pytest.skip("Empty subgraph array, cannot test sampling")

        if hasattr(subgraph_array, "sample"):
            sample_size = min(2, len(subgraph_array))
            sampled = subgraph_array.sample(sample_size)
            assert sampled is not None
            assert type(sampled).__name__ == "SubgraphArray"

    def test_subgraph_array_merge(self):
        """Test SubgraphArray merge operations"""
        subgraph_array = self.get_subgraph_array()

        if hasattr(subgraph_array, "merge"):
            merged = subgraph_array.merge()
            assert merged is not None
            assert hasattr(merged, "nodes")

    def test_subgraph_array_is_empty(self):
        """Test SubgraphArray is_empty check"""
        subgraph_array = self.get_subgraph_array()

        if hasattr(subgraph_array, "is_empty"):
            result = subgraph_array.is_empty()
            assert isinstance(result, bool)

    def test_subgraph_array_visualization(self):
        """Test SubgraphArray visualization property"""
        subgraph_array = self.get_subgraph_array()

        if hasattr(subgraph_array, "viz"):
            viz_accessor = subgraph_array.viz
            assert viz_accessor is not None


class SubgraphAttributeTestMixin:
    """Mixin for testing attribute extraction from subgraphs"""

    def test_subgraph_extract_node_attribute(self):
        """Test extracting node attributes from subgraphs"""
        subgraph_array = self.get_subgraph_array()

        if len(subgraph_array) == 0:
            pytest.skip("Empty subgraph array, cannot test attribute extraction")

        # Test extract_node_attribute - expected to fail if attribute doesn't exist
        if hasattr(subgraph_array, "extract_node_attribute"):
            # Try with a known attribute from test graph
            # Test graph has "cluster" and "label" attributes
            try:
                result = subgraph_array.extract_node_attribute("cluster")
                assert (
                    result is not None
                ), "extract_node_attribute('cluster') should return a result"
            except Exception as e:
                # May fail if attribute not present in all subgraphs
                assert (
                    "not found" in str(e).lower()
                ), f"Expected 'not found' error, got: {e}"

    def test_subgraph_group_by(self):
        """Test grouping subgraphs"""
        subgraph_array = self.get_subgraph_array()

        # Test group_by - requires element_type parameter
        if hasattr(subgraph_array, "group_by"):
            # Validate error path when element_type omitted
            with pytest.raises(TypeError):
                subgraph_array.group_by("cluster")

            # Group by nodes and edges explicitly
            node_groups = subgraph_array.group_by("cluster", "nodes")
            assert hasattr(
                node_groups, "__len__"
            ), "group_by should return SubgraphArray"
            assert len(node_groups) >= len(
                subgraph_array
            ), "Grouped result should not shrink subgraphs"

            edge_groups = subgraph_array.group_by("edge_type", "edges")
            assert hasattr(edge_groups, "__len__")
            assert (
                len(edge_groups) >= 1
            ), "Grouping edges should yield at least one subgraph"

    def test_subgraph_map(self):
        """Test mapping operations on subgraphs"""
        subgraph_array = self.get_subgraph_array()

        if len(subgraph_array) == 0:
            pytest.skip("Empty subgraph array, cannot test map")

        # Test map - requires function that returns int, float, str, or bool
        if hasattr(subgraph_array, "map"):
            # Test with valid function
            def count_nodes(subgraph):
                return len(list(subgraph.nodes))

            try:
                result = subgraph_array.map(count_nodes)
                assert result is not None, "map() should return a result"
                assert hasattr(result, "to_list"), "map() should yield BaseArray"
                values = result.to_list()
                assert len(values) == len(subgraph_array)
            except Exception as e:
                # May fail if function signature is wrong
                expected_errors = ["must return", "int, float, str, or bool"]
                assert any(
                    err in str(e) for err in expected_errors
                ), f"Expected type requirement error, got: {e}"


class SubgraphOperationsTestMixin:
    """Mixin covering core single-subgraph operations."""

    def _ensure_single_subgraph(self):
        candidate = self.get_subgraph_array()
        class_name = getattr(
            candidate.__class__, "__name__", candidate.__class__.__name__
        )

        # Handle genuine Subgraph objects directly
        if class_name in {"Subgraph", "PySubgraph"}:
            return candidate

        # SubgraphArray/ComponentsArray provide len()+indexing
        if class_name in {"SubgraphArray", "ComponentsArray"}:
            if len(candidate) == 0:
                pytest.skip("SubgraphArray is empty")
            subgraph = candidate[0]
            assert (
                getattr(subgraph.__class__, "__name__", subgraph.__class__.__name__)
                == "Subgraph"
            )
            return subgraph

        # Graceful fallback for other container-like objects
        if hasattr(candidate, "__len__") and hasattr(candidate, "__getitem__"):
            if len(candidate) == 0:
                pytest.skip("Subgraph container is empty")
            subgraph = candidate[0]
            assert (
                getattr(subgraph.__class__, "__name__", subgraph.__class__.__name__)
                == "Subgraph"
            )
            return subgraph

        # Last resort: treat candidate as a Subgraph-like object
        if hasattr(candidate, "node_count") and hasattr(candidate, "edge_count"):
            return candidate

        pytest.fail(f"Unable to derive subgraph from {class_name}")

    def test_subgraph_counts_and_density(self):
        subgraph = self._ensure_single_subgraph()

        assert subgraph.node_count() > 0, "Subgraph should contain nodes"
        assert (
            subgraph.edge_count() >= subgraph.node_count() - 1
        ), "Subgraph edges should form connected structure"
        density = subgraph.density()
        assert isinstance(density, float), "density() should return float"
        assert 0.0 <= density <= 1.0

    def test_subgraph_membership_checks(self):
        subgraph = self._ensure_single_subgraph()
        node_ids = subgraph.node_ids.to_list()
        edge_ids = subgraph.edge_ids.to_list()

        assert subgraph.has_node(node_ids[0]), "Expected node to exist"
        if edge_ids:
            assert subgraph.has_edge(edge_ids[0]), "Expected edge to exist"

    def test_subgraph_path_and_connectivity(self):
        subgraph = self._ensure_single_subgraph()
        node_ids = subgraph.node_ids.to_list()
        if len(node_ids) < 2:
            pytest.skip("Need at least two nodes to test paths")

        path_exists = subgraph.has_path(node_ids[0], node_ids[-1])
        assert isinstance(path_exists, bool)
        components = subgraph.connected_components()
        assert hasattr(components, "__len__")
        comp_count = len(components)
        assert comp_count >= 1
        if comp_count == 1:
            assert subgraph.is_connected()

    def test_subgraph_degree_variants(self):
        subgraph = self._ensure_single_subgraph()
        degree_all = subgraph.degree()
        assert hasattr(degree_all, "to_list"), "degree() should return NumArray"
        degrees = degree_all.to_list()
        assert len(degrees) == subgraph.node_count()

        first_node = subgraph.node_ids.to_list()[0]
        degree_single = subgraph.degree(first_node)
        assert isinstance(degree_single, int)

        in_degrees = subgraph.in_degree()
        out_degrees = subgraph.out_degree()
        assert hasattr(in_degrees, "to_list") and hasattr(out_degrees, "to_list")
        assert len(in_degrees.to_list()) == subgraph.node_count()
        assert len(out_degrees.to_list()) == subgraph.node_count()

    def test_subgraph_tables_and_filters(self):
        subgraph = self._ensure_single_subgraph()
        graph_table = subgraph.table()
        edges_table = subgraph.edges_table()
        assert graph_table is not None
        assert edges_table is not None

        filtered_nodes = subgraph.filter_nodes("cluster == 0")
        assert type(filtered_nodes).__name__ == "Subgraph"

        filtered_edges = subgraph.filter_edges('edge_type == "intra_cluster"')
        assert type(filtered_edges).__name__ == "Subgraph"

    def test_subgraph_neighborhood_results(self):
        subgraph = self._ensure_single_subgraph()
        node_ids = subgraph.node_ids.to_list()
        if not node_ids:
            pytest.skip("Subgraph has no nodes for neighborhood expansion")

        neighborhood_single = subgraph.neighborhood(node_ids[0], hops=1)
        class_name = getattr(
            neighborhood_single.__class__,
            "__name__",
            neighborhood_single.__class__.__name__,
        )
        assert (
            class_name == "SubgraphArray"
        ), "neighborhood() should return SubgraphArray"
        assert (
            len(neighborhood_single) >= 1
        ), "Neighborhood expansion should yield subgraphs"

        neighborhood_list = subgraph.neighborhood([node_ids[0]], hops=1)
        assert len(neighborhood_list) == len(neighborhood_single)

    def test_subgraph_neighborhood_collapse(self):
        subgraph = self._ensure_single_subgraph()
        node_ids = subgraph.node_ids.to_list()
        if not node_ids:
            pytest.skip("Subgraph has no nodes for neighborhood collapse")

        neighborhoods = subgraph.neighborhood(node_ids[0], hops=1)
        assert len(neighborhoods) >= 1

        meta_nodes = neighborhoods.collapse(
            node_aggs={"size": "count"}, node_strategy="extract"
        )
        assert len(meta_nodes) == len(neighborhoods)
        for meta in meta_nodes:
            assert hasattr(meta, "id"), "MetaNode should expose id attribute"


@pytest.mark.subgraph_ops
class TestSubgraphOperationsCore(SubgraphTestBase, SubgraphOperationsTestMixin):
    """Comprehensive tests for single subgraph operations."""

    def get_subgraph_array(self, graph=None):
        if gr is None:
            pytest.skip("groggy not available")

        if graph is None:
            graph = self.get_test_graph()

        # Full-graph view provides a rich Subgraph to exercise operations
        return graph.view()

    def _get_subgraph(self):
        return self._ensure_single_subgraph()

    def test_subgraph_adjacency_structures(self):
        subgraph = self._get_subgraph()

        adjacency = subgraph.adjacency_list()
        assert isinstance(adjacency, dict), "adjacency_list() should return dict"
        assert adjacency, "Adjacency list should include entries"

        # Use to_matrix() to get adjacency matrix representation
        matrix = subgraph.to_matrix()
        rows, cols = matrix.shape
        assert (
            rows == cols == subgraph.node_count()
        ), "Adjacency matrix dimensions should match node count"

    def test_subgraph_traversals_and_neighbors(self):
        subgraph = self._get_subgraph()
        node_ids = subgraph.node_ids.to_list()
        start = node_ids[0]

        bfs_result = subgraph.bfs(start)
        dfs_result = subgraph.dfs(start)
        assert bfs_result.node_count() >= 1
        assert dfs_result.node_count() >= 1

        neighbors = subgraph.neighbors(start)
        assert hasattr(neighbors, "to_list")
        assert neighbors.to_list(), "neighbors() should list adjacent nodes"

    def test_subgraph_induced_and_shortest_path(self):
        subgraph = self._get_subgraph()
        node_ids = subgraph.node_ids.to_list()

        induced = subgraph.induced_subgraph(node_ids[:2])
        assert induced.node_count() == 2

        source, target = node_ids[0], node_ids[-1]
        assert subgraph.has_path(source, target)
        path_subgraph = subgraph.shortest_path_subgraph(source, target)
        assert path_subgraph.node_count() >= 2
        assert path_subgraph.edge_count() >= 1

    def test_subgraph_accessors_and_conversions(self):
        subgraph = self._get_subgraph()

        nodes_accessor = subgraph.to_nodes()
        edges_accessor = subgraph.to_edges()
        assert hasattr(nodes_accessor, "table")
        assert hasattr(edges_accessor, "table")
        assert nodes_accessor.table() is not None
        assert edges_accessor.table() is not None

        matrix = subgraph.to_matrix()
        assert hasattr(matrix, "shape")
        assert not matrix.is_empty()

        graph_copy = subgraph.to_graph()
        assert hasattr(graph_copy, "nodes"), "to_graph() should return a Graph"

    def test_subgraph_group_by_and_sample(self):
        subgraph = self._get_subgraph()

        node_groups = subgraph.group_by("cluster", "nodes")
        assert hasattr(node_groups, "__len__")
        assert len(node_groups) >= 3

        edge_groups = subgraph.group_by("edge_type", "edges")
        assert hasattr(edge_groups, "__len__")
        assert len(edge_groups) >= 2

        sampled = subgraph.sample(2)
        assert type(sampled).__name__ == "Subgraph"
        assert sampled.node_count() <= subgraph.node_count()

    def test_subgraph_collapse_returns_meta_node(self):
        subgraph = self._get_subgraph()
        meta_node = subgraph.collapse(
            node_aggs={"size": "count"},
            node_strategy="collapse",
        )
        assert hasattr(meta_node, "id"), "collapse() should return MetaNode"
