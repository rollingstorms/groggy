"""
Module 6: GraphMatrix Testing - Milestone 6

Tests GraphMatrix type for linear algebra operations on graphs.
GraphMatrix represents adjacency, Laplacian, and other graph matrices.

Test Coverage:
GraphMatrix (124/192 methods pass, ~65% pass rate):
- Basic properties (shape, dtype, data, columns)
- Density conversion (dense, sparse)
- Arithmetic operations (abs, exp, log, sqrt)
- Activation functions (sigmoid, tanh, relu, elu, softmax)
- Matrix operations (transpose, matmul, multiply)
- Decomposition (SVD, eigenvalue, Cholesky)
- Normalization (normalize, standardize)
- Construction (from_data, from_flattened)
- Reduction operations (sum, mean, min, max)
- Advanced operations (apply, filter, dropout)

Success Criteria: 90%+ pass rate on available methods, proper error handling
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
from tests.modules.test_matrix_base import (MatrixConstructorTestMixin,
                                            MatrixOperationTestMixin,
                                            MatrixTestBase)


class GraphMatrixTest(
    MatrixTestBase, MatrixOperationTestMixin, MatrixConstructorTestMixin
):
    """Test class for GraphMatrix using shared test patterns"""

    def get_matrix_instance(self, graph=None):
        """Create GraphMatrix instance for testing"""
        if gr is None:
            pytest.skip("groggy not available")

        if graph is None:
            graph = self.get_test_graph()

        # Get GraphMatrix from graph
        return graph.to_matrix()


@pytest.mark.graph_matrix
class TestGraphMatrix(GraphMatrixTest):
    """Test GraphMatrix functionality"""

    def test_graph_matrix_creation(self):
        """Test creating GraphMatrix from graph"""
        graph = self.get_test_graph()

        # Test to_matrix
        matrix = graph.to_matrix()
        assert matrix is not None, "graph.to_matrix() should return a GraphMatrix"
        assert type(matrix).__name__ == "GraphMatrix", "Should return GraphMatrix type"

        # Test laplacian_matrix
        if hasattr(graph, "laplacian_matrix"):
            laplacian = graph.laplacian_matrix()
            assert (
                laplacian is not None
            ), "graph.laplacian_matrix() should return a GraphMatrix"

    def test_graph_matrix_from_accessors(self):
        """Test creating GraphMatrix from accessors"""
        graph = self.get_test_graph()

        # Test nodes.matrix()
        if hasattr(graph.nodes, "matrix"):
            nodes_matrix = graph.nodes.matrix()
            assert (
                nodes_matrix is not None
            ), "nodes.matrix() should return a GraphMatrix"

        # Test edges.matrix()
        if hasattr(graph.edges, "matrix"):
            edges_matrix = graph.edges.matrix()
            assert (
                edges_matrix is not None
            ), "edges.matrix() should return a GraphMatrix"

        # Test edges.weight_matrix()
        if hasattr(graph.edges, "weight_matrix"):
            weight_matrix = graph.edges.weight_matrix()
            assert (
                weight_matrix is not None
            ), "weight_matrix() should return a GraphMatrix"

    def test_graph_matrix_basic_properties(self):
        """Test basic GraphMatrix properties"""
        # Test inherited base methods
        self.test_matrix_basic_properties()

        matrix = self.get_matrix_instance()

        # Additional specific tests
        shape = matrix.shape
        assert (
            shape[0] > 0 and shape[1] > 0
        ), f"Matrix should have positive dimensions: {shape}"

        # Test data property
        data = matrix.data
        assert isinstance(data, list), "data should return a list"

        # Test columns property
        columns = matrix.columns
        assert isinstance(columns, list), "columns should return a list"
        assert (
            len(columns) == shape[1]
        ), f"columns length {len(columns)} should match shape[1] {shape[1]}"

    def test_graph_matrix_density_operations(self):
        """Test GraphMatrix density conversion"""
        # Test inherited base methods
        self.test_matrix_density_conversion()

        matrix = self.get_matrix_instance()

        # Test dense_html_repr for display
        if hasattr(matrix, "dense_html_repr"):
            html = matrix.dense_html_repr()
            assert isinstance(html, str), "dense_html_repr() should return string"
            assert len(html) > 0, "HTML representation should not be empty"

    def test_graph_matrix_flatten(self):
        """Test GraphMatrix flatten operation"""
        # Test inherited base methods
        self.test_matrix_flatten_operation()

        matrix = self.get_matrix_instance()
        shape = matrix.shape

        flattened = matrix.flatten()
        assert (
            type(flattened).__name__ == "NumArray"
        ), "flatten() should return NumArray"

    def test_graph_matrix_arithmetic(self):
        """Test GraphMatrix arithmetic operations"""
        # Test inherited base methods
        self.test_matrix_arithmetic_operations()

        matrix = self.get_matrix_instance()

        # Test operations that should work
        operations = [
            ("abs", lambda: matrix.abs()),
            ("exp", lambda: matrix.exp()),
        ]

        for op_name, op_func in operations:
            if hasattr(matrix, op_name):
                result = op_func()
                assert result is not None, f"{op_name}() should return a result"
                assert (
                    type(result).__name__ == "GraphMatrix"
                ), f"{op_name}() should return GraphMatrix"

    def test_graph_matrix_activations(self):
        """Test GraphMatrix activation functions"""
        # Test inherited base methods
        self.test_matrix_activation_functions()

        matrix = self.get_matrix_instance()

        # Test common activations
        activations = ["sigmoid", "tanh", "relu", "elu"]
        for activation in activations:
            if hasattr(matrix, activation):
                result = getattr(matrix, activation)()
                assert result is not None, f"{activation}() should return a GraphMatrix"
                assert type(result).__name__ == "GraphMatrix"

    def test_graph_matrix_apply(self):
        """Test GraphMatrix apply operation"""
        # Test inherited base methods
        self.test_matrix_apply_operation()

        matrix = self.get_matrix_instance()

        # Test with various functions
        def square(x):
            return x * x

        def add_one(x):
            return x + 1

        for func, name in [(square, "square"), (add_one, "add_one")]:
            applied = matrix.apply(func)
            assert applied is not None, f"apply({name}) should return a GraphMatrix"

    def test_graph_matrix_reduction_operations(self):
        """Test GraphMatrix reduction operations"""
        matrix = self.get_matrix_instance()

        # Test sum
        if hasattr(matrix, "sum"):
            total = matrix.sum()
            assert total is not None, "sum() should return a value"

        # Test mean
        if hasattr(matrix, "mean"):
            avg = matrix.mean()
            assert avg is not None, "mean() should return a value"

        # Test min
        if hasattr(matrix, "min"):
            minimum = matrix.min()
            assert minimum is not None, "min() should return a value"

        # Test max
        if hasattr(matrix, "max"):
            maximum = matrix.max()
            assert maximum is not None, "max() should return a value"

    def test_graph_matrix_transpose(self):
        """Test GraphMatrix transpose operation"""
        matrix = self.get_matrix_instance()

        if hasattr(matrix, "transpose"):
            transposed = matrix.transpose()
            assert transposed is not None, "transpose() should return a GraphMatrix"
            assert type(transposed).__name__ == "GraphMatrix"

            # Transposed shape should be swapped
            original_shape = matrix.shape
            transposed_shape = transposed.shape
            assert (
                transposed_shape[0] == original_shape[1]
            ), "Transposed rows should equal original columns"
            assert (
                transposed_shape[1] == original_shape[0]
            ), "Transposed columns should equal original rows"

    def test_graph_matrix_trace(self):
        """Test GraphMatrix trace operation"""
        matrix = self.get_matrix_instance()

        if hasattr(matrix, "trace"):
            try:
                trace_val = matrix.trace()
                # Trace may require square matrix
                if matrix.shape[0] == matrix.shape[1]:
                    assert (
                        trace_val is not None
                    ), "trace() should return a value for square matrix"
            except Exception as e:
                # May require square matrix
                if matrix.shape[0] != matrix.shape[1]:
                    assert "square" in str(e).lower()


@pytest.mark.graph_matrix
@pytest.mark.matrix_operations
class TestGraphMatrixOperations(GraphMatrixTest):
    """Test GraphMatrix advanced operations"""

    def test_graph_matrix_normalization(self):
        """Test GraphMatrix normalization"""
        # Test inherited mixin methods
        self.test_matrix_normalization()

    def test_graph_matrix_decomposition(self):
        """Test GraphMatrix decomposition operations"""
        # Test inherited mixin methods
        self.test_matrix_decomposition_operations()

    def test_graph_matrix_multiplication(self):
        """Test GraphMatrix multiplication"""
        # Test inherited mixin methods
        self.test_matrix_multiplication()


@pytest.mark.graph_matrix
@pytest.mark.integration
class TestGraphMatrixIntegration:
    """Test GraphMatrix integration with graph operations"""

    def test_graph_matrix_round_trip(self):
        """Test graph -> matrix -> operations"""
        if gr is None:
            pytest.skip("groggy not available")

        base_test = GraphMatrixTest()
        graph = base_test.get_test_graph()

        # Convert to matrix
        matrix = graph.to_matrix()
        assert matrix is not None, "Should create matrix from graph"

        # Perform operations
        dense = matrix.dense()
        assert dense is not None, "Should convert to dense"

        flattened = matrix.flatten()
        assert flattened is not None, "Should flatten matrix"

    def test_graph_matrix_from_multiple_sources(self):
        """Test creating matrices from different graph sources"""
        if gr is None:
            pytest.skip("groggy not available")

        base_test = GraphMatrixTest()
        graph = base_test.get_test_graph()

        # From graph
        graph_matrix = graph.to_matrix()
        assert graph_matrix is not None, "Should create from graph"

        # From nodes accessor
        if hasattr(graph.nodes, "matrix"):
            nodes_matrix = graph.nodes.matrix()
            assert nodes_matrix is not None, "Should create from nodes"

        # From edges accessor
        if hasattr(graph.edges, "matrix"):
            edges_matrix = graph.edges.matrix()
            assert edges_matrix is not None, "Should create from edges"

    def test_graph_matrix_chaining(self):
        """Test chaining matrix operations"""
        if gr is None:
            pytest.skip("groggy not available")

        base_test = GraphMatrixTest()
        graph = base_test.get_test_graph()

        # Chain: to_matrix -> dense -> abs -> flatten
        result = graph.to_matrix().dense().abs().flatten()
        assert result is not None, "Chained operations should succeed"
        assert type(result).__name__ == "NumArray", "Final result should be NumArray"

    def test_laplacian_matrix_properties(self):
        """Test Laplacian matrix specific properties"""
        if gr is None:
            pytest.skip("groggy not available")

        base_test = GraphMatrixTest()
        graph = base_test.get_test_graph()

        if hasattr(graph, "laplacian_matrix"):
            laplacian = graph.laplacian_matrix()
            assert laplacian is not None, "Should create Laplacian matrix"

            # Laplacian should be square
            shape = laplacian.shape
            assert shape[0] == shape[1], "Laplacian matrix should be square"


if __name__ == "__main__":
    # Allow running this module directly for development
    pytest.main([__file__, "-v"])
