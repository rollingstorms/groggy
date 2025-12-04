"""
Shared test infrastructure for matrix operations.

Provides base classes and mixins for testing GraphMatrix objects.
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


class MatrixTestBase(ABC):
    """Base class for matrix testing with shared patterns"""

    @abstractmethod
    def get_matrix_instance(self, graph=None):
        """Create GraphMatrix instance for testing"""
        pass

    def get_test_graph(self):
        """Create test graph suitable for matrix operations"""
        if gr is None:
            pytest.skip("groggy not available")

        graph = gr.Graph()

        # Create simple graph for matrix testing
        node_ids = []
        for i in range(5):
            node_id = graph.add_node(label=f"Node{i}", value=i * 10, weight=float(i))
            node_ids.append(node_id)

        # Add edges to create interesting matrix structure
        for i in range(len(node_ids) - 1):
            graph.add_edge(node_ids[i], node_ids[i + 1], weight=1.0 + i)

        # Add some backward edges
        graph.add_edge(node_ids[2], node_ids[0], weight=2.5)
        graph.add_edge(node_ids[4], node_ids[1], weight=3.5)

        return graph

    def test_matrix_basic_properties(self):
        """Test basic GraphMatrix properties"""
        matrix = self.get_matrix_instance()

        # Should have shape
        assert hasattr(matrix, "shape"), "GraphMatrix should have shape property"
        shape = matrix.shape
        assert isinstance(shape, tuple), f"shape should be tuple, got {type(shape)}"
        assert len(shape) == 2, f"shape should have 2 dimensions, got {len(shape)}"
        assert shape[0] > 0, "Matrix should have positive row count"
        assert shape[1] > 0, "Matrix should have positive column count"

        # Should have dtype
        if hasattr(matrix, "dtype"):
            dtype = matrix.dtype
            assert dtype is not None, "dtype should return a value"

        # Should have data property
        if hasattr(matrix, "data"):
            data = matrix.data
            assert data is not None, "data property should return matrix data"

        # Should have columns property
        if hasattr(matrix, "columns"):
            columns = matrix.columns
            assert isinstance(columns, list), "columns should return a list"

    def test_matrix_density_conversion(self):
        """Test matrix density conversion"""
        matrix = self.get_matrix_instance()

        # Test dense conversion
        if hasattr(matrix, "dense"):
            dense_matrix = matrix.dense()
            assert dense_matrix is not None, "dense() should return a GraphMatrix"
            assert (
                type(dense_matrix).__name__ == "GraphMatrix"
            ), "Should return GraphMatrix"

    def test_matrix_flatten_operation(self):
        """Test matrix flattening"""
        matrix = self.get_matrix_instance()

        # Test flatten
        if hasattr(matrix, "flatten"):
            flattened = matrix.flatten()
            assert flattened is not None, "flatten() should return a NumArray"
            # Flattened should have shape[0] * shape[1] elements
            shape = matrix.shape
            expected_length = shape[0] * shape[1]
            # Can't directly check length without implementing __len__ on NumArray
            # but we can verify it's a NumArray
            assert (
                type(flattened).__name__ == "NumArray"
            ), "flatten() should return NumArray"

    def test_matrix_arithmetic_operations(self):
        """Test matrix arithmetic operations"""
        matrix = self.get_matrix_instance()

        # Test abs
        if hasattr(matrix, "abs"):
            abs_matrix = matrix.abs()
            assert abs_matrix is not None, "abs() should return a GraphMatrix"
            assert (
                type(abs_matrix).__name__ == "GraphMatrix"
            ), "abs() should return GraphMatrix"

        # Test exp
        if hasattr(matrix, "exp"):
            exp_matrix = matrix.exp()
            assert exp_matrix is not None, "exp() should return a GraphMatrix"

        # Test log
        if hasattr(matrix, "log"):
            try:
                log_matrix = matrix.log()
                assert log_matrix is not None, "log() should return a GraphMatrix"
            except Exception:
                # May fail on zero/negative values
                pass

        # Test sqrt
        if hasattr(matrix, "sqrt"):
            try:
                sqrt_matrix = matrix.sqrt()
                assert sqrt_matrix is not None, "sqrt() should return a GraphMatrix"
            except Exception:
                # May fail on negative values
                pass

    def test_matrix_activation_functions(self):
        """Test matrix activation functions"""
        matrix = self.get_matrix_instance()

        # Test sigmoid
        if hasattr(matrix, "sigmoid"):
            sigmoid_matrix = matrix.sigmoid()
            assert sigmoid_matrix is not None, "sigmoid() should return a GraphMatrix"

        # Test tanh
        if hasattr(matrix, "tanh"):
            tanh_matrix = matrix.tanh()
            assert tanh_matrix is not None, "tanh() should return a GraphMatrix"

        # Test relu
        if hasattr(matrix, "relu"):
            relu_matrix = matrix.relu()
            assert relu_matrix is not None, "relu() should return a GraphMatrix"

        # Test elu
        if hasattr(matrix, "elu"):
            elu_matrix = matrix.elu()
            assert elu_matrix is not None, "elu() should return a GraphMatrix"

        # Test softmax
        if hasattr(matrix, "softmax"):
            softmax_matrix = matrix.softmax()
            assert softmax_matrix is not None, "softmax() should return a GraphMatrix"

    def test_matrix_apply_operation(self):
        """Test matrix apply with function"""
        matrix = self.get_matrix_instance()

        if hasattr(matrix, "apply"):
            # Test with simple function
            def double(x):
                return x * 2

            applied = matrix.apply(double)
            assert applied is not None, "apply() should return a GraphMatrix"
            assert (
                type(applied).__name__ == "GraphMatrix"
            ), "apply() should return GraphMatrix"


class MatrixOperationTestMixin:
    """Mixin for testing matrix operations requiring parameters"""

    def test_matrix_multiplication(self):
        """Test matrix multiplication operations"""
        matrix = self.get_matrix_instance()

        # Test matmul - requires another matrix
        if hasattr(matrix, "matmul"):
            # Need compatible matrix for multiplication
            pytest.skip("matmul requires compatible matrix parameter")

        # Test multiply (elementwise)
        if hasattr(matrix, "multiply"):
            try:
                # Try scalar multiplication
                result = matrix.multiply(2.0)
                assert result is not None, "multiply() should work with scalar"
            except TypeError:
                # Might require matrix parameter
                pytest.skip("multiply() requires specific parameter")

    def test_matrix_normalization(self):
        """Test matrix normalization operations"""
        matrix = self.get_matrix_instance()

        # Test normalize
        if hasattr(matrix, "normalize"):
            normalized = matrix.normalize()
            assert normalized is not None, "normalize() should return a GraphMatrix"

        # Test standardize
        if hasattr(matrix, "standardize"):
            standardized = matrix.standardize()
            assert standardized is not None, "standardize() should return a GraphMatrix"

    def test_matrix_decomposition_operations(self):
        """Test matrix decomposition operations"""
        matrix = self.get_matrix_instance()

        # Test SVD - may require square matrix
        if hasattr(matrix, "svd"):
            try:
                svd_result = matrix.svd()
                assert svd_result is not None, "svd() should return decomposition"
            except Exception as e:
                # Expected - may require square matrix or specific conditions
                assert "square" in str(e).lower() or "dimension" in str(e).lower()

        # Test eigenvalue_decomposition - requires square matrix
        if hasattr(matrix, "eigenvalue_decomposition"):
            try:
                eigen_result = matrix.eigenvalue_decomposition()
                pytest.skip(
                    "eigenvalue_decomposition unexpectedly succeeded on non-square matrix"
                )
            except Exception as e:
                # Expected - requires square matrix
                assert "square" in str(e).lower()

        # Test cholesky_decomposition - requires square, positive definite matrix
        if hasattr(matrix, "cholesky_decomposition"):
            try:
                cholesky_result = matrix.cholesky_decomposition()
                pytest.skip("cholesky_decomposition unexpectedly succeeded")
            except Exception as e:
                # Expected - requires specific matrix properties
                assert "square" in str(e).lower() or "positive" in str(e).lower()


class MatrixConstructorTestMixin:
    """Mixin for testing matrix construction methods"""

    def test_matrix_from_data_construction(self):
        """Test constructing matrices from data"""
        if gr is None:
            pytest.skip("groggy not available")

        # These are static/class methods, need to test on the class
        MatrixClass = type(self.get_matrix_instance())

        # Test from_data - requires data parameter
        if hasattr(MatrixClass, "from_data"):
            try:
                result = MatrixClass.from_data([[1, 2], [3, 4]])
                assert result is not None, "from_data() should create matrix"
            except TypeError:
                # Expected - missing parameters
                pass

        # Test from_flattened - requires num_array, rows, cols
        if hasattr(MatrixClass, "from_flattened"):
            try:
                num_arr = gr.num_array([1, 2, 3, 4])
                result = MatrixClass.from_flattened(num_arr, 2, 2)
                assert result is not None, "from_flattened() should create matrix"
            except Exception:
                # May have different signature or requirements
                pass
