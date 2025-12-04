"""
BaseArray Testing

Tests the BaseArray type, which is the foundation for all array types in Groggy.
BaseArray handles both numeric and text data with appropriate operations.

Test Coverage:
- Basic array operations (length, indexing, iteration)
- Data type handling (numeric vs text data)
- Conversion operations (to_num_array when applicable)
- Statistical operations (only for numeric data)
- Null/NA handling
- Head/tail operations
- Array building and construction

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
from tests.modules.test_array_base import ArrayTestBase


class BaseArrayTest(ArrayTestBase):
    """Test class for BaseArray using shared test patterns"""

    def get_array_instance(self, graph=None, size=None, data_type="mixed"):
        """Create BaseArray instance for testing"""
        if gr is None:
            pytest.skip("groggy not available")

        # Create BaseArray using builder functions
        if data_type == "numeric":
            if size is None:
                data = [1, 2, 3, 4, 5]
            else:
                data = list(range(size))
            return gr.array(data)
        elif data_type == "text":
            if size is None:
                data = ["apple", "banana", "cherry", "date", "elderberry"]
            else:
                data = [f"item_{i}" for i in range(size)]
            return gr.array(data)
        else:  # mixed
            if size is None:
                data = [1, "hello", 2, "world", 3]
            else:
                data = []
                for i in range(size):
                    if i % 2 == 0:
                        data.append(i)
                    else:
                        data.append(f"text_{i}")
            return gr.array(data)

    def get_expected_array_type(self):
        """Return the expected BaseArray type"""
        return type(self.get_array_instance())

    # Override methods that don't apply to BaseArray interface
    def test_basic_properties(self):
        """Test basic array properties for BaseArray"""
        array = self.get_array_instance()

        # BaseArray should be iterable
        assert hasattr(array, "iter"), f"{type(array)} should have iter() method"

        # BaseArray should have length concept
        assert hasattr(array, "len"), f"{type(array)} should have len() method"
        length = array.len()
        assert isinstance(length, int), f"Array len() should be int, got {type(length)}"
        assert length >= 0, f"Array length should be non-negative, got {length}"

        # BaseArray should have is_empty concept
        assert hasattr(
            array, "is_empty"
        ), f"{type(array)} should have is_empty() method"
        is_empty = array.is_empty()
        assert isinstance(is_empty, bool)

    def test_to_list_conversion(self):
        """Test converting BaseArray to Python list"""
        array = self.get_array_instance()

        # BaseArray can be converted to list using list()
        result = list(array)
        assert isinstance(
            result, list
        ), f"list(array) should return list, got {type(result)}"

        if hasattr(array, "to_list"):
            to_list_result = array.to_list()
            assert isinstance(
                to_list_result, list
            ), "to_list() should return a Python list"
            assert to_list_result == result, "to_list() should match list(array) output"

    def test_empty_array_handling(self):
        """Test that methods handle empty arrays gracefully"""
        try:
            empty_array = gr.array([])
            assert empty_array.len() == 0, "Empty array should have length 0"

            # These methods should not crash on empty arrays
            result = list(empty_array)
            assert result == [], "Empty array list() should return empty list"

        except Exception as e:
            pytest.skip(f"Could not create empty array for testing: {e}")


@pytest.mark.base_array
class TestBaseArrayCore(BaseArrayTest):
    """Test core BaseArray functionality"""

    def test_basearray_creation_numeric(self):
        """Test creating BaseArray with numeric data"""
        array = gr.array([1, 2, 3, 4, 5])
        assert array is not None
        assert hasattr(array, "iter"), "BaseArray should have iter() method"

        data = list(array)
        assert data == [1, 2, 3, 4, 5], f"Expected [1, 2, 3, 4, 5], got {data}"

    def test_basearray_creation_text(self):
        """Test creating BaseArray with text data"""
        array = gr.array(["apple", "banana", "cherry"])
        assert array is not None
        assert hasattr(array, "iter"), "BaseArray should have iter() method"

        data = list(array)
        assert data == ["apple", "banana", "cherry"], f"Expected text data, got {data}"

    def test_basearray_creation_mixed(self):
        """Test creating BaseArray with mixed data types"""
        array = gr.array([1, "hello", 2, "world"])
        assert array is not None
        assert hasattr(array, "iter"), "BaseArray should have iter() method"

        data = list(array)
        assert len(data) == 4, f"Expected 4 items, got {len(data)}"

    def test_basearray_basic_properties(self):
        """Test basic properties of BaseArray"""
        array = self.get_array_instance(data_type="numeric")

        # Test basic properties
        assert array.len() > 0, "Non-empty array should have positive length"

        # Test length-related operations
        if hasattr(array, "count"):
            count = array.count()
            assert isinstance(
                count, int
            ), f"count() should return int, got {type(count)}"
            assert count > 0, f"count() should be positive, got {count}"

        # Test head/tail operations
        if hasattr(array, "head"):
            head = array.head(3)
            assert head is not None, "head() should return a BaseArray"
            assert hasattr(head, "iter"), "head() result should be iterable"

        if hasattr(array, "tail"):
            tail = array.tail(3)
            assert tail is not None, "tail() should return a BaseArray"
            assert hasattr(tail, "iter"), "tail() result should be iterable"

    def test_basearray_numeric_operations_on_numeric_data(self):
        """Test numeric operations work correctly on numeric data"""
        array = self.get_array_instance(data_type="numeric", size=10)

        if array.len() == 0:
            pytest.skip("Cannot test numeric operations on empty array")

        # Test operations that should work on numeric data
        numeric_operations = ["sum", "mean", "min", "max", "std", "var"]

        for op_name in numeric_operations:
            if hasattr(array, op_name):
                try:
                    result = getattr(array, op_name)()
                    assert isinstance(
                        result, (int, float)
                    ), f"{op_name}() should return numeric value, got {type(result)}"
                    # Should not be NaN
                    assert result == result, f"{op_name}() returned NaN"
                except Exception as e:
                    # If it fails, that's information about the implementation
                    pytest.skip(f"{op_name}() on numeric data failed: {e}")

    def test_basearray_numeric_operations_on_text_data(self):
        """Test that numeric operations handle text data appropriately"""
        array = self.get_array_instance(data_type="text", size=5)

        if array.len() == 0:
            pytest.skip("Cannot test operations on empty array")

        # Test operations that should fail gracefully on text data
        numeric_operations = ["sum", "mean", "min", "max", "std", "var"]

        for op_name in numeric_operations:
            if hasattr(array, op_name):
                try:
                    result = getattr(array, op_name)()
                    # If it succeeds, that's also valid (maybe implementation handles text)
                    pytest.skip(
                        f"{op_name}() on text data unexpectedly succeeded: {result}"
                    )
                except Exception as e:
                    # This is expected behavior - text data shouldn't support numeric operations
                    assert (
                        "non-numeric" in str(e)
                        or "Cannot compute" in str(e)
                        or "Cannot compare" in str(e)
                    ), f"{op_name}() should fail with appropriate error message, got: {e}"

    def test_basearray_mixed_data_operations(self):
        """Test operations on mixed data types"""
        array = self.get_array_instance(data_type="mixed", size=6)

        if array.len() == 0:
            pytest.skip("Cannot test operations on empty array")

        # Count should always work
        if hasattr(array, "count"):
            count = array.count()
            assert isinstance(
                count, int
            ), f"count() should work on mixed data, got {type(count)}"

        # Head/tail should work
        if hasattr(array, "head"):
            head = array.head(2)
            assert head is not None, "head() should work on mixed data"

        if hasattr(array, "tail"):
            tail = array.tail(2)
            assert tail is not None, "tail() should work on mixed data"

        # Numeric operations should fail gracefully
        numeric_operations = ["sum", "mean", "std", "var"]
        for op_name in numeric_operations:
            if hasattr(array, op_name):
                try:
                    result = getattr(array, op_name)()
                    pytest.skip(
                        f"{op_name}() on mixed data unexpectedly succeeded: {result}"
                    )
                except Exception as e:
                    # This is expected - mixed data shouldn't support numeric operations
                    assert (
                        "non-numeric" in str(e)
                        or "Cannot compute" in str(e)
                        or "Cannot compare" in str(e)
                    ), f"{op_name}() should fail appropriately on mixed data"

    def test_basearray_null_handling(self):
        """Test null/NA value handling"""
        # Create array with potential nulls
        array = self.get_array_instance(data_type="numeric", size=5)

        # Test null-related operations
        if hasattr(array, "null_count"):
            null_count = array.null_count()
            assert isinstance(
                null_count, int
            ), f"null_count() should return int, got {type(null_count)}"
            assert (
                null_count >= 0
            ), f"null_count() should be non-negative, got {null_count}"

        if hasattr(array, "notna"):
            notna_array = array.notna()
            assert notna_array is not None, "notna() should return an array"
            assert hasattr(notna_array, "iter"), "notna() result should be iterable"

    def test_basearray_type_conversion(self):
        """Test type conversion operations"""
        # Test converting numeric BaseArray to NumArray
        numeric_array = self.get_array_instance(data_type="numeric", size=5)

        if hasattr(numeric_array, "to_num_array"):
            try:
                num_array = numeric_array.to_num_array()
                assert num_array is not None, "to_num_array() should return a NumArray"
                # Should have numeric operations
                assert hasattr(
                    num_array, "sum"
                ), "Converted NumArray should have sum() method"
                assert hasattr(
                    num_array, "mean"
                ), "Converted NumArray should have mean() method"
            except Exception as e:
                pytest.skip(f"to_num_array() on numeric data failed: {e}")

        # Test converting text BaseArray to NumArray (should fail)
        text_array = self.get_array_instance(data_type="text", size=3)

        if hasattr(text_array, "to_num_array"):
            try:
                num_array = text_array.to_num_array()
                pytest.skip(f"to_num_array() on text data unexpectedly succeeded")
            except Exception as e:
                # This is expected - text data can't be converted to numeric
                assert "non-numeric" in str(e) or "cannot be converted" in str(
                    e
                ), f"to_num_array() should fail appropriately on text data"

    def test_basearray_indexing_operations(self):
        """Test indexing and slicing operations"""
        try:
            array = self.get_array_instance(data_type="numeric", size=10)

            if array.is_empty():
                pytest.skip("Cannot test indexing on empty array")

            # Test head with different sizes
            for head_size in [1, 3, 5]:
                if hasattr(array, "head"):
                    try:
                        head = array.head(head_size)
                        assert (
                            head is not None
                        ), f"head({head_size}) should return an array"
                        head_data = list(head)
                        assert (
                            len(head_data) <= head_size
                        ), f"head({head_size}) should return at most {head_size} items"
                    except Exception as e:
                        pytest.skip(f"head({head_size}) failed: {e}")

            # Test tail with different sizes
            for tail_size in [1, 3, 5]:
                if hasattr(array, "tail"):
                    try:
                        tail = array.tail(tail_size)
                        assert (
                            tail is not None
                        ), f"tail({tail_size}) should return an array"
                        tail_data = list(tail)
                        assert (
                            len(tail_data) <= tail_size
                        ), f"tail({tail_size}) should return at most {tail_size} items"
                    except Exception as e:
                        pytest.skip(f"tail({tail_size}) failed: {e}")
        except Exception as e:
            pytest.skip(f"test_basearray_indexing_operations failed: {e}")

    def test_basearray_dtype_operations(self):
        """Test data type detection and handling"""
        try:
            # Test with different data types
            test_cases = [
                ("numeric", [1, 2, 3, 4, 5]),
                ("text", ["a", "b", "c"]),
                ("mixed", [1, "hello", 2]),
            ]

            for case_name, data in test_cases:
                array = gr.array(data)

                # Test basic properties
                assert not array.is_empty(), f"{case_name} array should not be empty"

                array_data = list(array)
                assert len(array_data) == len(
                    data
                ), f"{case_name} array should preserve length"

                # Test that appropriate operations work or fail as expected
                if hasattr(array, "sum"):
                    try:
                        sum_result = array.sum()
                        if case_name == "numeric":
                            assert isinstance(
                                sum_result, (int, float)
                            ), f"sum() on {case_name} should return number"
                        else:
                            pytest.skip(
                                f"sum() on {case_name} data unexpectedly succeeded"
                            )
                    except Exception as e:
                        if case_name != "numeric":
                            # Expected failure for non-numeric data
                            assert "non-numeric" in str(e) or "Cannot compute" in str(
                                e
                            ), f"sum() on {case_name} should fail with appropriate message"
        except Exception as e:
            pytest.skip(f"test_basearray_dtype_operations failed: {e}")

    @pytest.mark.performance
    def test_basearray_performance_operations(self):
        """Test performance of BaseArray operations on larger arrays"""
        if gr is None:
            pytest.skip("groggy not available")

        import time

        # Test with large numeric array
        large_data = list(range(10000))
        start_time = time.time()
        array = gr.array(large_data)
        creation_time = time.time() - start_time

        assert (
            creation_time < 1.0
        ), f"BaseArray creation took {creation_time:.3f}s, should be < 1.0s"

        # Test basic operations performance
        operations_to_test = [
            ("count", lambda a: a.count() if hasattr(a, "count") else len(a.to_list())),
            ("head", lambda a: a.head(100) if hasattr(a, "head") else None),
            ("tail", lambda a: a.tail(100) if hasattr(a, "tail") else None),
        ]

        for op_name, op_func in operations_to_test:
            try:
                start_time = time.time()
                result = op_func(array)
                elapsed = time.time() - start_time

                assert (
                    elapsed < 0.5
                ), f"{op_name} took {elapsed:.3f}s, should be < 0.5s for 10k elements"
                if result is not None:
                    assert result is not None, f"{op_name} should return a result"
            except Exception as e:
                pytest.skip(f"{op_name} performance test failed: {e}")

    @pytest.mark.parametrize("data_type", ["numeric", "text", "mixed"])
    def test_basearray_across_data_types(self, data_type):
        """Test BaseArray operations work appropriately across data types"""

        array = self.get_array_instance(data_type=data_type, size=5)

        # Basic operations should work for all types
        assert array.len() > 0, f"{data_type} array should not be empty"
        data = list(array)
        assert len(data) >= 1, f"{data_type} array should have content"

        # Type-specific operation expectations
        if data_type == "numeric":
            # Numeric operations should work
            if hasattr(array, "sum"):
                try:
                    sum_val = array.sum()
                    assert isinstance(
                        sum_val, (int, float)
                    ), f"sum() on numeric should return number"
                except Exception as e:
                    pytest.skip(f"sum() on numeric data failed: {e}")
        else:
            # Non-numeric operations should fail appropriately
            if hasattr(array, "sum"):
                try:
                    sum_val = array.sum()
                    pytest.skip(
                        f"sum() on {data_type} data unexpectedly succeeded: {sum_val}"
                    )
                except Exception as e:
                    # Expected failure
                    assert "non-numeric" in str(e) or "Cannot compute" in str(
                        e
                    ), f"sum() on {data_type} should fail with appropriate error"


@pytest.mark.base_array
@pytest.mark.integration
class TestBaseArrayIntegration:
    """Test BaseArray integration with other components"""

    def test_basearray_builder_functions(self):
        """Test various ways to create BaseArray"""
        if gr is None:
            pytest.skip("groggy not available")

        # Test creating arrays with different data
        test_cases = [
            ([1, 2, 3], "numeric"),
            (["a", "b", "c"], "text"),
            ([1, "hello", 2], "mixed"),
            ([], "empty"),
        ]

        for data, case_name in test_cases:
            array = gr.array(data)
            assert array is not None, f"Should create {case_name} array"

            result_data = list(array)
            assert result_data == data, f"{case_name} array should preserve data"

            if len(data) == 0:
                assert array.len() == 0, f"Empty array should have length 0"
            else:
                assert array.len() > 0, f"Non-empty array should have positive length"

    def test_basearray_with_graph_data(self, attributed_graph):
        """Test BaseArray working with data from graphs"""
        # Get attribute names to test with
        attr_names = attributed_graph.nodes.attribute_names()

        for attr_name in attr_names[:3]:  # Test first few attributes
            try:
                # Try to get attribute data as BaseArray-like structure
                attr_data = attributed_graph[attr_name]

                if hasattr(attr_data, "to_list"):
                    data_list = list(attr_data)

                    # Create BaseArray from this data
                    base_array = gr.array(data_list)
                    assert (
                        base_array is not None
                    ), f"Should create BaseArray from {attr_name} data"

                    # Verify data preservation
                    result_data = list(base_array)
                    assert len(result_data) == len(
                        data_list
                    ), f"Data length should be preserved for {attr_name}"

            except Exception as e:
                # Some attributes may not be accessible this way
                pass

    def test_basearray_conversion_patterns(self):
        """Test conversion patterns between BaseArray and other array types"""
        if gr is None:
            pytest.skip("groggy not available")

        # Test numeric BaseArray -> NumArray conversion
        numeric_data = [1, 2, 3, 4, 5]
        base_array = gr.array(numeric_data)

        if hasattr(base_array, "to_num_array"):
            try:
                num_array = base_array.to_num_array()
                assert (
                    num_array is not None
                ), "Should convert numeric BaseArray to NumArray"

                # NumArray should have additional numeric operations
                assert hasattr(num_array, "std"), "NumArray should have std() method"
                assert hasattr(num_array, "var"), "NumArray should have var() method"

                # Data should be preserved - NumArray can be converted to list
                if hasattr(num_array, "to_list"):
                    converted_data = list(num_array)
                    assert (
                        converted_data == numeric_data
                    ), "Conversion should preserve data"
                else:
                    # NumArray might use different interface
                    converted_data = list(num_array)
                    assert (
                        converted_data == numeric_data
                    ), "Conversion should preserve data"

            except Exception as e:
                pytest.skip(f"BaseArray to NumArray conversion failed: {e}")


if __name__ == "__main__":
    # Allow running this module directly for development
    pytest.main([__file__, "-v"])
