#!/usr/bin/env python3
"""
Test PyGraphMatrix with NumArray/BaseArray integration (Phase 2.1)
"""

import groggy as gg

def test_matrix_with_new_arrays():
    print("Testing PyGraphMatrix with NumArray/BaseArray integration...")
    
    # Create a graph with some data
    g = gg.Graph()
    
    # Add nodes with numerical attributes (use correct API)
    g.add_node(value=10, score=0.5)
    g.add_node(value=20, score=0.7)
    g.add_node(value=30, score=0.9)
    
    # Test 1: Create NumArrays (numerical data)
    print("\n1. Creating NumArrays:")
    values_array = gg.NumArray([10.0, 20.0, 30.0])
    scores_array = gg.NumArray([0.5, 0.7, 0.9])
    print(f"values_array: {values_array}")
    print(f"scores_array: {scores_array}")
    
    # Test 2: Create BaseArrays (mixed data)
    print("\n2. Creating BaseArrays:")
    mixed_array = gg.BaseArray([10, "hello", 0.5, True])
    print(f"mixed_array: {mixed_array}")
    
    # Test 3: Create matrix with NumArrays (all numerical)
    print("\n3. Creating matrix with NumArrays:")
    try:
        matrix_num = gg.GraphMatrix([values_array, scores_array])
        print(f"Numerical matrix: {matrix_num}")
        print(f"Matrix shape: {matrix_num.shape}")
        print(f"Matrix dtype: {matrix_num.dtype}")
    except Exception as e:
        print(f"Error creating numerical matrix: {e}")
    
    # Test 4: Create matrix with BaseArrays (mixed types)
    print("\n4. Creating matrix with BaseArrays:")
    try:
        name_array = gg.BaseArray(["A", "B", "C"])
        id_array = gg.BaseArray([1, 2, 3])
        matrix_mixed = gg.GraphMatrix([id_array, name_array, mixed_array[:3]])
        print(f"Mixed matrix: {matrix_mixed}")
        print(f"Matrix shape: {matrix_mixed.shape}")
        print(f"Matrix dtype: {matrix_mixed.dtype}")
    except Exception as e:
        print(f"Error creating mixed matrix: {e}")
    
    # Test 5: Access matrix rows and columns
    if 'matrix_num' in locals():
        print("\n5. Testing matrix access methods:")
        try:
            # Get row (should return NumArray for numerical matrix)
            row_0 = matrix_num[0]
            print(f"Row 0: {row_0} (type: {type(row_0).__name__})")
            
            # Get column by name (should return NumArray for numerical data)
            col_0 = matrix_num["col_0"]
            print(f"Column 0: {col_0} (type: {type(col_0).__name__})")
            
            # Statistical operations (should return NumArray)
            col_sum = matrix_num.sum_axis(1)  # sum along columns
            print(f"Column sums: {col_sum} (type: {type(col_sum).__name__})")
            
        except Exception as e:
            print(f"Error accessing matrix: {e}")
    
    # Test 6: Matrix operations
    if 'matrix_num' in locals():
        print("\n6. Testing matrix operations:")
        try:
            # Matrix power
            matrix_power = matrix_num.power(2)
            print(f"Matrix^2 shape: {matrix_power.shape}")
            
            # Matrix multiplication
            matrix_mult = matrix_num.multiply(matrix_num.transpose())
            print(f"Matrix multiply result shape: {matrix_mult.shape}")
            
        except Exception as e:
            print(f"Error in matrix operations: {e}")
    
    print("\n✅ Matrix Phase 2.1 integration test completed!")

if __name__ == "__main__":
    test_matrix_with_new_arrays()