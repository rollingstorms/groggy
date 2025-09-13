#!/usr/bin/env python3
"""
Test adjacency matrix functionality with NumArray/BaseArray integration (Phase 2.2)
"""

import groggy as gg

def test_adjacency_matrices():
    print("Testing adjacency matrix methods with NumArray integration...")
    
    # Create a simple graph
    g = gg.Graph()
    
    # Add nodes
    node1 = g.add_node(name="A", weight=1.0)
    node2 = g.add_node(name="B", weight=2.0)  
    node3 = g.add_node(name="C", weight=3.0)
    
    # Add edges 
    g.add_edge(node1, node2, weight=0.5)
    g.add_edge(node2, node3, weight=1.5)
    g.add_edge(node1, node3, weight=2.0)
    
    print(f"Graph created with {g.node_count()} nodes and {g.edge_count()} edges")
    
    # Test 1: Basic adjacency matrix
    print("\n1. Testing basic adjacency matrix:")
    try:
        adj_matrix = g.adjacency()
        print(f"Adjacency matrix: {adj_matrix}")
        print(f"Matrix shape: {adj_matrix.shape}")
        print(f"Matrix dtype: {adj_matrix.dtype}")
        
        # Test accessing elements
        row_0 = adj_matrix[0] 
        print(f"Row 0: {row_0} (type: {type(row_0).__name__})")
        
        # Check column names
        column_names = adj_matrix.columns
        print(f"Column names: {column_names}")
        
        # Try accessing by actual column name
        if column_names:
            first_col_name = column_names[0]
            col_0 = adj_matrix[first_col_name]
            print(f"Column '{first_col_name}': {col_0} (type: {type(col_0).__name__})")
        else:
            print("No column names found")
        
    except Exception as e:
        print(f"Error with basic adjacency matrix: {e}")
    
    # Test 2: Weighted adjacency matrix
    print("\n2. Testing weighted adjacency matrix:")
    try:
        weighted_adj = g.weighted_adjacency_matrix("weight")
        print(f"Weighted adjacency matrix: {weighted_adj}")
        print(f"Weighted shape: {weighted_adj.shape}")
        print(f"Weighted dtype: {weighted_adj.dtype}")
        
        # Test that weights are preserved
        row_0_weighted = weighted_adj[0]
        print(f"Weighted row 0: {row_0_weighted} (type: {type(row_0_weighted).__name__})")
        
    except Exception as e:
        print(f"Error with weighted adjacency matrix: {e}")
    
    # Test 3: Dense adjacency matrix
    print("\n3. Testing dense adjacency matrix:")
    try:
        dense_adj = g.dense_adjacency_matrix()
        print(f"Dense adjacency matrix: {dense_adj}")
        print(f"Dense shape: {dense_adj.shape}")
        
    except Exception as e:
        print(f"Error with dense adjacency matrix: {e}")
    
    # Test 4: Sparse adjacency matrix  
    print("\n4. Testing sparse adjacency matrix:")
    try:
        sparse_adj = g.sparse_adjacency_matrix()
        print(f"Sparse adjacency matrix: {sparse_adj}")
        print(f"Sparse type: {type(sparse_adj)}")
        
    except Exception as e:
        print(f"Error with sparse adjacency matrix: {e}")
    
    # Test 5: Matrix operations on adjacency matrix
    if 'adj_matrix' in locals():
        print("\n5. Testing matrix operations:")
        try:
            # Statistical operations should return NumArray
            col_sums = adj_matrix.sum_axis(1)  # sum along columns
            print(f"Column sums: {col_sums} (type: {type(col_sums).__name__})")
            
            # Matrix transpose
            adj_transposed = adj_matrix.transpose()
            print(f"Transposed shape: {adj_transposed.shape}")
            
            # Matrix operations (if square)
            if adj_matrix.is_square:
                try:
                    # Matrix power
                    adj_squared = adj_matrix.power(2)
                    print(f"Matrix^2 shape: {adj_squared.shape}")
                    
                except Exception as e:
                    print(f"Matrix power error: {e}")
            
        except Exception as e:
            print(f"Error with matrix operations: {e}")
    
    # Test 6: Conversion to NumPy (if available)
    if 'adj_matrix' in locals():
        print("\n6. Testing NumPy conversion:")
        try:
            import numpy as np
            np_matrix = adj_matrix.to_numpy()
            print(f"NumPy array shape: {np_matrix.shape}")
            print(f"NumPy array dtype: {np_matrix.dtype}")
            print("✅ NumPy conversion successful")
        except ImportError:
            print("NumPy not available - skipping conversion test")
        except Exception as e:
            print(f"Error with NumPy conversion: {e}")
    
    print("\n✅ Adjacency matrix Phase 2.2 integration test completed!")

if __name__ == "__main__":
    test_adjacency_matrices()