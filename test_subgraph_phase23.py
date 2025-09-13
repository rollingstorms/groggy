#!/usr/bin/env python3
"""
Test subgraph operations with NumArray/BaseArray integration (Phase 2.3)
"""

import groggy as gg

def test_subgraph_array_operations():
    print("Testing subgraph operations with NumArray/BaseArray integration...")
    
    # Create a simple graph
    g = gg.Graph()
    
    # Add nodes with attributes
    node1 = g.add_node(name="A", value=10, weight=1.0)
    node2 = g.add_node(name="B", value=20, weight=2.0)  
    node3 = g.add_node(name="C", value=30, weight=3.0)
    node4 = g.add_node(name="D", value=40, weight=4.0)
    
    # Add edges 
    g.add_edge(node1, node2, weight=0.5)
    g.add_edge(node2, node3, weight=1.5)
    g.add_edge(node1, node3, weight=2.0)
    g.add_edge(node3, node4, weight=1.0)
    
    print(f"Graph created with {g.node_count()} nodes and {g.edge_count()} edges")
    
    # Create a subgraph with first 3 nodes
    nodes_to_include = [node1, node2, node3]
    subgraph = g.induced_subgraph(nodes_to_include)
    print(f"Subgraph created with {subgraph.node_count()} nodes and {subgraph.edge_count()} edges")
    
    # Test 1: Node and Edge ID arrays
    print("\n1. Testing node and edge ID arrays:")
    try:
        node_ids = subgraph.node_ids
        edge_ids = subgraph.edge_ids
        print(f"Node IDs: {node_ids} (type: {type(node_ids).__name__})")
        print(f"Edge IDs: {edge_ids} (type: {type(edge_ids).__name__})")
    except Exception as e:
        print(f"Error with ID arrays: {e}")
    
    # Test 2: Degree operations - should return NumArray
    print("\n2. Testing degree operations:")
    try:
        # All nodes degree
        degrees = subgraph.degree()
        print(f"All degrees: {degrees} (type: {type(degrees).__name__})")
        
        # Single node degree
        single_degree = subgraph.degree(node1)
        print(f"Node {node1} degree: {single_degree} (type: {type(single_degree).__name__})")
        
        # Multiple nodes degree
        multi_degrees = subgraph.degree([node1, node2])
        print(f"Multi degrees: {multi_degrees} (type: {type(multi_degrees).__name__})")
        
    except Exception as e:
        print(f"Error with degree operations: {e}")
    
    # Test 3: In-degree and out-degree operations
    print("\n3. Testing in-degree and out-degree operations:")
    try:
        in_degrees = subgraph.in_degree()
        out_degrees = subgraph.out_degree()
        print(f"In-degrees: {in_degrees} (type: {type(in_degrees).__name__})")
        print(f"Out-degrees: {out_degrees} (type: {type(out_degrees).__name__})")
        
    except Exception as e:
        print(f"Error with in/out degree operations: {e}")
    
    # Test 4: Neighbors operation - should return NumArray
    print("\n4. Testing neighbors operation:")
    try:
        neighbors = subgraph.neighbors(node1)
        print(f"Neighbors of node {node1}: {neighbors} (type: {type(neighbors).__name__})")
        
    except Exception as e:
        print(f"Error with neighbors operation: {e}")
    
    # Test 5: Matrix conversion - now implemented
    print("\n5. Testing subgraph to matrix conversion:")
    try:
        matrix = subgraph.to_matrix()
        print(f"Subgraph matrix: {matrix}")
        print(f"Matrix shape: {matrix.shape}")
        print(f"Matrix dtype: {matrix.dtype}")
        
        # Test that matrix operations work
        row_0 = matrix[0]
        print(f"Matrix row 0: {row_0} (type: {type(row_0).__name__})")
        
        # Test column access
        column_names = matrix.columns
        if column_names:
            first_col = matrix[column_names[0]]
            print(f"First column: {first_col} (type: {type(first_col).__name__})")
        
    except Exception as e:
        print(f"Error with matrix conversion: {e}")
    
    # Test 6: Statistical operations on arrays
    if 'degrees' in locals():
        print("\n6. Testing statistical operations:")
        try:
            # NumArray should support statistical operations
            if hasattr(degrees, 'sum'):
                degree_sum = degrees.sum()
                print(f"Sum of degrees: {degree_sum}")
            
            if hasattr(degrees, 'mean'):
                degree_mean = degrees.mean()
                print(f"Mean degree: {degree_mean}")
                
        except Exception as e:
            print(f"Error with statistical operations: {e}")
    
    # Test 7: Subgraph properties
    print("\n7. Testing subgraph properties:")
    try:
        density = subgraph.density()
        is_connected = subgraph.is_connected()
        print(f"Subgraph density: {density}")
        print(f"Subgraph is connected: {is_connected}")
        
    except Exception as e:
        print(f"Error with subgraph properties: {e}")
    
    # Test 8: Graph traversal operations
    print("\n8. Testing graph traversal:")
    try:
        # BFS from first node  
        bfs_subgraph = subgraph.bfs(node1, max_depth=2)
        print(f"BFS result: {bfs_subgraph.node_count()} nodes, {bfs_subgraph.edge_count()} edges")
        
        # DFS from first node
        dfs_subgraph = subgraph.dfs(node1, max_depth=2) 
        print(f"DFS result: {dfs_subgraph.node_count()} nodes, {dfs_subgraph.edge_count()} edges")
        
    except Exception as e:
        print(f"Error with traversal operations: {e}")
    
    print("\n✅ Subgraph Phase 2.3 integration test completed!")

if __name__ == "__main__":
    test_subgraph_array_operations()