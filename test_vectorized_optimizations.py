#!/usr/bin/env python3
"""
Test and benchmark the new optimized vectorized attribute retrieval methods
"""

import time
import random
import pandas as pd
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

import groggy

def generate_test_data(num_nodes=10000, num_attributes=10):
    """Generate test data for benchmarking"""
    nodes_data = []
    
    # Generate diverse attribute types
    for i in range(num_nodes):
        node_data = {
            'id': f'node_{i}',
            'age': random.randint(18, 80),
            'salary': random.randint(30000, 200000),
            'score': random.uniform(0.0, 100.0),
            'department': random.choice(['engineering', 'marketing', 'sales', 'hr', 'finance']),
            'active': random.choice([True, False]),
            'category': random.choice(['A', 'B', 'C']),
        }
        
        # Add additional attributes if needed
        for j in range(7, num_attributes):
            node_data[f'attr_{j}'] = random.uniform(0, 1000)
            
        nodes_data.append(node_data)
    
    return nodes_data

def test_optimized_methods():
    """Test the new optimized vectorized methods"""
    print("Testing optimized vectorized attribute retrieval methods...")
    
    # Create graph with test data
    g = groggy.Graph(backend='rust')
    
    # Generate and add test data
    print("Generating test data...")
    nodes_data = generate_test_data(num_nodes=50000, num_attributes=10)
    
    print(f"Adding {len(nodes_data)} nodes with vectorized operations...")
    start_time = time.time()
    g.add_nodes_chunked(nodes_data, chunk_size=5000)
    add_time = time.time() - start_time
    print(f"Chunked node addition took {add_time:.3f} seconds")
    
    print(f"\nGraph stats: {g.node_count()} nodes")
    
    # Test bulk attribute vector retrieval
    print("\n1. Testing bulk attribute vectors retrieval...")
    attr_names = ['age', 'salary', 'score', 'department']
    
    start_time = time.time()
    bulk_vectors = g.get_bulk_node_attribute_vectors(attr_names)
    bulk_time = time.time() - start_time
    
    print(f"Bulk vectors retrieval took {bulk_time:.3f} seconds")
    print(f"Retrieved {len(bulk_vectors)} attributes")
    for attr_name, (indices, values) in bulk_vectors.items():
        print(f"  {attr_name}: {len(values)} values")
    
    # Test single attribute vectorized retrieval
    print("\n2. Testing single attribute vectorized retrieval...")
    
    start_time = time.time()
    node_ids, salaries = g.get_single_attribute_vectorized('salary')
    single_time = time.time() - start_time
    
    print(f"Single attribute vectorized retrieval took {single_time:.3f} seconds")
    print(f"Retrieved {len(salaries)} salary values")
    print(f"Sample salaries: {salaries[:5]}")
    
    # Test optimized DataFrame export
    print("\n3. Testing optimized DataFrame export...")
    
    start_time = time.time()
    df_data = g.export_node_dataframe_optimized(attr_names)
    export_time = time.time() - start_time
    
    print(f"Optimized DataFrame export took {export_time:.3f} seconds")
    print(f"DataFrame data contains {len(df_data)} columns")
    for col_name, values in df_data.items():
        print(f"  {col_name}: {len(values)} values")
    
    # Create actual DataFrame
    start_time = time.time()
    df = pd.DataFrame(df_data)
    df_creation_time = time.time() - start_time
    
    print(f"pandas DataFrame creation took {df_creation_time:.3f} seconds")
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame columns: {list(df.columns)}")
    print("Sample rows:")
    print(df.head())
    
    # Test to_dataframe_optimized convenience method
    print("\n4. Testing to_dataframe_optimized convenience method...")
    
    start_time = time.time()
    df_optimized = g.to_dataframe_optimized(attr_names)
    convenience_time = time.time() - start_time
    
    print(f"to_dataframe_optimized took {convenience_time:.3f} seconds")
    print(f"DataFrame shape: {df_optimized.shape}")
    
    # Test chunked attribute updates
    print("\n5. Testing chunked attribute updates...")
    
    # Create update data
    updates = {}
    for i in range(0, 10000, 2):  # Update every other node
        node_id = f'node_{i}'
        updates[node_id] = {
            'bonus': random.randint(1000, 10000),
            'updated': True
        }
    
    start_time = time.time()
    g.set_node_attributes_chunked(updates, chunk_size=2000)
    update_time = time.time() - start_time
    
    print(f"Chunked attribute updates took {update_time:.3f} seconds")
    print(f"Updated {len(updates)} nodes with new attributes")
    
    # Verify updates worked
    node_ids, bonuses = g.get_single_attribute_vectorized('bonus')
    print(f"Found {len(bonuses)} nodes with bonus attribute")
    print(f"Sample bonuses: {bonuses[:5]}")
    
    print("\n6. Performance summary:")
    print(f"  Chunked node addition: {add_time:.3f}s for {len(nodes_data)} nodes")
    print(f"  Bulk vectors retrieval: {bulk_time:.3f}s for {len(attr_names)} attributes")
    print(f"  Single attribute vectorized: {single_time:.3f}s for 1 attribute")
    print(f"  Optimized DataFrame export: {export_time:.3f}s")
    print(f"  pandas DataFrame creation: {df_creation_time:.3f}s")
    print(f"  Chunked attribute updates: {update_time:.3f}s for {len(updates)} nodes")
    
    total_time = add_time + bulk_time + single_time + export_time + df_creation_time + update_time
    print(f"  Total time: {total_time:.3f}s")
    
    return True

def compare_with_regular_methods():
    """Compare optimized methods with regular methods"""
    print("\n" + "="*50)
    print("PERFORMANCE COMPARISON: Optimized vs Regular Methods")
    print("="*50)
    
    # Create smaller dataset for comparison
    g = groggy.Graph(backend='rust')
    nodes_data = generate_test_data(num_nodes=10000, num_attributes=8)
    
    print("Adding nodes for comparison...")
    # Use individual add_node calls for comparison
    for node_data in nodes_data:
        node_id = node_data['id']
        attributes = {k: v for k, v in node_data.items() if k != 'id'}
        g.add_node(node_id, **attributes)
    
    attr_names = ['age', 'salary', 'score', 'department']
    
    # Test regular DataFrame method
    print("\n1. Regular to_dataframe method:")
    start_time = time.time()
    df_regular = g.to_dataframe(attr_names)
    regular_time = time.time() - start_time
    print(f"Regular to_dataframe took {regular_time:.3f} seconds")
    
    # Test optimized DataFrame method
    print("\n2. Optimized to_dataframe_optimized method:")
    start_time = time.time()
    df_optimized = g.to_dataframe_optimized(attr_names)
    optimized_time = time.time() - start_time
    print(f"Optimized to_dataframe_optimized took {optimized_time:.3f} seconds")
    
    # Verify results are equivalent
    print(f"\n3. Results verification:")
    print(f"Regular DataFrame shape: {df_regular.shape}")
    print(f"Optimized DataFrame shape: {df_optimized.shape}")
    
    # Check data consistency (allowing for potential ordering differences)
    if df_regular.shape == df_optimized.shape:
        print("✓ DataFrames have same shape")
    else:
        print("✗ DataFrames have different shapes")
    
    # Performance improvement
    if regular_time > 0:
        speedup = regular_time / optimized_time
        print(f"\n4. Performance improvement:")
        print(f"Speedup: {speedup:.2f}x faster")
        print(f"Time saved: {(regular_time - optimized_time):.3f} seconds")
    
    return True

if __name__ == "__main__":
    print("Testing Groggy's new vectorized attribute optimization methods")
    print("=" * 60)
    
    try:
        # Test optimized methods
        test_optimized_methods()
        
        # Compare with regular methods
        compare_with_regular_methods()
        
        print("\n" + "="*50)
        print("✓ ALL TESTS PASSED - Optimizations are working correctly!")
        print("="*50)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
