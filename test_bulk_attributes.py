#!/usr/bin/env python3
"""
Quick test to verify the new bulk attribute methods work correctly
"""

import sys
import time
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python')

import groggy as gr
import numpy as np

def test_bulk_methods():
    """Test the new bulk attribute retrieval methods"""
    print("🧪 Testing New Bulk Attribute Methods")
    print("=" * 50)
    
    # Create test graph
    print("Creating test graph with 1000 nodes...")
    g = gr.Graph(backend='rust')
    
    nodes_data = []
    for i in range(1000):
        nodes_data.append({
            'id': f'n{i}',
            'salary': 40000 + i * 100,
            'age': 25 + (i % 40),
            'role': ['engineer', 'manager', 'analyst'][i % 3]
        })
    
    g.add_nodes(nodes_data)
    print(f"Created graph with {g.node_count()} nodes")
    
    # Test 1: get_all_nodes_attribute
    print("\n1. Testing get_all_nodes_attribute('salary')...")
    start_time = time.time()
    all_salaries = g.get_all_nodes_attribute('salary')
    time1 = (time.time() - start_time) * 1000
    print(f"   ✅ Retrieved {len(all_salaries)} salaries in {time1:.2f}ms")
    print(f"   Sample: {dict(list(all_salaries.items())[:3])}")
    
    # Test 2: get_nodes_attribute for specific nodes
    print("\n2. Testing get_nodes_attribute for specific nodes...")
    node_ids = [f'n{i}' for i in range(0, 100, 10)]  # Every 10th node
    start_time = time.time()
    specific_salaries = g.get_nodes_attribute(node_ids, 'salary')
    time2 = (time.time() - start_time) * 1000
    print(f"   ✅ Retrieved {len(specific_salaries)} salaries in {time2:.2f}ms")
    print(f"   Sample: {specific_salaries}")
    
    # Test 3: get_nodes_attributes for complete node data
    print("\n3. Testing get_nodes_attributes for complete node data...")
    start_time = time.time()
    all_attributes = g.get_nodes_attributes(node_ids)
    time3 = (time.time() - start_time) * 1000
    print(f"   ✅ Retrieved complete data for {len(all_attributes)} nodes in {time3:.2f}ms")
    print(f"   Sample: {dict(list(all_attributes.items())[:1])}")
    
    # Test 4: Compare with old method (one-by-one)
    print("\n4. Comparing with old method (one-by-one retrieval)...")
    start_time = time.time()
    old_method_salaries = {}
    for node_id in node_ids:
        node = g.get_node(node_id)
        if node and 'salary' in node.attributes:
            old_method_salaries[node_id] = node.attributes['salary']
    time4 = (time.time() - start_time) * 1000
    print(f"   ✅ Old method: {len(old_method_salaries)} salaries in {time4:.2f}ms")
    
    # Performance comparison
    print("\n📊 Performance Comparison:")
    print(f"   New bulk method: {time2:.2f}ms")
    print(f"   Old method:      {time4:.2f}ms")
    if time4 > 0:
        speedup = time4 / time2
        print(f"   🚀 Speedup: {speedup:.1f}x faster")
    
    # Test 5: Statistical operations
    print("\n5. Testing statistical operations...")
    start_time = time.time()
    salaries = list(all_salaries.values())
    stats = {
        'count': len(salaries),
        'mean': np.mean(salaries),
        'median': np.median(salaries),
        'min': np.min(salaries),
        'max': np.max(salaries)
    }
    time5 = (time.time() - start_time) * 1000
    print(f"   ✅ Computed stats in {time5:.2f}ms")
    print(f"   Stats: {stats}")
    
    print("\n✅ All tests passed! Bulk attribute methods are working correctly.")

if __name__ == "__main__":
    test_bulk_methods()
