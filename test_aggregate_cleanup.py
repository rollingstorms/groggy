#!/usr/bin/env python3
"""
Test the new unified aggregate() method
"""

import groggy as gr

def test_unified_aggregate():
    """Test the new unified aggregate method"""
    
    print("🔢 === TESTING UNIFIED AGGREGATE METHOD ===")
    
    # Create a test graph with some numeric data
    g = gr.Graph()
    
    # Add nodes with salary data
    employees = [
        {"id": "alice", "salary": 120000, "years": 5},
        {"id": "bob", "salary": 140000, "years": 8},
        {"id": "carol", "salary": 95000, "years": 3},
        {"id": "david", "salary": 180000, "years": 12},
        {"id": "eve", "salary": 100000, "years": 2},
    ]
    
    employee_map = g.add_nodes(employees, uid_key="id")
    print(f"✅ Created {len(g.nodes)} employees")
    
    # Add some edges with weights
    g.add_edge(0, 1, weight=0.9, relationship="reports_to")
    g.add_edge(1, 2, weight=0.7, relationship="collaborates")
    g.add_edge(2, 3, weight=0.8, relationship="reports_to")
    g.add_edge(3, 4, weight=0.6, relationship="mentors")
    
    print(f"✅ Created {len(g.edges)} edges")
    
    # Test 1: All nodes aggregation (replaces aggregate_node_attribute)
    print(f"\n🧮 === Test 1: All Nodes Aggregation ===")
    try:
        result = g.aggregate("salary", "mean")
        print(f"✅ All salary mean: {result}")
        print(f"   Type: {type(result)}")
        
        result2 = g.aggregate("salary", "sum") 
        print(f"✅ All salary sum: {result2}")
        
    except Exception as e:
        print(f"❌ All nodes aggregation error: {e}")
    
    # Test 2: Custom node list aggregation (replaces aggregate_nodes)
    print(f"\n📊 === Test 2: Custom Node List Aggregation ===")
    try:
        # Test with specific node IDs 
        subset_nodes = [0, 1, 2]  # First 3 employees
        result = g.aggregate("salary", "sum", target="nodes", node_ids=subset_nodes)
        print(f"✅ Subset salary stats: {result}")
        print(f"   Type: {type(result)}")
        
        if hasattr(result, 'items') or isinstance(result, dict):
            # If it's a dict, show the contents
            print(f"   Contents: {result}")
    except Exception as e:
        print(f"❌ Custom node list aggregation error: {e}")
    
    # Test 3: Edge aggregation (replaces aggregate_edge_attribute)
    print(f"\n🔗 === Test 3: Edge Aggregation ===")
    try:
        result = g.aggregate("weight", "mean", target="edges")
        print(f"✅ Edge weight mean: {result}")
        print(f"   Type: {type(result)}")
        
        result2 = g.aggregate("weight", "sum", target="edges")
        print(f"✅ Edge weight sum: {result2}")
        
    except Exception as e:
        print(f"❌ Edge aggregation error: {e}")
    
    # Test 4: Error handling - invalid target
    print(f"\n❌ === Test 4: Error Handling ===")
    try:
        result = g.aggregate("salary", "mean", target="invalid")
        print(f"❌ Should have failed but got: {result}")
    except Exception as e:
        print(f"✅ Correctly caught error for invalid target: {e}")
    
    # Test 5: Check that old methods are gone
    print(f"\n🧹 === Test 5: Old Methods Removed ===")
    
    old_methods = ['aggregate_node_attribute', 'aggregate_edge_attribute', 'aggregate_nodes']
    
    for method in old_methods:
        if hasattr(g, method):
            print(f"❌ {method} still exists (should be removed)")
        else:
            print(f"✅ {method} correctly removed")
    
    # Check that new method exists
    if hasattr(g, 'aggregate'):
        print(f"✅ New unified 'aggregate' method exists")
    else:
        print(f"❌ New 'aggregate' method missing")
    
    print(f"\n🎉 Unified aggregate method testing complete!")

if __name__ == "__main__":
    test_unified_aggregate()