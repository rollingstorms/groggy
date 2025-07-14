#!/usr/bin/env python3
"""Debug script to check attribute data types in filtering"""

import sys
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python')
from groggy import Graph

def debug_filtering():
    """Debug the filtering issue"""
    print("🔍 Debugging Filtering Issue")
    print("=" * 40)
    
    # Create a simple graph
    g = Graph()
    
    # Add a few test nodes
    g.add_node('node1', age=25, salary=50000, department='engineering')
    g.add_node('node2', age=30, salary=75000, department='sales')
    g.add_node('node3', age=35, salary=100000, department='engineering')
    
    print(f"Created graph with {g.node_count()} nodes")
    
    # Check what type the attributes are when retrieved
    print("\n📊 Checking attribute types:")
    for node_id in ['node1', 'node2', 'node3']:
        node = g.get_node(node_id)
        print(f"  {node_id}:")
        print(f"    age: {node.attributes['age']} (type: {type(node.attributes['age'])})")
        print(f"    salary: {node.attributes['salary']} (type: {type(node.attributes['salary'])})")
        print(f"    department: {node.attributes['department']} (type: {type(node.attributes['department'])})")
    
    # Test different filtering approaches
    print("\n🔍 Testing Different Filtering Approaches:")
    
    # 1. Simple attribute filtering (should work)
    print("1. Simple exact filtering...")
    try:
        result = g.filter_nodes({'department': 'engineering'})
        print(f"   ✅ Found {len(result)} engineering nodes: {result}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 2. Direct numeric comparison (may fail if stored as strings)
    print("2. Direct numeric filtering...")
    try:
        result = g.filter_nodes({'salary': ('>', 60000)})
        print(f"   ✅ Found {len(result)} high-salary nodes: {result}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 3. Try the new multi-criteria method
    print("3. Multi-criteria filtering...")
    try:
        result = g.filter_nodes_multi_criteria(
            exact_matches={'department': 'engineering'},
            numeric_comparisons=[('salary', '>', 50000)]
        )
        print(f"   ✅ Found {len(result)} matching nodes: {result}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print(f"   🔍 Error details: {type(e).__name__}: {e}")
    
    # 4. Check raw attribute values using Rust backend
    print("4. Checking raw Rust backend data...")
    try:
        if g.use_rust:
            # Get all salary values to see their types
            salary_data = g._rust_core.get_all_nodes_attribute('salary')
            print(f"   📊 Raw salary data: {salary_data}")
            for node_id, value in list(salary_data.items())[:3]:
                print(f"      {node_id}: {value} (type: {type(value)})")
    except Exception as e:
        print(f"   ❌ Error accessing raw data: {e}")

if __name__ == "__main__":
    debug_filtering()
