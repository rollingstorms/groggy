#!/usr/bin/env python3
"""
Test Meta-Node Attributes Fix 🔍⚡
"""

import groggy

def test_meta_node_attributes_fix():
    print("🔍⚡ Testing META-NODE ATTRIBUTES ACCESS...")
    
    g = groggy.Graph()
    
    # Create nodes with attributes
    node1 = g.add_node(name="Alice", age=25, salary=50000)
    node2 = g.add_node(name="Bob", age=30, salary=60000)
    
    print(f"Created nodes: {node1}, {node2}")
    
    # Create subgraph
    subgraph = g.nodes[[node1, node2]]
    
    # Test the enhanced aggregation syntax
    meta_node = subgraph.add_to_graph({
        "salary": "sum",
        "avg_age": ("mean", "age")
    })
    
    print(f"✅ Meta-node created: {meta_node.node_id}")
    
    # Test 1: Check if avg_age shows up in meta_node.attributes()
    print(f"\n📋 Test 1: Check meta_node.attributes() method")
    try:
        attributes = meta_node.attributes()
        print(f"Meta-node attributes: {attributes}")
        
        # Check for expected attributes
        if "avg_age" in attributes:
            print(f"✅ Found avg_age: {attributes['avg_age']}")
        else:
            print(f"❌ avg_age NOT found in attributes!")
            
        if "salary" in attributes:
            print(f"✅ Found salary: {attributes['salary']}")
        else:
            print(f"❌ salary NOT found in attributes!")
            
    except Exception as e:
        print(f"❌ Error getting attributes: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Check table access (should also work)
    print(f"\n📋 Test 2: Check table access")
    try:
        all_nodes = g.nodes[list(g.node_ids)]
        print(f"Available node IDs: {list(g.node_ids)}")
        
        meta_node_id = meta_node.node_id
        print(f"Meta-node ID: {meta_node_id}")
        
        # Check if we can get attributes via table access
        try:
            avg_age_via_table = all_nodes.get_node_attribute(meta_node_id, "avg_age")
            print(f"✅ avg_age via table: {avg_age_via_table}")
        except Exception as e:
            print(f"❌ avg_age via table failed: {e}")
            
        try:
            salary_via_table = all_nodes.get_node_attribute(meta_node_id, "salary")
            print(f"✅ salary via table: {salary_via_table}")
        except Exception as e:
            print(f"❌ salary via table failed: {e}")
            
    except Exception as e:
        print(f"❌ Error with table access: {e}")
    
    print(f"\n🔍⚡ ATTRIBUTES FIX TEST COMPLETED!")
    return True

if __name__ == "__main__":
    success = test_meta_node_attributes_fix()
    if success:
        print(f"\n🎉 ATTRIBUTES FIX: SUCCESS! 🔍⚡")
    else:
        print(f"\n💥 ATTRIBUTES FIX FAILURE!")