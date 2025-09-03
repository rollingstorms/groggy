#!/usr/bin/env python3
"""
Test Auto-Sliced Table Views with NaN Filtering 📊⚡
"""

import groggy

def test_auto_sliced_tables():
    print("📊⚡ Testing AUTO-SLICED TABLE VIEWS...")
    
    g = groggy.Graph()
    
    # Create base nodes with different attributes
    node1 = g.add_node(name="Alice", age=25, salary=50000)
    node2 = g.add_node(name="Bob", age=30, salary=60000)  
    node3 = g.add_node(name="Charlie", age=35, base_only_attr="special")
    
    # Create some edges
    edge1 = g.add_edge(node1, node2, weight=1.0, relation="friend")
    edge2 = g.add_edge(node2, node3, weight=2.0, relation="colleague")
    
    print(f"Initial nodes: {list(g.node_ids)}")
    print(f"Initial edges: {list(g.edge_ids)}")
    
    # Test 1: Check current table behavior before meta-nodes
    print(f"\n📋 Test 1: Base table before meta-nodes")
    try:
        base_table_before = g.nodes.base.table()
        print(f"✅ Base nodes table created")
        print(f"Base table columns (should include all base attributes):")
        # Check what columns are available - this will help us understand current behavior
        
        meta_table_before = g.nodes.meta.table()
        print(f"✅ Meta nodes table created")
        print(f"Meta table should be empty (no meta-nodes yet)")
        
    except Exception as e:
        print(f"❌ Table creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Create meta-nodes with meta-specific attributes
    print(f"\n📋 Test 2: Create meta-nodes and check table slicing")
    try:
        # Collapse nodes 1,2 into meta-node with meta-specific attributes
        subgraph = g.nodes[[node1, node2]]
        meta_node = subgraph.add_to_graph({
            "avg_age": ("mean", "age"),
            "total_salary": ("sum", "salary"),
            "node_count": "count",
            "meta_only_attr": "count"  # This should only exist on meta-nodes
        })
        meta_node_id = meta_node.node_id
        
        print(f"✅ Meta-node created: {meta_node_id}")
        print(f"All nodes after collapse: {list(g.node_ids)}")
        
        # Now we should have:
        # - Base nodes (node3) with: name, age, base_only_attr, entity_type  
        # - Meta node with: avg_age, total_salary, node_count, meta_only_attr, entity_type
        
    except Exception as e:
        print(f"❌ Meta-node creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Check auto-sliced table behavior
    print(f"\n📋 Test 3: Test auto-sliced table views")
    try:
        print("Testing base nodes table (should exclude meta-only attributes):")
        base_table_after = g.nodes.base.table()
        print(f"✅ Base nodes table created after meta-node")
        
        print("\nTesting meta nodes table (should exclude base-only attributes):")
        meta_table_after = g.nodes.meta.table()
        print(f"✅ Meta nodes table created")
        
        print("\nTesting full nodes table (should include all attributes with NaN where missing):")
        full_table = g.nodes.table()
        print(f"✅ Full nodes table created")
        
        # TODO: Once auto-slicing is implemented, these should show different column sets
        
    except Exception as e:
        print(f"❌ Auto-sliced table test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Test the same for edges if meta-edges were created
    print(f"\n📋 Test 4: Test auto-sliced edge tables")
    try:
        print("Testing base edges table:")
        base_edges_table = g.edges.base.table()
        print(f"✅ Base edges table created")
        
        print("\nTesting meta edges table:")
        meta_edges_table = g.edges.meta.table()  
        print(f"✅ Meta edges table created")
        
        print("\nTesting full edges table:")
        full_edges_table = g.edges.table()
        print(f"✅ Full edges table created")
        
    except Exception as e:
        print(f"❌ Edge table test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Test auto_slice parameter (when implemented)
    print(f"\n📋 Test 5: Test auto_slice parameter control")
    try:
        # These should work once the parameter is implemented
        print("Testing auto_slice=True (default, should exclude NaN-only columns):")
        auto_sliced = g.nodes.base.table()  # Default behavior
        print(f"✅ Auto-sliced table created")
        
        print("\nTesting auto_slice=False (should include all columns):")
        # This parameter doesn't exist yet, but this is what we want to implement
        # full_columns = g.nodes.base.table(auto_slice=False)
        print("⚠️ auto_slice parameter not implemented yet")
        
    except Exception as e:
        print(f"❌ Auto-slice parameter test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n📊⚡ AUTO-SLICED TABLE TESTS COMPLETED!")
    print(f"✅ Current table functionality working")
    print(f"⏳ Auto-slicing NaN filtering needs implementation")
    return True

if __name__ == "__main__":
    success = test_auto_sliced_tables()
    if success:
        print(f"\n🎉 AUTO-SLICED TABLE TESTING: BASIC FUNCTIONALITY OK!")
        print(f"📋 NEXT: Implement NaN filtering logic")
    else:
        print(f"\n💥 AUTO-SLICED TABLE TESTING FAILURE!")