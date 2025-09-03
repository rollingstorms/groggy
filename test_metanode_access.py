#!/usr/bin/env python3
"""
Test MetaNode Access from Enhanced Aggregation 🔍
"""

import groggy

def test_metanode_access():
    print("🔍 Testing METANODE ACCESS from Enhanced Aggregation...")
    
    g = groggy.Graph()
    
    # Create test data
    node1 = g.add_node(name="Alice", age=25, salary=50000)
    node2 = g.add_node(name="Bob", age=30, salary=60000)
    node3 = g.add_node(name="Charlie", age=35, salary=70000)
    
    subgraph = g.nodes[[node1, node2, node3]]
    
    print(f"Created subgraph with nodes: {[node1, node2, node3]}")
    
    # Test 1: Enhanced syntax should return PyMetaNode
    print(f"\n📋 Test 1: Enhanced syntax MetaNode access")
    try:
        meta_node = subgraph.add_to_graph({
            "avg_age": ("mean", "age"),
            "total_salary": ("sum", "salary"),
            "person_count": ("count", None)
        })
        
        print(f"✅ Meta-node created: {meta_node}")
        print(f"  Type: {type(meta_node)}")
        
        # Test MetaNode properties
        print(f"  node_id: {meta_node.node_id}")
        print(f"  has_subgraph: {meta_node.has_subgraph}")
        print(f"  subgraph_id: {meta_node.subgraph_id}")
        
        # Test MetaNode methods
        attributes = meta_node.attributes()
        print(f"  attributes: {attributes}")
        
        # Test expand method
        expanded = meta_node.expand()
        print(f"  expand(): {expanded}")
        print(f"  expand() type: {type(expanded) if expanded else None}")
        
        assert hasattr(meta_node, 'node_id'), "MetaNode should have node_id property"
        assert hasattr(meta_node, 'has_subgraph'), "MetaNode should have has_subgraph property"
        assert hasattr(meta_node, 'expand'), "MetaNode should have expand method"
        assert hasattr(meta_node, 'attributes'), "MetaNode should have attributes method"
        
        print("✅ MetaNode has all expected properties and methods!")
        
    except Exception as e:
        print(f"❌ Enhanced syntax MetaNode test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Access from graph node accessor should also work
    print(f"\n📋 Test 2: MetaNode access via graph nodes accessor")
    try:
        all_nodes = g.nodes[list(g.node_ids)]
        meta_nodes = []
        
        for node_id in g.node_ids:
            entity_type = all_nodes.get_node_attribute(node_id, "entity_type")
            if entity_type == "meta":
                print(f"  Found meta-node: {node_id}")
                # For now, we can't directly create PyMetaNode from accessor
                # But we can verify the attributes exist
                contained_subgraph = all_nodes.get_node_attribute(node_id, "contained_subgraph")
                avg_age = all_nodes.get_node_attribute(node_id, "avg_age")
                total_salary = all_nodes.get_node_attribute(node_id, "total_salary")
                person_count = all_nodes.get_node_attribute(node_id, "person_count")
                
                print(f"    contained_subgraph: {contained_subgraph}")
                print(f"    avg_age: {avg_age}")
                print(f"    total_salary: {total_salary}")
                print(f"    person_count: {person_count}")
                
                meta_nodes.append(node_id)
                
        assert len(meta_nodes) > 0, "Should have found at least one meta-node"
        print(f"✅ Found {len(meta_nodes)} meta-node(s) via accessor!")
        
    except Exception as e:
        print(f"❌ Accessor MetaNode test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n🔍 ALL METANODE ACCESS TESTS PASSED!")
    print(f"✅ Enhanced syntax returns proper PyMetaNode")
    print(f"✅ MetaNode has all expected properties and methods")
    print(f"✅ Meta-node attributes accessible via graph accessor")
    return True

if __name__ == "__main__":
    success = test_metanode_access()
    if success:
        print(f"\n🎉 METANODE ACCESS: OPERATIONAL! 🔍⚡")
    else:
        print(f"\n💥 METANODE ACCESS FAILURE!")