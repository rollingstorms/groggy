#!/usr/bin/env python3
"""
Test Worf's Airtight Entity Type System 🛡️
"""

import groggy

def test_worf_safety_system():
    print("🛡️ Testing WORF'S AIRTIGHT ENTITY TYPE SYSTEM...")
    
    g = groggy.Graph()
    
    # Test 1: Automatic entity_type setting for new nodes
    print(f"\n📋 Test 1: Automatic entity_type setting")
    node1 = g.add_node(name="Alice", age=25)
    node2 = g.add_node(name="Bob", age=30)
    
    all_nodes = g.nodes[list(g.node_ids)]
    entity_type_1 = all_nodes.get_node_attribute(node1, "entity_type")
    entity_type_2 = all_nodes.get_node_attribute(node2, "entity_type")
    
    print(f"  Node {node1} entity_type: {entity_type_1}")
    print(f"  Node {node2} entity_type: {entity_type_2}")
    
    assert entity_type_1 == "base", f"Expected 'base', got {entity_type_1}"
    assert entity_type_2 == "base", f"Expected 'base', got {entity_type_2}"
    print("✅ All new nodes automatically have entity_type='base'")
    
    # Test 2: Safety guard prevents direct entity_type modification
    print(f"\n📋 Test 2: Safety guard against direct entity_type modification")
    try:
        g.set_node_attr(node1, "entity_type", "hacker_type")
        print(f"❌ SECURITY BREACH: Direct entity_type modification succeeded!")
        return False
    except Exception as e:
        print(f"✅ WORF SECURITY: {e}")
        assert "entity_type is immutable" in str(e), f"Wrong error message: {e}"
    
    # Test 3: Meta-node creation through collapse_to_node
    print(f"\n📋 Test 3: Safe meta-node creation via collapse_to_node")
    subgraph = g.nodes[[node1, node2]]
    
    try:
        meta_node = subgraph.collapse_to_node({"person_count": "count", "age": "sum"})
        
        # Verify meta-node has correct entity_type
        all_nodes = g.nodes[list(g.node_ids)]
        meta_entity_type = all_nodes.get_node_attribute(meta_node, "entity_type")
        contained_subgraph = all_nodes.get_node_attribute(meta_node, "contained_subgraph")
        person_count = all_nodes.get_node_attribute(meta_node, "person_count")
        age_sum = all_nodes.get_node_attribute(meta_node, "age")
        
        print(f"✅ Meta-node created: {meta_node}")
        print(f"  entity_type: {meta_entity_type}")
        print(f"  contained_subgraph: {contained_subgraph}")
        print(f"  person_count: {person_count}")
        print(f"  age_sum: {age_sum}")
        
        # Validate meta-node properties
        assert meta_entity_type == "meta", f"Expected entity_type='meta', got {meta_entity_type}"
        assert contained_subgraph is not None, "Meta-node should have contained_subgraph"
        assert person_count == 2, f"Expected person_count=2, got {person_count}"
        assert age_sum == 55, f"Expected age_sum=55, got {age_sum}"
        
        print("✅ Meta-node created safely with all required attributes!")
        
    except Exception as e:
        print(f"❌ Meta-node creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Verify mixed entity types in graph
    print(f"\n📋 Test 4: Mixed entity types in graph")
    all_node_ids = list(g.node_ids)
    print(f"All nodes in graph: {all_node_ids}")
    
    base_count = 0
    meta_count = 0
    
    for node_id in all_node_ids:
        entity_type = all_nodes.get_node_attribute(node_id, "entity_type")
        if entity_type == "base":
            base_count += 1
        elif entity_type == "meta":
            meta_count += 1
        print(f"  Node {node_id}: entity_type={entity_type}")
    
    print(f"Final count - Base nodes: {base_count}, Meta nodes: {meta_count}")
    assert base_count == 2, f"Expected 2 base nodes, got {base_count}"
    assert meta_count == 1, f"Expected 1 meta node, got {meta_count}"
    
    print(f"\n🛡️ ALL WORF SAFETY TESTS PASSED!")
    print(f"✅ Automatic entity_type setting")  
    print(f"✅ Immutable entity_type protection")
    print(f"✅ Safe meta-node creation")
    print(f"✅ Mixed entity type management")
    return True

if __name__ == "__main__":
    success = test_worf_safety_system()
    if success:
        print(f"\n🎉 WORF'S AIRTIGHT ENTITY TYPE SYSTEM: OPERATIONAL! 🛡️⚔️")
    else:
        print(f"\n💥 SECURITY FAILURE!")
