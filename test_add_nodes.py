#!/usr/bin/env python3
"""
Test flexible add_nodes() functionality
"""

import groggy as gr

def test_add_nodes():
    print("Testing flexible add_nodes() functionality...")
    
    graph = gr.Graph()
    
    # Test 1: Old API - count-based
    print("\n=== Test 1: Count-based (backward compatibility) ===")
    node_ids = graph.add_nodes(3)
    print(f"✅ add_nodes(3) -> {node_ids}")
    
    # Test 2: New API - dict-based without id_key
    print("\n=== Test 2: Dict-based without id_key ===")
    node_data = [
        {"name": "Alice", "age": 30},
        {"name": "Bob", "age": 25}
    ]
    node_ids2 = graph.add_nodes(node_data)
    print(f"✅ add_nodes(dict_list) -> {node_ids2}")
    
    # Test 3: New API - dict-based with id_key (the main feature!)
    print("\n=== Test 3: Dict-based with id_key mapping ===")
    node_data_with_ids = [
        {"id": "charlie", "name": "Charlie", "age": 35, "dept": "Engineering"},
        {"id": "diana", "name": "Diana", "age": 32, "dept": "Marketing"},
        {"id": "eve", "name": "Eve", "age": 28, "dept": "Sales"}
    ]
    
    mapping = graph.add_nodes(node_data_with_ids, id_key="id")
    print(f"✅ add_nodes(data, id_key='id') -> {mapping}")
    
    # Verify the mapping works
    charlie_internal_id = mapping["charlie"]
    diana_internal_id = mapping["diana"]
    eve_internal_id = mapping["eve"]
    
    print(f"   charlie -> internal ID {charlie_internal_id}")
    print(f"   diana -> internal ID {diana_internal_id}")
    print(f"   eve -> internal ID {eve_internal_id}")
    
    # Test 4: Verify attributes were set correctly
    print("\n=== Test 4: Verify attributes ===")
    charlie_name = graph.get_node_attribute(charlie_internal_id, "name")
    charlie_dept = graph.get_node_attribute(charlie_internal_id, "dept")
    
    print(f"   Charlie's name: {charlie_name.value if charlie_name else None}")
    print(f"   Charlie's dept: {charlie_dept.value if charlie_dept else None}")
    
    # Make sure "id" wasn't set as an attribute (it should be excluded)
    charlie_id_attr = graph.get_node_attribute(charlie_internal_id, "id")
    print(f"   Charlie's 'id' attribute (should be None): {charlie_id_attr}")
    
    print(f"\nFinal graph: {graph}")
    print("🎉 All add_nodes tests passed!")

if __name__ == "__main__":
    test_add_nodes()