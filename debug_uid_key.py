#!/usr/bin/env python3
"""
Debug the uid_key issue
"""

import groggy as gr

def debug_uid_key():
    """Debug why uid_key isn't working"""
    
    print("=== Debugging uid_key Issue ===")
    
    # Create a simple test
    g = gr.Graph()
    
    # Add nodes with string IDs
    employee_data = [
        {"id": "alice", "name": "Alice Chen"},
        {"id": "bob", "name": "Bob Smith"}
    ]
    
    # Create nodes and get mapping
    node_mapping = g.add_nodes(employee_data, id_key="id")
    print(f"Node mapping: {node_mapping}")
    
    # Check what attributes each node actually has
    print(f"\n=== Node Attributes ===")
    for node_id in g.nodes:
        print(f"Node {node_id}:")
        
        # Check individual attributes
        id_attr = g.get_node_attribute(node_id, "id")
        name_attr = g.get_node_attribute(node_id, "name")
        
        print(f"  id attribute: {id_attr.value if id_attr else 'None'}")
        print(f"  name attribute: {name_attr.value if name_attr else 'None'}")
    
    # Check g.attributes access
    print(f"\n=== Column Attributes ===")
    print(f"Available attribute keys: {g.attributes.keys()}")
    
    if "id" in g.attributes:
        ids = g.attributes["id"]
        print(f"ID column: {ids}")
    else:
        print("No 'id' attribute found in g.attributes!")
    
    if "name" in g.attributes:
        names = g.attributes["name"]
        print(f"Name column: {names}")
    
    # Try manual resolution like the resolve_string_id_to_node method
    print(f"\n=== Manual Resolution Test ===")
    target_id = "alice"
    uid_key = "id"
    
    for node_id in g.nodes:
        id_attr = g.get_node_attribute(node_id, uid_key)
        if id_attr:
            print(f"Node {node_id} has {uid_key}='{id_attr.value}' (type: {type(id_attr.value)})")
            if str(id_attr.value) == target_id:
                print(f"  -> MATCH! Node {node_id} has {uid_key}='{target_id}'")
        else:
            print(f"Node {node_id} has no '{uid_key}' attribute")

if __name__ == "__main__":
    debug_uid_key()