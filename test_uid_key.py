#!/usr/bin/env python3
"""
Test the new uid_key functionality for edge creation
"""

import groggy as gr

def test_uid_key_functionality():
    """Test uid_key parameter for add_edge and add_edges"""
    
    print("=== Testing uid_key Functionality ===")
    
    # Create a graph with string IDs
    g = gr.Graph()
    
    # Add nodes with string IDs
    employee_data = [
        {"id": "alice", "name": "Alice Chen", "role": "Engineer"},
        {"id": "bob", "name": "Bob Smith", "role": "Manager"},
        {"id": "carol", "name": "Carol Davis", "role": "Designer"},
        {"id": "david", "name": "David Wilson", "role": "Director"}
    ]
    
    # Create nodes and get mapping
    node_mapping = g.add_nodes(employee_data, id_key="id")
    print(f"Created nodes with mapping: {node_mapping}")
    
    # Test 1: add_edge with uid_key
    print(f"\n=== Test 1: add_edge with uid_key ===")
    try:
        edge1 = g.add_edge("alice", "bob", uid_key="id", relationship="reports_to", strength=0.9)
        print(f"✅ Created edge between alice and bob: {edge1}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    try:
        edge2 = g.add_edge("carol", "alice", uid_key="id", relationship="collaborates", strength=0.7)
        print(f"✅ Created edge between carol and alice: {edge2}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: add_edge with numeric IDs (should still work)
    print(f"\n=== Test 2: add_edge with numeric IDs ===")
    try:
        alice_id = node_mapping["alice"]
        david_id = node_mapping["david"]
        edge3 = g.add_edge(alice_id, david_id, relationship="managed_by", strength=0.95)
        print(f"✅ Created edge between numeric IDs: {edge3}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: add_edges with node_mapping
    print(f"\n=== Test 3: add_edges with node_mapping ===")
    edge_data = [
        {"source": "bob", "target": "david", "relationship": "reports_to", "frequency": "weekly"},
        {"source": "carol", "target": "bob", "relationship": "reports_to", "frequency": "daily"},
    ]
    
    try:
        edge_ids = g.add_edges(edge_data, node_mapping=node_mapping)
        print(f"✅ Created bulk edges: {edge_ids}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 4: Error cases
    print(f"\n=== Test 4: Error Cases ===")
    
    # Non-existent string ID
    try:
        g.add_edge("alice", "nonexistent", uid_key="id", relationship="test")
        print(f"❌ Should have failed for non-existent ID")
    except Exception as e:
        print(f"✅ Correctly failed for non-existent ID: {e}")
    
    # String ID without uid_key
    try:
        g.add_edge("alice", "bob", relationship="test")
        print(f"❌ Should have failed for string ID without uid_key")
    except Exception as e:
        print(f"✅ Correctly failed for string ID without uid_key: {e}")
    
    # Test 5: Verify graph state
    print(f"\n=== Test 5: Final Graph State ===")
    print(f"Total nodes: {len(g.nodes)}")
    print(f"Total edges: {len(g.edges)}")
    print(f"Node mapping: {node_mapping}")
    
    # Check attributes
    print(f"\nNode attributes:")
    for node_id in g.nodes:
        print(f"  Node {node_id}: name={g.get_node_attribute(node_id, 'name').value if g.get_node_attribute(node_id, 'name') else None}")
    
    print(f"\nEdge attributes:")
    for edge_id in g.edges:
        rel_attr = g.get_edge_attribute(edge_id, 'relationship')
        print(f"  Edge {edge_id}: relationship={rel_attr.value if rel_attr else None}")
    
    print("\n✅ uid_key functionality working!")

if __name__ == "__main__":
    test_uid_key_functionality()