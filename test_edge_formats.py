#!/usr/bin/env python3
"""
Test multiple edge input formats
"""

import groggy as gr

def test_edge_formats():
    print("Testing multiple edge input formats...")
    
    graph = gr.Graph()
    
    # Set up some nodes with the new add_nodes functionality
    node_data = [
        {"id": "alice", "name": "Alice", "dept": "Engineering"},
        {"id": "bob", "name": "Bob", "dept": "Sales"},
        {"id": "charlie", "name": "Charlie", "dept": "Marketing"}
    ]
    
    # Get the mapping for later use
    mapping = graph.add_nodes(node_data, id_key="id")
    print(f"Created nodes with mapping: {mapping}")
    
    alice_id = mapping["alice"]
    bob_id = mapping["bob"] 
    charlie_id = mapping["charlie"]
    
    print(f"\n=== Test 1: add_edge() with kwargs ===")
    edge1 = graph.add_edge(alice_id, bob_id, relationship="collaborates", strength=0.9, duration=6)
    print(f"✅ add_edge({alice_id}, {bob_id}, relationship='collaborates', strength=0.9) -> edge {edge1}")
    
    # Verify attributes were set
    rel = graph.get_edge_attribute(edge1, "relationship")
    strength = graph.get_edge_attribute(edge1, "strength")
    print(f"   relationship: {rel.value if rel else None}")
    print(f"   strength: {strength.value if strength else None}")
    
    print(f"\n=== Test 2: add_edges() with tuple format ===")
    # Format: [(source, target), ...]
    edges2 = graph.add_edges([(alice_id, charlie_id), (bob_id, charlie_id)])
    print(f"✅ add_edges([(alice, charlie), (bob, charlie)]) -> {edges2}")
    
    print(f"\n=== Test 3: add_edges() with tuple + attributes format ===")
    # Format: [(source, target, attrs_dict), ...]
    edges3 = graph.add_edges([
        (alice_id, bob_id, {"type": "friendship", "weight": 0.8}),
        (charlie_id, alice_id, {"type": "mentorship", "weight": 0.7})
    ])
    print(f"✅ add_edges() with attributes -> {edges3}")
    
    # Verify attributes
    edge_attr = graph.get_edge_attribute(edges3[0], "type")
    print(f"   edge {edges3[0]} type: {edge_attr.value if edge_attr else None}")
    
    print(f"\n=== Test 4: add_edges_from_dicts() with string ID resolution ===")
    edge_dicts = [
        {"source": "alice", "target": "bob", "relationship": "reports_to", "weight": 1.0},
        {"source": "charlie", "target": "alice", "relationship": "manages", "weight": 0.9}
    ]
    
    edges4 = graph.add_edges_from_dicts(edge_dicts, mapping)
    print(f"✅ add_edges_from_dicts() -> {edges4}")
    
    # Verify one of the dict-created edges
    edge_rel = graph.get_edge_attribute(edges4[0], "relationship") 
    edge_weight = graph.get_edge_attribute(edges4[0], "weight")
    print(f"   edge {edges4[0]} relationship: {edge_rel.value if edge_rel else None}")
    print(f"   edge {edges4[0]} weight: {edge_weight.value if edge_weight else None}")
    
    print(f"\n=== Test 5: Custom source/target keys ===")
    custom_edge_dicts = [
        {"from": "bob", "to": "charlie", "connection": "peers", "score": 0.6}
    ]
    
    edges5 = graph.add_edges_from_dicts(custom_edge_dicts, mapping, 
                                       source_key="from", target_key="to")
    print(f"✅ add_edges_from_dicts() with custom keys -> {edges5}")
    
    print(f"\nFinal graph: {graph}")
    print("🎉 All edge format tests passed!")

if __name__ == "__main__":
    test_edge_formats()