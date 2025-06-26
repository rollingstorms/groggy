#!/usr/bin/env python3
"""
Test the EdgeView iterator functionality
"""

import sys
sys.path.insert(0, '/Users/michaelroth/Documents/Code/gli')

import gli

def test_edge_iteration():
    print("Testing edge iteration...")
    
    gli.set_backend('rust')
    g = gli.Graph.empty()
    
    # Add some nodes and edges
    alice_id = g.add_node(name="Alice", age=30)
    bob_id = g.add_node(name="Bob", age=25)
    charlie_id = g.add_node(name="Charlie", age=35)
    
    edge1_id = g.add_edge(alice_id, bob_id, relationship="friend", strength=0.8)
    edge2_id = g.add_edge(bob_id, charlie_id, relationship="colleague", strength=0.6)
    edge3_id = g.add_edge(alice_id, charlie_id, relationship="family", strength=0.9)
    
    print(f"Added edges: {edge1_id}, {edge2_id}, {edge3_id}")
    
    # Test edge iteration
    print("\nIterating over g.edges:")
    for edge_id in g.edges:
        print(f"  Edge ID: {edge_id} (type: {type(edge_id)})")
        edge = g.get_edge(edge_id)
        print(f"  Edge: {edge.source} -> {edge.target}, attrs: {dict(edge.attributes)}")
    
    print(f"\nTotal edges: {len(g.edges)}")

if __name__ == "__main__":
    test_edge_iteration()
