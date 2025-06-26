#!/usr/bin/env python3
"""
Test the NodeView iterator functionality
"""

import sys
sys.path.insert(0, '/Users/michaelroth/Documents/Code/gli')

import gli

def test_node_iteration():
    print("Testing node iteration...")
    
    gli.set_backend('rust')
    g = gli.Graph.empty()
    
    # Add some nodes
    alice_id = g.add_node(name="Alice", age=30)
    bob_id = g.add_node(name="Bob", age=25)
    charlie_id = g.add_node(name="Charlie", age=35)
    
    print(f"Added nodes: {alice_id}, {bob_id}, {charlie_id}")
    
    # Test iteration
    print("\nIterating over g.nodes:")
    for node_id in g.nodes:
        print(f"  Node ID: {node_id} (type: {type(node_id)})")
        node = g.get_node(node_id)
        print(f"  Node: {node.id} - {dict(node.attributes)}")
    
    # Test edge iteration
    g.add_edge(alice_id, bob_id, relationship="friend")
    g.add_edge(bob_id, charlie_id, relationship="colleague")
    
    print("\nGraph structure:")
    for node_id in g.nodes:
        node = g.get_node(node_id)
        neighbors = g.get_neighbors(node_id)
        print(f"{node.id}: {dict(node.attributes)} -> neighbors: {neighbors}")

if __name__ == "__main__":
    test_node_iteration()
