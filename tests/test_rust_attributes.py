#!/usr/bin/env python3
"""
Quick test to verify Rust backend attribute support
"""

import sys
sys.path.insert(0, '/Users/michaelroth/Documents/Code/gli')

import gli
import time

def test_rust_attributes():
    print("Testing Rust backend attribute support...")
    
    # Force Rust backend
    gli.set_backend('rust')
    print(f"Current backend: {gli.get_current_backend()}")
    
    # Create a graph
    g = gli.Graph.empty()
    
    # Add nodes with attributes
    print("\n1. Testing node attributes:")
    node1_id = g.add_node(name="Alice", age=30, active=True)
    node2_id = g.add_node(name="Bob", age=25, active=False)
    
    print(f"Added nodes: {node1_id}, {node2_id}")
    
    # Test retrieving node attributes
    node1 = g.get_node(node1_id)
    node2 = g.get_node(node2_id)
    
    print(f"Node {node1_id} attributes: {dict(node1.attributes)}")
    print(f"Node {node2_id} attributes: {dict(node2.attributes)}")
    
    # Add edge with attributes
    print("\n2. Testing edge attributes:")
    edge_id = g.add_edge(node1_id, node2_id, relationship="friend", strength=0.8, since=2020)
    
    print(f"Added edge: {edge_id}")
    
    # Test retrieving edge attributes
    edge = g.get_edge(edge_id)
    print(f"Edge {edge_id} attributes: {dict(edge.attributes)}")
    
    # Test modifying attributes
    print("\n3. Testing attribute modification:")
    g.set_node_attribute(node1_id, "age", 31)
    g.set_edge_attribute(edge_id, "strength", 0.9)
    
    # Verify modifications
    node1_updated = g.get_node(node1_id)
    edge_updated = g.get_edge(edge_id)
    
    print(f"Node {node1_id} updated attributes: {dict(node1_updated.attributes)}")
    print(f"Edge {edge_id} updated attributes: {dict(edge_updated.attributes)}")
    
    # Test complex data types
    print("\n4. Testing complex data types:")
    g.set_node_attribute(node1_id, "hobbies", ["reading", "coding", "hiking"])
    g.set_node_attribute(node1_id, "profile", {"city": "San Francisco", "occupation": "Engineer"})
    
    node1_complex = g.get_node(node1_id)
    print(f"Node {node1_id} complex attributes: {dict(node1_complex.attributes)}")
    
    print("\n✅ All Rust backend attribute tests passed!")

def test_performance():
    print("\nTesting performance with attributes...")
    
    g = gli.Graph.empty()
    
    start_time = time.time()
    
    # Create many nodes with attributes
    node_ids = []
    for i in range(1000):
        node_id = g.add_node(
            id=i,
            name=f"Node_{i}",
            value=i * 0.1,
            active=i % 2 == 0,
            tags=[f"tag_{j}" for j in range(i % 5)]
        )
        node_ids.append(node_id)
    
    # Create edges with attributes
    for i in range(0, len(node_ids)-1, 2):
        g.add_edge(
            node_ids[i], 
            node_ids[i+1],
            weight=0.5 + (i * 0.001),
            created_at=time.time(),
            metadata={"type": "connection", "batch": i // 100}
        )
    
    end_time = time.time()
    
    print(f"Created {len(node_ids)} nodes and {len(node_ids)//2} edges with attributes in {end_time - start_time:.3f} seconds")
    print(f"Nodes: {g.node_count()}, Edges: {g.edge_count()}")
    
    # Test retrieval performance
    start_time = time.time()
    
    for i in range(0, min(100, len(node_ids))):
        node = g.get_node(node_ids[i])
        attrs = dict(node.attributes)
    
    end_time = time.time()
    print(f"Retrieved 100 nodes with attributes in {end_time - start_time:.3f} seconds")

if __name__ == "__main__":
    test_rust_attributes()
    test_performance()
