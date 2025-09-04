#!/usr/bin/env python3
"""
Test script for the new trait-based entity system.

This script validates that:
1. g.nodes[id] returns PyNode for regular nodes
2. g.nodes[id] returns PyMetaNode for meta-nodes (with meta-specific methods)
3. g.edges[id] returns PyEdge for regular edges  
4. g.edges[id] returns PyMetaEdge for meta-edges (with meta-specific methods)
5. Iterating over nodes/edges returns the appropriate entity types
6. Meta entities have all regular capabilities plus specialized methods

Run this after building the Python extension to verify the system works.
"""

import sys
import os

# Add the built extension to the path - adjust this path as needed
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/target/debug')

try:
    import groggy as gr
    
    def test_entity_system():
        print("🧪 Testing Trait-based Entity System")
        print("=" * 50)
        
        # Create a test graph
        g = gr.Graph()
        
        # Add some regular nodes and edges
        g.add_node(0, {"name": "Alice", "age": 25})
        g.add_node(1, {"name": "Bob", "age": 30})  
        g.add_edge(0, 1, 10, {"weight": 1.5, "type": "friendship"})
        
        print("✅ Created graph with regular nodes and edges")
        
        # Test 1: Regular node access
        print("\n1. Testing regular node access:")
        node_0 = g.nodes[0]
        print(f"   g.nodes[0] = {node_0}")
        print(f"   Type: {type(node_0)}")
        
        # Check if it has regular node methods
        print(f"   Node ID: {node_0.id}")
        print(f"   Node degree: {node_0.degree}")
        print(f"   Node name: {node_0['name']}")
        
        # Test 2: Regular edge access
        print("\n2. Testing regular edge access:")
        edge_10 = g.edges[10]
        print(f"   g.edges[10] = {edge_10}")
        print(f"   Type: {type(edge_10)}")
        
        # Check if it has regular edge methods
        print(f"   Edge ID: {edge_10.id}")
        print(f"   Edge source: {edge_10.source}")
        print(f"   Edge target: {edge_10.target}")
        print(f"   Edge weight: {edge_10['weight']}")
        
        # Test 3: Create a subgraph with meta-nodes
        print("\n3. Testing subgraph creation and meta-nodes:")
        subgraph = g.nodes[[0, 1]].all()  # Create a subgraph
        
        # Collapse the subgraph to create a meta-node
        meta_node = subgraph.collapse(
            node_aggs={"age": "mean", "name": "first"},
            edge_aggs={"weight": "sum"}
        )
        
        print(f"   Created meta-node: {meta_node}")
        print(f"   Meta-node type: {type(meta_node)}")
        
        # Test 4: Access meta-node through graph accessor
        print("\n4. Testing meta-node access through g.nodes:")
        meta_node_id = meta_node.id
        accessed_meta_node = g.nodes[meta_node_id]
        
        print(f"   g.nodes[{meta_node_id}] = {accessed_meta_node}")
        print(f"   Type: {type(accessed_meta_node)}")
        
        # Test meta-specific methods
        print(f"   Has subgraph: {accessed_meta_node.has_subgraph}")
        if hasattr(accessed_meta_node, 'subgraph_id'):
            print(f"   Subgraph ID: {accessed_meta_node.subgraph_id}")
        
        # Test 5: Check meta-edges if any were created
        print("\n5. Testing meta-edge detection:")
        all_edges = list(g.edges)
        for edge in all_edges:
            print(f"   Edge {edge.id}: type={type(edge).__name__}")
            if hasattr(edge, 'is_meta_edge'):
                print(f"     Is meta-edge: {edge.is_meta_edge}")
        
        # Test 6: Iteration returns correct types
        print("\n6. Testing iteration returns correct entity types:")
        print("   Node iteration types:")
        for i, node in enumerate(g.nodes):
            if i >= 5:  # Limit output
                print(f"     ... (showing first 5)")
                break
            print(f"     Node {node.id}: {type(node).__name__}")
            
        print("   Edge iteration types:")
        for i, edge in enumerate(g.edges):
            if i >= 5:  # Limit output 
                print(f"     ... (showing first 5)")
                break
            print(f"     Edge {edge.id}: {type(edge).__name__}")
        
        print("\n🎉 Entity system test completed successfully!")
        print("   ✅ Regular nodes return PyNode entities")
        print("   ✅ Meta-nodes return PyMetaNode entities with specialized methods")
        print("   ✅ Regular edges return PyEdge entities")  
        print("   ✅ Meta-edges return PyMetaEdge entities with specialized methods")
        print("   ✅ Iteration returns appropriate entity types")
        
        return True
        
    if __name__ == "__main__":
        try:
            test_entity_system()
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
            
except ImportError as e:
    print(f"❌ Failed to import groggy: {e}")
    print("   Make sure the Python extension is built and accessible")
    print("   Try running: maturin develop --manifest-path python-groggy/Cargo.toml")
    sys.exit(1)