#!/usr/bin/env python3
"""Test meta-node subgraph property"""

import sys
sys.path.append('.')
import groggy as gr

def test_meta_node_subgraph_property():
    """Test that meta-node.subgraph returns the original subgraph"""
    print("=== Testing Meta-Node Subgraph Property ===")
    
    # Create test graph
    g = gr.Graph(directed=False)
    
    # Add nodes
    g.add_node(name="A", group="cluster", value=10)
    g.add_node(name="B", group="cluster", value=20) 
    g.add_node(name="C", group="cluster", value=30)
    g.add_node(name="X", group="external", value=100)
    
    # Add edges
    g.add_edge(0, 1, weight=0.9)
    g.add_edge(1, 2, weight=0.8)
    g.add_edge(1, 3, weight=0.6)  # B -> X
    
    print(f"Original graph: {g.node_count()} nodes, {g.edge_count()} edges")
    
    # Create subgraph
    original_subgraph = g.nodes[[0, 1, 2]]
    print(f"Original subgraph: {original_subgraph.node_count()} nodes, {original_subgraph.edge_count()} edges")
    print(f"Original subgraph details:\n{original_subgraph}")
    
    # Collapse subgraph to create meta-node
    meta_node = original_subgraph.collapse(
        node_aggs={"size": "count", "total_value": ("sum", "value")},
        edge_strategy="aggregate",
        allow_missing_attributes=True
    )
    
    print(f"Meta-node created with ID: {meta_node.node_id}")
    print(f"Graph after collapse: {g.node_count()} nodes, {g.edge_count()} edges")
    
    # Test 1: Direct access to subgraph property
    print(f"\n1. meta_node.subgraph: {meta_node.subgraph}")
    
    # Test 2: Try to get meta-node from graph and check its subgraph
    try:
        retrieved_meta_node = g.nodes.get_meta_node(meta_node.node_id)
        print(f"2. Retrieved meta-node: {retrieved_meta_node}")
        if retrieved_meta_node:
            print(f"   Retrieved meta-node.subgraph: {retrieved_meta_node.subgraph}")
        else:
            print("   ❌ get_meta_node returned None")
    except Exception as e:
        print(f"   ❌ Error retrieving meta-node: {e}")
    
    # Test 3: Check if meta-node has the subgraph data in its attributes
    try:
        contains_subgraph_attr = g.get_node_attr(meta_node.node_id, 'contains_subgraph')
        print(f"3. contains_subgraph attribute: {contains_subgraph_attr}")
        print(f"   Type: {type(contains_subgraph_attr)}")
    except Exception as e:
        print(f"   ❌ Error getting contains_subgraph attr: {e}")
    
    # Test 3b: Check if meta-node has subgraph ID
    try:
        subgraph_id = meta_node.subgraph_id
        print(f"3b. meta_node.subgraph_id: {subgraph_id}")
        has_subgraph = meta_node.has_subgraph
        print(f"    meta_node.has_subgraph: {has_subgraph}")
    except Exception as e:
        print(f"   ❌ Error getting subgraph_id: {e}")
    
    # Test 4: Check what attributes the meta-node has
    try:
        print("4. Meta-node attributes:")
        for attr in ['entity_type', 'contains_subgraph', 'size', 'total_value']:
            try:
                value = g.get_node_attr(meta_node.node_id, attr)
                print(f"   {attr}: {value}")
            except:
                print(f"   {attr}: <not found>")
    except Exception as e:
        print(f"   ❌ Error checking attributes: {e}")
    
    return meta_node.subgraph is not None

def main():
    result = test_meta_node_subgraph_property()
    
    print(f"\n{'='*50}")
    if result:
        print("✅ Meta-node subgraph property working correctly")
        return 0
    else:
        print("❌ Meta-node subgraph property is None")
        return 1

if __name__ == "__main__":
    sys.exit(main())