#!/usr/bin/env python3
"""
Test attribute access in detail.
"""

import groggy as gr

def test_detailed_attributes():
    """Test accessing attributes in detail."""
    print("=== DETAILED ATTRIBUTE ACCESS TEST ===")
    
    g = gr.Graph()
    
    # Add nodes with explicit attributes
    node_0 = g.add_node(name="A", custom_attr="test_value")
    node_1 = g.add_node(name="B", custom_attr="test_value_2") 
    g.add_edge(node_0, node_1, weight=1.0)
    
    print("Before meta-node creation:")
    for node in g.nodes:
        print(f"Node {node.id}: {dict(node.attrs)}")
    
    # Create meta-node 
    subgraph = g.nodes[[node_0, node_1]]
    meta_node = subgraph.collapse(
        node_aggs={"size": "count"},
        mark_entity_type=True,
        entity_type="meta"
    )
    
    print(f"\nMeta-node created: {meta_node.id}")
    
    # Check all nodes after meta-node creation
    print("\nAfter meta-node creation:")
    for node in g.nodes:
        print(f"Node {node.id}: type={type(node).__name__}")
        attrs = dict(node.attrs) if hasattr(node, 'attrs') else {}
        print(f"  attrs: {attrs}")
        
        # Check specific attributes we're looking for
        if 'entity_type' in attrs:
            print(f"  ✓ has entity_type: '{attrs['entity_type']}'")
        else:
            print(f"  ✗ missing entity_type")

if __name__ == "__main__":
    test_detailed_attributes()
