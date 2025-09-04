#!/usr/bin/env python3
"""Debug meta-node attributes"""

import sys
sys.path.append('.')
import groggy as gr

def test_debug_attrs():
    print("=== Meta-Node Attribute Debug ===")
    
    # Create simple graph
    g = gr.Graph(directed=False)
    g.add_node(name="A")
    g.add_node(name="B") 
    g.add_edge(0, 1, weight=1.0)
    
    # Create subgraph
    subgraph = g.nodes[[0, 1]]
    
    # Collapse and check attributes
    meta_node = subgraph.collapse(
        node_aggs={"size": "count"},
        allow_missing_attributes=True
    )
    
    print(f"Meta-node ID: {meta_node.node_id}")
    
    # Check all attributes on the meta-node
    try:
        all_attrs = g.node_attrs(meta_node.node_id)
        print(f"All attributes on meta-node: {all_attrs}")
    except Exception as e:
        print(f"Error getting all attributes: {e}")
    
    # Test specific attributes
    test_attrs = ['entity_type', 'contains_subgraph', 'contained_subgraph', 'size']
    for attr in test_attrs:
        try:
            value = g.get_node_attr(meta_node.node_id, attr)
            print(f"  {attr}: {value} (type: {type(value)})")
        except Exception as e:
            print(f"  {attr}: Error - {e}")
    
    return meta_node

if __name__ == "__main__":
    test_debug_attrs()