#!/usr/bin/env python3
"""Test direct meta-node return from collapse"""

import sys
sys.path.append('.')
import groggy as gr

def test_direct_meta_node():
    print("=== Direct Meta-Node Test ===")
    
    # Create simple graph
    g = gr.Graph(directed=False)
    g.add_node(name="A")
    g.add_node(name="B") 
    g.add_edge(0, 1, weight=1.0)
    
    # Create subgraph
    subgraph = g.nodes[[0, 1]]
    print(f"Original subgraph: {subgraph}")
    
    # Collapse and immediately test the returned meta-node (without retrieving from graph)
    meta_node = subgraph.collapse(
        node_aggs={"size": "count"},
        allow_missing_attributes=True
    )
    
    print(f"\nDirect meta-node from collapse:")
    print(f"  ID: {meta_node.node_id}")
    print(f"  has_subgraph: {meta_node.has_subgraph}")
    print(f"  subgraph_id: {meta_node.subgraph_id}")
    print(f"  subgraph: {meta_node.subgraph}")
    
    return meta_node.subgraph is not None

if __name__ == "__main__":
    success = test_direct_meta_node()
    print(f"\n{'✅' if success else '❌'} Direct meta-node subgraph property: {'working' if success else 'broken'}")