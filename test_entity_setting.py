#!/usr/bin/env python3
"""
Test if entity_type is being set correctly during meta-node creation.
"""

import groggy as gr

def test_entity_type_setting():
    """Test if entity_type attribute is actually set."""
    print("=== TESTING ENTITY_TYPE SETTING ===")
    
    # Simple case
    g = gr.Graph()
    g.add_node(name="A")  # 0
    g.add_node(name="B")  # 1 
    g.add_edge(0, 1, weight=1.0)
    
    print(f"Initial: {len(g.nodes)} nodes")
    
    # Get all nodes and collapse them
    all_nodes = [node.id for node in g.nodes]
    subgraph = g.nodes[all_nodes]
    
    # Create meta-node with explicit mark_entity_type=True
    meta_node = subgraph.collapse(
        node_aggs={"size": "count"},
        edge_strategy='keep_external',
        node_strategy='extract',
        mark_entity_type=True,  # Explicitly enable
        entity_type="meta"      # Explicit entity type
    )
    
    print(f"Meta-node created: {meta_node.id}")
    print(f"Meta-node type: {type(meta_node)}")
    print(f"Meta-node has_subgraph: {meta_node.has_subgraph}")
    
    # Check the meta-node's attributes directly
    print(f"\n--- Meta-node attributes ---")
    if hasattr(meta_node, 'attrs'):
        print(f"Meta-node attrs: {dict(meta_node.attrs)}")
    
    # Access the meta-node through the graph  
    print(f"\n--- Accessing meta-node through graph ---")
    for node in g.nodes:
        print(f"Node {node.id}: type={type(node).__name__}")
        if hasattr(node, 'attrs'):
            print(f"  attrs: {dict(node.attrs)}")
        if node.id == meta_node.id:
            print(f"  This is the meta-node we created!")
    
    # Try to access entity_type using the core Graph API
    print(f"\n--- Direct Graph API check ---")
    # We'll check this by trying to look at all node attributes
    print("All node attributes in graph:")
    for node in g.nodes:
        print(f"Node {node.id} available attributes:", list(node.attrs.keys()) if hasattr(node, 'attrs') else "no attrs")

if __name__ == "__main__":
    test_entity_type_setting()
