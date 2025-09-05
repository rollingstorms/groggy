#!/usr/bin/env python3
"""
Debug the node_strategy issue - check what's actually happening.
"""

import groggy as gr

def debug_node_strategy():
    """Debug what's happening with node_strategy parameter."""
    print("=== DEBUGGING NODE STRATEGY ===")
    
    # Create a simple graph
    g = gr.Graph()
    
    # Add 5 nodes
    for i in range(5):
        g.add_node(name=f"Node_{i}", value=i * 10)
    
    # Connect them in a line
    for i in range(4):
        g.add_edge(i, i+1, weight=1.0)
    
    print(f"Initial graph: {len(g.nodes)} nodes, {len(g.edges)} edges")
    
    # Get all nodes as subgraph
    all_node_ids = [node.id for node in g.nodes]
    subgraph = g.nodes[all_node_ids]
    print(f"Subgraph: {subgraph.node_count()} nodes")
    
    # Test EXTRACT strategy (should keep original nodes)
    print(f"\n--- Testing EXTRACT strategy ---")
    g_extract = gr.Graph()
    for i in range(5):
        g_extract.add_node(name=f"Node_{i}", value=i * 10)
    for i in range(4):
        g_extract.add_edge(i, i+1, weight=1.0)
    
    print(f"Before extract collapse: {len(g_extract.nodes)} nodes")
    extract_subgraph = g_extract.nodes[all_node_ids]
    
    meta_node = extract_subgraph.collapse(
        node_aggs={"size": "count", "total_value": ("sum", "value")},
        edge_strategy='keep_external',
        node_strategy='extract',  # Should keep originals
    )
    
    print(f"After extract collapse: {len(g_extract.nodes)} nodes")
    print(f"Meta-node created: {meta_node.id}")
    print(f"Meta-node has_subgraph: {hasattr(meta_node, 'has_subgraph') and meta_node.has_subgraph}")
    
    # Test COLLAPSE strategy (should remove original nodes)
    print(f"\n--- Testing COLLAPSE strategy ---")
    g_collapse = gr.Graph()
    for i in range(5):
        g_collapse.add_node(name=f"Node_{i}", value=i * 10)
    for i in range(4):
        g_collapse.add_edge(i, i+1, weight=1.0)
    
    print(f"Before collapse collapse: {len(g_collapse.nodes)} nodes")
    collapse_subgraph = g_collapse.nodes[all_node_ids]
    
    meta_node_2 = collapse_subgraph.collapse(
        node_aggs={"size": "count", "total_value": ("sum", "value")},
        edge_strategy='keep_external',
        node_strategy='collapse',  # Should remove originals
    )
    
    print(f"After collapse collapse: {len(g_collapse.nodes)} nodes")
    print(f"Meta-node created: {meta_node_2.id}")
    print(f"Meta-node has_subgraph: {hasattr(meta_node_2, 'has_subgraph') and meta_node_2.has_subgraph}")
    
    # Check what nodes exist in each graph
    print(f"\n--- Node details ---")
    print("Extract graph node IDs:", [node.id for node in g_extract.nodes])
    print("Collapse graph node IDs:", [node.id for node in g_collapse.nodes])
    
    # Check meta-node attributes
    print(f"\n--- Meta-node attributes ---")
    print("Extract meta-node attrs:", dict(meta_node.attrs))
    print("Collapse meta-node attrs:", dict(meta_node_2.attrs))
    
    return len(g_extract.nodes), len(g_collapse.nodes)

if __name__ == "__main__":
    extract_count, collapse_count = debug_node_strategy()
    print(f"\n=== SUMMARY ===")
    print(f"Extract strategy final count: {extract_count} nodes")
    print(f"Collapse strategy final count: {collapse_count} nodes")
    print(f"Expected: extract=6 (5 original + 1 meta), collapse=1 (1 meta only)")
    
    if extract_count == 6 and collapse_count == 1:
        print("✅ Node strategies working correctly")
    else:
        print("❌ Node strategies not working as expected")
