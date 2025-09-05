#!/usr/bin/env python3
"""
Debug test to understand what's happening with collapse strategy.
"""

import groggy as gr

def debug_collapse():
    print("Debug: Collapse strategy behavior")
    
    # Create a simple graph
    g = gr.Graph()
    g.add_node(name="Node1", value=10)
    g.add_node(name="Node2", value=20)
    g.add_node(name="Node3", value=30)
    
    print(f"Initial graph: {len(g.nodes)} nodes")
    print("Initial node IDs:", [n.id for n in g.nodes])
    
    # Create subgraph with first two nodes
    subgraph = g.nodes[[0, 1]]
    print(f"Subgraph contains node IDs: {[0, 1]}")
    
    # Collapse with collapse strategy
    try:
        meta_node = subgraph.collapse(
            node_aggs={"total": ("sum", "value")},
            node_strategy="collapse"
        )
        
        print(f"Meta-node created with ID: {meta_node.id}")
        print(f"Final graph: {len(g.nodes)} nodes")
        print("Final node IDs:", [n.id for n in g.nodes])
        print("Final node types:", [type(n).__name__ for n in g.nodes])
        
        # Check if original nodes are still there
        for node in g.nodes:
            if hasattr(node, 'has_subgraph'):
                print(f"  Meta-node {node.id}: {node}")
            else:
                print(f"  Regular node {node.id}: {node}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_collapse()
