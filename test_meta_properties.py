#!/usr/bin/env python3
"""
Debug the exact issue - are meta-nodes losing their meta-node properties?
"""

import groggy as gr

def debug_meta_node_properties():
    """Debug what happens to meta-node properties."""
    print("=== DEBUGGING META-NODE PROPERTIES ===")
    
    # Simple case with 2 triangles
    g = gr.Graph()
    
    # Triangle 1
    g.add_node(name="A1")  # 0
    g.add_node(name="A2")  # 1 
    g.add_node(name="A3")  # 2
    g.add_edge(0, 1, weight=1.0)
    g.add_edge(1, 2, weight=1.0)
    g.add_edge(2, 0, weight=1.0)
    
    # Triangle 2  
    g.add_node(name="B1")  # 3
    g.add_node(name="B2")  # 4
    g.add_node(name="B3")  # 5
    g.add_edge(3, 4, weight=1.0)
    g.add_edge(4, 5, weight=1.0) 
    g.add_edge(5, 3, weight=1.0)
    
    print(f"Initial: {len(g.nodes)} nodes")
    
    # Create components
    all_nodes = [node.id for node in g.nodes]
    subgraph = g.nodes[all_nodes]
    components = subgraph.connected_components()
    print(f"Components: {len(components)}")
    
    # Collapse each component
    meta_nodes = []
    for i, comp in enumerate(components):
        meta_node = comp.collapse(
            node_aggs={"size": "count"},
            edge_strategy='keep_external',
            node_strategy='extract'
        )
        meta_nodes.append(meta_node)
        print(f"Component {i} -> meta-node {meta_node.id}")
        print(f"  Type: {type(meta_node)}")
        print(f"  has_subgraph: {meta_node.has_subgraph}")
    
    print(f"\nAfter components: {len(g.nodes)} nodes")
    
    # Check what types of objects exist in the graph
    print(f"\nNode types in graph:")
    for node in g.nodes:
        print(f"  Node {node.id}: {type(node)}, has_subgraph attr: {hasattr(node, 'has_subgraph')}")
        if hasattr(node, 'has_subgraph'):
            print(f"    has_subgraph value: {node.has_subgraph}")
    
    # Now collapse the meta-nodes
    meta_node_ids = [mn.id for mn in meta_nodes]
    print(f"\nCollapsing meta-nodes {meta_node_ids}...")
    
    meta_subgraph = g.nodes[meta_node_ids]
    super_meta = meta_subgraph.collapse(
        node_aggs={"total": "count"},
        edge_strategy='keep_external',
        node_strategy='extract'  # Extract should keep them
    )
    
    print(f"Super meta-node: {super_meta.id}")
    print(f"Final graph: {len(g.nodes)} nodes")
    
    # Check node types again
    print(f"\nNode types after super collapse:")
    for node in g.nodes:
        print(f"  Node {node.id}: {type(node)}, has_subgraph attr: {hasattr(node, 'has_subgraph')}")
        if hasattr(node, 'has_subgraph'):
            print(f"    has_subgraph value: {node.has_subgraph}")
            if hasattr(node, 'subgraph') and node.has_subgraph:
                try:
                    print(f"    subgraph nodes: {node.subgraph.node_count()}")
                except Exception as e:
                    print(f"    subgraph access error: {e}")
        
        # Check all attributes
        print(f"    attributes: {list(node.attrs.keys()) if hasattr(node, 'attrs') else 'no attrs'}")

if __name__ == "__main__":
    debug_meta_node_properties()
