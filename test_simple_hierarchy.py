#!/usr/bin/env python3
"""
Simple test to understand the actual issue with node_strategy and meta-node subgraphs.
"""

import groggy as gr

def test_simple_hierarchy():
    """Test a simple hierarchy issue."""
    print("=== SIMPLE HIERARCHY TEST ===")
    
    # Create a graph with distinct clusters
    g = gr.Graph()
    
    # Cluster 1: nodes 0-2 (triangle)
    g.add_node(name="A1", cluster=1)  # id=0
    g.add_node(name="A2", cluster=1)  # id=1
    g.add_node(name="A3", cluster=1)  # id=2
    g.add_edge(0, 1, weight=1.0)
    g.add_edge(1, 2, weight=1.0)
    g.add_edge(2, 0, weight=1.0)
    
    # Cluster 2: nodes 3-4 (connected pair)
    g.add_node(name="B1", cluster=2)  # id=3
    g.add_node(name="B2", cluster=2)  # id=4
    g.add_edge(3, 4, weight=1.0)
    
    # Add connection between clusters
    g.add_edge(0, 3, weight=0.5)
    
    print(f"Initial: {len(g.nodes)} nodes, {len(g.edges)} edges")
    for node in g.nodes:
        print(f"  Node {node.id}: degree {node.degree}")
    
    # Get high degree nodes (degree > 1)
    high_degree_nodes = [node.id for node in g.nodes if node.degree > 1]
    print(f"\nHigh degree nodes: {high_degree_nodes}")
    
    subgraph = g.nodes[high_degree_nodes]
    components = subgraph.connected_components()
    print(f"Components: {len(components)}")
    
    # Test each component
    meta_nodes = []
    for i, component in enumerate(components):
        print(f"\nComponent {i}: {component.node_count()} nodes")
        
        # Use EXTRACT first 
        meta_node = component.collapse(
            node_aggs={"size": "count"},
            edge_strategy='keep_external', 
            node_strategy='extract'
        )
        
        print(f"  Meta-node {meta_node.id} created")
        print(f"  has_subgraph: {meta_node.has_subgraph}")
        meta_nodes.append(meta_node)
    
    print(f"\nAfter component collapse: {len(g.nodes)} nodes")
    
    # Now try to find and access the meta-nodes from the graph
    print(f"\n--- Checking meta-nodes in graph ---")
    meta_node_ids = []
    for node in g.nodes:
        if hasattr(node, 'has_subgraph') and node.has_subgraph:
            print(f"Found meta-node: {node.id}")
            meta_node_ids.append(node.id)
            if hasattr(node, 'subgraph'):
                print(f"  subgraph has {node.subgraph.node_count()} nodes")
            else:
                print(f"  no subgraph attribute")
    
    # Test collapsing meta-nodes if we have multiple
    if len(meta_node_ids) > 1:
        print(f"\n--- Collapsing {len(meta_node_ids)} meta-nodes ---")
        meta_subgraph = g.nodes[meta_node_ids] 
        
        super_meta = meta_subgraph.collapse(
            node_aggs={"component_count": "count"},
            edge_strategy='keep_external',
            node_strategy='extract'  # Keep the meta-nodes
        )
        
        print(f"Super meta-node: {super_meta.id}")
        print(f"Final graph: {len(g.nodes)} nodes")
        
        # Check if original meta-nodes still have their subgraphs
        print(f"\n--- Checking original meta-nodes after super collapse ---")
        for node in g.nodes:
            if hasattr(node, 'has_subgraph') and node.has_subgraph:
                print(f"Meta-node {node.id}: has_subgraph={node.has_subgraph}")
                if hasattr(node, 'subgraph'):
                    try:
                        count = node.subgraph.node_count()
                        print(f"  subgraph: {count} nodes")
                    except Exception as e:
                        print(f"  subgraph error: {e}")

if __name__ == "__main__":
    test_simple_hierarchy()
