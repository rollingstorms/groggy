#!/usr/bin/env python3
"""
Test both strategies in the hierarchical scenario.
"""

import groggy as gr

def create_test_graph():
    """Create a test graph with multiple connected components of high-degree nodes."""
    g = gr.Graph()
    
    # Create 2 separate clusters, each with high-degree nodes
    nodes = []
    
    # Cluster 1: nodes 0-6 (high degree within cluster)
    cluster1_nodes = []
    for i in range(7):
        node = g.add_node(name=f"Cluster1_Node_{i}", cluster=1)
        nodes.append(node)
        cluster1_nodes.append(node)
    
    # Make cluster 1 densely connected (each node will have degree 6)
    for i in range(len(cluster1_nodes)):
        for j in range(i+1, len(cluster1_nodes)):
            g.add_edge(cluster1_nodes[i], cluster1_nodes[j], weight=1.0)
    
    # Cluster 2: nodes 7-12 (high degree within cluster)  
    cluster2_nodes = []
    for i in range(6):
        node = g.add_node(name=f"Cluster2_Node_{i}", cluster=2)
        nodes.append(node)
        cluster2_nodes.append(node)
    
    # Make cluster 2 densely connected (each node will have degree 5)
    for i in range(len(cluster2_nodes)):
        for j in range(i+1, len(cluster2_nodes)):
            g.add_edge(cluster2_nodes[i], cluster2_nodes[j], weight=1.0)
    
    # Add a few low-degree nodes that won't be in components
    for i in range(3):
        low_degree_node = g.add_node(name=f"Low_Degree_{i}", cluster=0)
        nodes.append(low_degree_node)
        # Connect to just one other node (low degree)
        if len(nodes) > 1:
            g.add_edge(low_degree_node, nodes[0], weight=0.1)
    
    return g

def test_extract_strategy():
    """Test with 'extract' strategy - keeps original nodes."""
    print("=== EXTRACT STRATEGY (keeps originals) ===")
    
    g = create_test_graph()
    print(f"Initial: {len(g.nodes)} nodes")
    
    # Find high degree nodes and create components
    high_degree_node_ids = [node.id for node in g.nodes if node.degree > 4]
    high_degree_subgraph = g.nodes[high_degree_node_ids]
    components = high_degree_subgraph.connected_components()
    
    print(f"Found {len(components)} components of high-degree nodes")
    
    # Collapse each component with 'extract'
    meta_node_ids = []
    for i, component in enumerate(components):
        meta_node = component.collapse(
            node_aggs={"size": "count"},
            edge_strategy='keep_external',
            node_strategy='extract',  # Keep originals
        )
        meta_node_ids.append(meta_node.id)
        print(f"  Component {i+1}: collapsed {component.node_count()} nodes -> meta-node {meta_node.id}")
    
    print(f"After component collapse: {len(g.nodes)} nodes")
    
    # Now collapse meta-nodes into super meta-node
    if len(meta_node_ids) > 1:
        meta_subgraph = g.nodes[meta_node_ids]
        super_meta = meta_subgraph.collapse(
            node_aggs={"total_components": "count"},
            edge_strategy='keep_external',
            node_strategy='extract',  # Keep component meta-nodes
        )
        print(f"Super meta-node: {super_meta.id}")
    
    final_meta_count = sum(1 for node in g.nodes if hasattr(node, 'has_subgraph') and node.has_subgraph)
    print(f"Final: {len(g.nodes)} nodes, {final_meta_count} meta-nodes")
    
    return len(g.nodes), final_meta_count

def test_collapse_strategy():
    """Test with 'collapse' strategy - removes original nodes."""
    print("\n=== COLLAPSE STRATEGY (removes originals) ===")
    
    g = create_test_graph()
    print(f"Initial: {len(g.nodes)} nodes")
    
    # Find high degree nodes and create components
    high_degree_node_ids = [node.id for node in g.nodes if node.degree > 4]
    high_degree_subgraph = g.nodes[high_degree_node_ids]
    components = high_degree_subgraph.connected_components()
    
    print(f"Found {len(components)} components of high-degree nodes")
    
    # Collapse each component with 'collapse'
    meta_node_ids = []
    for i, component in enumerate(components):
        meta_node = component.collapse(
            node_aggs={"size": "count"},
            edge_strategy='keep_external',
            node_strategy='collapse',  # Remove originals
        )
        meta_node_ids.append(meta_node.id)
        print(f"  Component {i+1}: collapsed {component.node_count()} nodes -> meta-node {meta_node.id}")
    
    print(f"After component collapse: {len(g.nodes)} nodes")
    
    # Now collapse meta-nodes into super meta-node
    if len(meta_node_ids) > 1:
        meta_subgraph = g.nodes[meta_node_ids]
        super_meta = meta_subgraph.collapse(
            node_aggs={"total_components": "count"},
            edge_strategy='keep_external',
            node_strategy='collapse',  # Remove component meta-nodes
        )
        print(f"Super meta-node: {super_meta.id}")
    
    final_meta_count = sum(1 for node in g.nodes if hasattr(node, 'has_subgraph') and node.has_subgraph)
    print(f"Final: {len(g.nodes)} nodes, {final_meta_count} meta-nodes")
    
    return len(g.nodes), final_meta_count

if __name__ == "__main__":
    extract_nodes, extract_metas = test_extract_strategy()
    collapse_nodes, collapse_metas = test_collapse_strategy()
    
    print(f"\n=== COMPARISON ===")
    print(f"Extract strategy:  {extract_nodes} total nodes, {extract_metas} meta-nodes")
    print(f"Collapse strategy: {collapse_nodes} total nodes, {collapse_metas} meta-nodes")
    print(f"Node difference: {extract_nodes - collapse_nodes} (extract keeps more)")
