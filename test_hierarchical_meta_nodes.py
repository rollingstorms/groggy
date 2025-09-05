#!/usr/bin/env python3
"""
Test hierarchical meta-node creation scenario.
"""

import groggy as gr

def test_hierarchical_collapse():
    """Test the hierarchical meta-node scenario from the user."""
    print("=== Testing Hierarchical Meta-Node Creation ===")
    
    # Create a graph with high-degree nodes
    g = gr.Graph()
    
    # Add nodes - some with high degree, some with low
    nodes = []
    for i in range(20):
        node = g.add_node(name=f"Node_{i}", group=i % 3)
        nodes.append(node)
    
    # Create edges to make some nodes have degree > 4
    # Connect nodes 0-4 to many others (high degree)
    for i in range(5):  # First 5 nodes will have high degree
        for j in range(5, 15):  # Connect to nodes 5-14
            g.add_edge(nodes[i], nodes[j], weight=0.5)
    
    # Add some additional edges for the high-degree nodes
    for i in range(5):
        for j in range(i+1, 5):
            g.add_edge(nodes[i], nodes[j], weight=0.8)
    
    print(f"Initial graph: {len(g.nodes)} nodes, {len(g.edges)} edges")
    
    # Check degrees
    degrees = {}
    for node in g.nodes:
        degrees[node.id] = node.degree
    
    print("Node degrees:")
    for node_id, degree in sorted(degrees.items()):
        print(f"  Node {node_id}: degree {degree}")
    
    high_degree_nodes = [node_id for node_id, degree in degrees.items() if degree > 4]
    print(f"High degree nodes (>4): {high_degree_nodes}")
    
    # Step 1: Get connected components of high-degree nodes
    try:
        # This should create a subgraph of nodes with degree > 4
        high_degree_subgraph = g.nodes[high_degree_nodes]
        print(f"High degree subgraph: {high_degree_subgraph.node_count()} nodes")
        
        # Get connected components of the high-degree subgraph
        components = high_degree_subgraph.connected_components()
        print(f"Connected components found: {len(components)}")
        
        # Collapse each component
        meta_nodes_created = []
        for i, component in enumerate(components):
            print(f"Collapsing component {i+1}: {component.node_count()} nodes")
            try:
                meta_node = component.collapse(
                    node_aggs={"size": "count", "avg_group": ("mean", "group")},
                    edge_strategy='keep_external',
                    node_strategy='extract',  # Keep original nodes
                )
                meta_nodes_created.append(meta_node)
                print(f"  Created meta-node {meta_node.id}")
            except Exception as e:
                print(f"  Error collapsing component {i+1}: {e}")
        
        print(f"After first collapse: {len(g.nodes)} nodes, {len(g.edges)} edges")
        print(f"Meta-nodes created in first step: {len(meta_nodes_created)}")
        
        # Step 2: Get all meta-nodes and collapse them into one
        print("\n=== Step 2: Collapsing all meta-nodes ===")
        
        # Find all meta-nodes
        meta_nodes = []
        for node in g.nodes:
            if hasattr(node, 'has_subgraph') and node.has_subgraph:
                meta_nodes.append(node.id)
        
        print(f"Found {len(meta_nodes)} meta-nodes: {meta_nodes}")
        
        if len(meta_nodes) > 1:
            # Create subgraph of all meta-nodes
            meta_subgraph = g.nodes[meta_nodes]
            print(f"Meta-subgraph: {meta_subgraph.node_count()} nodes")
            
            try:
                # Collapse all meta-nodes into one
                super_meta_node = meta_subgraph.collapse(
                    node_aggs={"total_components": "count"},
                    edge_strategy='keep_external',
                    node_strategy='extract',  # Keep the individual meta-nodes
                )
                print(f"Created super meta-node: {super_meta_node.id}")
                
                # Final count
                final_meta_count = 0
                for node in g.nodes:
                    if hasattr(node, 'has_subgraph') and node.has_subgraph:
                        final_meta_count += 1
                
                print(f"Final result: {len(g.nodes)} total nodes, {final_meta_count} meta-nodes")
                
            except Exception as e:
                print(f"Error in second collapse: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Only one meta-node found, no second collapse needed")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_hierarchical_collapse()
