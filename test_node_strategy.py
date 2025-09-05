#!/usr/bin/env python3

import groggy

def test_node_strategy():
    """Test the new node_strategy parameter"""
    
    # Create a simple graph
    g = groggy.Graph()
    
    # Add some nodes
    n1 = g.add_node()
    n2 = g.add_node()
    n3 = g.add_node()
    
    # Set node attributes
    g.set_node_attrs({"name": {n1: "Alice", n2: "Bob", n3: "Carol"}})
    g.set_node_attrs({"age": {n1: 30, n2: 25, n3: 35}})
    
    # Add some edges
    e1 = g.add_edge(n1, n2)
    e2 = g.add_edge(n2, n3)
    e3 = g.add_edge(n1, n3)
    
    # Set edge attributes
    g.set_edge_attrs({"relationship": {e1: "friend", e2: "colleague", e3: "friend"}})
    g.set_edge_attrs({"weight": {e1: 0.8, e2: 0.6, e3: 0.9}})
    
    print(f"Original graph: {g.node_count()} nodes, {g.edge_count()} edges")
    print(f"Nodes: {[n1, n2, n3]}")
    
    # Create a subgraph using view() - get all nodes as subgraph
    subgraph = g.view()
    print(f"Subgraph: {subgraph.node_count()} nodes, {subgraph.edge_count()} edges")
    
    # Test Extract strategy (default - should keep original nodes)
    print("\n=== Testing Extract Strategy ===")
    try:
        meta_node_extract = subgraph.collapse(
            node_aggs={"avg_age": ("mean", "age"), "total_people": "count"},
            edge_aggs={"avg_weight": "mean"},
            node_strategy="extract"
        )
        print(f"Extract successful! Meta-node ID: {meta_node_extract.id()}")
        print(f"Graph after extract: {g.node_count()} nodes, {g.edge_count()} edges")
        
        # Check if original nodes still exist
        original_nodes_exist = []
        for node_id in [n1, n2, n3]:
            try:
                attrs = g.get_node_attrs(node_id)
                original_nodes_exist.append(f"Node {node_id}: {attrs}")
            except:
                original_nodes_exist.append(f"Node {node_id}: NOT FOUND")
        
        print("Original nodes after extract:")
        for status in original_nodes_exist:
            print(f"  {status}")
            
    except Exception as e:
        print(f"Extract failed: {e}")
    
    # Create a fresh graph for collapse test
    print("\n=== Testing Collapse Strategy ===")
    g2 = groggy.Graph()
    n1_2 = g2.add_node()
    n2_2 = g2.add_node()
    n3_2 = g2.add_node()
    
    g2.set_node_attrs({"name": {n1_2: "Alice", n2_2: "Bob", n3_2: "Carol"}})
    g2.set_node_attrs({"age": {n1_2: 30, n2_2: 25, n3_2: 35}})
    
    e1_2 = g2.add_edge(n1_2, n2_2)
    e2_2 = g2.add_edge(n2_2, n3_2)
    e3_2 = g2.add_edge(n1_2, n3_2)
    
    g2.set_edge_attrs({"relationship": {e1_2: "friend", e2_2: "colleague", e3_2: "friend"}})
    g2.set_edge_attrs({"weight": {e1_2: 0.8, e2_2: 0.6, e3_2: 0.9}})
    
    subgraph2 = g2.view()
    print(f"Fresh graph: {g2.node_count()} nodes, {g2.edge_count()} edges")
    
    try:
        meta_node_collapse = subgraph2.collapse(
            node_aggs={"avg_age": ("mean", "age"), "total_people": "count"},
            edge_aggs={"avg_weight": "mean"},
            node_strategy="collapse"
        )
        print(f"Collapse successful! Meta-node ID: {meta_node_collapse.id()}")
        print(f"Graph after collapse: {g2.node_count()} nodes, {g2.edge_count()} edges")
        
        # Check if original nodes still exist
        original_nodes_exist_2 = []
        for node_id in [n1_2, n2_2, n3_2]:
            try:
                attrs = g2.get_node_attrs(node_id)
                original_nodes_exist_2.append(f"Node {node_id}: {attrs}")
            except:
                original_nodes_exist_2.append(f"Node {node_id}: NOT FOUND")
        
        print("Original nodes after collapse:")
        for status in original_nodes_exist_2:
            print(f"  {status}")
            
    except Exception as e:
        print(f"Collapse failed: {e}")

if __name__ == "__main__":
    test_node_strategy()
