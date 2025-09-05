#!/usr/bin/env python3

import groggy
import traceback

def test_node_strategy_simple():
    """Simple test to debug the node_strategy parameter"""
    
    # Create a simple graph
    g = groggy.Graph()
    
    # Add some nodes
    n1 = g.add_node()
    n2 = g.add_node()
    n3 = g.add_node()
    
    # Set node attributes
    g.set_node_attrs({"name": {n1: "Alice", n2: "Bob", n3: "Carol"}})
    g.set_node_attrs({"age": {n1: 30, n2: 25, n3: 35}})
    
    # Add edges
    e1 = g.add_edge(n1, n2)
    e2 = g.add_edge(n2, n3)
    e3 = g.add_edge(n1, n3)
    
    print(f"Original graph: {g.node_count()} nodes, {g.edge_count()} edges")
    
    # Create subgraph
    try:
        subgraph = g.induced_subgraph([n1, n2, n3])
        print(f"Subgraph created: {subgraph.node_count()} nodes, {subgraph.edge_count()} edges")
        
        # Try to call collapse method
        print("\nTesting collapse method...")
        result = subgraph.collapse(
            node_aggs={"total_people": "count"},
            node_strategy="extract"
        )
        print(f"Success! Result type: {type(result)}")
        print(f"Meta-node ID: {result.id()}")
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_node_strategy_simple()
