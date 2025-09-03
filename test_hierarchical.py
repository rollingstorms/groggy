#!/usr/bin/env python3
"""
Test script for hierarchical subgraph functionality.

This script tests the FFI bindings for hierarchical subgraphs including:
- Creating subgraphs
- Collapsing them to meta-nodes
- Accessing subgraph properties
- Aggregation functions
"""

def test_hierarchical_subgraphs():
    try:
        import groggy
        
        # Create a test graph
        g = groggy.Graph()
        
        # Add some nodes with attributes
        g.add_node(name="Alice", age=25, salary=50000)
        g.add_node(name="Bob", age=30, salary=60000) 
        g.add_node(name="Charlie", age=35, salary=70000)
        g.add_node(name="Diana", age=28, salary=55000)
        
        # Add some edges
        g.add_edge(0, 1, weight=0.8)
        g.add_edge(1, 2, weight=0.7)
        g.add_edge(2, 3, weight=0.9)
        g.add_edge(3, 0, weight=0.6)
        
        print("Created test graph with 4 nodes and 4 edges")
        
        # Create a subgraph with the first 3 nodes
        subgraph_nodes = [0, 1, 2]
        subgraph = g.nodes[subgraph_nodes]
        
        print(f"Created subgraph: {subgraph}")
        print(f"Subgraph has {len(subgraph)} nodes")
        
        # Test aggregation functions
        from groggy import AggregationFunction
        
        # Test various aggregation function creation methods
        sum_agg = AggregationFunction.sum()
        mean_agg = AggregationFunction.mean()
        count_agg = AggregationFunction.count()
        
        print(f"Created aggregation functions: {sum_agg}, {mean_agg}, {count_agg}")
        
        # Test add_to_graph method (collapse subgraph to meta-node)
        agg_functions = {
            "total_salary": "sum",
            "avg_age": "mean", 
            "person_count": "count"
        }
        
        meta_node = subgraph.add_to_graph(agg_functions)
        print(f"Created meta-node: {meta_node}")
        
        # Test accessing meta-node properties
        print(f"Meta-node ID: {meta_node.node_id}")
        print(f"Has subgraph: {meta_node.has_subgraph}")
        if meta_node.has_subgraph:
            print(f"Subgraph ID: {meta_node.subgraph_id}")
        
        # Test accessing attributes
        attrs = meta_node.attributes()
        print(f"Meta-node attributes: {attrs}")
        
        # Test nodes.subgraphs property
        subgraph_nodes = g.nodes.subgraphs
        print(f"Found {len(subgraph_nodes)} subgraph nodes in graph")
        
        print("✅ All hierarchical subgraph tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Could not import groggy: {e}")
        print("Make sure the Python package is built with: maturin develop")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_hierarchical_subgraphs()
    exit(0 if success else 1)