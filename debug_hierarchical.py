#!/usr/bin/env python3
"""
Debug hierarchical subgraph functionality.
"""

import groggy

def debug_hierarchical():
    # Create graph and subgraphs
    g = groggy.Graph()
    
    # Add nodes with attributes using individual add_node calls
    node0 = g.add_node(name="Alice", age=25, salary=50000)
    node1 = g.add_node(name="Bob", age=30, salary=60000)  
    node2 = g.add_node(name="Charlie", age=35, salary=70000)
    
    print(f"Added nodes: {node0}, {node1}, {node2}")
    
    # Check what attributes exist
    print("Node 0 attributes:", g.nodes[0].__dict__ if hasattr(g.nodes[0], '__dict__') else "Not available")
    
    # Check node attributes directly
    try:
        print("Node 0 name:", g.nodes[0].name)  
        print("Node 0 age:", g.nodes[0].age)
        print("Node 0 salary:", g.nodes[0].salary)
    except Exception as e:
        print(f"Error accessing node attributes: {e}")
    
    # Create subgraph
    subgraph = g.nodes[[0, 1, 2]]
    print(f"Created subgraph: {subgraph}")
    
    # Check subgraph node attributes 
    try:
        print("Subgraph node 0 attributes:")
        node_attrs = subgraph.get_node_attribute(0, "name")
        print(f"  name: {node_attrs}")
        
        node_attrs = subgraph.get_node_attribute(0, "age") 
        print(f"  age: {node_attrs}")
        
        node_attrs = subgraph.get_node_attribute(0, "salary")
        print(f"  salary: {node_attrs}")
        
    except Exception as e:
        print(f"Error getting subgraph node attributes: {e}")
    
    # Test aggregation function creation
    print("\nTesting aggregation functions:")
    sum_agg = groggy.AggregationFunction.sum()
    mean_agg = groggy.AggregationFunction.mean()
    count_agg = groggy.AggregationFunction.count()
    print(f"Created: {sum_agg}, {mean_agg}, {count_agg}")
    
    # Try collapsing to meta-node
    print("\nCollapsing to meta-node:")
    try:
        meta_node = subgraph.add_to_graph({
            "total_salary": "sum",
            "avg_age": "mean",
            "person_count": "count"
        })
        
        print(f"Meta-node created: {meta_node}")
        print(f"Meta-node ID: {meta_node.node_id}")
        print(f"Has subgraph: {meta_node.has_subgraph}")
        
        # Get attributes 
        attrs = meta_node.attributes()
        print(f"Attributes: {attrs}")
        
        # Check if the meta-node itself has the aggregated attributes as node attributes
        print(f"Meta-node as regular node:")
        try:
            print(f"  total_salary: {g.nodes[meta_node.node_id].total_salary if hasattr(g.nodes[meta_node.node_id], 'total_salary') else 'Not found'}")
        except:
            print("  Could not access meta-node attributes as regular node")
            
    except Exception as e:
        print(f"Error creating meta-node: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_hierarchical()