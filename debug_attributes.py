#!/usr/bin/env python3
"""
Debug attribute access in hierarchical subgraphs.
"""

import groggy

def debug_attributes():
    # Create graph and add nodes
    g = groggy.Graph()
    
    node0 = g.add_node(name="Alice", age=25, salary=50000)
    node1 = g.add_node(name="Bob", age=30, salary=60000)  
    node2 = g.add_node(name="Charlie", age=35, salary=70000)
    
    print(f"Added nodes: {node0}, {node1}, {node2}")
    
    # Create subgraph and collapse
    subgraph = g.nodes[[0, 1, 2]]
    
    print("Before collapsing, checking attribute access...")
    try:
        # Test different ways of attribute access
        print("Using subgraph.get_node_attribute():")
        for node_id in [0, 1, 2]:
            name = subgraph.get_node_attribute(node_id, "name")
            age = subgraph.get_node_attribute(node_id, "age")  
            salary = subgraph.get_node_attribute(node_id, "salary")
            print(f"  Node {node_id}: name={name}, age={age}, salary={salary}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nCollapsing to meta-node...")
    meta_node = subgraph.add_to_graph({
        "total_salary": "sum",
        "avg_age": "mean",
        "person_count": "count"
    })
    
    print(f"Meta-node created: ID={meta_node.node_id}")
    
    # Check different ways to access meta-node attributes
    print("\nChecking meta-node attributes:")
    
    # Method 1: meta_node.attributes()
    attrs = meta_node.attributes()
    print(f"1. meta_node.attributes(): {attrs}")
    
    # Method 2: Access via graph nodes directly  
    try:
        print("2. Direct graph access:")
        total_salary = g.nodes[meta_node.node_id].__getattribute__("total_salary") if hasattr(g.nodes[meta_node.node_id], 'total_salary') else None
        print(f"   total_salary: {total_salary}")
        
        # Try to access using subgraph method
        total_salary2 = subgraph.get_node_attribute(meta_node.node_id, "total_salary")
        print(f"   total_salary via subgraph: {total_salary2}")
        
    except Exception as e:
        print(f"   Error accessing directly: {e}")
    
    # Method 3: Check if attributes are set on the meta-node in the graph
    try:
        print("3. All graph nodes and their attributes:")
        for node_id in g.node_ids():
            print(f"   Node {node_id}: checking attributes...")
            
            # Try to get specific attributes we set
            try:
                total_salary = g.nodes[node_id].total_salary if hasattr(g.nodes[node_id], 'total_salary') else "Not found"
                avg_age = g.nodes[node_id].avg_age if hasattr(g.nodes[node_id], 'avg_age') else "Not found" 
                person_count = g.nodes[node_id].person_count if hasattr(g.nodes[node_id], 'person_count') else "Not found"
                print(f"      total_salary={total_salary}, avg_age={avg_age}, person_count={person_count}")
            except Exception as e:
                print(f"      Error: {e}")
                
    except Exception as e:
        print(f"Error checking all nodes: {e}")

if __name__ == "__main__":
    debug_attributes()