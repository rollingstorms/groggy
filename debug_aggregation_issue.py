#!/usr/bin/env python3
"""
Debug aggregation values to understand why they're 0.
"""

import groggy as gr

def debug_aggregation_issue():
    print("Debug: Why aggregation values are 0")
    
    # Create a graph with nodes that have numeric attributes
    g = gr.Graph()
    g.add_node(name="Alice", age=25, salary=85000)
    g.add_node(name="Bob", age=30, salary=95000)
    g.add_node(name="Carol", age=28, salary=80000)
    
    print("Original nodes:")
    for i, node in enumerate(g.nodes):
        age = g.get_node_attr(node.id, "age")
        salary = g.get_node_attr(node.id, "salary")
        name = g.get_node_attr(node.id, "name")
        print(f"  Node {node.id}: name={name}, age={age}, salary={salary}")
    
    # Create subgraph and collapse
    subgraph = g.nodes[[0, 1, 2]]
    print(f"Subgraph created: {subgraph.node_count()} nodes")
    
    try:
        meta_node = subgraph.collapse(
            node_aggs={
                "team_size": "count",
                "avg_age": ("mean", "age"),        
                "total_salary": ("sum", "salary"),
            },
            node_strategy="extract"
        )
        
        print(f"\nMeta-node created: {meta_node}")
        
        # Check attributes directly on meta-node
        team_size = g.get_node_attr(meta_node.id, "team_size")
        avg_age = g.get_node_attr(meta_node.id, "avg_age")
        total_salary = g.get_node_attr(meta_node.id, "total_salary")
        
        print(f"team_size: {team_size} (type: {type(team_size)})")
        print(f"avg_age: {avg_age} (type: {type(avg_age)})")
        print(f"total_salary: {total_salary} (type: {type(total_salary)})")
        
        # Expected values
        expected_avg_age = (25 + 30 + 28) / 3
        expected_total_salary = 85000 + 95000 + 80000
        print(f"\nExpected avg_age: {expected_avg_age}")
        print(f"Expected total_salary: {expected_total_salary}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_aggregation_issue()
