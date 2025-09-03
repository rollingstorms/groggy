#!/usr/bin/env python3
"""
Test what happens when we try to aggregate non-existent attributes.
"""

import groggy

def test_missing_attributes():
    g = groggy.Graph()
    
    # Create nodes with salary but try to aggregate "income" (doesn't exist)
    node1 = g.add_node(name="Alice", salary=75000)
    node2 = g.add_node(name="Bob", salary=85000) 
    
    print(f"Nodes: {node1}, {node2}")
    print("Node attributes:")
    all_nodes = g.nodes[list(g.node_ids)]
    for node_id in [node1, node2]:
        salary = all_nodes.get_node_attribute(node_id, "salary")
        income = all_nodes.get_node_attribute(node_id, "income")  # doesn't exist
        print(f"  Node {node_id}: salary={salary}, income={income}")
    
    subgraph = g.nodes[[node1, node2]]
    
    # Test 1: Try to aggregate existing attribute
    print(f"\nTest 1: Aggregate existing 'salary' attribute")
    meta_node1 = subgraph.collapse_to_node({"salary": "sum"})
    
    all_nodes = g.nodes[list(g.node_ids)]
    salary_sum = all_nodes.get_node_attribute(meta_node1, "salary")
    print(f"Meta-node {meta_node1} salary: {salary_sum}")
    
    # Test 2: Try to aggregate non-existent attribute 
    print(f"\nTest 2: Try to aggregate non-existent 'income' attribute")
    node3 = g.add_node(name="Charlie", salary=95000)
    node4 = g.add_node(name="David", salary=65000)
    
    subgraph2 = g.nodes[[node3, node4]]
    meta_node2 = subgraph2.collapse_to_node({"income": "sum"})  # 'income' doesn't exist
    
    all_nodes = g.nodes[list(g.node_ids)]
    income_sum = all_nodes.get_node_attribute(meta_node2, "income")
    print(f"Meta-node {meta_node2} income: {income_sum}")  # Should be None
    
    # Test 3: Mix existing and non-existent attributes
    print(f"\nTest 3: Mix existing and non-existent attributes")
    node5 = g.add_node(name="Eve", salary=70000)
    node6 = g.add_node(name="Frank", salary=80000)
    
    subgraph3 = g.nodes[[node5, node6]]
    meta_node3 = subgraph3.collapse_to_node({
        "count": "count",           # Should work (doesn't need existing attr)
        "salary": "sum",            # Should work (exists)
        "bonus": "sum",             # Should NOT work (doesn't exist)
        "total_compensation": "sum" # Should NOT work (doesn't exist)
    })
    
    all_nodes = g.nodes[list(g.node_ids)]
    count = all_nodes.get_node_attribute(meta_node3, "count")
    salary_sum = all_nodes.get_node_attribute(meta_node3, "salary")
    bonus_sum = all_nodes.get_node_attribute(meta_node3, "bonus")
    total_comp = all_nodes.get_node_attribute(meta_node3, "total_compensation")
    
    print(f"Meta-node {meta_node3} results:")
    print(f"  count: {count} (should be 2)")
    print(f"  salary: {salary_sum} (should be 150000)")
    print(f"  bonus: {bonus_sum} (should be None)")
    print(f"  total_compensation: {total_comp} (should be None)")

if __name__ == "__main__":
    test_missing_attributes()
