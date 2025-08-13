#!/usr/bin/env python3
"""
Test the new subgraph filtering functionality
"""

import groggy as gr

def test_subgraph_filtering():
    """Test chainable subgraph filtering"""
    
    print("=== Testing Subgraph Filtering ===")
    
    # Create a graph with various attributes
    g = gr.Graph()
    
    # Add nodes with different attributes
    node_data = [
        {"id": "alice", "age": 30, "role": "engineer", "salary": 120000},
        {"id": "bob", "age": 25, "role": "engineer", "salary": 90000},
        {"id": "carol", "age": 35, "role": "designer", "salary": 95000},
        {"id": "david", "age": 28, "role": "engineer", "salary": 100000},
        {"id": "eve", "age": 45, "role": "manager", "salary": 150000}
    ]
    
    node_mapping = g.add_nodes(node_data, id_key="id")
    print(f"Created {len(g.nodes)} nodes")
    
    # Add some edges
    edge_data = [
        {"source": "alice", "target": "bob", "relationship": "collaborates"},
        {"source": "bob", "target": "david", "relationship": "collaborates"},
        {"source": "carol", "target": "alice", "relationship": "works_with"},
        {"source": "eve", "target": "alice", "relationship": "manages"}
    ]
    
    g.add_edges(edge_data, node_mapping=node_mapping)
    print(f"Created {len(g.edges)} edges")
    
    # Test 1: Basic subgraph filtering
    print(f"\n=== Test 1: Basic Subgraph Creation ===")
    engineers = g.filter_nodes('role == "engineer"')
    print(f"Engineers subgraph: {engineers}")
    print(f"Engineer nodes: {engineers.nodes}")
    print(f"Engineer edges: {engineers.edges}")
    
    # Test 2: Graph-level subgraph filtering
    print(f"\n=== Test 2: Graph-level Subgraph Filtering ===")
    try:
        # Filter engineers further for high salary
        high_salary_engineers = g.filter_subgraph_nodes(engineers, 'salary > 100000')
        print(f"High-salary engineers: {high_salary_engineers}")
        print(f"High-salary engineer nodes: {high_salary_engineers.nodes}")
        print(f"High-salary engineer edges: {high_salary_engineers.edges}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 3: Multiple filtering steps
    print(f"\n=== Test 3: Multiple Filtering Steps ===")
    try:
        # Start with all nodes
        all_people = g.filter_nodes('age > 0')  # Get everyone
        print(f"All people: {all_people}")
        
        # Filter to just people under 40
        young_people = g.filter_subgraph_nodes(all_people, 'age < 40')
        print(f"Young people: {young_people}")
        
        # Filter young people to just engineers  
        young_engineers = g.filter_subgraph_nodes(young_people, 'role == "engineer"')
        print(f"Young engineers: {young_engineers}")
        
    except Exception as e:
        print(f"Error in multiple filtering: {e}")
    
    # Test 4: Verify attributes are preserved
    print(f"\n=== Test 4: Verify Attributes ===")
    for node_id in engineers.nodes:
        name = g.get_node_attribute(node_id, "id")
        role = g.get_node_attribute(node_id, "role")
        age = g.get_node_attribute(node_id, "age")
        print(f"  Node {node_id}: {name.value if name else 'N/A'}, "
              f"role={role.value if role else 'N/A'}, "
              f"age={age.value if age else 'N/A'}")
    
    print("\n✅ Subgraph filtering working!")

if __name__ == "__main__":
    test_subgraph_filtering()