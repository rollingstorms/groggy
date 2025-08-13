#!/usr/bin/env python3
"""
Test string query parsing functionality
"""

import groggy as gr

def test_string_queries():
    print("Testing string query parsing...")
    
    graph = gr.Graph()
    
    # Create some test data
    node_data = [
        {"id": "alice", "name": "Alice", "salary": 120000, "department": "Engineering"},
        {"id": "bob", "name": "Bob", "salary": 90000, "department": "Sales"},
        {"id": "charlie", "name": "Charlie", "salary": 150000, "department": "Engineering"},
        {"id": "diana", "name": "Diana", "salary": 80000, "department": "Marketing"}
    ]
    
    mapping = graph.add_nodes(node_data, id_key="id")
    print(f"Created nodes: {mapping}")
    
    # Add some edges
    edge_data = [
        {"source": "alice", "target": "bob", "relationship": "reports_to", "weight": 0.8},
        {"source": "charlie", "target": "alice", "relationship": "collaborates", "weight": 0.9},
        {"source": "diana", "target": "bob", "relationship": "collaborates", "weight": 0.7}
    ]
    graph.add_edges_from_dicts(edge_data, mapping)
    
    print(f"\nGraph created: {graph}")
    
    # Test 1: String query for high earners
    print(f"\n=== Test 1: String query for high earners (salary > 100000) ===")
    try:
        high_earners = graph.filter_nodes("salary > 100000")
        print(f"✅ Found high earners: {high_earners}")
        
        # Check the results
        for node_id in high_earners.get_all_nodes():
            name = graph.get_node_attribute(node_id, "name")
            salary = graph.get_node_attribute(node_id, "salary")
            print(f"   {name.value if name else 'Unknown'}: ${salary.value if salary else 'Unknown'}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: String query for department
    print(f"\n=== Test 2: String query for Engineering department ===")
    try:
        engineers = graph.filter_nodes("department == 'Engineering'")
        print(f"✅ Found engineers: {engineers}")
        
        for node_id in engineers.get_all_nodes():
            name = graph.get_node_attribute(node_id, "name")
            dept = graph.get_node_attribute(node_id, "department")
            print(f"   {name.value if name else 'Unknown'}: {dept.value if dept else 'Unknown'}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: Edge filtering with strings
    print(f"\n=== Test 3: String query for high-weight relationships ===")
    try:
        strong_edges = graph.filter_edges("weight > 0.75")
        print(f"✅ Found strong relationships: {strong_edges}")
        
        for edge_id in strong_edges:
            weight = graph.get_edge_attribute(edge_id, "weight")
            relationship = graph.get_edge_attribute(edge_id, "relationship")
            print(f"   Edge {edge_id}: {relationship.value if relationship else 'Unknown'} (weight: {weight.value if weight else 'Unknown'})")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 4: Backward compatibility - still works with filter objects
    print(f"\n=== Test 4: Backward compatibility with filter objects ===")
    try:
        attr_filter = gr.AttributeFilter.greater_than(gr.AttrValue(100000))
        node_filter = gr.NodeFilter.attribute_filter("salary", attr_filter)
        high_earners_old = graph.filter_nodes(node_filter)
        print(f"✅ Old-style filter still works: {high_earners_old}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print(f"\n🎉 String query tests completed!")

if __name__ == "__main__":
    test_string_queries()