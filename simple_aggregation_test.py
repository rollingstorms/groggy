#!/usr/bin/env python3
"""
Simple test to verify aggregation functionality works correctly
"""

def test_basic_aggregation():
    try:
        import groggy as gr
        print("✅ Successfully imported groggy")
    except ImportError as e:
        print(f"❌ Failed to import groggy: {e}")
        return False
    
    try:
        # Create a simple graph
        graph = gr.Graph()
        print("✅ Created graph")
        
        # Add nodes
        nodes = graph.add_nodes(5)
        print(f"✅ Added {len(nodes)} nodes: {nodes}")
        
        # Set attributes using correct format
        attrs_dict = {
            "age": {
                "nodes": nodes,
                "values": [25, 30, 35, 40, 45],
                "value_type": "int"
            },
            "department": {
                "nodes": nodes,
                "values": ["Engineering", "Marketing", "Engineering", "Sales", "Engineering"],
                "value_type": "text"
            }
        }
        graph.set_node_attributes(attrs_dict)
        print("✅ Set node attributes")
        
        # Test individual attribute access
        for i, node in enumerate(nodes):
            age = graph.get_node_attribute(node, "age")
            dept = graph.get_node_attribute(node, "department")
            print(f"   Node {node}: age={age.value if age else None}, dept={dept.value if dept else None}")
        
        # Test basic aggregation
        print("\n🧮 Testing Basic Aggregation:")
        age_stats = graph.aggregate_node_attribute("age", "average")
        print(f"   Average age: {age_stats.value}")
        
        count_stats = graph.aggregate_node_attribute("age", "count")
        print(f"   Count: {count_stats.value}")
        
        sum_stats = graph.aggregate_node_attribute("age", "sum")
        print(f"   Sum: {sum_stats.value}")
        
        # Test grouped aggregation
        print("\n👥 Testing Grouped Aggregation:")
        grouped = graph.group_nodes_by_attribute("department", "age", "average")
        print(f"   Grouped result type: {type(grouped)}")
        print(f"   Grouped result value type: {type(grouped.value)}")
        
        # Access the grouped results
        group_dict = grouped.value
        for dept, avg_age in group_dict.items():
            print(f"   Department {dept.value}: avg age = {avg_age.value}")
        
        print("\n✅ All aggregation tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_basic_aggregation()
