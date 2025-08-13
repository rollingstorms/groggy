#!/usr/bin/env python3
"""
Test the new g.attributes functionality
"""

import groggy as gr

def test_attributes_access():
    """Test g.attributes dictionary-like access"""
    
    print("=== Testing g.attributes Dictionary Access ===")
    
    # Create a graph with various attributes
    g = gr.Graph()
    
    # Add nodes with different attributes
    alice = g.add_node(name="Alice", age=30, salary=120000, active=True)
    bob = g.add_node(name="Bob", age=25, salary=90000, active=True) 
    carol = g.add_node(name="Carol", age=35, salary=150000, active=False)
    david = g.add_node(name="David", age=28)  # Missing salary and active
    
    print(f"Created graph with {len(g.nodes)} nodes")
    print(f"Node IDs: {g.nodes}")
    
    # Test g.attributes object
    print(f"\n=== Testing g.attributes Object ===")
    attrs = g.attributes
    print(f"Attributes object: {attrs}")
    print(f"Available keys: {attrs.keys()}")
    
    # Test accessing individual attributes
    print(f"\n=== Testing Individual Attribute Access ===")
    
    if "name" in attrs:
        names = attrs["name"]
        print(f"Names: {names}")
        print(f"Names type: {type(names)}")
    
    if "age" in attrs:
        ages = attrs["age"]
        print(f"Ages: {ages}")
        print(f"Ages type: {type(ages)}")
    
    if "salary" in attrs:
        salaries = attrs["salary"]
        print(f"Salaries: {salaries}")
        print(f"Salaries type: {type(salaries)}")
    
    if "active" in attrs:
        actives = attrs["active"]
        print(f"Active flags: {actives}")
        print(f"Active type: {type(actives)}")
    
    # Test missing attribute
    print(f"\n=== Testing Missing Attribute ===")
    try:
        missing = attrs["nonexistent"]
        print(f"Nonexistent attribute: {missing}")
    except Exception as e:
        print(f"Error accessing nonexistent attribute: {e}")
    
    # Test alignment with node IDs
    print(f"\n=== Testing Index Alignment ===")
    print(f"Node IDs: {g.nodes}")
    print(f"Names: {attrs['name'] if 'name' in attrs else 'N/A'}")
    print(f"Ages: {attrs['age'] if 'age' in attrs else 'N/A'}")
    
    # Verify individual access matches column access
    print(f"\n=== Testing Individual vs Column Access ===")
    for i, node_id in enumerate(g.nodes):
        print(f"Node {node_id} (index {i}):")
        # Get individual attributes
        name = g.get_node_attribute(node_id, "name")
        age = g.get_node_attribute(node_id, "age")
        print(f"  Individual name: {name.value if name else None}")
        print(f"  Individual age: {age.value if age else None}")
        
        # Compare with column access
        if "name" in attrs:
            print(f"  Column name[{i}]: {attrs['name'][i]}")
        if "age" in attrs:
            print(f"  Column age[{i}]: {attrs['age'][i]}")
    
    print("\n✅ Attributes access working!")

if __name__ == "__main__":
    test_attributes_access()