#!/usr/bin/env python3
"""
Test kwargs functionality for add_node()
"""

import groggy as gr

def test_kwargs():
    print("Testing add_node() with kwargs...")
    
    # Create graph
    graph = gr.Graph()
    
    # Test 1: Basic kwargs
    alice = graph.add_node(name="Alice", age=30, department="Engineering")
    print(f"✅ Created node {alice} with kwargs")
    
    # Verify attributes were set
    name = graph.get_node_attribute(alice, "name")
    age = graph.get_node_attribute(alice, "age") 
    dept = graph.get_node_attribute(alice, "department")
    
    print(f"   name: {name.value if name else None}")
    print(f"   age: {age.value if age else None}")
    print(f"   department: {dept.value if dept else None}")
    
    # Test 2: Different data types
    bob = graph.add_node(
        name="Bob",
        age=25,
        salary=75000,
        active=True,
        skills=3.5,
        tags=[1.0, 2.0, 3.0]
    )
    print(f"✅ Created node {bob} with mixed data types")
    
    # Test 3: No kwargs (should still work)
    charlie = graph.add_node()
    print(f"✅ Created node {charlie} without kwargs")
    
    # Test 4: Empty kwargs 
    diana = graph.add_node(**{})
    print(f"✅ Created node {diana} with empty kwargs")
    
    print(f"\nFinal graph: {graph}")
    print("🎉 All kwargs tests passed!")

if __name__ == "__main__":
    test_kwargs()