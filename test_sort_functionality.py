#!/usr/bin/env python3
"""
Test script to verify the sort_by functionality
"""
import groggy

def test_sort_functionality():
    print("Creating test graph with varied data...")
    g = groggy.Graph()
    
    # Add nodes with different ages for sorting
    n1 = g.add_node(name='Alice', age=25)
    n2 = g.add_node(name='Bob', age=30)  
    n3 = g.add_node(name='Charlie', age=20)
    n4 = g.add_node(name='Diana', age=35)
    
    # Add edges with different strengths
    e1 = g.add_edge(n1, n2, strength=5, type='friend')
    e2 = g.add_edge(n2, n3, strength=3, type='colleague')  
    e3 = g.add_edge(n3, n4, strength=8, type='friend')
    e4 = g.add_edge(n4, n1, strength=2, type='family')
    
    table = g.table()
    
    print(f"\n=== Original Nodes Table ===")
    print(table.nodes)
    
    print(f"\n=== Original Edges Table ===")  
    print(table.edges)
    
    # Test 1: Sort nodes by age (ascending)
    print(f"\n=== Test 1: Sort nodes by age (ascending) ===")
    try:
        sorted_nodes = table.nodes.sort_by('age', ascending=True)
        print(f"✅ nodes.sort_by('age', ascending=True) works:")
        print(sorted_nodes)
    except Exception as e:
        print(f"❌ nodes.sort_by('age', ascending=True) failed: {e}")
    
    # Test 2: Sort nodes by age (descending)
    print(f"\n=== Test 2: Sort nodes by age (descending) ===")
    try:
        sorted_nodes_desc = table.nodes.sort_by('age', ascending=False)
        print(f"✅ nodes.sort_by('age', ascending=False) works:")
        print(sorted_nodes_desc)
    except Exception as e:
        print(f"❌ nodes.sort_by('age', ascending=False) failed: {e}")
    
    # Test 3: Sort nodes by name (ascending)
    print(f"\n=== Test 3: Sort nodes by name (ascending) ===")
    try:
        sorted_by_name = table.nodes.sort_by('name', ascending=True)
        print(f"✅ nodes.sort_by('name', ascending=True) works:")
        print(sorted_by_name)
    except Exception as e:
        print(f"❌ nodes.sort_by('name', ascending=True) failed: {e}")
    
    # Test 4: Sort edges by strength (ascending)
    print(f"\n=== Test 4: Sort edges by strength (ascending) ===")
    try:
        sorted_edges = table.edges.sort_by('strength', ascending=True)
        print(f"✅ edges.sort_by('strength', ascending=True) works:")
        print(sorted_edges)
    except Exception as e:
        print(f"❌ edges.sort_by('strength', ascending=True) failed: {e}")
    
    # Test 5: Sort edges by type (ascending)
    print(f"\n=== Test 5: Sort edges by type (ascending) ===")
    try:
        sorted_edges_type = table.edges.sort_by('type', ascending=True)
        print(f"✅ edges.sort_by('type', ascending=True) works:")
        print(sorted_edges_type)
    except Exception as e:
        print(f"❌ edges.sort_by('type', ascending=True) failed: {e}")
    
    # Test 6: Sort base table directly
    print(f"\n=== Test 6: Sort base table directly ===")
    try:
        base_nodes = table.nodes.base_table()
        sorted_base = base_nodes.sort_by('age', ascending=True)
        print(f"✅ base_table.sort_by('age', ascending=True) works:")
        print(sorted_base)
    except Exception as e:
        print(f"❌ base_table.sort_by('age', ascending=True) failed: {e}")
    
    # Test 7: Error handling - invalid column
    print(f"\n=== Test 7: Error handling - invalid column ===")
    try:
        invalid_sort = table.nodes.sort_by('invalid_column', ascending=True)
        print(f"❌ Should have failed but didn't")
    except Exception as e:
        print(f"✅ nodes.sort_by('invalid_column') correctly failed: {e}")
        
    print(f"\n=== All sort tests completed ===")

if __name__ == '__main__':
    test_sort_functionality()
