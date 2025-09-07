#!/usr/bin/env python3
"""
Debug script to check filter_by_attr issue
"""
import groggy

def debug_filter():
    print("Creating test graph...")
    g = groggy.Graph()
    
    # Add some nodes and edges with attributes
    n1 = g.add_node(name='Alice', age=25)
    n2 = g.add_node(name='Bob', age=30)  
    
    e1 = g.add_edge(n1, n2, strength=5, type='friend')
    
    table = g.table()
    print("Edge table:")
    print(table.edges)
    
    print("\nColumn details:")
    for col_name in table.edges.base_table().column_names():
        column = table.edges.base_table().column(col_name)
        if column:
            print(f"{col_name}: {column.data()}")
    
    print(f"\nTesting filter_by_attr with different types:")
    
    # Try different data types
    try:
        result1 = table.edges.filter_by_attr('strength', 5)
        print(f"filter_by_attr('strength', 5) -> {result1.nrows()} rows")
    except Exception as e:
        print(f"filter_by_attr('strength', 5) failed: {e}")
        
    try:
        result2 = table.edges.filter_by_attr('strength', 5.0)
        print(f"filter_by_attr('strength', 5.0) -> {result2.nrows()} rows")
    except Exception as e:
        print(f"filter_by_attr('strength', 5.0) failed: {e}")
        
    try:
        result3 = table.edges.filter_by_attr('type', 'friend')
        print(f"filter_by_attr('type', 'friend') -> {result3.nrows()} rows")
    except Exception as e:
        print(f"filter_by_attr('type', 'friend') failed: {e}")
        
    # Check unique values
    try:
        unique_strengths = table.edges.unique_attr_values('strength')
        print(f"\nUnique strength values: {[val.value for val in unique_strengths]}")
    except Exception as e:
        print(f"unique_attr_values('strength') failed: {e}")

if __name__ == '__main__':
    debug_filter()
