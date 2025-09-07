#!/usr/bin/env python3
"""
Quick debug of filter issue
"""
import groggy

def debug_filter_quick():
    g = groggy.Graph()
    n1 = g.add_node(name='Alice', age=25)
    n2 = g.add_node(name='Bob', age=30)  
    e1 = g.add_edge(n1, n2, strength=5, type='friend')
    
    table = g.table()
    print("Edge table:")
    print(table.edges)
    
    # Check what values exist
    unique_strengths = table.edges.unique_attr_values('strength')
    print(f"\nUnique strength values: {[str(val.value) + f' (type: {type(val.value)})' for val in unique_strengths]}")
    
    # Try filtering
    result = table.edges.filter_by_attr('strength', 5)
    print(f"Filtering by strength=5 (int): {result.nrows()} rows")
    
    # Try with float
    result2 = table.edges.filter_by_attr('strength', 5.0)  
    print(f"Filtering by strength=5.0 (float): {result2.nrows()} rows")

if __name__ == '__main__':
    debug_filter_quick()
