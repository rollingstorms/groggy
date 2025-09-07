#!/usr/bin/env python3
"""
Test script to verify the table method fixes
"""
import groggy

def test_table_fixes():
    print("Creating test graph...")
    g = groggy.Graph()
    
    # Add some nodes and edges with attributes
    n1 = g.add_node(name='Alice', age=25)
    n2 = g.add_node(name='Bob', age=30)  
    n3 = g.add_node(name='Charlie', age=35)
    
    e1 = g.add_edge(n1, n2, strength=5, type='friend')
    e2 = g.add_edge(n2, n3, strength=3, type='colleague')
    e3 = g.add_edge(n1, n3, strength=8, type='friend')
    
    # Get table
    table = g.table()
    print(f"Graph table created: {table}")
    print(f"Nodes: {table.nodes.nrows()} rows, {table.nodes.ncols()} cols")
    print(f"Edges: {table.edges.nrows()} rows, {table.edges.ncols()} cols")
    
    # Test 1: Column selection with required columns preserved
    print("\n=== Test 1: Column selection ===")
    try:
        edges_subset = table.edges[['strength']]
        print(f"✅ edges[['strength']] works: {edges_subset.shape()} rows")
        print(f"   Columns: {', '.join(edges_subset.base_table().column_names())}")
    except Exception as e:
        print(f"❌ edges[['strength']] failed: {e}")
    
    # Test 2: filter_by_attr method  
    print("\n=== Test 2: filter_by_attr ===")
    try:
        filtered_edges = table.edges.filter_by_attr('strength', 5)
        print(f"✅ edges.filter_by_attr('strength', 5) works: {filtered_edges.nrows()} rows")
    except Exception as e:
        print(f"❌ edges.filter_by_attr('strength', 5) failed: {e}")
    
    # Test 3: as_tuples method
    print("\n=== Test 3: as_tuples ===")
    try:
        tuples = table.edges.as_tuples()
        print(f"✅ edges.as_tuples() works: {len(tuples)} tuples")
        print(f"   First tuple: {tuples[0] if tuples else 'None'}")
    except Exception as e:
        print(f"❌ edges.as_tuples() failed: {e}")
    
    # Test 4: edge_ids method
    print("\n=== Test 4: edge_ids ===")
    try:
        edge_ids = table.edges.edge_ids()
        print(f"✅ edges.edge_ids() works: {len(edge_ids)} IDs")
        print(f"   Edge IDs: {edge_ids}")
    except Exception as e:
        print(f"❌ edges.edge_ids() failed: {e}")
    
    # Test 5: sources and targets methods
    print("\n=== Test 5: sources and targets ===")
    try:
        sources = table.edges.sources()
        targets = table.edges.targets()
        print(f"✅ edges.sources() works: {len(sources)} sources")
        print(f"✅ edges.targets() works: {len(targets)} targets")
        print(f"   Sources: {sources}")
        print(f"   Targets: {targets}")
    except Exception as e:
        print(f"❌ sources/targets failed: {e}")
    
    # Test 6: unique_attr_values method
    print("\n=== Test 6: unique_attr_values ===")
    try:
        unique_types = table.edges.unique_attr_values('type')
        print(f"✅ edges.unique_attr_values('type') works: {len(unique_types)} unique values")
        print(f"   Unique types: {[val.value for val in unique_types]}")
    except Exception as e:
        print(f"❌ edges.unique_attr_values('type') failed: {e}")
        
    print("\n=== All tests completed ===")

if __name__ == '__main__':
    test_table_fixes()
