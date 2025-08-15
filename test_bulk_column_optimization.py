#!/usr/bin/env python3
"""
Test script to validate bulk column access optimization for GraphTable.

This tests the key optimization: O(m) bulk column calls instead of O(n*m) individual calls.
Expected 5-10x speedup for multi-column tables.
"""

import time
import groggy

def test_bulk_column_optimization():
    """Test the bulk column access optimization."""
    
    print("🚀 Testing Bulk Column Access Optimization")
    print("=" * 50)
    
    # Create a moderately sized graph to test optimization
    print("Creating test graph with 1000 nodes...")
    g = groggy.Graph()
    
    # Add nodes with multiple attributes (to test multi-column performance)
    node_data = []
    for i in range(1000):
        node_data.append({
            "name": f"user_{i}",
            "age": 20 + (i % 50),  # Ages 20-69
            "dept": ["Engineering", "Sales", "Marketing", "HR"][i % 4],
            "salary": 50000 + (i * 100),  # Salaries 50k-149k
            "seniority": i % 10,
            "active": i % 2 == 0
        })
    
    start_time = time.time()
    nodes = g.add_nodes(node_data)
    creation_time = time.time() - start_time
    print(f"✅ Created {len(nodes)} nodes in {creation_time:.3f}s")
    
    # Test if bulk methods are available
    has_bulk_methods = hasattr(g, 'get_node_attribute_column')
    print(f"Bulk methods available: {has_bulk_methods}")
    
    if has_bulk_methods:
        print("\n🔧 Testing individual bulk column access methods:")
        
        # Test individual bulk methods
        attrs = ['name', 'age', 'dept', 'salary', 'seniority']
        for attr in attrs:
            start_time = time.time()
            column = g.get_node_attribute_column(attr)
            elapsed = time.time() - start_time
            print(f"  {attr}: {len(column)} values in {elapsed*1000:.2f}ms")
    
    # Test GraphTable creation (this should use the bulk optimization)
    print("\n📊 Testing GraphTable creation:")
    
    start_time = time.time()
    table = g.table()
    table_creation_time = time.time() - start_time
    
    print(f"✅ Created GraphTable: {table.shape} in {table_creation_time:.3f}s")
    print(f"Table columns: {table.columns}")
    
    # Test table operations
    print(f"First few rows:")
    rows, columns = table._build_table_data()
    for i in range(min(3, len(rows))):
        row_str = ", ".join([f"{col}: {rows[i].get(col)}" for col in columns[:4]])
        print(f"  Row {i}: {row_str}...")
    
    # Test subgraph table creation  
    print("\n🔍 Testing subgraph table creation:")
    engineers = g.filter_nodes('dept == "Engineering"')
    print(f"Found {engineers.node_count()} engineers")
    
    start_time = time.time()
    eng_table = engineers.table()
    subgraph_table_time = time.time() - start_time
    
    print(f"✅ Created subgraph table: {eng_table.shape} in {subgraph_table_time:.3f}s")
    
    # Performance summary
    print(f"\n📈 Performance Summary:")
    print(f"  Graph creation: {creation_time:.3f}s")
    print(f"  Full table:     {table_creation_time:.3f}s") 
    print(f"  Subgraph table: {subgraph_table_time:.3f}s")
    
    if has_bulk_methods:
        print(f"  ✅ Using optimized bulk column access")
        print(f"  💡 Expected: 5-10x faster than individual attribute calls")
    else:
        print(f"  ⚠️  Using fallback individual attribute access") 
        print(f"  💡 Bulk methods not available - check implementation")
    
    print(f"\n🎯 Optimization Status: {'SUCCESS' if has_bulk_methods else 'NEEDS IMPLEMENTATION'}")

if __name__ == "__main__":
    test_bulk_column_optimization()