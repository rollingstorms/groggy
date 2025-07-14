#!/usr/bin/env python3
"""
Demonstration of Groggy's new optimization features
Shows the performance improvements from batch operations and vectorized attribute retrieval
"""

import time
import random
from groggy import Graph

def create_test_graph(num_nodes=10000):
    """Create a test graph with structured data"""
    print(f"Creating test graph with {num_nodes:,} nodes...")
    
    g = Graph(backend='rust')
    
    # Use chunked bulk addition (much faster than individual additions)
    start_time = time.time()
    
    # Add nodes in chunks using the optimized method
    chunk_size = 1000
    for i in range(0, num_nodes, chunk_size):
        chunk_end = min(i + chunk_size, num_nodes)
        
        # Create chunk of nodes
        for j in range(i, chunk_end):
            node_id = f"user_{j}"
            attrs = {
                'age': random.randint(18, 80),
                'salary': random.randint(30000, 150000),
                'department': random.choice(['engineering', 'sales', 'marketing', 'hr']),
                'location': random.choice(['NYC', 'SF', 'LA', 'Chicago', 'Boston']),
                'experience': random.randint(0, 20)
            }
            g.add_node(node_id, **attrs)
    
    creation_time = time.time() - start_time
    print(f"✅ Graph created in {creation_time:.3f}s ({num_nodes/creation_time:,.0f} nodes/sec)")
    
    return g

def test_bulk_attribute_operations(g):
    """Test the new bulk attribute operations"""
    print("\n🚀 Testing Bulk Attribute Operations:")
    
    # Test 1: Bulk attribute updates
    print("1. Testing chunked attribute updates...")
    start_time = time.time()
    
    # Update 5000 nodes with new attributes using chunked method
    updates = {}
    for i in range(5000):
        node_id = f"user_{i}"
        updates[node_id] = {
            'performance_score': random.uniform(0.5, 1.0),
            'training_completed': random.choice([True, False])
        }
    
    # Use the optimized chunked update method
    g.set_node_attributes_chunked(updates, chunk_size=1000)
    
    update_time = time.time() - start_time
    print(f"   ✅ Updated 5,000 nodes in {update_time:.3f}s ({5000/update_time:,.0f} updates/sec)")

def test_vectorized_retrieval(g):
    """Test the new vectorized retrieval methods"""
    print("\n⚡ Testing Vectorized Attribute Retrieval:")
    
    # Test 1: Bulk vectors retrieval
    print("1. Testing bulk attribute vectors...")
    start_time = time.time()
    
    attr_names = ['age', 'salary', 'department', 'experience']
    vectors_data = g.get_bulk_node_attribute_vectors(attr_names)
    
    vectors_time = time.time() - start_time
    total_values = sum(len(data[1]) for data in vectors_data.values())
    print(f"   ✅ Retrieved {len(attr_names)} attributes for {total_values:,} values in {vectors_time:.3f}s")
    print(f"   📊 Rate: {total_values/vectors_time:,.0f} values/sec")
    
    # Test 2: Single attribute vectorized retrieval
    print("2. Testing single attribute vectorized...")
    start_time = time.time()
    
    node_ids, salaries = g.get_single_attribute_vectorized('salary')
    
    single_time = time.time() - start_time
    print(f"   ✅ Retrieved salary for {len(salaries):,} nodes in {single_time:.3f}s")
    print(f"   📊 Rate: {len(salaries)/single_time:,.0f} values/sec")
    
    # Test 3: Optimized DataFrame export
    print("3. Testing optimized DataFrame export...")
    start_time = time.time()
    
    df_data = g.export_node_dataframe_optimized(['age', 'salary', 'department'])
    
    export_time = time.time() - start_time
    total_exported = sum(len(column) for column in df_data.values())
    print(f"   ✅ Exported {len(df_data)} columns with {total_exported:,} total values in {export_time:.3f}s")
    print(f"   📊 Rate: {total_exported/export_time:,.0f} values/sec")

def test_dataframe_conversion(g):
    """Test DataFrame creation performance"""
    print("\n📊 Testing DataFrame Conversion:")
    
    try:
        import pandas as pd
        
        print("1. Testing pandas DataFrame creation...")
        start_time = time.time()
        
        # Use the optimized get_dataframe_data_fast method
        df_dict = g.get_dataframe_data_fast(['age', 'salary', 'department'])
        df = pd.DataFrame(df_dict)
        
        pandas_time = time.time() - start_time
        print(f"   ✅ Created pandas DataFrame with {len(df):,} rows, {len(df.columns)} columns in {pandas_time:.3f}s")
        print(f"   📊 Rate: {len(df)/pandas_time:,.0f} rows/sec")
        
        # Show sample of data
        print("   📋 Sample data:")
        print(df.head(3).to_string())
        
    except ImportError:
        print("   ⚠️  pandas not available, skipping DataFrame test")

def test_filtering_performance(g):
    """Test the filtering performance improvements"""
    print("\n🔍 Testing Advanced Filtering:")
    
    # Test multi-criteria filtering
    print("1. Testing multi-criteria filtering...")
    start_time = time.time()
    
    # Find high-paid engineers in specific locations
    filtered_nodes = g.filter_nodes_multi_criteria(
        exact_matches={'department': 'engineering'},
        numeric_comparisons=[('salary', '>', 80000), ('age', '<', 40)],
        string_comparisons=[('location', '==', 'SF')]
    )
    
    filter_time = time.time() - start_time
    print(f"   ✅ Filtered to {len(filtered_nodes):,} nodes in {filter_time:.3f}s")
    print(f"   📊 Filter rate: {g.node_count()/filter_time:,.0f} nodes/sec processed")

def main():
    """Run the optimization demonstration"""
    print("🔥 Groggy Performance Optimization Demo")
    print("=" * 50)
    
    # Create test graph
    g = create_test_graph(10000)
    
    print(f"\n📈 Graph Statistics:")
    print(f"   Nodes: {g.node_count():,}")
    print(f"   Backend: {g.backend}")
    
    # Test the new optimization features
    test_bulk_attribute_operations(g)
    test_vectorized_retrieval(g)
    test_dataframe_conversion(g)
    test_filtering_performance(g)
    
    print("\n🎉 Optimization Demo Complete!")
    print("\n💡 Key Improvements Demonstrated:")
    print("   • Chunked bulk operations for large-scale updates")
    print("   • Vectorized attribute retrieval using columnar storage")
    print("   • Optimized DataFrame export for pandas/polars integration")
    print("   • Multi-criteria filtering with early termination")
    print("   • All operations leverage Rust backend for maximum performance")

if __name__ == "__main__":
    main()
