#!/usr/bin/env python3
"""
Quick demonstration of Groggy's optimization features
Tests the new vectorized and bulk methods on a smaller graph
"""

import time
import random
from groggy import Graph

def create_small_test_graph():
    """Create a small test graph quickly"""
    print("Creating small test graph...")
    
    g = Graph(backend='rust')
    
    # Add just 100 nodes quickly to test the optimizations
    for i in range(100):
        node_id = f"user_{i}"
        attrs = {
            'age': random.randint(18, 80),
            'salary': random.randint(30000, 150000),
            'department': random.choice(['engineering', 'sales', 'marketing']),
            'location': random.choice(['NYC', 'SF', 'LA']),
        }
        g.add_node(node_id, **attrs)
    
    print(f"✅ Created graph with {g.node_count()} nodes")
    return g

def test_new_optimization_methods(g):
    """Test the newly added optimization methods"""
    print("\n🚀 Testing New Optimization Methods:")
    
    # Test 1: Check if the new methods are available
    print("1. Checking available optimization methods...")
    
    methods_to_test = [
        'get_bulk_node_attribute_vectors',
        'get_single_attribute_vectorized', 
        'export_node_dataframe_optimized',
        'set_node_attributes_chunked'
    ]
    
    available_methods = []
    for method in methods_to_test:
        if hasattr(g._rust_core, method):
            available_methods.append(method)
            print(f"   ✅ {method} - Available")
        else:
            print(f"   ❌ {method} - Not available")
    
    # Test 2: Test bulk vectors retrieval if available
    if 'get_bulk_node_attribute_vectors' in available_methods:
        print("\n2. Testing bulk attribute vectors...")
        try:
            start_time = time.time()
            attr_names = ['age', 'salary', 'department']
            vectors_data = g._rust_core.get_bulk_node_attribute_vectors(attr_names, None)
            vectors_time = time.time() - start_time
            
            print(f"   ✅ Retrieved {len(attr_names)} attributes in {vectors_time:.4f}s")
            print(f"   📊 Data structure: {type(vectors_data)}")
            for attr_name in vectors_data.keys():
                data = vectors_data[attr_name]
                print(f"      {attr_name}: {len(data[0])} indices, {len(data[1])} values")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Test 3: Test single attribute vectorized if available
    if 'get_single_attribute_vectorized' in available_methods:
        print("\n3. Testing single attribute vectorized...")
        try:
            start_time = time.time()
            node_ids, salaries = g._rust_core.get_single_attribute_vectorized('salary', None)
            single_time = time.time() - start_time
            
            print(f"   ✅ Retrieved salary for {len(salaries)} nodes in {single_time:.4f}s")
            print(f"   📊 Sample values: {salaries[:3] if len(salaries) >= 3 else salaries}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Test 4: Test DataFrame export if available
    if 'export_node_dataframe_optimized' in available_methods:
        print("\n4. Testing optimized DataFrame export...")
        try:
            start_time = time.time()
            df_data = g._rust_core.export_node_dataframe_optimized(['age', 'salary'], None)
            export_time = time.time() - start_time
            
            print(f"   ✅ Exported DataFrame data in {export_time:.4f}s")
            print(f"   📊 Columns: {list(df_data.keys())}")
            for col_name, col_data in df_data.items():
                print(f"      {col_name}: {len(col_data)} values")
        except Exception as e:
            print(f"   ❌ Error: {e}")

def test_existing_optimized_methods(g):
    """Test the existing optimized methods that should work"""
    print("\n⚡ Testing Existing Optimized Methods:")
    
    # Test 1: get_dataframe_data_fast
    print("1. Testing get_dataframe_data_fast...")
    try:
        start_time = time.time()
        df_data = g.get_dataframe_data_fast(['age', 'salary'])
        df_time = time.time() - start_time
        
        print(f"   ✅ Retrieved DataFrame data in {df_time:.4f}s")
        print(f"   📊 Columns: {list(df_data.keys())}")
        for col_name, col_data in df_data.items():
            print(f"      {col_name}: {len(col_data)} values")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: get_attribute_column  
    print("\n2. Testing get_attribute_column...")
    try:
        start_time = time.time()
        node_indices, values = g.get_attribute_column('salary')
        col_time = time.time() - start_time
        
        print(f"   ✅ Retrieved attribute column in {col_time:.4f}s")
        print(f"   📊 Got {len(values)} values")
        print(f"   📊 Sample values: {values[:3] if len(values) >= 3 else values}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

def test_filtering_methods(g):
    """Test filtering performance"""
    print("\n🔍 Testing Filtering Methods:")
    
    # Test 1: Multi-criteria filtering
    print("1. Testing multi-criteria filtering...")
    try:
        start_time = time.time()
        filtered_nodes = g.filter_nodes_multi_criteria(
            exact_matches={'department': 'engineering'},
            numeric_comparisons=[('salary', '>', 50000)],
            string_comparisons=[]
        )
        filter_time = time.time() - start_time
        
        print(f"   ✅ Filtered to {len(filtered_nodes)} nodes in {filter_time:.4f}s")
    except Exception as e:
        print(f"   ❌ Error: {e}")

def main():
    """Run the quick optimization test"""
    print("🔥 Groggy Optimization Quick Test")
    print("=" * 40)
    
    # Create small test graph
    g = create_small_test_graph()
    
    # Test the new optimization methods
    test_new_optimization_methods(g)
    
    # Test existing optimized methods
    test_existing_optimized_methods(g)
    
    # Test filtering
    test_filtering_methods(g)
    
    print("\n🎉 Quick Test Complete!")
    print("\n💡 This test validates that the optimization methods are properly exposed and working.")

if __name__ == "__main__":
    main()
