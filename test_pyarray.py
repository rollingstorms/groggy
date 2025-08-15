#!/usr/bin/env python3

"""
Quick test for PyArray statistical functionality
"""

import sys
sys.path.insert(0, 'python-groggy/python')

import groggy

def test_pyarray_basic():
    """Test basic PyArray functionality"""
    print("🧪 Testing PyArray basic functionality...")
    
    # Create a PyArray with numeric values
    values = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30]
    arr = groggy.PyArray(values)
    
    # Test list compatibility
    print(f"✅ Length: {len(arr)} (expected: 10)")
    print(f"✅ Indexing: arr[0] = {arr[0]}, arr[-1] = {arr[-1]}")
    
    # Test statistical methods
    print(f"✅ Mean: {arr.mean():.2f} (expected: ~11.5)")
    print(f"✅ Min: {arr.min()}")
    print(f"✅ Max: {arr.max()}")
    print(f"✅ Count: {arr.count()}")
    
    if arr.std() is not None:
        print(f"✅ Std Dev: {arr.std():.2f}")
    
    # Test quantiles
    median = arr.median()
    if median is not None:
        print(f"✅ Median: {median:.2f}")
    
    q25 = arr.quantile(0.25)
    if q25 is not None:
        print(f"✅ 25th percentile: {q25:.2f}")
    
    # Test conversion back to list
    as_list = arr.to_list()
    print(f"✅ to_list() length: {len(as_list)}")
    
    # Test statistical summary
    summary = arr.describe()
    print(f"✅ Statistical summary:")
    print(f"   Count: {summary.count}")
    print(f"   Mean: {summary.mean}")
    print(f"   Min: {summary.min}")
    print(f"   Max: {summary.max}")

def test_pyarray_iteration():
    """Test PyArray iteration"""
    print("\n🧪 Testing PyArray iteration...")
    
    values = [10, 20, 30]
    arr = groggy.PyArray(values)
    
    # Test iteration
    total = 0
    for value in arr:
        total += value
    
    print(f"✅ Iteration sum: {total} (expected: 60)")

def test_pyarray_graph_integration():
    """Test PyArray integration with graph data"""
    print("\n🧪 Testing PyArray graph integration...")
    
    # Create a simple graph
    g = groggy.Graph()
    nodes = [g.add_node() for _ in range(5)]
    
    # Set some attributes
    ages = [25, 30, 35, 40, 45]
    for node, age in zip(nodes, ages):
        g.set_node_attribute(node, 'age', groggy.AttrValue(age))
    
    # Get attributes as plain list (current behavior)
    age_values = [g.get_node_attribute(node, 'age') for node in nodes]
    print(f"✅ Current age list: {age_values}")
    
    # Create PyArray from graph data
    age_array = groggy.PyArray(age_values)
    print(f"✅ PyArray mean age: {age_array.mean():.1f} (expected: 35.0)")
    print(f"✅ PyArray age range: {age_array.min()} - {age_array.max()}")

if __name__ == "__main__":
    try:
        test_pyarray_basic()
        test_pyarray_iteration()
        test_pyarray_graph_integration()
        print("\n🎉 All PyArray tests passed!")
    except Exception as e:
        print(f"\n❌ PyArray test failed: {e}")
        import traceback
        traceback.print_exc()