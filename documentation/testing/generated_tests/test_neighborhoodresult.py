#!/usr/bin/env python3
"""
Comprehensive test script for Groggy NeighborhoodResult
Generated on: 2025-09-07 21:42:37

This script tests ALL methods of the NeighborhoodResult class with proper argument patterns.
Edit the TODO sections to provide correct arguments for each method.
"""

import sys
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python-groggy/python')
import groggy as gr
import traceback
from datetime import datetime

def create_test_objects():
    """Create test objects for NeighborhoodResult testing"""
    print("🏗️ Creating test objects...")
    
    # Core graph with rich data
    g = gr.Graph()
    nodes_data = g.add_nodes([
        {'name': 'Alice', 'age': 25, 'salary': 75000, 'active': True, 'team': 'Engineering', 'level': 'Senior'},
        {'name': 'Bob', 'age': 30, 'salary': 85000, 'active': True, 'team': 'Sales', 'level': 'Manager'},
        {'name': 'Charlie', 'age': 35, 'salary': 95000, 'active': False, 'team': 'Marketing', 'level': 'Director'},
        {'name': 'Diana', 'age': 28, 'salary': 80000, 'active': True, 'team': 'Engineering', 'level': 'Senior'},
        {'name': 'Eve', 'age': 32, 'salary': 90000, 'active': True, 'team': 'Product', 'level': 'Manager'},
    ])
    
    edges_data = g.add_edges([
        (nodes_data[0], nodes_data[1], {'weight': 1.5, 'type': 'collaboration', 'strength': 'strong'}),
        (nodes_data[1], nodes_data[2], {'weight': 2.0, 'type': 'reports_to', 'strength': 'formal'}),
        (nodes_data[2], nodes_data[3], {'weight': 0.8, 'type': 'peer', 'strength': 'weak'}),
        (nodes_data[0], nodes_data[3], {'weight': 1.2, 'type': 'collaboration', 'strength': 'medium'}),
        (nodes_data[1], nodes_data[4], {'weight': 1.8, 'type': 'cross_team', 'strength': 'strong'}),
    ])
    
    # Create the specific test object for NeighborhoodResult
    test_obj = g.neighborhood([nodes_data[0]], radius=2)
    
    return test_obj, nodes_data, edges_data

def test_method(obj, method_name, method_func, nodes_data, edges_data):
    """Test a single method with error handling"""
    print(f"Testing {method_name}...")
    
    try:
        # Call the method - EDIT THE ARGUMENTS AS NEEDED
        if method_name == 'PLACEHOLDER_METHOD':
            # Example: result = method_func(arg1, arg2, kwarg1=value)
            result = method_func()
        else:
            # Default call with no arguments
            result = method_func()
        
        print(f"  ✅ {method_name}() → {type(result).__name__}: {result}")
        return True, result
        
    except Exception as e:
        print(f"  ❌ {method_name}() → Error: {str(e)}")
        return False, str(e)


def test___getitem__(test_obj, nodes_data, edges_data):
    """Test NeighborhoodResult.__getitem__(key, /)"""
    # Arguments for __getitem__(key, /)
    try:
        if hasattr(test_obj, '__getitem__'):
            method = getattr(test_obj, '__getitem__')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method(0)
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ __getitem__() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if '__getitem__' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ __getitem__() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ __getitem__() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ __getitem__ not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ __getitem__() → Error: {str(e)}")
        return False, str(e)

def test___iter__(test_obj, nodes_data, edges_data):
    """Test NeighborhoodResult.__iter__()"""
    # No arguments needed
    try:
        if hasattr(test_obj, '__iter__'):
            method = getattr(test_obj, '__iter__')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ __iter__() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if '__iter__' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ __iter__() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ __iter__() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ __iter__ not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ __iter__() → Error: {str(e)}")
        return False, str(e)

def test___len__(test_obj, nodes_data, edges_data):
    """Test NeighborhoodResult.__len__()"""
    # No arguments needed
    try:
        if hasattr(test_obj, '__len__'):
            method = getattr(test_obj, '__len__')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ __len__() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if '__len__' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ __len__() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ __len__() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ __len__ not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ __len__() → Error: {str(e)}")
        return False, str(e)

def test___repr__(test_obj, nodes_data, edges_data):
    """Test NeighborhoodResult.__repr__()"""
    # No arguments needed
    try:
        if hasattr(test_obj, '__repr__'):
            method = getattr(test_obj, '__repr__')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ __repr__() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if '__repr__' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ __repr__() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ __repr__() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ __repr__ not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ __repr__() → Error: {str(e)}")
        return False, str(e)

def test___str__(test_obj, nodes_data, edges_data):
    """Test NeighborhoodResult.__str__()"""
    # No arguments needed
    try:
        if hasattr(test_obj, '__str__'):
            method = getattr(test_obj, '__str__')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ __str__() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if '__str__' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ __str__() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ __str__() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ __str__ not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ __str__() → Error: {str(e)}")
        return False, str(e)

def test_execution_time_ms(test_obj, nodes_data, edges_data):
    """Test NeighborhoodResult.execution_time_ms(property)"""
    # Error getting signature: 0.0 is not a callable object
    try:
        if hasattr(test_obj, 'execution_time_ms'):
            method = getattr(test_obj, 'execution_time_ms')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ execution_time_ms() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'execution_time_ms' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ execution_time_ms() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ execution_time_ms() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ execution_time_ms not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ execution_time_ms() → Error: {str(e)}")
        return False, str(e)

def test_largest_neighborhood_size(test_obj, nodes_data, edges_data):
    """Test NeighborhoodResult.largest_neighborhood_size(property)"""
    # Error getting signature: 5 is not a callable object
    try:
        if hasattr(test_obj, 'largest_neighborhood_size'):
            method = getattr(test_obj, 'largest_neighborhood_size')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ largest_neighborhood_size() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'largest_neighborhood_size' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ largest_neighborhood_size() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ largest_neighborhood_size() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ largest_neighborhood_size not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ largest_neighborhood_size() → Error: {str(e)}")
        return False, str(e)

def test_neighborhoods(test_obj, nodes_data, edges_data):
    """Test NeighborhoodResult.neighborhoods(property)"""
    # Error getting signature: [NeighborhoodSubgraph(central_nodes=[0], hops=2, nodes=5, edges=5)] is not a callable object
    try:
        if hasattr(test_obj, 'neighborhoods'):
            method = getattr(test_obj, 'neighborhoods')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ neighborhoods() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'neighborhoods' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ neighborhoods() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ neighborhoods() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ neighborhoods not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ neighborhoods() → Error: {str(e)}")
        return False, str(e)

def test_total_neighborhoods(test_obj, nodes_data, edges_data):
    """Test NeighborhoodResult.total_neighborhoods(property)"""
    # Error getting signature: 1 is not a callable object
    try:
        if hasattr(test_obj, 'total_neighborhoods'):
            method = getattr(test_obj, 'total_neighborhoods')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ total_neighborhoods() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'total_neighborhoods' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ total_neighborhoods() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ total_neighborhoods() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ total_neighborhoods not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ total_neighborhoods() → Error: {str(e)}")
        return False, str(e)

def run_all_tests():
    """Run all NeighborhoodResult method tests"""
    print(f"# NeighborhoodResult Comprehensive Test Suite")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Testing 9 methods\n")
    
    # Create test objects
    test_obj, nodes_data, edges_data = create_test_objects()
    
    if test_obj is None:
        print("❌ Failed to create test object")
        return
    
    results = []
    working_count = 0
    total_count = 0
    
    print(f"## Testing NeighborhoodResult Methods\n")
    
    # Run all method tests
    # Test __getitem__
    success, result = test___getitem__(test_obj, nodes_data, edges_data)
    results.append({'method': '__getitem__', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test __iter__
    success, result = test___iter__(test_obj, nodes_data, edges_data)
    results.append({'method': '__iter__', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test __len__
    success, result = test___len__(test_obj, nodes_data, edges_data)
    results.append({'method': '__len__', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test __repr__
    success, result = test___repr__(test_obj, nodes_data, edges_data)
    results.append({'method': '__repr__', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test __str__
    success, result = test___str__(test_obj, nodes_data, edges_data)
    results.append({'method': '__str__', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test execution_time_ms
    success, result = test_execution_time_ms(test_obj, nodes_data, edges_data)
    results.append({'method': 'execution_time_ms', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test largest_neighborhood_size
    success, result = test_largest_neighborhood_size(test_obj, nodes_data, edges_data)
    results.append({'method': 'largest_neighborhood_size', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test neighborhoods
    success, result = test_neighborhoods(test_obj, nodes_data, edges_data)
    results.append({'method': 'neighborhoods', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test total_neighborhoods
    success, result = test_total_neighborhoods(test_obj, nodes_data, edges_data)
    results.append({'method': 'total_neighborhoods', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    
    # Print summary
    print(f"\n# NeighborhoodResult Test Summary")
    print(f"**Results**: {working_count}/{total_count} methods working ({working_count/total_count*100:.1f}%)")
    
    # Show working methods
    working = [r for r in results if r['success']]
    failing = [r for r in results if not r['success']]
    
    print(f"\n**Working Methods ({len(working)}):**")
    for r in working:  # Show all
        print(f"  ✅ {r['method']}")
    
    print(f"\n**Failing Methods ({len(failing)}):**")  
    for r in failing:  # Show all
        print(f"  ❌ {r['method']}: {r['result']}")
    
    return results

if __name__ == "__main__":
    results = run_all_tests()
