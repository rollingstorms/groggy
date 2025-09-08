#!/usr/bin/env python3
"""
Comprehensive test script for Groggy GraphArray
Generated on: 2025-09-07 21:42:37

This script tests ALL methods of the GraphArray class with proper argument patterns.
Edit the TODO sections to provide correct arguments for each method.
"""

import sys
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python-groggy/python')
import groggy as gr
import traceback
from datetime import datetime

def create_test_objects():
    """Create test objects for GraphArray testing"""
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
    
    # Create the specific test object for GraphArray
    test_obj = g.nodes.table()['node_id']
    
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
    """Test GraphArray.__getitem__(key, /)"""
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
    """Test GraphArray.__iter__()"""
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
    """Test GraphArray.__len__()"""
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
    """Test GraphArray.__repr__()"""
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
    """Test GraphArray.__str__()"""
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

def test_count(test_obj, nodes_data, edges_data):
    """Test GraphArray.count()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'count'):
            method = getattr(test_obj, 'count')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ count() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'count' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ count() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ count() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ count not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ count() → Error: {str(e)}")
        return False, str(e)

def test_describe(test_obj, nodes_data, edges_data):
    """Test GraphArray.describe()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'describe'):
            method = getattr(test_obj, 'describe')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ describe() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'describe' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ describe() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ describe() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ describe not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ describe() → Error: {str(e)}")
        return False, str(e)

def test_drop_na(test_obj, nodes_data, edges_data):
    """Test GraphArray.drop_na()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'drop_na'):
            method = getattr(test_obj, 'drop_na')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ drop_na() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'drop_na' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ drop_na() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ drop_na() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ drop_na not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ drop_na() → Error: {str(e)}")
        return False, str(e)

def test_fill_na(test_obj, nodes_data, edges_data):
    """Test GraphArray.fill_na(fill_value)"""
    # Arguments for fill_na(fill_value)
    try:
        if hasattr(test_obj, 'fill_na'):
            method = getattr(test_obj, 'fill_na')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method(0)
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ fill_na() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'fill_na' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ fill_na() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ fill_na() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ fill_na not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ fill_na() → Error: {str(e)}")
        return False, str(e)

def test_has_null(test_obj, nodes_data, edges_data):
    """Test GraphArray.has_null()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'has_null'):
            method = getattr(test_obj, 'has_null')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ has_null() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'has_null' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ has_null() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ has_null() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ has_null not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ has_null() → Error: {str(e)}")
        return False, str(e)

def test_is_sparse(test_obj, nodes_data, edges_data):
    """Test GraphArray.is_sparse(property)"""
    # Error getting signature: False is not a callable object
    try:
        if hasattr(test_obj, 'is_sparse'):
            method = getattr(test_obj, 'is_sparse')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ is_sparse() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'is_sparse' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ is_sparse() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ is_sparse() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ is_sparse not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ is_sparse() → Error: {str(e)}")
        return False, str(e)

def test_items(test_obj, nodes_data, edges_data):
    """Test GraphArray.items()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'items'):
            method = getattr(test_obj, 'items')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ items() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'items' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ items() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ items() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ items not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ items() → Error: {str(e)}")
        return False, str(e)

def test_max(test_obj, nodes_data, edges_data):
    """Test GraphArray.max()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'max'):
            method = getattr(test_obj, 'max')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ max() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'max' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ max() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ max() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ max not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ max() → Error: {str(e)}")
        return False, str(e)

def test_mean(test_obj, nodes_data, edges_data):
    """Test GraphArray.mean()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'mean'):
            method = getattr(test_obj, 'mean')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ mean() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'mean' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ mean() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ mean() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ mean not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ mean() → Error: {str(e)}")
        return False, str(e)

def test_median(test_obj, nodes_data, edges_data):
    """Test GraphArray.median()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'median'):
            method = getattr(test_obj, 'median')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ median() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'median' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ median() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ median() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ median not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ median() → Error: {str(e)}")
        return False, str(e)

def test_min(test_obj, nodes_data, edges_data):
    """Test GraphArray.min()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'min'):
            method = getattr(test_obj, 'min')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ min() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'min' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ min() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ min() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ min not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ min() → Error: {str(e)}")
        return False, str(e)

def test_null_count(test_obj, nodes_data, edges_data):
    """Test GraphArray.null_count()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'null_count'):
            method = getattr(test_obj, 'null_count')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ null_count() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'null_count' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ null_count() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ null_count() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ null_count not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ null_count() → Error: {str(e)}")
        return False, str(e)

def test_percentile(test_obj, nodes_data, edges_data):
    """Test GraphArray.percentile(p)"""
    # Arguments for percentile(p)
    try:
        if hasattr(test_obj, 'percentile'):
            method = getattr(test_obj, 'percentile')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method(0.5)
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ percentile() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'percentile' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ percentile() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ percentile() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ percentile not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ percentile() → Error: {str(e)}")
        return False, str(e)

def test_preview(test_obj, nodes_data, edges_data):
    """Test GraphArray.preview(limit=None)"""
    # Arguments for preview(limit=None)
    try:
        if hasattr(test_obj, 'preview'):
            method = getattr(test_obj, 'preview')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ preview() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'preview' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ preview() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ preview() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ preview not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ preview() → Error: {str(e)}")
        return False, str(e)

def test_quantile(test_obj, nodes_data, edges_data):
    """Test GraphArray.quantile(q)"""
    # Arguments for quantile(q)
    try:
        if hasattr(test_obj, 'quantile'):
            method = getattr(test_obj, 'quantile')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method(0.5)
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ quantile() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'quantile' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ quantile() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ quantile() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ quantile not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ quantile() → Error: {str(e)}")
        return False, str(e)

def test_rich_display(test_obj, nodes_data, edges_data):
    """Test GraphArray.rich_display(config=None)"""
    # Arguments for rich_display(config=None)
    try:
        if hasattr(test_obj, 'rich_display'):
            method = getattr(test_obj, 'rich_display')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ rich_display() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'rich_display' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ rich_display() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ rich_display() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ rich_display not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ rich_display() → Error: {str(e)}")
        return False, str(e)

def test_std(test_obj, nodes_data, edges_data):
    """Test GraphArray.std()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'std'):
            method = getattr(test_obj, 'std')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ std() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'std' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ std() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ std() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ std not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ std() → Error: {str(e)}")
        return False, str(e)

def test_summary(test_obj, nodes_data, edges_data):
    """Test GraphArray.summary()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'summary'):
            method = getattr(test_obj, 'summary')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ summary() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'summary' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ summary() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ summary() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ summary not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ summary() → Error: {str(e)}")
        return False, str(e)

def test_to_list(test_obj, nodes_data, edges_data):
    """Test GraphArray.to_list()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'to_list'):
            method = getattr(test_obj, 'to_list')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ to_list() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'to_list' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ to_list() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ to_list() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ to_list not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ to_list() → Error: {str(e)}")
        return False, str(e)

def test_to_numpy(test_obj, nodes_data, edges_data):
    """Test GraphArray.to_numpy()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'to_numpy'):
            method = getattr(test_obj, 'to_numpy')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ to_numpy() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'to_numpy' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ to_numpy() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ to_numpy() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ to_numpy not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ to_numpy() → Error: {str(e)}")
        return False, str(e)

def test_to_pandas(test_obj, nodes_data, edges_data):
    """Test GraphArray.to_pandas()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'to_pandas'):
            method = getattr(test_obj, 'to_pandas')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ to_pandas() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'to_pandas' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ to_pandas() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ to_pandas() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ to_pandas not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ to_pandas() → Error: {str(e)}")
        return False, str(e)

def test_to_scipy_sparse(test_obj, nodes_data, edges_data):
    """Test GraphArray.to_scipy_sparse()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'to_scipy_sparse'):
            method = getattr(test_obj, 'to_scipy_sparse')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ to_scipy_sparse() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'to_scipy_sparse' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ to_scipy_sparse() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ to_scipy_sparse() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ to_scipy_sparse not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ to_scipy_sparse() → Error: {str(e)}")
        return False, str(e)

def test_true_indices(test_obj, nodes_data, edges_data):
    """Test GraphArray.true_indices()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'true_indices'):
            method = getattr(test_obj, 'true_indices')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ true_indices() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'true_indices' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ true_indices() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ true_indices() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ true_indices not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ true_indices() → Error: {str(e)}")
        return False, str(e)

def test_unique(test_obj, nodes_data, edges_data):
    """Test GraphArray.unique()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'unique'):
            method = getattr(test_obj, 'unique')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ unique() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'unique' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ unique() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ unique() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ unique not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ unique() → Error: {str(e)}")
        return False, str(e)

def test_value_counts(test_obj, nodes_data, edges_data):
    """Test GraphArray.value_counts()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'value_counts'):
            method = getattr(test_obj, 'value_counts')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ value_counts() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'value_counts' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ value_counts() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ value_counts() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ value_counts not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ value_counts() → Error: {str(e)}")
        return False, str(e)

def test_values(test_obj, nodes_data, edges_data):
    """Test GraphArray.values(property)"""
    # Error getting signature: [3, 1, 0, 4, 2] is not a callable object
    try:
        if hasattr(test_obj, 'values'):
            method = getattr(test_obj, 'values')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ values() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'values' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ values() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ values() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ values not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ values() → Error: {str(e)}")
        return False, str(e)

def run_all_tests():
    """Run all GraphArray method tests"""
    print(f"# GraphArray Comprehensive Test Suite")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Testing 31 methods\n")
    
    # Create test objects
    test_obj, nodes_data, edges_data = create_test_objects()
    
    if test_obj is None:
        print("❌ Failed to create test object")
        return
    
    results = []
    working_count = 0
    total_count = 0
    
    print(f"## Testing GraphArray Methods\n")
    
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
    
    # Test count
    success, result = test_count(test_obj, nodes_data, edges_data)
    results.append({'method': 'count', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test describe
    success, result = test_describe(test_obj, nodes_data, edges_data)
    results.append({'method': 'describe', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test drop_na
    success, result = test_drop_na(test_obj, nodes_data, edges_data)
    results.append({'method': 'drop_na', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test fill_na
    success, result = test_fill_na(test_obj, nodes_data, edges_data)
    results.append({'method': 'fill_na', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test has_null
    success, result = test_has_null(test_obj, nodes_data, edges_data)
    results.append({'method': 'has_null', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test is_sparse
    success, result = test_is_sparse(test_obj, nodes_data, edges_data)
    results.append({'method': 'is_sparse', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test items
    success, result = test_items(test_obj, nodes_data, edges_data)
    results.append({'method': 'items', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test max
    success, result = test_max(test_obj, nodes_data, edges_data)
    results.append({'method': 'max', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test mean
    success, result = test_mean(test_obj, nodes_data, edges_data)
    results.append({'method': 'mean', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test median
    success, result = test_median(test_obj, nodes_data, edges_data)
    results.append({'method': 'median', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test min
    success, result = test_min(test_obj, nodes_data, edges_data)
    results.append({'method': 'min', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test null_count
    success, result = test_null_count(test_obj, nodes_data, edges_data)
    results.append({'method': 'null_count', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test percentile
    success, result = test_percentile(test_obj, nodes_data, edges_data)
    results.append({'method': 'percentile', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test preview
    success, result = test_preview(test_obj, nodes_data, edges_data)
    results.append({'method': 'preview', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test quantile
    success, result = test_quantile(test_obj, nodes_data, edges_data)
    results.append({'method': 'quantile', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test rich_display
    success, result = test_rich_display(test_obj, nodes_data, edges_data)
    results.append({'method': 'rich_display', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test std
    success, result = test_std(test_obj, nodes_data, edges_data)
    results.append({'method': 'std', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test summary
    success, result = test_summary(test_obj, nodes_data, edges_data)
    results.append({'method': 'summary', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test to_list
    success, result = test_to_list(test_obj, nodes_data, edges_data)
    results.append({'method': 'to_list', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test to_numpy
    success, result = test_to_numpy(test_obj, nodes_data, edges_data)
    results.append({'method': 'to_numpy', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test to_pandas
    success, result = test_to_pandas(test_obj, nodes_data, edges_data)
    results.append({'method': 'to_pandas', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test to_scipy_sparse
    success, result = test_to_scipy_sparse(test_obj, nodes_data, edges_data)
    results.append({'method': 'to_scipy_sparse', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test true_indices
    success, result = test_true_indices(test_obj, nodes_data, edges_data)
    results.append({'method': 'true_indices', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test unique
    success, result = test_unique(test_obj, nodes_data, edges_data)
    results.append({'method': 'unique', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test value_counts
    success, result = test_value_counts(test_obj, nodes_data, edges_data)
    results.append({'method': 'value_counts', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test values
    success, result = test_values(test_obj, nodes_data, edges_data)
    results.append({'method': 'values', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    
    # Print summary
    print(f"\n# GraphArray Test Summary")
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
