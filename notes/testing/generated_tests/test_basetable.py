#!/usr/bin/env python3
"""
Comprehensive test script for Groggy BaseTable
Generated on: 2025-09-07 21:42:37

This script tests ALL methods of the BaseTable class with proper argument patterns.
Edit the TODO sections to provide correct arguments for each method.
"""

import sys
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python-groggy/python')
import groggy as gr
import traceback
from datetime import datetime

def create_test_objects():
    """Create test objects for BaseTable testing"""
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
    
    # Create the specific test object for BaseTable
    test_obj = g.nodes.table().base_table()
    
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
    """Test BaseTable.__getitem__(key, /)"""
    # Arguments for __getitem__(key, /)
    try:
        if hasattr(test_obj, '__getitem__'):
            method = getattr(test_obj, '__getitem__')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method('salary')
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
    """Test BaseTable.__iter__()"""
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
    """Test BaseTable.__len__()"""
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
    """Test BaseTable.__repr__()"""
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
    """Test BaseTable.__str__()"""
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

def test_column_names(test_obj, nodes_data, edges_data):
    """Test BaseTable.column_names()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'column_names'):
            method = getattr(test_obj, 'column_names')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ column_names() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'column_names' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ column_names() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ column_names() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ column_names not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ column_names() → Error: {str(e)}")
        return False, str(e)

def test_drop_columns(test_obj, nodes_data, edges_data):
    """Test BaseTable.drop_columns(columns)"""
    # Arguments for drop_columns(columns)
    try:
        if hasattr(test_obj, 'drop_columns'):
            method = getattr(test_obj, 'drop_columns')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method(['salary'])
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ drop_columns() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'drop_columns' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ drop_columns() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ drop_columns() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ drop_columns not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ drop_columns() → Error: {str(e)}")
        return False, str(e)

def test_filter(test_obj, nodes_data, edges_data):
    """Test BaseTable.filter(predicate)"""
    # Arguments for filter(predicate)
    try:
        if hasattr(test_obj, 'filter'):
            method = getattr(test_obj, 'filter')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method(lambda x: x.get('salary', None) is not None)
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ filter() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'filter' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ filter() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ filter() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ filter not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ filter() → Error: {str(e)}")
        return False, str(e)

def test_group_by(test_obj, nodes_data, edges_data):
    """Test BaseTable.group_by(columns)"""
    # Arguments for group_by(columns)
    try:
        if hasattr(test_obj, 'group_by'):
            method = getattr(test_obj, 'group_by')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method(['salary'])
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ group_by() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'group_by' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ group_by() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ group_by() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ group_by not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ group_by() → Error: {str(e)}")
        return False, str(e)

def test_has_column(test_obj, nodes_data, edges_data):
    """Test BaseTable.has_column(name)"""
    # Arguments for has_column(name)
    try:
        if hasattr(test_obj, 'has_column'):
            method = getattr(test_obj, 'has_column')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method('salary')
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ has_column() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'has_column' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ has_column() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ has_column() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ has_column not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ has_column() → Error: {str(e)}")
        return False, str(e)

def test_head(test_obj, nodes_data, edges_data):
    """Test BaseTable.head(n=5)"""
    # Arguments for head(n=5)
    try:
        if hasattr(test_obj, 'head'):
            method = getattr(test_obj, 'head')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method(5)
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ head() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'head' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ head() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ head() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ head not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ head() → Error: {str(e)}")
        return False, str(e)

def test_iter(test_obj, nodes_data, edges_data):
    """Test BaseTable.iter()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'iter'):
            method = getattr(test_obj, 'iter')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ iter() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'iter' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ iter() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ iter() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ iter not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ iter() → Error: {str(e)}")
        return False, str(e)

def test_ncols(test_obj, nodes_data, edges_data):
    """Test BaseTable.ncols()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'ncols'):
            method = getattr(test_obj, 'ncols')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ ncols() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'ncols' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ ncols() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ ncols() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ ncols not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ ncols() → Error: {str(e)}")
        return False, str(e)

def test_nrows(test_obj, nodes_data, edges_data):
    """Test BaseTable.nrows()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'nrows'):
            method = getattr(test_obj, 'nrows')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ nrows() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'nrows' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ nrows() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ nrows() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ nrows not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ nrows() → Error: {str(e)}")
        return False, str(e)

def test_rich_display(test_obj, nodes_data, edges_data):
    """Test BaseTable.rich_display(config=None)"""
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

def test_select(test_obj, nodes_data, edges_data):
    """Test BaseTable.select(columns)"""
    # Arguments for select(columns)
    try:
        if hasattr(test_obj, 'select'):
            method = getattr(test_obj, 'select')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method(['salary'])
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ select() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'select' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ select() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ select() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ select not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ select() → Error: {str(e)}")
        return False, str(e)

def test_shape(test_obj, nodes_data, edges_data):
    """Test BaseTable.shape()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'shape'):
            method = getattr(test_obj, 'shape')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ shape() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'shape' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ shape() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ shape() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ shape not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ shape() → Error: {str(e)}")
        return False, str(e)

def test_slice(test_obj, nodes_data, edges_data):
    """Test BaseTable.slice(start, end)"""
    # Arguments for slice(start, end)
    try:
        if hasattr(test_obj, 'slice'):
            method = getattr(test_obj, 'slice')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method(0, 5)
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ slice() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'slice' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ slice() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ slice() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ slice not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ slice() → Error: {str(e)}")
        return False, str(e)

def test_sort_by(test_obj, nodes_data, edges_data):
    """Test BaseTable.sort_by(column, ascending=True)"""
    # Arguments for sort_by(column, ascending=True)
    try:
        if hasattr(test_obj, 'sort_by'):
            method = getattr(test_obj, 'sort_by')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method('salary')
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ sort_by() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'sort_by' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ sort_by() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ sort_by() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ sort_by not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ sort_by() → Error: {str(e)}")
        return False, str(e)

def test_tail(test_obj, nodes_data, edges_data):
    """Test BaseTable.tail(n=5)"""
    # Arguments for tail(n=5)
    try:
        if hasattr(test_obj, 'tail'):
            method = getattr(test_obj, 'tail')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method(5)
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ tail() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'tail' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ tail() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ tail() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ tail not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ tail() → Error: {str(e)}")
        return False, str(e)

def test_to_pandas(test_obj, nodes_data, edges_data):
    """Test BaseTable.to_pandas()"""
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

def run_all_tests():
    """Run all BaseTable method tests"""
    print(f"# BaseTable Comprehensive Test Suite")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Testing 21 methods\n")
    
    # Create test objects
    test_obj, nodes_data, edges_data = create_test_objects()
    
    if test_obj is None:
        print("❌ Failed to create test object")
        return
    
    results = []
    working_count = 0
    total_count = 0
    
    print(f"## Testing BaseTable Methods\n")
    
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
    
    # Test column_names
    success, result = test_column_names(test_obj, nodes_data, edges_data)
    results.append({'method': 'column_names', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test drop_columns
    success, result = test_drop_columns(test_obj, nodes_data, edges_data)
    results.append({'method': 'drop_columns', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test filter
    success, result = test_filter(test_obj, nodes_data, edges_data)
    results.append({'method': 'filter', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test group_by
    success, result = test_group_by(test_obj, nodes_data, edges_data)
    results.append({'method': 'group_by', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test has_column
    success, result = test_has_column(test_obj, nodes_data, edges_data)
    results.append({'method': 'has_column', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test head
    success, result = test_head(test_obj, nodes_data, edges_data)
    results.append({'method': 'head', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test iter
    success, result = test_iter(test_obj, nodes_data, edges_data)
    results.append({'method': 'iter', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test ncols
    success, result = test_ncols(test_obj, nodes_data, edges_data)
    results.append({'method': 'ncols', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test nrows
    success, result = test_nrows(test_obj, nodes_data, edges_data)
    results.append({'method': 'nrows', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test rich_display
    success, result = test_rich_display(test_obj, nodes_data, edges_data)
    results.append({'method': 'rich_display', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test select
    success, result = test_select(test_obj, nodes_data, edges_data)
    results.append({'method': 'select', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test shape
    success, result = test_shape(test_obj, nodes_data, edges_data)
    results.append({'method': 'shape', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test slice
    success, result = test_slice(test_obj, nodes_data, edges_data)
    results.append({'method': 'slice', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test sort_by
    success, result = test_sort_by(test_obj, nodes_data, edges_data)
    results.append({'method': 'sort_by', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test tail
    success, result = test_tail(test_obj, nodes_data, edges_data)
    results.append({'method': 'tail', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test to_pandas
    success, result = test_to_pandas(test_obj, nodes_data, edges_data)
    results.append({'method': 'to_pandas', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    
    # Print summary
    print(f"\n# BaseTable Test Summary")
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
