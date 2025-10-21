#!/usr/bin/env python3
"""
Comprehensive test script for Groggy GraphMatrix
Generated on: 2025-09-07 21:42:37

This script tests ALL methods of the GraphMatrix class with proper argument patterns.
Edit the TODO sections to provide correct arguments for each method.
"""

import sys
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python-groggy/python')
import groggy as gr
import traceback
from datetime import datetime

def create_test_objects():
    """Create test objects for GraphMatrix testing"""
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
    
    # Create the specific test object for GraphMatrix
    test_obj = g.adjacency_matrix()
    
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
    """Test GraphMatrix.__getitem__(...)"""
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
    """Test GraphMatrix.__iter__()"""
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
    """Test GraphMatrix.__len__()"""
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
    """Test GraphMatrix.__repr__()"""
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
    """Test GraphMatrix.__str__()"""
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

def test_clear(test_obj, nodes_data, edges_data):
    """Test GraphMatrix.clear(...)"""
    # Error getting signature: no signature found for builtin <built-in method clear of dict object at 0x1051e7880>
    try:
        if hasattr(test_obj, 'clear'):
            method = getattr(test_obj, 'clear')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ clear() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'clear' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ clear() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ clear() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ clear not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ clear() → Error: {str(e)}")
        return False, str(e)

def test_copy(test_obj, nodes_data, edges_data):
    """Test GraphMatrix.copy(...)"""
    # Error getting signature: no signature found for builtin <built-in method copy of dict object at 0x1051e7880>
    try:
        if hasattr(test_obj, 'copy'):
            method = getattr(test_obj, 'copy')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ copy() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'copy' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ copy() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ copy() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ copy not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ copy() → Error: {str(e)}")
        return False, str(e)

def test_fromkeys(test_obj, nodes_data, edges_data):
    """Test GraphMatrix.fromkeys(iterable, value=None, /)"""
    # Arguments for fromkeys(iterable, value=None, /)
    try:
        if hasattr(test_obj, 'fromkeys'):
            method = getattr(test_obj, 'fromkeys')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method(['key1', 'key2'], 'default_value')
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ fromkeys() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'fromkeys' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ fromkeys() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ fromkeys() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ fromkeys not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ fromkeys() → Error: {str(e)}")
        return False, str(e)

def test_get(test_obj, nodes_data, edges_data):
    """Test GraphMatrix.get(key, default=None, /)"""
    # Arguments for get(key, default=None, /)
    try:
        if hasattr(test_obj, 'get'):
            method = getattr(test_obj, 'get')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method('key')
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ get() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'get' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ get() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ get() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ get not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ get() → Error: {str(e)}")
        return False, str(e)

def test_items(test_obj, nodes_data, edges_data):
    """Test GraphMatrix.items(...)"""
    # Error getting signature: no signature found for builtin <built-in method items of dict object at 0x1051e7880>
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

def test_keys(test_obj, nodes_data, edges_data):
    """Test GraphMatrix.keys(...)"""
    # Error getting signature: no signature found for builtin <built-in method keys of dict object at 0x1051e7880>
    try:
        if hasattr(test_obj, 'keys'):
            method = getattr(test_obj, 'keys')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ keys() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'keys' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ keys() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ keys() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ keys not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ keys() → Error: {str(e)}")
        return False, str(e)

def test_pop(test_obj, nodes_data, edges_data):
    """Test GraphMatrix.pop(...)"""
    # Error getting signature: <built-in method pop of dict object at 0x1051e7880> builtin has invalid signature
    try:
        if hasattr(test_obj, 'pop'):
            method = getattr(test_obj, 'pop')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ pop() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'pop' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ pop() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ pop() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ pop not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ pop() → Error: {str(e)}")
        return False, str(e)

def test_popitem(test_obj, nodes_data, edges_data):
    """Test GraphMatrix.popitem()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'popitem'):
            method = getattr(test_obj, 'popitem')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ popitem() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'popitem' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ popitem() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ popitem() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ popitem not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ popitem() → Error: {str(e)}")
        return False, str(e)

def test_setdefault(test_obj, nodes_data, edges_data):
    """Test GraphMatrix.setdefault(key, default=None, /)"""
    # Arguments for setdefault(key, default=None, /)
    try:
        if hasattr(test_obj, 'setdefault'):
            method = getattr(test_obj, 'setdefault')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method('new_key', 'default_value')
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ setdefault() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'setdefault' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ setdefault() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ setdefault() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ setdefault not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ setdefault() → Error: {str(e)}")
        return False, str(e)

def test_update(test_obj, nodes_data, edges_data):
    """Test GraphMatrix.update(...)"""
    # Error getting signature: no signature found for builtin <built-in method update of dict object at 0x1051e7880>
    try:
        if hasattr(test_obj, 'update'):
            method = getattr(test_obj, 'update')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ update() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'update' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ update() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ update() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ update not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ update() → Error: {str(e)}")
        return False, str(e)

def test_values(test_obj, nodes_data, edges_data):
    """Test GraphMatrix.values(...)"""
    # Error getting signature: no signature found for builtin <built-in method values of dict object at 0x1051e7880>
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
    """Run all GraphMatrix method tests"""
    print(f"# GraphMatrix Comprehensive Test Suite")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Testing 16 methods\n")
    
    # Create test objects
    test_obj, nodes_data, edges_data = create_test_objects()
    
    if test_obj is None:
        print("❌ Failed to create test object")
        return
    
    results = []
    working_count = 0
    total_count = 0
    
    print(f"## Testing GraphMatrix Methods\n")
    
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
    
    # Test clear
    success, result = test_clear(test_obj, nodes_data, edges_data)
    results.append({'method': 'clear', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test copy
    success, result = test_copy(test_obj, nodes_data, edges_data)
    results.append({'method': 'copy', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test fromkeys
    success, result = test_fromkeys(test_obj, nodes_data, edges_data)
    results.append({'method': 'fromkeys', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test get
    success, result = test_get(test_obj, nodes_data, edges_data)
    results.append({'method': 'get', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test items
    success, result = test_items(test_obj, nodes_data, edges_data)
    results.append({'method': 'items', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test keys
    success, result = test_keys(test_obj, nodes_data, edges_data)
    results.append({'method': 'keys', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test pop
    success, result = test_pop(test_obj, nodes_data, edges_data)
    results.append({'method': 'pop', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test popitem
    success, result = test_popitem(test_obj, nodes_data, edges_data)
    results.append({'method': 'popitem', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test setdefault
    success, result = test_setdefault(test_obj, nodes_data, edges_data)
    results.append({'method': 'setdefault', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test update
    success, result = test_update(test_obj, nodes_data, edges_data)
    results.append({'method': 'update', 'success': success, 'result': result})
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
    print(f"\n# GraphMatrix Test Summary")
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
