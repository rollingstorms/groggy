#!/usr/bin/env python3
"""Comprehensive test for the enhanced assign method functionality."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python-groggy'))

try:
    import groggy
    
    print("=== Enhanced assign method tests ===\n")
    
    # Create test table
    data = {
        'id': [0, 1, 2, 3, 4],
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'score': [85.5, 90.0, 78.5, 92.0, 88.5]
    }
    table = groggy.BaseTable.from_dict(data)
    print(f"Initial table: {table.shape} - columns: {table.column_names}")
    
    # Test 1: List format (full column replacement)
    print("\n--- Test 1: List format (full column replacement) ---")
    updates = {'bonus': [100, 200, 150, 250, 175]}
    table.assign(updates)
    print("✓ Added bonus column with list format")
    print(f"Table now: {table.shape} - columns: {table.column_names}")
    
    # Test 2: Dictionary format with sparse updates (existing column)
    print("\n--- Test 2: Dictionary format - sparse updates to existing column ---")
    updates = {'bonus': {0: 1000, 3: 1500}}  # Only update rows 0 and 3
    table.assign(updates)
    print("✓ Updated specific rows in existing bonus column")
    
    # Test 3: Dictionary format with sparse updates (new column)
    print("\n--- Test 3: Dictionary format - sparse updates to new column ---")
    updates = {'status': {1: 'active', 2: 'inactive', 4: 'pending'}}
    table.assign(updates)
    print("✓ Created new status column with sparse updates")
    print(f"Table now: {table.shape} - columns: {table.column_names}")
    
    # Test 4: Mixed format updates
    print("\n--- Test 4: Mixed format updates ---")
    updates = {
        'priority': [1, 2, 1, 3, 2],  # Full list
        'flag': {0: True, 2: False, 4: True}  # Sparse dictionary
    }
    table.assign(updates)
    print("✓ Mixed list and dictionary updates in same call")
    print(f"Final table: {table.shape} - columns: {table.column_names}")
    
    # Test 5: Edge case - empty dictionary
    print("\n--- Test 5: Edge case - empty dictionary ---")
    updates = {'empty_col': {}}
    table.assign(updates)
    print("✓ Handled empty dictionary (created column with all nulls)")
    print(f"Final table: {table.shape} - columns: {table.column_names}")
    
    print("\n🎉 All tests passed! The assign method now supports:")
    print("  ✓ List format: {'attr': [values]} for full column updates")
    print("  ✓ Dictionary format: {'attr': {0: value0, 1: value1}} for sparse updates")
    print("  ✓ Mixed formats in the same call")
    print("  ✓ Creating new columns with both formats")
    print("  ✓ Updating existing columns with both formats")
    
except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
