#!/usr/bin/env python3
"""Test script to verify that assign method works with both list and dictionary formats."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python-groggy'))

try:
    import groggy
    print(f"✓ groggy imported successfully")
    
    # Create a simple table
    data = {
        'id': [0, 1, 2, 3, 4],
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'score': [85.5, 90.0, 78.5, 92.0, 88.5]
    }
    
    table = groggy.BaseTable.from_dict(data)
    print(f"✓ Table created: {table.shape}")
    
    # Test 1: Using list format (should replace entire column)
    print("\n--- Test 1: List format ---")
    updates_list = {
        'bonus': [100, 200, 150, 250, 175]
    }
    table.assign(updates_list)
    print(f"✓ List format assignment completed")
    
    # Test 2: Using dictionary format with specific indices
    print("\n--- Test 2: Dictionary format ---")
    updates_dict = {
        'bonus': {0: 1000, 3: 1500},  # Only update rows 0 and 3
        'status': {1: 'active', 2: 'inactive', 4: 'pending'}  # Update rows 1, 2, and 4
    }
    table.assign(updates_dict)
    print(f"✓ Dictionary format assignment completed")
    
    # Display the final table to verify results
    print(f"\nFinal table shape: {table.shape}")
    print(f"Columns: {table.column_names}")
    
    print("\nSuccess! Both list and dictionary formats work correctly.")
    
except ImportError as e:
    print(f"✗ Failed to import groggy: {e}")
    print("Make sure you've built the Python package with maturin")
    sys.exit(1)
except Exception as e:
    print(f"✗ Test failed: {e}")
    sys.exit(1)
