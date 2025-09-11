#!/usr/bin/env python3

import groggy

def test_group_by_agg_functionality():
    """Test group_by().agg() functionality following PHASE_1_2_USAGE_EXAMPLE patterns."""
    
    # Create a more comprehensive test table similar to the example
    data = {
        'user_id': [1, 2, 3, 4, 5, 6],
        'department': ['Engineering', 'Sales', 'Engineering', 'Sales', 'Marketing', 'Engineering'],
        'salary': [95000, 65000, 88000, 72000, 58000, 102000],
        'age': [28, 35, 31, 29, 42, 33],
        'years_exp': [3, 8, 5, 4, 15, 7]
    }
    
    table = groggy.BaseTable.from_dict(data)
    print("Original employee table:")
    print(f"Shape: {table.shape}")
    print(f"Columns: {table.column_names}")
    
    # Test basic group_by (returns TableArray)
    print("\n=== Testing basic group_by (returns TableArray) ===")
    grouped = table.group_by(['department'])
    
    print(f"Type of grouped result: {type(grouped)}")
    print(f"Is it a TableArray? {type(grouped).__name__ == 'TableArray'}")
    print(f"Number of groups: {len(grouped)}")
    
    # Test group_by().agg() pattern
    print("\n=== Testing group_by().agg() functionality ===")
    try:
        dept_stats = grouped.agg({
            'salary': 'avg',
            'age': 'mean', 
            'years_exp': 'avg',
            'user_id': 'count'
        })
        
        print("✓ Aggregation succeeded!")
        print(f"Result type: {type(dept_stats)}")
        print(f"Result shape: {dept_stats.shape}")
        print(f"Result columns: {dept_stats.column_names}")
        
    except Exception as e:
        print(f"✗ Aggregation failed: {e}")
    
    # Test individual group examination
    print("\n=== Examining individual groups ===")
    for i, group in enumerate(grouped):
        print(f"Group {i}:")
        print(f"  Shape: {group.shape}")
        print(f"  Columns: {group.column_names}")
        # Try to get department name from first row if possible
        try:
            dept_col = group['department']
            if len(dept_col) > 0:
                dept_name = list(dept_col)[0]  # Get first department name
                print(f"  Department: {dept_name}")
        except:
            pass

def test_assign_both_formats():
    """Test both list and dictionary formats for assign method."""
    
    print("\n" + "="*50)
    print("TESTING UPDATED ASSIGN METHOD")
    print("="*50)
    
    # Create a test table
    data = {
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve']
    }
    
    table = groggy.BaseTable.from_dict(data)
    print("Original table:")
    print(f"Shape: {table.shape}")
    
    # Test 1: List format (existing functionality)
    print("\n=== Test 1: List format ===")
    updates_list = {
        'bonus': [1000, 1500, 2000, 2500, 3000],
        'active': [True, False, True, True, False]
    }
    
    try:
        table.assign(updates_list)
        print("✓ List format assign succeeded")
        print(f"New shape: {table.shape}")
        print(f"Columns: {table.column_names}")
    except Exception as e:
        print(f"✗ List format assign failed: {e}")
    
    # Test 2: Dictionary format (new functionality)
    print("\n=== Test 2: Dictionary format (sparse updates) ===")
    updates_dict = {
        'score': {0: 95, 2: 87, 4: 92},  # Only update rows 0, 2, 4
        'status': {1: 'premium', 3: 'premium'}  # Only update rows 1, 3
    }
    
    try:
        table.assign(updates_dict)
        print("✓ Dictionary format assign succeeded")
        print(f"New shape: {table.shape}")
        print(f"Columns: {table.column_names}")
        
        # Check the 'score' column 
        score_col = table['score']
        print(f"Score column: {list(score_col)}")
        
        # Check the 'status' column
        status_col = table['status'] 
        print(f"Status column: {list(status_col)}")
        
    except Exception as e:
        print(f"✗ Dictionary format assign failed: {e}")

if __name__ == "__main__":
    test_group_by_agg_functionality()
    test_assign_both_formats()
