#!/usr/bin/env python3
"""
Test double bracket notation on group_by: g.nodes.group_by('col')[['val']]
Should return TableArray with one column extracted from each group
"""

import groggy as gr

g = gr.Graph()

# Create nodes with departments
for i in range(6):
    dept = 'Eng' if i < 3 else 'Sales'
    g.add_node(name=f'Person{i}', department=dept, salary=float(100 + i*10))

print(f"Created graph with {g.node_count()} nodes")

# Group by department
dept_groups = g.nodes.group_by('department')
print(f"\nGrouped by department: {len(dept_groups)} groups")
print(f"Type: {type(dept_groups)}")

# Test 1: Single bracket (should return ArrayArray)
print("\n--- Test 1: Single bracket ['salary'] ---")
try:
    salary_arrays = dept_groups['salary']
    print(f"Result type: {type(salary_arrays)}")
    print(f"Result: {salary_arrays}")
    print("✓ Single bracket works")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 2: Double bracket (should return TableArray with salary column)
print("\n--- Test 2: Double bracket [['salary']] ---")
try:
    salary_tables = dept_groups[['salary']]
    print(f"Result type: {type(salary_tables)}")
    print(f"Result: {salary_tables}")
    print("✓ Double bracket works")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Multiple columns with double bracket
print("\n--- Test 3: Multiple columns [['name', 'salary']] ---")
try:
    multi_col_tables = dept_groups[['name', 'salary']]
    print(f"Result type: {type(multi_col_tables)}")
    print(f"Result: {multi_col_tables}")
    print("✓ Multiple column extraction works")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
