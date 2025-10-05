#!/usr/bin/env python3
"""Detailed examination of describe() output structure"""

import sys
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python-groggy/python')

import groggy as gr

def examine_describe_structure():
    """Examine the structure and content of describe() output"""
    print("=== Examining describe() output structure ===")

    data = [
        {"age": 30, "salary": 50000.0, "name": "Alice", "height": 165.5},
        {"age": 25, "salary": 45000.0, "name": "Bob", "height": 175.2},
        {"age": 35, "salary": 60000.0, "name": "Carol", "height": 160.8},
        {"age": 28, "salary": 52000.0, "name": "David", "height": 180.1},
        {"age": 32, "salary": 55000.0, "name": "Eve", "height": 168.3},
    ]

    table = gr.table(data)
    print(f"Original table:\n{table}\n")

    # Get the description
    description = table.describe()
    print(f"Description table:\n{description}\n")

    # Try to examine the structure in detail
    print("=== Detailed examination ===")

    # Check if we can access the description as a regular table
    try:
        print(f"Description type: {type(description)}")
        print(f"Description class: {description.__class__}")

        # Try to get column names
        if hasattr(description, 'columns'):
            print(f"Description columns: {description.columns}")

        # Try to get individual columns
        try:
            age_desc = description['age']
            print(f"Age column description: {age_desc}")
            print(f"Age column type: {type(age_desc)}")
        except Exception as e:
            print(f"Cannot access age column: {e}")

        try:
            salary_desc = description['salary']
            print(f"Salary column description: {salary_desc}")
        except Exception as e:
            print(f"Cannot access salary column: {e}")

        try:
            height_desc = description['height']
            print(f"Height column description: {height_desc}")
        except Exception as e:
            print(f"Cannot access height column: {e}")

        # Try to iterate over the description table
        print("\n--- Attempting to examine structure ---")
        try:
            # Check if we can convert to pandas for easier inspection
            if hasattr(description, 'to_pandas'):
                df = description.to_pandas()
                print(f"Description as pandas DataFrame:\n{df}")
        except Exception as e:
            print(f"Cannot convert to pandas: {e}")

    except Exception as e:
        print(f"Error examining description structure: {e}")

    return description

def compare_with_manual_calculations():
    """Compare describe() output with manual statistical calculations"""
    print("\n=== Comparing with manual calculations ===")

    data = [
        {"values": 10, "amounts": 100.0},
        {"values": 20, "amounts": 200.0},
        {"values": 30, "amounts": 300.0},
        {"values": 40, "amounts": 400.0},
        {"values": 50, "amounts": 500.0},
    ]

    table = gr.table(data)
    values_col = table['values']

    # Manual calculations
    manual_sum = sum([10, 20, 30, 40, 50])
    manual_mean = manual_sum / 5
    manual_min = min([10, 20, 30, 40, 50])
    manual_max = max([10, 20, 30, 40, 50])
    manual_count = 5

    print(f"Manual calculations for values column:")
    print(f"  Sum: {manual_sum}")
    print(f"  Mean: {manual_mean}")
    print(f"  Min: {manual_min}")
    print(f"  Max: {manual_max}")
    print(f"  Count: {manual_count}")

    # Groggy calculations
    groggy_sum = values_col.sum()
    groggy_mean = values_col.mean()
    groggy_min = values_col.min()
    groggy_max = values_col.max()
    groggy_count = values_col.count()

    print(f"\nGroggy calculations for values column:")
    print(f"  Sum: {groggy_sum}")
    print(f"  Mean: {groggy_mean}")
    print(f"  Min: {groggy_min}")
    print(f"  Max: {groggy_max}")
    print(f"  Count: {groggy_count}")

    # Check if they match
    print(f"\nComparison:")
    print(f"  Sum matches: {manual_sum == float(groggy_sum)}")
    print(f"  Mean matches: {manual_mean == float(groggy_mean)}")
    print(f"  Min matches: {manual_min == float(groggy_min)}")
    print(f"  Max matches: {manual_max == float(groggy_max)}")
    print(f"  Count matches: {manual_count == int(groggy_count)}")

    # Get describe output
    description = table.describe()
    print(f"\nDescribe output:\n{description}")

def main():
    """Run detailed examination"""
    print("Detailed examination of table.describe() functionality\n")

    description = examine_describe_structure()
    compare_with_manual_calculations()

    print("\n✓ Detailed examination completed")

if __name__ == "__main__":
    main()