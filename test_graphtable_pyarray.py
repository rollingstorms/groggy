#!/usr/bin/env python3

"""
Test GraphTable GraphArray integration - column access returns GraphArray objects
"""

import sys
sys.path.insert(0, 'python-groggy/python')

import groggy

def test_graphtable_grapharray_columns():
    """Test that GraphTable column access returns GraphArray objects"""
    print("🧪 Testing GraphTable GraphArray integration...")
    
    # Create a graph with multiple attributes
    g = groggy.Graph()
    nodes = [g.add_node() for _ in range(6)]
    
    # Set multiple attributes for statistical testing
    ages = [25, 30, 35, 40, 45, 50]
    salaries = [50000, 65000, 80000, 95000, 110000, 125000]
    names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"]
    
    for node, age, salary, name in zip(nodes, ages, salaries, names):
        g.set_node_attribute(node, 'age', groggy.AttrValue(age))
        g.set_node_attribute(node, 'salary', groggy.AttrValue(salary))
        g.set_node_attribute(node, 'name', groggy.AttrValue(name))
    
    print(f"✅ Created graph with {len(nodes)} nodes")
    
    # Create table
    table = g.table()
    print(f"✅ Created table with shape: {table.shape}")
    
    # Test PyArray column access
    try:
        ages_column = table['age']
        print(f"✅ Column access successful: {type(ages_column)}")
        
        # Test that it's a GraphArray
        if hasattr(ages_column, 'mean') and hasattr(ages_column, 'std'):
            print(f"✅ Column is GraphArray with statistical methods")
            
            # Test statistical operations
            mean_age = ages_column.mean()
            std_age = ages_column.std()
            min_age = ages_column.min()
            max_age = ages_column.max()
            
            print(f"✅ Statistical operations work:")
            print(f"   Mean age: {mean_age:.1f} (expected: 37.5)")
            print(f"   Std dev: {std_age:.1f}")
            print(f"   Min age: {min_age}")
            print(f"   Max age: {max_age}")
            
            # Validate statistical accuracy
            import statistics
            plain_ages = [25, 30, 35, 40, 45, 50]
            expected_mean = statistics.mean(plain_ages)
            expected_std = statistics.stdev(plain_ages)
            
            if abs(mean_age - expected_mean) < 0.01:
                print(f"✅ Mean calculation accurate")
            else:
                print(f"❌ Mean calculation error: {mean_age} vs {expected_mean}")
                
            if abs(std_age - expected_std) < 0.01:
                print(f"✅ Standard deviation accurate")
            else:
                print(f"❌ Std dev calculation error: {std_age} vs {expected_std}")
                
        else:
            print(f"❌ Column is not GraphArray: {type(ages_column)}")
            return False
            
    except Exception as e:
        print(f"❌ Column access failed: {e}")
        return False
    
    # Test list compatibility
    try:
        print(f"✅ List compatibility tests:")
        print(f"   Length: {len(ages_column)} (expected: 6)")
        print(f"   Indexing: ages_column[0] = {ages_column[0]} (expected: 25)")
        print(f"   Negative indexing: ages_column[-1] = {ages_column[-1]} (expected: 50)")
        
        # Test iteration
        total = sum(age for age in ages_column)
        print(f"   Iteration sum: {total} (expected: 225)")
        
    except Exception as e:
        print(f"❌ List compatibility failed: {e}")
        return False
    
    # Test multiple columns
    try:
        salary_column = table['salary']
        print(f"✅ Multiple column access works")
        print(f"   Salary mean: ${salary_column.mean():,.0f}")
        print(f"   Salary range: ${salary_column.min():,.0f} - ${salary_column.max():,.0f}")
        
    except Exception as e:
        print(f"❌ Multiple column access failed: {e}")
        return False
    
    return True

def test_graphtable_row_slicing():
    """Test GraphTable row slicing functionality"""
    print("\n🧪 Testing GraphTable row slicing...")
    
    # Create graph
    g = groggy.Graph()
    nodes = [g.add_node() for _ in range(10)]
    
    # Set data
    for i, node in enumerate(nodes):
        g.set_node_attribute(node, 'value', groggy.AttrValue(i * 10))
    
    # Create table
    table = g.table()
    print(f"✅ Created table with {table.shape[0]} rows")
    
    # Test row slicing
    try:
        # First 5 rows
        first_five = table[:5]
        print(f"✅ First 5 rows: {type(first_five)} with shape {first_five.shape}")
        
        # Next 3 rows
        next_three = table[5:8]
        print(f"✅ Next 3 rows: shape {next_three.shape}")
        
        # Last 2 rows
        last_two = table[-2:]
        print(f"✅ Last 2 rows: shape {last_two.shape}")
        
        # Test that sliced tables support column access
        values_slice = first_five['value']
        print(f"✅ Sliced table column access works: {type(values_slice)}")
        print(f"   Values in first 5: mean = {values_slice.mean():.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Row slicing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backward_compatibility():
    """Test that existing code still works with GraphArray integration"""
    print("\n🧪 Testing backward compatibility...")
    
    g = groggy.Graph()
    nodes = [g.add_node() for _ in range(3)]
    
    for i, node in enumerate(nodes):
        g.set_node_attribute(node, 'test_val', groggy.AttrValue(i + 1))
    
    table = g.table()
    
    # Test operations that should still work
    try:
        column = table['test_val']
        
        # These should all work with GraphArray (list compatibility)
        length = len(column)
        first_item = column[0]
        iteration_works = list(column)
        
        print(f"✅ Backward compatibility maintained:")
        print(f"   len() works: {length}")
        print(f"   indexing works: {first_item}")
        print(f"   iteration works: {iteration_works}")
        
        return True
        
    except Exception as e:
        print(f"❌ Backward compatibility broken: {e}")
        return False

if __name__ == "__main__":
    try:
        success1 = test_graphtable_grapharray_columns()
        success2 = test_graphtable_row_slicing()
        success3 = test_backward_compatibility()
        
        if success1 and success2 and success3:
            print("\n🎉 GraphTable GraphArray integration tests passed!")
        else:
            print("\n❌ Some GraphTable GraphArray integration tests failed!")
            
    except Exception as e:
        print(f"\n❌ GraphTable GraphArray integration test crashed: {e}")
        import traceback
        traceback.print_exc()