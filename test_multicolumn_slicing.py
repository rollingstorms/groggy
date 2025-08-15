#!/usr/bin/env python3

"""
Test multi-column slicing functionality: g.nodes[:n][['age', 'height']]
"""

import sys
sys.path.insert(0, 'python-groggy/python')

import groggy

def test_multicolumn_slicing():
    """Test multi-column slicing on subgraphs"""
    print("🧪 Testing multi-column slicing functionality...")
    
    # Create a graph with multiple attributes
    g = groggy.Graph()
    nodes = [g.add_node() for _ in range(5)]
    
    # Set multiple attributes
    ages = [25, 30, 35, 40, 45]
    heights = [170, 165, 180, 175, 160]  # cm
    names = ["Alice", "Bob", "Charlie", "Diana", "Eve"]
    
    for node, age, height, name in zip(nodes, ages, heights, names):
        g.set_node_attribute(node, 'age', groggy.AttrValue(age))
        g.set_node_attribute(node, 'height', groggy.AttrValue(height)) 
        g.set_node_attribute(node, 'name', groggy.AttrValue(name))
    
    print(f"✅ Created graph with {len(nodes)} nodes")
    
    # Test single column access (existing behavior)
    subgraph = g.nodes[:3]  # First 3 nodes
    ages_column = subgraph['age']
    print(f"✅ Single column access: ages = {ages_column}")
    
    # Test multi-column access (new feature)
    try:
        multi_columns = subgraph[['age', 'height']]
        print(f"✅ Multi-column access successful!")
        print(f"   Type: {type(multi_columns)}")
        print(f"   Content: {multi_columns}")
        
        # Check that it returns a 2D structure
        if isinstance(multi_columns, list) and len(multi_columns) == 2:
            print(f"✅ Correct 2D structure: {len(multi_columns)} columns")
            print(f"   Ages column length: {len(multi_columns[0])}")
            print(f"   Heights column length: {len(multi_columns[1])}")
        else:
            print(f"⚠️ Unexpected structure: {multi_columns}")
            
    except Exception as e:
        print(f"❌ Multi-column access failed: {e}")
        return False
    
    # Test single-item list (should behave like single string)
    try:
        single_in_list = subgraph[['age']]
        print(f"✅ Single item in list: {single_in_list}")
    except Exception as e:
        print(f"❌ Single item in list failed: {e}")
        return False
    
    # Test empty list (should fail gracefully)
    try:
        empty_list = subgraph[[]]
        print(f"❌ Empty list should have failed but returned: {empty_list}")
        return False
    except Exception as e:
        print(f"✅ Empty list correctly failed: {e}")
    
    # Test invalid key type
    try:
        invalid_key = subgraph[123]
        print(f"❌ Invalid key should have failed but returned: {invalid_key}")
        return False
    except Exception as e:
        print(f"✅ Invalid key correctly failed: {e}")

def test_full_workflow():
    """Test the full g.nodes[:n][['age', 'height']] workflow"""
    print("\n🧪 Testing full g.nodes[:n][['age', 'height']] workflow...")
    
    # Create graph
    g = groggy.Graph()
    nodes = [g.add_node() for _ in range(6)]
    
    # Set data
    for i, node in enumerate(nodes):
        g.set_node_attribute(node, 'age', groggy.AttrValue(20 + i * 5))
        g.set_node_attribute(node, 'height', groggy.AttrValue(160 + i * 3))
    
    # Test the exact syntax you requested
    try:
        # g.nodes[:n][['age', 'height']] - get first 4 nodes, 2 attributes
        result = g.nodes[:4][['age', 'height']]
        print(f"✅ g.nodes[:4][['age', 'height']] successful!")
        print(f"   Result type: {type(result)}")
        print(f"   Shape: {len(result)} columns x {len(result[0])} rows")
        print(f"   Ages: {result[0]}")
        print(f"   Heights: {result[1]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Full workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success1 = test_multicolumn_slicing()
        success2 = test_full_workflow()
        
        if success1 and success2:
            print("\n🎉 Multi-column slicing tests passed!")
        else:
            print("\n❌ Some multi-column slicing tests failed!")
            
    except Exception as e:
        print(f"\n❌ Multi-column slicing test crashed: {e}")
        import traceback
        traceback.print_exc()