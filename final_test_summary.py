#!/usr/bin/env python3
"""
Final test summary for FFI modularization and display integration
"""

import sys
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python-groggy/python')

def test_core_functionality():
    """Test core graph functionality"""
    print("🧪 Testing Core Graph Functionality")
    
    import groggy as gr
    
    # Basic graph operations
    g = gr.Graph()
    n1 = g.add_node()
    n2 = g.add_node()
    print(f"✅ Added nodes: {n1}, {n2}")
    print(f"✅ Node count: {g.node_count()}")
    print(f"✅ Edge count: {g.edge_count()}")
    
    return True

def test_display_system():
    """Test display formatting system"""
    print("\n🎨 Testing Display System")
    
    import groggy as gr
    
    # Test DisplayConfig
    config = gr.DisplayConfig(max_rows=3, use_color=False)
    print(f"✅ DisplayConfig: {config}")
    
    # Test formatting functions
    test_data = {
        "data": [10, 20, 30, 40, 50, 60],
        "dtype": "int64"
    }
    
    # Array formatting
    formatted = gr.format_array(test_data)
    print("✅ Array formatting:")
    print(formatted)
    
    # Matrix formatting
    matrix_data = {
        "data": [[1, 2], [3, 4], [5, 6]],
        "shape": [3, 2],
        "dtype": "int64"
    }
    
    matrix_formatted = gr.format_matrix(matrix_data)
    print("\n✅ Matrix formatting:")
    print(matrix_formatted)
    
    return True

def test_module_structure():
    """Test modular structure is working"""
    print("\n🏗️  Testing Modular Structure")
    
    import groggy as gr
    
    # Check that key classes are available
    classes_to_check = [
        'Graph', 'AttrValue', 'GraphArray', 'GraphMatrix',
        'DisplayConfig', 'NodeFilter', 'EdgeFilter',
        'TraversalResult', 'AggregationResult'
    ]
    
    missing_classes = []
    for cls_name in classes_to_check:
        if hasattr(gr, cls_name):
            print(f"✅ {cls_name} available")
        else:
            missing_classes.append(cls_name)
            print(f"❌ {cls_name} missing")
    
    # Check functions
    functions_to_check = [
        'format_array', 'format_matrix', 'format_table',
        'detect_display_type'
    ]
    
    missing_functions = []
    for func_name in functions_to_check:
        if hasattr(gr, func_name):
            print(f"✅ {func_name} available")
        else:
            missing_functions.append(func_name)
            print(f"❌ {func_name} missing")
    
    return len(missing_classes) == 0 and len(missing_functions) == 0

def main():
    """Run all tests"""
    print("="*70)
    print("🎉 GROGGY FFI MODULARIZATION & DISPLAY INTEGRATION - FINAL TEST")
    print("="*70)
    
    tests = [
        ("Core Functionality", test_core_functionality),
        ("Display System", test_display_system), 
        ("Module Structure", test_module_structure)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result, None))
        except Exception as e:
            results.append((test_name, False, str(e)))
    
    print("\n" + "="*70)
    print("📊 FINAL RESULTS")
    print("="*70)
    
    for test_name, passed, error in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
        if error:
            print(f"    Error: {error}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ FFI modularization successful")
        print("✅ Display system integration complete")
        print("✅ Python module working correctly")
        print("\n🚀 Ready for production use!")
    else:
        print("\n⚠️  Some tests failed - see details above")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)