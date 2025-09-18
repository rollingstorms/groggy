#!/usr/bin/env python3
"""
Basic Viz Import Validation

Simple test to check if viz module imports are working.
Run this first to check basic integration:
    python test_basic_viz_import.py
"""

import sys

def test_basic_imports():
    """Test basic groggy and viz imports."""
    print("🔍 Testing basic imports...")
    
    try:
        import groggy as gr
        print("✓ groggy imported successfully")
        
        # Check if viz module is available
        if hasattr(gr, 'viz'):
            print("✓ gr.viz module is available")
        else:
            print("✗ gr.viz module is missing")
            return False
        
        # Check if VizConfig is available
        if hasattr(gr, 'VizConfig'):
            print("✓ gr.VizConfig is available")
        else:
            print("✗ gr.VizConfig is missing")
            return False
        
        # Check if VizModule is available
        if hasattr(gr, 'VizModule'):
            print("✓ gr.VizModule is available")
        else:
            print("✗ gr.VizModule is missing") 
            return False
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def test_graph_viz_accessor():
    """Test if Graph has .viz accessor."""
    print("\n🔍 Testing Graph.viz accessor...")
    
    try:
        import groggy as gr
        
        # Create simple graph
        g = gr.Graph()
        
        # Check if viz accessor exists
        if hasattr(g, 'viz'):
            print("✓ Graph has .viz accessor")
            
            # Check if viz() method works and returns accessor with expected methods
            try:
                viz = g.viz()  # Call as method
                methods = ['interactive', 'static', 'info']
                for method in methods:
                    if hasattr(viz, method):
                        print(f"✓ viz.{method}() available")
                    else:
                        print(f"✗ viz.{method}() missing")
                        return False
                
                return True
            except Exception as e:
                print(f"✗ g.viz() method call failed: {e}")
                return False
        else:
            print("✗ Graph.viz accessor is missing")
            return False
            
    except Exception as e:
        print(f"✗ Graph viz accessor test failed: {e}")
        return False


def test_module_structure():
    """Test the overall module structure."""
    print("\n🔍 Testing module structure...")
    
    try:
        import groggy as gr
        
        # Test if we can create VizConfig
        try:
            config = gr.VizConfig()
            print("✓ VizConfig() can be instantiated")
        except Exception as e:
            print(f"✗ VizConfig() failed: {e}")
            return False
        
        # Test if we can access viz module functions
        try:
            if hasattr(gr.viz, 'interactive'):
                print("✓ gr.viz.interactive() function available")
            else:
                print("⚠️  gr.viz.interactive() function not available")
        except Exception as e:
            print(f"⚠️  Error accessing gr.viz functions: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Module structure test failed: {e}")
        return False


def main():
    """Run basic validation tests."""
    print("🚀 Basic Viz Import Validation")
    print("=" * 40)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Graph Viz Accessor", test_graph_viz_accessor),
        ("Module Structure", test_module_structure),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 BASIC IMPORT RESULTS")
    print("=" * 40)
    
    passed = 0
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:8} {test_name}")
        if success:
            passed += 1
    
    total = len(results)
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 BASIC IMPORTS WORKING!")
        print("   Ready for full viz testing.")
        return 0
    else:
        print(f"\n⚠️  {total-passed} basic tests failed.")
        print("   Fix compilation issues before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())