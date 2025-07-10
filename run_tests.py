#!/usr/bin/env python3
"""
Test runner for Groggy graph engine.
Run with: python run_tests.py [functionality|stress|all]
"""

import sys
import argparse
import time


def run_functionality_tests():
    """Run basic functionality tests."""
    print("🧪 Running Functionality Tests")
    print("=" * 40)
    
    try:
        from tests.test_functionality import (
            test_basic_graph_operations,
            test_state_management,
            test_filtering,
            test_node_updates,
            test_edge_operations,
            test_graph_properties
        )
        
        tests = [
            ("Basic Graph Operations", test_basic_graph_operations),
            ("State Management", test_state_management),
            ("Filtering", test_filtering),
            ("Node Updates", test_node_updates),
            ("Edge Operations", test_edge_operations),
            ("Graph Properties", test_graph_properties),
        ]
        
        passed = 0
        for test_name, test_func in tests:
            try:
                print(f"Running {test_name}...", end=" ")
                test_func()
                print("✅ PASSED")
                passed += 1
            except Exception as e:
                print(f"❌ FAILED: {e}")
        
        print(f"\n📊 Results: {passed}/{len(tests)} tests passed")
        return passed == len(tests)
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def run_stress_tests():
    """Run stress tests."""
    print("🧪 Running Stress Tests")
    print("=" * 40)
    
    try:
        from tests.test_stress import run_quick_stress_test
        
        start_time = time.time()
        run_quick_stress_test()
        end_time = time.time()
        
        print(f"\n⏱️ Total stress test time: {end_time - start_time:.2f} seconds")
        return True
        
    except Exception as e:
        print(f"❌ Stress test failed: {e}")
        return False


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Run Groggy tests")
    parser.add_argument(
        "test_type",
        choices=["functionality", "stress", "all"],
        nargs="?",
        default="all",
        help="Type of tests to run (default: all)"
    )
    
    args = parser.parse_args()
    
    print("🚀 GROGGY TEST SUITE")
    print("=" * 50)
    
    overall_start = time.time()
    success = True
    
    if args.test_type in ["functionality", "all"]:
        success &= run_functionality_tests()
        print()
    
    if args.test_type in ["stress", "all"]:
        success &= run_stress_tests()
        print()
    
    overall_time = time.time() - overall_start
    
    print("🏁 TEST SUITE COMPLETE")
    print("=" * 50)
    print(f"⏱️ Total time: {overall_time:.2f} seconds")
    
    if success:
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
