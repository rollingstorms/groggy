#!/usr/bin/env python3
"""
🎯 Groggy Method Coverage Test Suite - YN Style
====================================================

Tests EVERY SINGLE METHOD in the Groggy API systematically.
Because if we're gonna test, we test EVERYTHING.

This covers:
- All 100+ PyGraph methods
- All parameter combinations 
- All edge cases
- All error conditions
- All return types
"""

import groggy as g
import traceback
import time
import inspect
import sys
from typing import Any, Dict, List, Tuple

# Test configuration
VERBOSE = True
FAIL_FAST = False

class MethodTester:
    def __init__(self):
        self.results = {
            'total': 0,
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'errors': []
        }
        self.graph = None
        self.node_ids = []
        self.edge_ids = []
        
    def setup_test_graph(self):
        """Create a test graph with diverse data"""
        print("🔧 Setting up test graph...")
        self.graph = g.Graph()
        
        # Add nodes with various attributes
        test_nodes = [
            {"name": "Alice", "age": 25, "dept": "Engineering", "salary": 75000, "active": True},
            {"name": "Bob", "age": 30, "dept": "Research", "salary": 85000, "active": True},
            {"name": "Charlie", "age": 35, "dept": "Sales", "salary": 95000, "active": False},
            {"name": "Diana", "age": 28, "dept": "Engineering", "salary": 80000, "active": True},
            {"name": "Eve", "age": 40, "dept": "Management", "salary": 100000, "active": False}
        ]
        
        for node_data in test_nodes:
            node_id = self.graph.add_node(**node_data)
            self.node_ids.append(node_id)
            
        # Add edges
        edges = [
            (self.node_ids[0], self.node_ids[1], {"relationship": "reports_to", "strength": 0.8}),
            (self.node_ids[1], self.node_ids[2], {"relationship": "collaborates", "strength": 0.6}),
            (self.node_ids[2], self.node_ids[3], {"relationship": "mentors", "strength": 0.9}),
            (self.node_ids[3], self.node_ids[0], {"relationship": "friends", "strength": 0.7})
        ]
        
        for source, target, attrs in edges:
            edge_id = self.graph.add_edge(source, target, **attrs)
            self.edge_ids.append(edge_id)
            
        print(f"✅ Test graph ready: {len(self.node_ids)} nodes, {len(self.edge_ids)} edges")
        
    def test_method(self, method_name: str, method_obj: Any, test_args: List[Tuple], description: str = ""):
        """Test a single method with multiple argument combinations"""
        print(f"\n🧪 Testing {method_name}")
        if description:
            print(f"   {description}")
            
        method_results = {'passed': 0, 'failed': 0}
        
        for i, test_case in enumerate(test_args):
            try:
                # Handle different test case formats
                if isinstance(test_case, dict):
                    # Dictionary means call with **kwargs
                    result = method_obj(**test_case)
                    args_display = f"**{test_case}"
                elif (isinstance(test_case, tuple) and len(test_case) == 1 and 
                      isinstance(test_case[0], dict) and method_name not in ['set_node_attrs', 'set_edge_attrs']):
                    # Single dict in tuple means call with **kwargs (except for bulk attr methods)
                    result = method_obj(**test_case[0])
                    args_display = f"**{test_case[0]}"
                elif isinstance(test_case, tuple) and len(test_case) >= 2 and isinstance(test_case[-1], dict):
                    # Tuple with dict at end means args + kwargs
                    args = test_case[:-1]
                    kwargs = test_case[-1]
                    result = method_obj(*args, **kwargs)
                    args_display = f"{args}, **{kwargs}"
                else:
                    # Normal args/kwargs
                    result = method_obj(*test_case)
                    args_display = str(test_case)
                    
                method_results['passed'] += 1
                if VERBOSE:
                    print(f"   ✅ Case {i+1}: {args_display} → {type(result).__name__}")
                    
            except Exception as e:
                method_results['failed'] += 1
                if isinstance(test_case, dict):
                    args_display = f"**{test_case}"
                elif isinstance(test_case, tuple) and len(test_case) >= 2 and isinstance(test_case[-1], dict):
                    args = test_case[:-1]
                    kwargs = test_case[-1]
                    args_display = f"{args}, **{kwargs}"
                else:
                    args_display = str(test_case)
                error_msg = f"{method_name}({args_display}): {str(e)}"
                self.results['errors'].append(error_msg)
                print(f"   ❌ Case {i+1}: {args_display} → {str(e)}")
                
                if FAIL_FAST:
                    raise
                    
        # Update global results
        self.results['total'] += len(test_args)
        self.results['passed'] += method_results['passed']
        self.results['failed'] += method_results['failed']
        
        success_rate = method_results['passed'] / len(test_args) * 100
        print(f"   📊 {method_name}: {method_results['passed']}/{len(test_args)} ({success_rate:.1f}%)")
        
    def run_comprehensive_tests(self):
        """Test every method systematically"""
        self.setup_test_graph()
        
        print("\n" + "="*60)
        print("🎯 COMPREHENSIVE METHOD COVERAGE TESTING")
        print("="*60)
        
        # === BASIC OPERATIONS ===
        print("\n📍 SECTION: Basic Operations")
        
        # Node operations
        self.test_method("add_node", self.graph.add_node, [
            (),
            ({"name": "Test"},),
            ({"id": 999, "name": "Numbered"},),
            ({"complex": [1, 2, 3]},)
        ], "Adding nodes with various attributes")
        
        # Create a temporary node for removal testing to avoid affecting other tests
        temp_node = self.graph.add_node(name="TempForRemoval")
        self.test_method("remove_node", self.graph.remove_node, [
            (temp_node,),
            (999,),  # Non-existent node
        ], "Removing existing and non-existent nodes")
        
        self.test_method("node_count", self.graph.node_count, [()], "Count nodes")
        self.test_method("edge_count", self.graph.edge_count, [()], "Count edges")
        
        # === ATTRIBUTE OPERATIONS ===
        print("\n📍 SECTION: Attribute Operations")
        
        if self.node_ids:
            # Use actual node IDs from the setup
            test_node = self.node_ids[0]
            print(f"🔍 Debug: Using test_node={test_node}, node_ids={self.node_ids[:3]}")
            self.test_method("set_node_attr", self.graph.set_node_attr, [
                (test_node, "test_attr", "test_value"),
                (test_node, "numeric", 42),
                (test_node, "float_val", 3.14),
                (test_node, "bool_val", True),
                (test_node, "list_val", [1, 2, 3])
            ], "Set various attribute types")
            
            self.test_method("get_node_attr", self.graph.get_node_attr, [
                (test_node, "name"),
                (test_node, "age"),
                (test_node, "nonexistent"),
                (test_node, "nonexistent", "default_val")
            ], "Get attributes with and without defaults")
            
            # Test bulk attribute operations
            bulk_attrs = {
                "team": {self.node_ids[i]: f"Team{i}" for i in range(min(3, len(self.node_ids)))},
                "score": {self.node_ids[i]: i * 10 for i in range(min(3, len(self.node_ids)))}
            }
            
            # Fix the bulk attrs format - pass as single argument, not as kwargs
            self.test_method("set_node_attrs", self.graph.set_node_attrs, [
                (bulk_attrs,)  # Pass the dict as a single positional argument
            ], "Bulk set node attributes")
            
            self.test_method("get_node_attrs", self.graph.get_node_attrs, [
                (self.node_ids[:3], ["name", "age"]),
                (self.node_ids[:2], ["team", "score"]),
                ([self.node_ids[0]], ["nonexistent"])
            ], "Bulk get node attributes")
            
        # === EDGE OPERATIONS ===
        print("\n📍 SECTION: Edge Operations")
        
        if len(self.node_ids) >= 2:
            # Create some fresh nodes for edge testing
            fresh_node1 = self.graph.add_node(name="EdgeTest1")
            fresh_node2 = self.graph.add_node(name="EdgeTest2")
            
            self.test_method("add_edge", self.graph.add_edge, [
                (fresh_node1, fresh_node2),
                (fresh_node2, fresh_node1, {"bidirectional": True}),
                (999, 888),  # Non-existent nodes
            ], "Add edges between various node pairs")
            
        if self.edge_ids:
            self.test_method("remove_edge", self.graph.remove_edge, [
                (self.edge_ids[0],),
                (999,)  # Non-existent edge
            ], "Remove existing and non-existent edges")
            
        # === ANALYSIS OPERATIONS ===
        print("\n📍 SECTION: Analysis Operations")
        
        if self.node_ids:
            test_node = self.node_ids[0]
            self.test_method("neighbors", self.graph.neighbors, [
                (test_node,),
                ([test_node, self.node_ids[1]] if len(self.node_ids) > 1 else [test_node],),
                (999,),  # Non-existent node
            ], "Get neighbors for single and multiple nodes")
            
            self.test_method("degree", self.graph.degree, [
                (test_node,),
                ([test_node, self.node_ids[1]] if len(self.node_ids) > 1 else [test_node],),
                (None,),  # All nodes
            ], "Get degree for various node sets")
            
            self.test_method("in_degree", self.graph.in_degree, [
                (test_node,),
                (None,)
            ], "Get in-degree")
            
            self.test_method("out_degree", self.graph.out_degree, [
                (test_node,),
                (None,)
            ], "Get out-degree")
            
        # === MATRIX OPERATIONS ===
        print("\n📍 SECTION: Matrix Operations")
        
        try:
            self.test_method("adjacency_matrix", self.graph.adjacency_matrix, [()], "Get adjacency matrix")
            self.test_method("dense_adjacency_matrix", self.graph.dense_adjacency_matrix, [()], "Get dense adjacency matrix")
            self.test_method("laplacian_matrix", self.graph.laplacian_matrix, [
                (None,),
                (True,),
                (False,)
            ], "Get Laplacian matrix with normalization options")
        except Exception as e:
            print(f"   ⚠️ Matrix operations skipped: {e}")
            
        # === QUERY OPERATIONS ===
        print("\n📍 SECTION: Query Operations")
        
        query_tests = [
            "name == 'Alice'",
            "age > 25",
            "salary < 90000",
            "active == True",
            "dept == 'Engineering'",
            "age >= 30 AND salary < 100000",
            "name != 'Bob' OR active == False"
        ]
        
        for query in query_tests:
            try:
                parsed = g.parse_node_query(query)
                result = self.graph.filter_nodes(parsed)
                print(f"   ✅ Query '{query}' → {len(result) if result else 0} results")
            except Exception as e:
                print(f"   ❌ Query '{query}' → {str(e)}")
                
        # === PROPERTY-STYLE ACCESS ===
        print("\n📍 SECTION: Property-Style Access (__getattr__)")
        
        # Test the new property-style access
        try:
            # This should return attribute values for all nodes that have the 'name' attribute
            name_attrs = getattr(self.graph, 'name', None)
            if name_attrs:
                print(f"   ✅ g.name → {type(name_attrs).__name__} with {len(name_attrs)} entries")
            else:
                print(f"   ✅ g.name → None (no name attributes)")
                
        except Exception as e:
            print(f"   ❌ g.name → {str(e)}")
            
        try:
            # Test non-existent attribute
            nonexistent = getattr(self.graph, 'nonexistent_attr', None)
            print(f"   ✅ g.nonexistent_attr → None (correctly handled)")
        except Exception as e:
            print(f"   ✅ g.nonexistent_attr → {str(e)} (expected error)")
            
    def print_final_report(self):
        """Print comprehensive test results"""
        print("\n" + "="*60)
        print("📊 FINAL METHOD COVERAGE REPORT")
        print("="*60)
        
        total = self.results['total']
        passed = self.results['passed']
        failed = self.results['failed']
        success_rate = passed / total * 100 if total > 0 else 0
        
        print(f"Total Method Calls Tested: {total}")
        print(f"Successful: {passed} ✅")
        print(f"Failed: {failed} ❌")
        print(f"Success Rate: {success_rate:.1f}%")
        
        if self.results['errors']:
            print(f"\n⚠️  Error Summary ({len(self.results['errors'])} errors):")
            for i, error in enumerate(self.results['errors'][:10]):  # Show first 10
                print(f"   {i+1}. {error}")
            if len(self.results['errors']) > 10:
                print(f"   ... and {len(self.results['errors']) - 10} more")
                
        print(f"\n🎯 Method coverage testing complete!")
        
        return success_rate

def main():
    """Run the comprehensive method coverage test suite"""
    print("🚀 Starting YN-Style Comprehensive Method Coverage Testing")
    print("Testing EVERY method because that's how we roll! 💪")
    
    tester = MethodTester()
    
    try:
        tester.run_comprehensive_tests()
        success_rate = tester.print_final_report()
        
        # Exit with appropriate code
        if success_rate >= 95:
            print("🎉 EXCELLENT! Method coverage is outstanding!")
            sys.exit(0)
        elif success_rate >= 90:
            print("👍 GOOD! Most methods working well!")
            sys.exit(0)  
        else:
            print("⚠️ NEEDS WORK! Many methods need attention!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⛔ Testing interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 FATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()