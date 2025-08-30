#!/usr/bin/env python3
"""
🔥 Groggy Edge Case Torture Test Suite - YN Style
=================================================

Break things in creative ways! Test every possible edge case.
Because if it's gonna break, let's find out NOW.

Tests:
- Every possible error condition
- Boundary values (None, empty, huge, negative, etc.)
- Invalid inputs and malformed data
- Concurrent access scenarios
- Memory pressure situations
- Unicode and special characters
"""

import groggy as g
import sys
import math
import threading
import time
from typing import Any, List
import gc

class EdgeCaseTortureTester:
    def __init__(self):
        self.results = {'total': 0, 'passed': 0, 'failed': 0, 'errors': []}
        
    def expect_error(self, test_name: str, test_func, *args, **kwargs):
        """Test that expects an error - passes if error occurs"""
        try:
            result = test_func(*args, **kwargs)
            self.results['total'] += 1
            self.results['failed'] += 1
            print(f"❌ {test_name}: Expected error but got result: {result}")
            return False
        except Exception as e:
            self.results['total'] += 1
            self.results['passed'] += 1
            print(f"✅ {test_name}: Got expected error: {type(e).__name__}")
            return True
            
    def expect_success(self, test_name: str, func, *args, **kwargs):
        """Test that expects success"""
        try:
            result = func(*args, **kwargs)
            self.results['total'] += 1  
            self.results['passed'] += 1
            print(f"✅ {test_name}: Success - {type(result).__name__}")
            return True, result
        except Exception as e:
            self.results['total'] += 1
            self.results['failed'] += 1
            error_msg = f"{test_name}: {str(e)}"
            self.results['errors'].append(error_msg)
            print(f"❌ {test_name}: Unexpected error: {str(e)}")
            return False, None
            
    def torture_test_graph_creation(self):
        """Test every possible way to break graph creation"""
        print("\n" + "="*60)
        print("🔥 TORTURE TEST: Graph Creation Edge Cases")
        print("="*60)
        
        # Valid graph creation
        self.expect_success("Empty directed graph", g.Graph)
        self.expect_success("Empty undirected graph", g.Graph, directed=False)
        
        # Edge case parameters
        self.expect_success("Explicit directed=True", g.Graph, directed=True)
        self.expect_success("Explicit directed=None", g.Graph, directed=None)
        
        # Invalid parameters (should these error or be ignored?)
        try:
            weird_graph = g.Graph(invalid_param=True, another_param="weird")
            print(f"⚠️  Graph accepts unknown parameters: {type(weird_graph)}")
        except Exception as e:
            print(f"✅ Graph rejects unknown parameters: {type(e).__name__}")
            
    def torture_test_node_edge_cases(self):
        """Test every possible way nodes can break"""
        print("\n" + "="*60)
        print("🔥 TORTURE TEST: Node Edge Cases")
        print("="*60)
        
        graph = g.Graph()
        
        # === BOUNDARY VALUE TESTING ===
        
        # None values
        self.expect_success("Node with None attr", graph.add_node, name=None)
        
        # Empty values
        self.expect_success("Node with empty string", graph.add_node, name="")
        self.expect_success("Node with empty list", graph.add_node, items=[])
        
        # Extreme values
        self.expect_success("Node with huge int", graph.add_node, big=sys.maxsize)
        self.expect_success("Node with negative int", graph.add_node, neg=-sys.maxsize)
        self.expect_success("Node with zero", graph.add_node, zero=0)
        
        # Float edge cases
        self.expect_success("Node with infinity", graph.add_node, inf=float('inf'))
        self.expect_success("Node with negative infinity", graph.add_node, ninf=float('-inf'))
        self.expect_success("Node with NaN", graph.add_node, nan=float('nan'))
        self.expect_success("Node with tiny float", graph.add_node, tiny=1e-100)
        self.expect_success("Node with huge float", graph.add_node, huge=1e100)
        
        # String edge cases
        self.expect_success("Node with unicode", graph.add_node, uni="🚀🔥💪")
        self.expect_success("Node with newlines", graph.add_node, multiline="line1\\nline2\\nline3")
        self.expect_success("Node with quotes", graph.add_node, quotes="'single' and \\\"double\\\"")
        
        # Very long strings
        long_string = "x" * 100000
        self.expect_success("Node with 100k char string", graph.add_node, long=long_string)
        
        # === UNSUPPORTED TYPES ===
        
        # Complex data structures (should fail)
        self.expect_error("Node with dict", graph.add_node, data={"nested": "dict"})
        self.expect_error("Node with set", graph.add_node, data={1, 2, 3})
        self.expect_error("Node with tuple", graph.add_node, data=(1, 2, 3))
        
        # Functions and objects
        self.expect_error("Node with function", lambda: graph.add_node(test_func=lambda x: x))
        self.expect_error("Node with object", lambda: graph.add_node(obj=graph))
        
        # === MASSIVE ATTRIBUTE TESTS ===
        
        # Node with tons of attributes
        massive_attrs = {f"attr_{i}": i for i in range(1000)}
        self.expect_success("Node with 1000 attributes", graph.add_node, **massive_attrs)
        
        # Node with very long attribute names
        long_attr_name = "attr_" + "x" * 1000
        attrs = {long_attr_name: "value"}
        self.expect_success("Node with 1000-char attr name", graph.add_node, **attrs)
        
    def torture_test_attribute_edge_cases(self):
        """Test every possible way attributes can break"""
        print("\n" + "="*60)
        print("🔥 TORTURE TEST: Attribute Edge Cases")
        print("="*60)
        
        graph = g.Graph()
        node_id = graph.add_node(name="test")
        
        # === ATTRIBUTE NAME EDGE CASES ===
        
        # Special character attribute names
        special_names = ["", " ", "\\t", "\\n", "🚀", "name with spaces", "name-with-dashes", 
                        "name.with.dots", "name_with_underscores", "123numeric", "!@#$%"]
        
        for name in special_names:
            if name:  # Skip empty name
                self.expect_success(f"Attr name: '{name}'", graph.set_node_attr, node_id, name, "value")
                
        # Very long attribute name  
        long_name = "attr_" + "x" * 10000
        self.expect_success("10k char attr name", graph.set_node_attr, node_id, long_name, "value")
        
        # === ATTRIBUTE VALUE EDGE CASES ===
        
        # Test every supported type's edge cases
        edge_values = [
            ("None", None),
            ("Empty string", ""),
            ("Long string", "x" * 100000),
            ("Unicode string", "🚀💪🔥"),
            ("Zero int", 0),
            ("Max int", sys.maxsize),
            ("Min int", -sys.maxsize),
            ("Zero float", 0.0),
            ("Infinity", float('inf')),
            ("Negative infinity", float('-inf')),
            ("NaN", float('nan')),
            ("Tiny float", 1e-100),
            ("Huge float", 1e100),
            ("True bool", True),
            ("False bool", False),
            ("Empty list", []),
            ("Large int list", list(range(10000))),
            ("Large float list", [i * 0.1 for i in range(10000)]),
            ("Large string list", [f"item_{i}" for i in range(1000)])
        ]
        
        for desc, value in edge_values:
            self.expect_success(f"Set {desc}", graph.set_node_attr, node_id, f"test_{desc}", value)
            
        # === BULK ATTRIBUTE EDGE CASES ===
        
        if hasattr(graph, 'set_node_attrs'):
            # Empty bulk operations
            self.expect_success("Empty bulk attrs", graph.set_node_attrs, {})
            
            # Bulk with missing nodes
            fake_node_id = 999999
            bulk_attrs = {
                "test_attr": {fake_node_id: "value"}
            }
            self.expect_error("Bulk attrs with fake node", graph.set_node_attrs, bulk_attrs)
            
            # Huge bulk operation
            many_nodes = [graph.add_node(id=i) for i in range(100)]
            huge_bulk = {
                f"attr_{j}": {node: f"value_{i}_{j}" for i, node in enumerate(many_nodes)}
                for j in range(10)  # 10 attrs * 100 nodes = 1000 operations
            }
            self.expect_success("Huge bulk attrs (1000 ops)", graph.set_node_attrs, huge_bulk)
            
    def torture_test_edge_operations(self):
        """Test every possible way edges can break"""
        print("\n" + "="*60)
        print("🔥 TORTURE TEST: Edge Edge Cases")
        print("="*60)
        
        graph = g.Graph()
        
        # Create some nodes for edge testing
        node1 = graph.add_node(name="node1")
        node2 = graph.add_node(name="node2")
        
        # === VALID EDGE CASES ===
        
        # Basic edges
        self.expect_success("Simple edge", graph.add_edge, node1, node2)
        
        # Self-loops
        self.expect_success("Self-loop", graph.add_edge, node1, node1)
        
        # Edge with attributes
        edge_id = None
        success, result = self.expect_success("Edge with attrs", graph.add_edge, node1, node2, 
                                            weight=1.5, type="connection")
        if success:
            edge_id = result
            
        # === INVALID EDGE CASES ===
        
        # Non-existent nodes
        fake_node = 999999
        self.expect_error("Edge to fake node", graph.add_edge, node1, fake_node)
        self.expect_error("Edge from fake node", graph.add_edge, fake_node, node2)
        self.expect_error("Edge between fake nodes", graph.add_edge, fake_node, fake_node)
        
        # Invalid edge IDs for operations
        fake_edge = 999999
        if hasattr(graph, 'remove_edge'):
            self.expect_error("Remove fake edge", graph.remove_edge, fake_edge)
            
        if hasattr(graph, 'set_edge_attr'):
            self.expect_error("Set attr on fake edge", graph.set_edge_attr, fake_edge, "attr", "value")
            
        # === EDGE ATTRIBUTE EDGE CASES ===
        
        if edge_id and hasattr(graph, 'set_edge_attr'):
            # Same edge cases as nodes
            edge_values = [
                ("None", None),
                ("Empty string", ""),
                ("Unicode", "🔥💪"),
                ("Infinity", float('inf')),
                ("NaN", float('nan')),
                ("Large list", list(range(1000)))
            ]
            
            for desc, value in edge_values:
                self.expect_success(f"Edge {desc}", graph.set_edge_attr, edge_id, f"test_{desc}", value)
                
    def torture_test_query_edge_cases(self):
        """Test every possible way queries can break"""
        print("\n" + "="*60)
        print("🔥 TORTURE TEST: Query Edge Cases")
        print("="*60)
        
        graph = g.Graph()
        
        # Set up test data with edge case values
        test_nodes = [
            graph.add_node(name="Alice", age=25, salary=None, active=True),
            graph.add_node(name="", age=0, salary=0, active=False),  # Empty name, zero values
            graph.add_node(name="🚀Bob", age=-5, salary=float('inf'), active=True),  # Unicode, negative, infinity
        ]
        
        # === MALFORMED QUERY STRINGS ===
        
        malformed_queries = [
            ("Empty query", ""),
            ("Whitespace only", "   "),
            ("Just operator", "=="),
            ("Unclosed quotes", "name == 'Alice"),
            ("Mismatched parens", "((age > 25)"),
            ("Invalid operator", "age >> 25"),
            ("Missing operand", "age > "),
            ("Invalid syntax", "age age age"),
            ("SQL injection attempt", "'; DROP TABLE nodes; --"),
        ]
        
        for desc, query_str in malformed_queries:
            self.expect_error(f"Parse {desc}", g.parse_node_query, query_str)
            
        # === EDGE CASE QUERY VALUES ===
        
        edge_case_queries = [
            ("Query None", "salary == None"),
            ("Query empty string", "name == ''"),
            ("Query zero", "age == 0"),
            ("Query negative", "age < 0"),
            ("Query infinity", "salary == inf"),
            ("Query unicode", "name == '🚀Bob'"),
            ("Query with spaces", "name == 'Alice Smith'"),
        ]
        
        for desc, query_str in edge_case_queries:
            try:
                parsed = g.parse_node_query(query_str)
                self.expect_success(f"Parse {desc}", lambda: parsed)
                
                # Try to execute the query
                self.expect_success(f"Execute {desc}", graph.filter_nodes, parsed)
            except Exception as e:
                print(f"❌ {desc}: {str(e)}")
                
    def torture_test_concurrent_access(self):
        """Test concurrent access patterns"""
        print("\n" + "="*60)
        print("🔥 TORTURE TEST: Concurrent Access")
        print("="*60)
        
        graph = g.Graph()
        errors = []
        
        def add_nodes_worker(worker_id, count):
            """Worker that adds nodes"""
            try:
                for i in range(count):
                    node_id = graph.add_node(worker=worker_id, index=i)
                    # Small delay to increase chance of race conditions
                    time.sleep(0.001)
            except Exception as e:
                errors.append(f"Worker {worker_id}: {e}")
                
        def modify_nodes_worker(worker_id, node_ids):
            """Worker that modifies existing nodes"""
            try:
                for i, node_id in enumerate(node_ids):
                    graph.set_node_attr(node_id, f"modified_by_{worker_id}", True)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(f"Modifier {worker_id}: {e}")
                
        # Start multiple threads
        threads = []
        
        # Node creation threads
        for i in range(3):
            t = threading.Thread(target=add_nodes_worker, args=(i, 10))
            threads.append(t)
            t.start()
            
        # Wait for creation to finish
        for t in threads:
            t.join()
            
        # Get node IDs for modification
        try:
            all_nodes = graph.node_ids()
            if all_nodes:
                # Node modification threads
                threads = []
                for i in range(2):
                    t = threading.Thread(target=modify_nodes_worker, args=(i, all_nodes[:5]))
                    threads.append(t)
                    t.start()
                    
                for t in threads:
                    t.join()
        except:
            pass  # node_ids() might not exist
            
        if errors:
            print(f"❌ Concurrent access errors: {len(errors)}")
            for error in errors[:3]:  # Show first 3
                print(f"   {error}")
        else:
            print("✅ Concurrent access: No obvious race conditions detected")
            
    def torture_test_memory_pressure(self):
        """Test behavior under memory pressure"""
        print("\n" + "="*60)
        print("🔥 TORTURE TEST: Memory Pressure")
        print("="*60)
        
        # Create many graphs to pressure memory
        graphs = []
        
        try:
            for i in range(100):
                g_instance = g.Graph()
                
                # Add nodes and edges to each graph
                nodes = [g_instance.add_node(id=j, data=f"data_{j}") for j in range(100)]
                
                # Add edges
                for j in range(50):
                    if len(nodes) >= 2:
                        g_instance.add_edge(nodes[j % len(nodes)], nodes[(j+1) % len(nodes)])
                        
                graphs.append(g_instance)
                
                # Force garbage collection periodically
                if i % 10 == 0:
                    gc.collect()
                    
            print(f"✅ Memory pressure: Created {len(graphs)} graphs successfully")
            
            # Test operations under pressure
            test_graph = graphs[0]
            self.expect_success("Operation under memory pressure", 
                              test_graph.node_count)
                              
        except MemoryError:
            print("❌ Memory pressure: Hit memory limit")
        except Exception as e:
            print(f"❌ Memory pressure: Unexpected error: {e}")
        finally:
            # Cleanup
            del graphs
            gc.collect()
            
    def print_final_report(self):
        """Print torture test results"""
        print("\n" + "="*60)
        print("🔥 EDGE CASE TORTURE TEST REPORT")
        print("="*60)
        
        total = self.results['total']
        passed = self.results['passed'] 
        failed = self.results['failed']
        
        if total == 0:
            print("No tests were run!")
            return False
            
        success_rate = passed / total * 100
        
        print(f"Total Torture Tests: {total}")
        print(f"Passed: {passed} ✅")
        print(f"Failed: {failed} ❌")
        print(f"Success Rate: {success_rate:.1f}%")
        
        if self.results['errors']:
            print(f"\\n⚠️  Error Summary ({len(self.results['errors'])} errors):")
            for error in self.results['errors'][:5]:
                print(f"   {error}")
            if len(self.results['errors']) > 5:
                print(f"   ... and {len(self.results['errors']) - 5} more")
                
        if success_rate >= 90:
            print("\\n🎉 EXCELLENT! The system handles edge cases very well!")
            return True
        elif success_rate >= 75:
            print("\\n👍 GOOD! Most edge cases are handled properly!")
            return True
        else:
            print("\\n⚠️ NEEDS WORK! Many edge cases cause problems!")
            return False

def main():
    """Run the edge case torture test suite"""
    print("🔥 Starting YN-Style Edge Case Torture Testing")
    print("Break ALL the things! Let's see what's really broken! 💥")
    
    tester = EdgeCaseTortureTester()
    
    try:
        # Run all torture tests
        tester.torture_test_graph_creation()
        tester.torture_test_node_edge_cases()  
        tester.torture_test_attribute_edge_cases()
        tester.torture_test_edge_operations()
        tester.torture_test_query_edge_cases()
        tester.torture_test_concurrent_access()
        tester.torture_test_memory_pressure()
        
        # Final report
        robust = tester.print_final_report()
        
        if robust:
            print("\\n🎯 Edge case torture testing complete - SYSTEM IS ROBUST! 💪")
            sys.exit(0)
        else:
            print("\\n⚠️ Edge case torture testing complete - NEEDS HARDENING! 🔧")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\\n⛔ Torture testing interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\\n💥 FATAL ERROR during torture testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()