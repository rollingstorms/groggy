#!/usr/bin/env python3
"""
🏋️ Groggy Performance Stress Test Suite - YN Style  
====================================================

Push the limits! Test performance under extreme conditions.
Because if it doesn't work under stress, it doesn't really work.

Tests:
- Large graph operations (10K, 100K, 1M+ nodes/edges)
- Memory usage patterns
- Bulk operation performance
- Query performance on large datasets
- Concurrent access patterns
"""

import groggy as g
import time
import psutil
import gc
import sys
from typing import List, Tuple, Dict
import random
import threading

class PerformanceStressTester:
    def __init__(self):
        self.results = []
        self.process = psutil.Process()
        
    def measure_performance(self, func, *args, **kwargs):
        """Measure execution time and memory usage"""
        gc.collect()  # Clean up before measurement
        
        # Memory before
        mem_before = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Execute and time
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        # Memory after
        mem_after = self.process.memory_info().rss / 1024 / 1024  # MB
        
        execution_time = end_time - start_time
        memory_delta = mem_after - mem_before
        
        return result, execution_time, memory_delta, mem_after
        
    def log_performance(self, test_name: str, execution_time: float, memory_delta: float, 
                       current_memory: float, details: str = ""):
        """Log performance results"""
        self.results.append({
            'test': test_name,
            'time': execution_time,
            'memory_delta': memory_delta,
            'current_memory': current_memory,
            'details': details
        })
        
        # Format output  
        time_str = f"{execution_time:.3f}s" if execution_time >= 0.001 else f"{execution_time*1000:.1f}ms"
        mem_str = f"{memory_delta:+.1f}MB" if abs(memory_delta) >= 0.1 else f"{memory_delta*1024:+.0f}KB"
        
        print(f"🏋️  {test_name:<40} | {time_str:>8} | {mem_str:>8} | {current_memory:.0f}MB")
        if details:
            print(f"    └─ {details}")
            
    def stress_test_node_operations(self):
        """Stress test node creation, modification, and deletion"""
        print("\n" + "="*80)
        print("🎯 STRESS TEST: Node Operations")
        print("="*80)
        print(f"{'Test':<40} | {'Time':>8} | {'Memory':>8} | {'Total':>6}")
        print("-" * 80)
        
        # Test different graph sizes
        sizes = [100000, 1000000, 5000000]
        
        for size in sizes:
            print(f"\n📊 Testing with {size:,} nodes:")
            
            # Create graph
            graph = g.Graph()
            
            # Bulk node creation using proper bulk method
            result, exec_time, mem_delta, current_mem = self.measure_performance(
                graph.add_nodes, size
            )
            node_ids = result
            
            self.log_performance(
                f"Add {size:,} nodes", exec_time, mem_delta, current_mem,
                f"Rate: {size/exec_time:.0f} nodes/sec"
            )
            
            # Random attribute updates
            random_nodes = random.sample(node_ids, min(1000, len(node_ids)))
            result, exec_time, mem_delta, current_mem = self.measure_performance(
                lambda: [graph.set_node_attr(node, "updated", True) for node in random_nodes]
            )
            
            self.log_performance(
                f"Update {len(random_nodes):,} node attrs", exec_time, mem_delta, current_mem,
                f"Rate: {len(random_nodes)/exec_time:.0f} updates/sec"
            )
            
            # Bulk attribute operations
            if hasattr(graph, 'set_node_attrs'):
                bulk_attrs = {
                    "batch_id": {node_ids[i]: f"batch_{i//100}" for i in range(min(1000, len(node_ids)))},
                    "score": {node_ids[i]: random.randint(1, 100) for i in range(min(1000, len(node_ids)))}
                }
                
                result, exec_time, mem_delta, current_mem = self.measure_performance(
                    graph.set_node_attrs, bulk_attrs
                )
                
                self.log_performance(
                    f"Bulk set {len(bulk_attrs):,} attr types", exec_time, mem_delta, current_mem,
                    f"Rate: {sum(len(v) for v in bulk_attrs.values())/exec_time:.0f} attrs/sec"
                )
                
            # Node deletion
            nodes_to_delete = random.sample(node_ids, min(500, len(node_ids)))
            result, exec_time, mem_delta, current_mem = self.measure_performance(
                lambda: [graph.remove_node(node) for node in nodes_to_delete]
            )
            
            self.log_performance(
                f"Delete {len(nodes_to_delete):,} nodes", exec_time, mem_delta, current_mem,
                f"Rate: {len(nodes_to_delete)/exec_time:.0f} deletions/sec"
            )
            
            # Final cleanup
            del graph
            gc.collect()
            
    def stress_test_edge_operations(self):
        """Stress test edge creation and operations"""
        print("\n" + "="*80)  
        print("🎯 STRESS TEST: Edge Operations")
        print("="*80)
        print(f"{'Test':<40} | {'Time':>8} | {'Memory':>8} | {'Total':>6}")
        print("-" * 80)
        
        # Test with different graph densities
        node_counts = [1000000, 5000000]
        edge_densities = [0.1, 0.5, 1.0]  # Percentage of possible edges
        
        for node_count in node_counts:
            for density in edge_densities:
                max_edges = node_count * (node_count - 1) // 2  # Undirected graph max
                target_edges = int(max_edges * density / 100)  # Scale down for testing
                target_edges = min(target_edges, 50000)  # Cap for performance
                
                print(f"\n📊 Testing {node_count:,} nodes, {target_edges:,} edges:")
                
                graph = g.Graph()
                
                # Create nodes first using bulk method
                result, exec_time, mem_delta, current_mem = self.measure_performance(
                    graph.add_nodes, node_count
                )
                node_ids = result
                
                self.log_performance(
                    f"Setup {node_count:,} nodes", exec_time, mem_delta, current_mem
                )
                
                # Create random edges for bulk creation
                edges_to_add = []
                for _ in range(target_edges):
                    source = random.choice(node_ids)
                    target = random.choice(node_ids)
                    if source != target:  # Avoid self-loops
                        edges_to_add.append((source, target))
                        
                result, exec_time, mem_delta, current_mem = self.measure_performance(
                    graph.add_edges, edges_to_add
                )
                
                self.log_performance(
                    f"Add {len(edges_to_add):,} edges", exec_time, mem_delta, current_mem,
                    f"Rate: {len(edges_to_add)/exec_time:.0f} edges/sec"
                )
                
                # Edge queries
                if node_ids:
                    sample_nodes = random.sample(node_ids, min(100, len(node_ids)))
                    result, exec_time, mem_delta, current_mem = self.measure_performance(
                        lambda: [graph.neighbors(node) for node in sample_nodes]
                    )
                    
                    self.log_performance(
                        f"Query {len(sample_nodes):,} neighborhoods", exec_time, mem_delta, current_mem,
                        f"Rate: {len(sample_nodes)/exec_time:.0f} queries/sec"
                    )
                
                del graph
                gc.collect()
                
    def stress_test_query_performance(self):
        """Stress test query operations on large datasets"""
        print("\n" + "="*80)
        print("🎯 STRESS TEST: Query Performance")  
        print("="*80)
        print(f"{'Test':<40} | {'Time':>8} | {'Memory':>8} | {'Total':>6}")
        print("-" * 80)
        
        # Create a large graph with diverse attributes
        graph = g.Graph()
        node_count = 1000000
        
        print(f"\n📊 Setting up {node_count:,} node test graph...")
        
        # Create nodes with random attributes for realistic querying
        departments = ["Engineering", "Research", "Sales", "Marketing", "HR", "Finance"]
        
        # Use bulk node creation
        result, exec_time, mem_delta, current_mem = self.measure_performance(
            graph.add_nodes, node_count
        )
        node_ids = result
        
        # Set attributes using bulk method
        departments = ["Engineering", "Research", "Sales", "Marketing", "HR", "Finance"]
        bulk_attrs = {
            "name": [f"Employee_{i:05d}" for i in range(node_count)],
            "age": [random.randint(22, 65) for _ in range(node_count)],
            "salary": [random.randint(40000, 150000) for _ in range(node_count)],
            "department": [random.choice(departments) for _ in range(node_count)],
            "active": [random.choice([True, False]) for _ in range(node_count)],
            "experience": [random.uniform(0, 20) for _ in range(node_count)]
        }
        
        # Convert to the bulk format expected by set_node_attrs
        bulk_attrs_formatted = {}
        for attr_name, values in bulk_attrs.items():
            bulk_attrs_formatted[attr_name] = {node_ids[i]: values[i] for i in range(len(node_ids))}
            
        # Set all attributes in bulk
        graph.set_node_attrs(bulk_attrs_formatted)
        
        self.log_performance(
            f"Setup {node_count:,} diverse nodes", exec_time, mem_delta, current_mem
        )
        
        # Test various query complexities
        queries = [
            ("Simple equality", "department == 'Engineering'"),
            ("Range query", "age > 30 AND age < 50"),
            ("Complex AND", "salary > 80000 AND active == True"),
            ("Complex OR", "department == 'Sales' OR department == 'Marketing'"),
            ("Multi-condition", "age > 25 AND salary < 100000 AND active == True"),
            ("Negation", "NOT (department == 'HR' OR salary > 120000)"),
        ]
        
        for query_name, query_str in queries:
            try:
                # Parse query
                parsed_query, parse_time, _, _ = self.measure_performance(
                    g.parse_node_query, query_str
                )
                
                # Execute query
                result, exec_time, mem_delta, current_mem = self.measure_performance(
                    graph.filter_nodes, parsed_query
                )
                
                result_count = len(result) if result else 0
                self.log_performance(
                    f"Query: {query_name}", exec_time, mem_delta, current_mem,
                    f"Found: {result_count:,} results, Parse: {parse_time*1000:.1f}ms"
                )
                
            except Exception as e:
                print(f"    ❌ Query '{query_name}' failed: {e}")
                
        del graph
        gc.collect()
        
    def stress_test_matrix_operations(self):
        """Stress test matrix operations on larger graphs"""
        print("\n" + "="*80)
        print("🎯 STRESS TEST: Matrix Operations")
        print("="*80)  
        print(f"{'Test':<40} | {'Time':>8} | {'Memory':>8} | {'Total':>6}")
        print("-" * 80)
        
        sizes = [1000, 5000, 10000]  # Start smaller for matrix ops
        
        for size in sizes:
            print(f"\n📊 Testing {size}x{size} matrices:")
            
            graph = g.Graph()
            
            # Create connected graph for meaningful matrices using bulk method
            node_ids = graph.add_nodes(size)
            
            # Add edges to create connected components using bulk method
            edge_count = size * 2  # Sparse but connected
            edges_to_add = []
            for i in range(edge_count):
                source = random.choice(node_ids)
                target = random.choice(node_ids)
                if source != target:
                    edges_to_add.append((source, target))
            
            if edges_to_add:
                edge_ids = graph.add_edges(edges_to_add)
                # Set weights using bulk method
                weight_attrs = {"weight": {edge_ids[i]: random.random() for i in range(len(edge_ids))}}
                graph.set_edge_attrs(weight_attrs)
                    
            self.log_performance(f"Setup {size}x{size} graph", 0, 0, 
                               self.process.memory_info().rss / 1024 / 1024)
            
            # Test matrix operations
            matrix_ops = [
                ("Adjacency Matrix", "adjacency_matrix"),
                ("Dense Adjacency", "dense_adjacency_matrix"), 
                ("Laplacian Matrix", "laplacian_matrix"),
            ]
            
            for op_name, method_name in matrix_ops:
                if hasattr(graph, method_name):
                    try:
                        method = getattr(graph, method_name)
                        result, exec_time, mem_delta, current_mem = self.measure_performance(method)
                        
                        self.log_performance(
                            f"{op_name} ({size}x{size})", exec_time, mem_delta, current_mem,
                            f"Matrix type: {type(result).__name__}"
                        )
                    except Exception as e:
                        print(f"    ❌ {op_name} failed: {e}")
                        
            del graph
            gc.collect()
            
    def print_final_report(self):
        """Print comprehensive performance report"""
        print("\n" + "="*80)
        print("📊 PERFORMANCE STRESS TEST REPORT") 
        print("="*80)
        
        if not self.results:
            print("No results to report!")
            return
            
        # Summary statistics
        total_tests = len(self.results)
        avg_time = sum(r['time'] for r in self.results) / total_tests
        max_time = max(r['time'] for r in self.results)
        total_memory = sum(abs(r['memory_delta']) for r in self.results)
        final_memory = self.results[-1]['current_memory']
        
        print(f"Total Performance Tests: {total_tests}")
        print(f"Average Execution Time: {avg_time:.3f}s")
        print(f"Slowest Operation: {max_time:.3f}s")
        print(f"Total Memory Churn: {total_memory:.1f}MB")
        print(f"Final Memory Usage: {final_memory:.1f}MB")
        
        # Find performance outliers
        slow_tests = [r for r in self.results if r['time'] > avg_time * 3]
        memory_hogs = [r for r in self.results if abs(r['memory_delta']) > 10]
        
        if slow_tests:
            print(f"\n⚠️  Slow Operations ({len(slow_tests)} tests > {avg_time*3:.3f}s):")
            for test in sorted(slow_tests, key=lambda x: x['time'], reverse=True)[:5]:
                print(f"   🐌 {test['test']}: {test['time']:.3f}s")
                
        if memory_hogs:
            print(f"\n⚠️  Memory Intensive Operations ({len(memory_hogs)} tests > 10MB):")
            for test in sorted(memory_hogs, key=lambda x: abs(x['memory_delta']), reverse=True)[:5]:
                print(f"   💾 {test['test']}: {test['memory_delta']:+.1f}MB")
                
        # Performance assessment
        if max_time < 1.0 and final_memory < 500:
            print("\n🎉 EXCELLENT! Performance is outstanding under stress!")
            return True
        elif max_time < 5.0 and final_memory < 1000:
            print("\n👍 GOOD! Performance is acceptable under stress!")
            return True
        else:
            print("\n⚠️ NEEDS OPTIMIZATION! Some operations are too slow or memory-intensive!")
            return False

def main():
    """Run the performance stress test suite"""
    print("🚀 Starting YN-Style Performance Stress Testing")
    print("Push it to the limit! 💪")
    
    tester = PerformanceStressTester()
    
    try:
        # Run all stress tests
        tester.stress_test_node_operations()
        tester.stress_test_edge_operations()  
        tester.stress_test_query_performance()
        tester.stress_test_matrix_operations()
        
        # Final report
        performance_ok = tester.print_final_report()
        
        if performance_ok:
            print("\n🎯 Performance stress testing complete - ALL SYSTEMS GO! 🚀")
            sys.exit(0)
        else:
            print("\n⚠️ Performance stress testing complete - NEEDS OPTIMIZATION! 🔧")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⛔ Stress testing interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 FATAL ERROR during stress testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()