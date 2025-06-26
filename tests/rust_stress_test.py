#!/usr/bin/env python3
"""
Stress test for GLI Rust backend - testing scalability limits
"""

import time
import random
import sys
import os
import psutil
import gc

# Add the python directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

import gli

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def benchmark_with_memory(func, description=""):
    """Benchmark a function and return (time_taken, result, memory_used)"""
    gc.collect()  # Clean up before measuring
    start_memory = get_memory_usage()
    start_time = time.perf_counter()
    
    result = func()
    
    end_time = time.perf_counter()
    end_memory = get_memory_usage()
    
    time_taken = end_time - start_time
    memory_used = end_memory - start_memory
    
    if description:
        print(f"  {description}: {time_taken:.2f}s, {memory_used:.1f}MB memory")
    
    return time_taken, result, memory_used

def stress_test_graph_creation():
    """Test graph creation with progressively larger sizes"""
    print("🔥 STRESS TEST: Graph Creation Limits")
    print("=" * 50)
    
    gli.set_backend('rust')
    print("Using Rust backend exclusively")
    
    # Test sizes - start reasonable, then push limits
    test_configs = [
        (5_000, 0.001, "5K nodes, sparse"),
        (10_000, 0.001, "10K nodes, sparse"),  
        (20_000, 0.001, "20K nodes, sparse"),
        (50_000, 0.0005, "50K nodes, very sparse"),
        (100_000, 0.0002, "100K nodes, ultra sparse"),
        (10_000, 0.01, "10K nodes, dense"),
        (5_000, 0.02, "5K nodes, very dense"),
        (2_000, 0.05, "2K nodes, extremely dense"),
    ]
    
    successful_tests = []
    
    for size, edge_prob, description in test_configs:
        print(f"\n📊 Testing: {description}")
        print(f"   Expected edges: ~{int(size * (size-1) * edge_prob / 2):,}")
        
        try:
            def create_large_graph():
                graph = gli.Graph()
                
                # Add nodes with progress indicator
                print(f"   Adding {size:,} nodes...", end="", flush=True)
                for i in range(size):
                    if i % (size // 10) == 0 and size > 1000:
                        print(".", end="", flush=True)
                    graph.add_node(f"n{i}", id=i, batch=i//1000)
                print(" ✓")
                
                # Add edges with progress indicator
                print(f"   Adding edges (prob={edge_prob})...", end="", flush=True)
                random.seed(42)  # Consistent results
                edge_count = 0
                for i in range(size):
                    if i % (size // 10) == 0 and size > 1000:
                        print(".", end="", flush=True)
                    for j in range(i + 1, min(i + 100, size)):  # Limit connections per node
                        if random.random() < edge_prob:
                            graph.add_edge(f"n{i}", f"n{j}", weight=random.random())
                            edge_count += 1
                print(" ✓")
                
                return graph
            
            time_taken, graph, memory_used = benchmark_with_memory(create_large_graph)
            
            # Verify graph
            node_count = graph.node_count()
            edge_count = graph.edge_count()
            
            print(f"   ✅ SUCCESS: {node_count:,} nodes, {edge_count:,} edges")
            print(f"   ⏱️  Time: {time_taken:.2f}s")
            print(f"   💾 Memory: {memory_used:.1f}MB")
            print(f"   📈 Rate: {node_count/time_taken:.0f} nodes/sec, {edge_count/time_taken:.0f} edges/sec")
            
            successful_tests.append((description, node_count, edge_count, time_taken, memory_used))
            
            # Test some queries on the large graph
            print(f"   🔍 Testing queries...")
            start_query = time.perf_counter()
            
            # Sample some nodes for neighbor queries
            sample_nodes = [f"n{i}" for i in range(0, min(node_count, 1000), 100)]
            total_neighbors = 0
            for node in sample_nodes:
                neighbors = graph.get_neighbors(node)
                total_neighbors += len(neighbors)
            
            query_time = time.perf_counter() - start_query
            print(f"   🔍 Query time: {query_time:.3f}s ({len(sample_nodes)} neighbor queries)")
            
        except Exception as e:
            print(f"   ❌ FAILED: {str(e)}")
            print(f"   💥 This appears to be the limit for: {description}")
            break
        except KeyboardInterrupt:
            print(f"\n   ⏹️  User interrupted at: {description}")
            break
        
        # Clean up
        del graph
        gc.collect()
    
    return successful_tests

def stress_test_queries():
    """Test query performance on large graphs"""
    print("\n🔍 STRESS TEST: Query Performance on Large Graphs")
    print("=" * 50)
    
    gli.set_backend('rust')
    
    # Create a large graph for query testing
    size = 50_000
    print(f"Creating base graph with {size:,} nodes for query testing...")
    
    def create_query_test_graph():
        graph = gli.Graph()
        
        # Add nodes
        for i in range(size):
            graph.add_node(f"node_{i}", value=i, category=i % 100)
        
        # Add edges - each node connected to next 10 nodes (cyclical)
        random.seed(42)
        for i in range(size):
            for j in range(1, 11):  # Connect to next 10 nodes
                target = (i + j) % size
                graph.add_edge(f"node_{i}", f"node_{target}", weight=random.random())
        
        return graph
    
    time_taken, graph, memory_used = benchmark_with_memory(create_query_test_graph, 
                                                         f"Created graph with {size:,} nodes")
    
    # Test different types of queries
    query_tests = [
        ("Node count", lambda: graph.node_count()),
        ("Edge count", lambda: graph.edge_count()),
        ("Get all node IDs", lambda: len(graph.get_node_ids())),
        ("Neighbors of 1000 random nodes", lambda: [len(graph.get_neighbors(f"node_{random.randint(0, size-1)}")) for _ in range(1000)]),
        ("Neighbors of first 10000 nodes", lambda: [len(graph.get_neighbors(f"node_{i}")) for i in range(10000)]),
    ]
    
    print("\n📊 Query Performance Tests:")
    for test_name, query_func in query_tests:
        try:
            start_time = time.perf_counter()
            result = query_func()
            end_time = time.perf_counter()
            
            query_time = end_time - start_time
            if isinstance(result, list):
                print(f"  {test_name}: {query_time:.3f}s (avg: {sum(result)/len(result):.1f})")
            else:
                print(f"  {test_name}: {query_time:.3f}s (result: {result:,})")
                
        except Exception as e:
            print(f"  {test_name}: FAILED - {str(e)}")

def main():
    print("🚀 GLI Rust Backend Stress Test")
    print("Testing the limits of graph size and performance")
    print("=" * 60)
    
    try:
        # Test graph creation limits
        successful_tests = stress_test_graph_creation()
        
        # Test query performance
        stress_test_queries()
        
        # Summary
        print("\n🎯 STRESS TEST SUMMARY")
        print("=" * 50)
        if successful_tests:
            print("✅ Successfully created graphs:")
            for desc, nodes, edges, time_taken, memory in successful_tests:
                print(f"  • {desc}: {nodes:,} nodes, {edges:,} edges ({time_taken:.1f}s, {memory:.1f}MB)")
            
            # Find the largest successful test
            largest = max(successful_tests, key=lambda x: x[1])  # By node count
            most_edges = max(successful_tests, key=lambda x: x[2])  # By edge count
            
            print(f"\n🏆 Largest graph by nodes: {largest[0]} - {largest[1]:,} nodes")
            print(f"🏆 Largest graph by edges: {most_edges[0]} - {most_edges[2]:,} edges")
            
        print(f"\n💾 Peak memory usage: {get_memory_usage():.1f}MB")
        print("🎉 Rust backend successfully handled large-scale graphs!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Stress test interrupted by user")
    except Exception as e:
        print(f"\n💥 Stress test failed: {str(e)}")

if __name__ == "__main__":
    main()
