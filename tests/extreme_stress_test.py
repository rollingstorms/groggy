#!/usr/bin/env python3
"""
EXTREME stress test for GLI Rust backend - finding the breaking point!
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

def extreme_stress_test():
    """Push the Rust backend to its absolute limits"""
    print("💥 EXTREME STRESS TEST: Finding the Breaking Point!")
    print("=" * 60)
    
    gli.set_backend('rust')
    print("🦀 Using Rust backend - LET'S PUSH THE LIMITS!")
    
    # Progressive stress test - keep going until we break
    test_configs = [
        # Start with what we know works
        (100_000, 0.0001, "100K nodes, minimal edges"),
        (250_000, 0.0001, "250K nodes, minimal edges"),
        (500_000, 0.0001, "500K nodes, minimal edges"),
        (750_000, 0.0001, "750K nodes, minimal edges"),  
        (1_000_000, 0.0001, "🎯 1 MILLION nodes, minimal edges"),
        (1_500_000, 0.0001, "🚀 1.5 MILLION nodes, minimal edges"),
        (2_000_000, 0.0001, "🔥 2 MILLION nodes, minimal edges"),
        
        # Now test with more edges
        (100_000, 0.001, "100K nodes, more edges"),
        (250_000, 0.0005, "250K nodes, medium edges"),
        (500_000, 0.0003, "500K nodes, some edges"),
        
        # Dense smaller graphs
        (50_000, 0.01, "50K nodes, dense"),
        (25_000, 0.02, "25K nodes, very dense"),
        (10_000, 0.05, "10K nodes, extremely dense"),
        (5_000, 0.1, "5K nodes, ULTRA dense"),
    ]
    
    successful_configs = []
    breaking_point = None
    
    for size, edge_prob, description in test_configs:
        print(f"\n🎯 ATTEMPTING: {description}")
        expected_edges = int(size * (size-1) * edge_prob / 2)
        print(f"   📊 Expected: {size:,} nodes, ~{expected_edges:,} edges")
        print(f"   💾 Current memory: {get_memory_usage():.1f}MB")
        
        try:
            start_memory = get_memory_usage()
            start_time = time.perf_counter()
            
            # Create the graph
            print("   🔨 Creating graph...", end="", flush=True)
            graph = gli.Graph()
            
            # Add nodes in batches for better progress tracking
            batch_size = max(1000, size // 50)
            for i in range(0, size, batch_size):
                end_batch = min(i + batch_size, size)
                for j in range(i, end_batch):
                    graph.add_node(f"n{j}", id=j, batch=j//10000)
                
                if i % (size // 10) == 0:
                    print(".", end="", flush=True)
            
            print("✓ Nodes added!", end="", flush=True)
            
            # Add edges - but limit to avoid O(n²) explosion
            print(" Adding edges...", end="", flush=True)
            random.seed(42)
            edge_count = 0
            
            # For very large graphs, use a smarter edge addition strategy
            if size > 100_000:
                # Each node connects to a limited number of random nodes
                max_connections_per_node = min(100, int(expected_edges / size * 2))
                for i in range(size):
                    if i % (size // 20) == 0:
                        print(".", end="", flush=True)
                    
                    # Connect to random nodes within a reasonable range
                    connections = 0
                    for _ in range(max_connections_per_node):
                        if random.random() < edge_prob:
                            # Connect to a node within reasonable distance
                            target_range = min(size - i - 1, 10000)  # Limit range
                            if target_range > 0:
                                j = i + 1 + random.randint(0, target_range - 1)
                                graph.add_edge(f"n{i}", f"n{j}", weight=random.random())
                                edge_count += 1
                                connections += 1
                        if connections >= 50:  # Limit connections per node
                            break
            else:
                # Original algorithm for smaller graphs
                for i in range(size):
                    if i % (size // 10) == 0:
                        print(".", end="", flush=True)
                    for j in range(i + 1, min(i + 1000, size)):  # Limit range
                        if random.random() < edge_prob:
                            graph.add_edge(f"n{i}", f"n{j}", weight=random.random())
                            edge_count += 1
            
            print("✓")
            
            # Measure final stats
            end_time = time.perf_counter()
            end_memory = get_memory_usage()
            
            actual_nodes = graph.node_count()
            actual_edges = graph.edge_count()
            time_taken = end_time - start_time
            memory_used = end_memory - start_memory
            
            print(f"   ✅ SUCCESS!")
            print(f"   📊 Result: {actual_nodes:,} nodes, {actual_edges:,} edges")
            print(f"   ⏱️  Time: {time_taken:.1f}s")
            print(f"   💾 Memory used: {memory_used:.1f}MB (total: {end_memory:.1f}MB)")
            print(f"   🚀 Rate: {actual_nodes/time_taken:.0f} nodes/sec")
            
            # Quick query test on large graphs
            if actual_nodes >= 100_000:
                print("   🔍 Quick query test...", end="", flush=True)
                query_start = time.perf_counter()
                
                # Test basic queries
                node_count = graph.node_count()
                edge_count = graph.edge_count()
                
                # Test some neighbor queries
                sample_size = min(100, actual_nodes)
                neighbor_counts = []
                for i in range(0, sample_size, 10):
                    neighbors = graph.get_neighbors(f"n{i}")
                    neighbor_counts.append(len(neighbors))
                
                query_time = time.perf_counter() - query_start
                avg_neighbors = sum(neighbor_counts) / len(neighbor_counts) if neighbor_counts else 0
                print(f"✓ ({query_time:.3f}s, avg {avg_neighbors:.1f} neighbors)")
            
            successful_configs.append((description, actual_nodes, actual_edges, time_taken, memory_used))
            
            # Clean up
            del graph
            gc.collect()
            
        except MemoryError as e:
            print(f"\n   💥 MEMORY ERROR: {str(e)}")
            breaking_point = (description, "Memory limit reached")
            break
        except Exception as e:
            print(f"\n   💥 FAILED: {str(e)}")
            breaking_point = (description, str(e))
            break
        except KeyboardInterrupt:
            print(f"\n   ⏹️  Interrupted by user")
            breaking_point = (description, "User interrupted")
            break
    
    return successful_configs, breaking_point

def main():
    print("🔥 GLI EXTREME STRESS TEST 🔥")
    print("Finding the absolute limits of the Rust backend!")
    print("=" * 60)
    
    initial_memory = get_memory_usage()
    print(f"🏁 Starting memory: {initial_memory:.1f}MB")
    
    try:
        successful_configs, breaking_point = extreme_stress_test()
        
        print("\n" + "="*60)
        print("🏆 EXTREME STRESS TEST RESULTS")
        print("="*60)
        
        if successful_configs:
            print("✅ SUCCESSFULLY HANDLED:")
            total_time = 0
            for desc, nodes, edges, time_taken, memory in successful_configs:
                print(f"  🎯 {desc}")
                print(f"     📊 {nodes:,} nodes, {edges:,} edges")
                print(f"     ⏱️  {time_taken:.1f}s, 💾 {memory:.1f}MB")
                total_time += time_taken
            
            # Find records
            largest_nodes = max(successful_configs, key=lambda x: x[1])
            largest_edges = max(successful_configs, key=lambda x: x[2])
            fastest = min(successful_configs, key=lambda x: x[3])
            
            print(f"\n🏆 RECORDS SET:")
            print(f"  🥇 Most nodes: {largest_nodes[0]} - {largest_nodes[1]:,} nodes")
            print(f"  🥇 Most edges: {largest_edges[0]} - {largest_edges[2]:,} edges")
            print(f"  🥇 Fastest: {fastest[0]} - {fastest[3]:.1f}s")
            print(f"  ⏱️  Total test time: {total_time:.1f}s")
        
        if breaking_point:
            print(f"\n💥 BREAKING POINT FOUND:")
            print(f"  🚫 Failed at: {breaking_point[0]}")
            print(f"  💀 Reason: {breaking_point[1]}")
        
        final_memory = get_memory_usage()
        print(f"\n💾 Memory usage: {initial_memory:.1f}MB → {final_memory:.1f}MB")
        print(f"🎉 The Rust backend is INCREDIBLY CAPABLE!")
        
    except Exception as e:
        print(f"\n💥 Test framework error: {str(e)}")

if __name__ == "__main__":
    main()
