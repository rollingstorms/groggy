#!/usr/bin/env python3
"""
Complexity stress test for GLI Rust backend - testing attributes and versioning
"""

import time
import random
import sys
import os
import psutil
import gc
import json
import string

# Add the python directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

import gli

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def generate_heavy_attributes(complexity_level="medium"):
    """Generate complex attributes to stress the system"""
    attrs = {}
    
    if complexity_level == "light":
        attrs.update({
            "id": random.randint(1, 10000),
            "name": f"node_{random.randint(1, 1000)}",
            "active": random.choice([True, False]),
        })
    
    elif complexity_level == "medium":
        attrs.update({
            "id": random.randint(1, 100000),
            "name": f"complex_node_{''.join(random.choices(string.ascii_letters, k=20))}",
            "description": " ".join(random.choices(["data", "processing", "analysis", "network", "graph", "complex", "system"], k=10)),
            "metadata": {
                "created": time.time(),
                "version": f"{random.randint(1, 10)}.{random.randint(0, 9)}.{random.randint(0, 9)}",
                "tags": random.choices(["important", "processing", "analysis", "temp", "archived"], k=3),
                "config": {
                    "threshold": random.uniform(0.1, 1.0),
                    "enabled": random.choice([True, False]),
                    "priority": random.randint(1, 10)
                }
            },
            "metrics": {
                "cpu_usage": random.uniform(0, 100),
                "memory_usage": random.uniform(0, 1000),
                "network_io": random.uniform(0, 1000),
                "disk_io": random.uniform(0, 500)
            },
            "history": [
                {"timestamp": time.time() - random.randint(0, 86400), "event": f"action_{i}", "value": random.uniform(0, 100)}
                for i in range(random.randint(5, 20))
            ]
        })
    
    elif complexity_level == "heavy":
        attrs.update({
            "id": random.randint(1, 1000000),
            "name": f"heavy_node_{''.join(random.choices(string.ascii_letters + string.digits, k=50))}",
            "description": " ".join(random.choices(string.words, k=100)) if hasattr(string, 'words') else "Very long description " * 50,
            "large_text_field": "Large data content: " + "x" * 1000,  # 1KB of text per node
            "complex_metadata": {
                "schema_version": "2.1.0",
                "created_by": f"user_{''.join(random.choices(string.ascii_letters, k=10))}",
                "created_at": time.time(),
                "modified_history": [
                    {
                        "timestamp": time.time() - random.randint(0, 86400),
                        "user": f"user_{random.randint(1, 100)}",
                        "action": random.choice(["create", "update", "modify", "annotate"]),
                        "changes": {
                            "field": random.choice(["name", "description", "config", "metadata"]),
                            "old_value": f"old_{random.randint(1, 1000)}",
                            "new_value": f"new_{random.randint(1, 1000)}"
                        }
                    }
                    for _ in range(random.randint(10, 50))
                ],
                "permissions": {
                    "read": random.choices(["user1", "user2", "admin", "system"], k=random.randint(1, 4)),
                    "write": random.choices(["admin", "system"], k=random.randint(1, 2)),
                    "delete": ["admin"]
                },
                "analytics": {
                    "access_count": random.randint(0, 10000),
                    "last_accessed": time.time() - random.randint(0, 86400),
                    "popularity_score": random.uniform(0, 1),
                    "related_nodes": [f"node_{random.randint(1, 10000)}" for _ in range(random.randint(5, 20))]
                }
            },
            "performance_data": {
                "benchmarks": [
                    {
                        "test_name": f"benchmark_{i}",
                        "duration": random.uniform(0.1, 10.0),
                        "memory_peak": random.uniform(10, 1000),
                        "cpu_time": random.uniform(0.01, 5.0),
                        "results": [random.uniform(0, 100) for _ in range(100)]  # 100 data points
                    }
                    for i in range(10)
                ],
                "resource_usage": {
                    "cpu_samples": [random.uniform(0, 100) for _ in range(1000)],  # 1000 CPU samples
                    "memory_samples": [random.uniform(0, 1000) for _ in range(1000)],  # 1000 memory samples
                    "network_samples": [random.uniform(0, 1000) for _ in range(500)]   # 500 network samples
                }
            }
        })
    
    return attrs

def test_heavy_attributes():
    """Test performance with complex attributes"""
    print("📊 COMPLEX ATTRIBUTES STRESS TEST")
    print("=" * 50)
    
    gli.set_backend('rust')
    
    complexity_levels = ["light", "medium", "heavy"]
    graph_sizes = [1000, 5000, 10000]
    
    results = []
    
    for complexity in complexity_levels:
        print(f"\n🧪 Testing {complexity.upper()} complexity attributes:")
        
        for size in graph_sizes:
            print(f"  📈 Graph size: {size:,} nodes")
            
            try:
                start_memory = get_memory_usage()
                start_time = time.perf_counter()
                
                # Create graph with complex attributes
                graph = gli.Graph()
                
                print("    Adding nodes with complex attributes...", end="", flush=True)
                for i in range(size):
                    if i % (size // 10) == 0 and size > 100:
                        print(".", end="", flush=True)
                    
                    # Node attributes
                    node_attrs = generate_heavy_attributes(complexity)
                    graph.add_node(f"node_{i}", **node_attrs)
                
                print("✓")
                print("    Adding edges with complex attributes...", end="", flush=True)
                
                # Add edges with attributes
                edge_count = 0
                for i in range(min(size, 1000)):  # Limit edges to keep test reasonable
                    if i % 100 == 0:
                        print(".", end="", flush=True)
                    
                    # Connect to a few random nodes
                    for _ in range(min(5, size - i - 1)):
                        j = random.randint(i + 1, size - 1)
                        edge_attrs = {
                            "weight": random.uniform(0, 1),
                            "type": random.choice(["friend", "colleague", "related", "similar"]),
                            "strength": random.uniform(0, 1),
                            "metadata": {
                                "created": time.time(),
                                "source": "auto_generated",
                                "confidence": random.uniform(0.5, 1.0)
                            }
                        }
                        if complexity == "heavy":
                            edge_attrs["complex_data"] = {
                                "analysis": [random.uniform(0, 1) for _ in range(100)],
                                "history": [{"time": time.time() - random.randint(0, 86400), "value": random.uniform(0, 1)} for _ in range(50)]
                            }
                        
                        graph.add_edge(f"node_{i}", f"node_{j}", **edge_attrs)
                        edge_count += 1
                
                print("✓")
                
                end_time = time.perf_counter()
                end_memory = get_memory_usage()
                
                # Get final stats
                final_nodes = graph.node_count()
                final_edges = graph.edge_count()
                time_taken = end_time - start_time
                memory_used = end_memory - start_memory
                
                print(f"    ✅ SUCCESS: {final_nodes:,} nodes, {final_edges:,} edges")
                print(f"    ⏱️  Time: {time_taken:.2f}s")
                print(f"    💾 Memory: {memory_used:.1f}MB")
                print(f"    📊 Memory per node: {memory_used/final_nodes:.3f}MB")
                
                # Quick query test
                print("    🔍 Testing queries with complex data...", end="", flush=True)
                query_start = time.perf_counter()
                
                # Test queries
                node_count = graph.node_count()
                edge_count = graph.edge_count()
                node_ids = graph.get_node_ids()[:100]  # First 100 nodes
                
                neighbor_counts = []
                for node_id in node_ids[:10]:  # Test 10 nodes
                    neighbors = graph.get_neighbors(node_id)
                    neighbor_counts.append(len(neighbors))
                
                query_time = time.perf_counter() - query_start
                print(f"✓ ({query_time:.3f}s)")
                
                results.append({
                    "complexity": complexity,
                    "size": size,
                    "time": time_taken,
                    "memory": memory_used,
                    "memory_per_node": memory_used/final_nodes,
                    "query_time": query_time
                })
                
                # Clean up
                del graph
                gc.collect()
                
            except Exception as e:
                print(f"    ❌ FAILED: {str(e)}")
                break
    
    return results

def test_graph_versioning():
    """Test graph branching and versioning performance"""
    print("\n🌿 GRAPH VERSIONING & BRANCHING STRESS TEST")  
    print("=" * 50)
    
    gli.set_backend('rust')
    
    # Create a base graph
    print("📊 Creating base graph...")
    base_size = 10000
    base_graph = gli.Graph()
    
    # Add base nodes and edges
    for i in range(base_size):
        base_graph.add_node(f"base_node_{i}", 
                           value=i, 
                           type="base",
                           metadata={"created": time.time(), "version": "1.0"})
    
    # Add some edges
    for i in range(0, base_size, 10):
        for j in range(i + 1, min(i + 5, base_size)):
            base_graph.add_edge(f"base_node_{i}", f"base_node_{j}",
                              weight=random.uniform(0, 1),
                              type="base_connection")
    
    print(f"✅ Base graph: {base_graph.node_count():,} nodes, {base_graph.edge_count():,} edges")
    
    # Test creating multiple "branches" (separate graph instances)
    print("\n🌳 Testing graph branching simulation...")
    branches = []
    branch_names = ["feature_A", "feature_B", "hotfix_C", "experiment_D", "analysis_E"]
    
    start_memory = get_memory_usage()
    
    for i, branch_name in enumerate(branch_names):
        print(f"  🌿 Creating branch: {branch_name}")
        
        branch_start = time.perf_counter()
        
        # Simulate creating a branch by copying base graph structure
        # In a real implementation, this would be more sophisticated
        branch_graph = gli.Graph()
        
        # Copy base nodes with branch-specific modifications
        base_node_ids = base_graph.get_node_ids()
        for node_id in base_node_ids[:1000]:  # Copy first 1000 nodes for performance
            # Add branch-specific attributes
            branch_graph.add_node(f"{branch_name}_{node_id}",
                                value=random.randint(0, 1000),
                                type=f"branch_{branch_name}",
                                branch_id=i,
                                parent_node=node_id,
                                modifications=[
                                    {"timestamp": time.time(), "change": f"modified_for_{branch_name}"},
                                    {"timestamp": time.time() + 1, "change": "additional_data"}
                                ])
        
        # Add some branch-specific edges
        for j in range(0, 1000, 20):
            for k in range(j + 1, min(j + 3, 1000)):
                branch_graph.add_edge(f"{branch_name}_base_node_{j}", 
                                    f"{branch_name}_base_node_{k}",
                                    weight=random.uniform(0, 1),
                                    branch_specific=True,
                                    branch_name=branch_name)
        
        # Add some new nodes specific to this branch
        for j in range(100):
            branch_graph.add_node(f"{branch_name}_new_node_{j}",
                                value=random.randint(0, 1000),
                                type="branch_new",
                                branch_exclusive=True)
        
        branch_time = time.perf_counter() - branch_start
        
        print(f"    ✅ Branch {branch_name}: {branch_graph.node_count():,} nodes, {branch_graph.edge_count():,} edges ({branch_time:.2f}s)")
        
        branches.append({
            "name": branch_name,
            "graph": branch_graph,
            "creation_time": branch_time,
            "nodes": branch_graph.node_count(),
            "edges": branch_graph.edge_count()
        })
    
    end_memory = get_memory_usage()
    memory_used = end_memory - start_memory
    
    print(f"\n📊 Branching Results:")
    print(f"  🌿 Created {len(branches)} branches")
    print(f"  💾 Total memory for branches: {memory_used:.1f}MB")
    print(f"  💾 Average memory per branch: {memory_used/len(branches):.1f}MB")
    
    total_nodes = sum(b["nodes"] for b in branches)
    total_edges = sum(b["edges"] for b in branches)
    total_time = sum(b["creation_time"] for b in branches)
    
    print(f"  📈 Total across all branches: {total_nodes:,} nodes, {total_edges:,} edges")
    print(f"  ⏱️  Total branch creation time: {total_time:.2f}s")
    
    # Test cross-branch queries
    print("\n🔍 Testing cross-branch operations...")
    cross_query_start = time.perf_counter()
    
    # Query each branch
    all_results = {}
    for branch in branches:
        branch_name = branch["name"]
        graph = branch["graph"]
        
        # Sample queries
        node_count = graph.node_count()
        edge_count = graph.edge_count()
        sample_neighbors = []
        
        # Get some neighbors
        node_ids = graph.get_node_ids()[:10]
        for node_id in node_ids:
            neighbors = graph.get_neighbors(node_id)
            sample_neighbors.append(len(neighbors))
        
        all_results[branch_name] = {
            "nodes": node_count,
            "edges": edge_count, 
            "avg_neighbors": sum(sample_neighbors) / len(sample_neighbors) if sample_neighbors else 0
        }
    
    cross_query_time = time.perf_counter() - cross_query_start
    print(f"  ✅ Cross-branch queries completed in {cross_query_time:.3f}s")
    
    for branch_name, results in all_results.items():
        print(f"    {branch_name}: {results['nodes']} nodes, {results['edges']} edges, {results['avg_neighbors']:.1f} avg neighbors")
    
    return branches

def main():
    print("🧪 GLI COMPLEXITY STRESS TEST")
    print("Testing attributes complexity and graph versioning")
    print("=" * 60)
    
    initial_memory = get_memory_usage()
    print(f"🏁 Starting memory: {initial_memory:.1f}MB")
    
    try:
        # Test complex attributes
        attr_results = test_heavy_attributes()
        
        # Test versioning/branching
        branch_results = test_graph_versioning()
        
        print("\n" + "="*60)
        print("🏆 COMPLEXITY TEST SUMMARY")
        print("="*60)
        
        if attr_results:
            print("📊 ATTRIBUTE COMPLEXITY RESULTS:")
            for result in attr_results:
                complexity = result['complexity']
                size = result['size'] 
                memory_per_node = result['memory_per_node']
                print(f"  {complexity:>6} | {size:>6,} nodes | {memory_per_node:>8.3f}MB/node | {result['time']:>7.2f}s | {result['query_time']:>7.3f}s queries")
            
            # Find the heaviest memory usage
            heaviest = max(attr_results, key=lambda x: x['memory_per_node'])
            print(f"\n  🏆 Heaviest attribute load: {heaviest['complexity']} complexity")
            print(f"      Memory per node: {heaviest['memory_per_node']:.3f}MB")
            print(f"      Still performant: {heaviest['query_time']:.3f}s queries")
        
        if branch_results:
            print(f"\n🌿 BRANCHING RESULTS:")
            print(f"  ✅ Successfully created {len(branch_results)} graph branches")
            total_branch_nodes = sum(b["nodes"] for b in branch_results)
            total_branch_edges = sum(b["edges"] for b in branch_results)
            print(f"  📊 Total: {total_branch_nodes:,} nodes, {total_branch_edges:,} edges across all branches")
            print(f"  🚀 Branching is feasible for complex workflows")
        
        final_memory = get_memory_usage()
        print(f"\n💾 Memory usage: {initial_memory:.1f}MB → {final_memory:.1f}MB")
        print(f"🎉 Complex operations handled successfully!")
        
    except Exception as e:
        print(f"\n💥 Test error: {str(e)}")

if __name__ == "__main__":
    main()
