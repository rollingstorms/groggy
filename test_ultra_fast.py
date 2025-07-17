#!/usr/bin/env python3
"""
Test Ultra-Fast Optimizations for 10x Performance Goal
"""

import groggy as gr
import time
import json

def test_ultra_fast_operations():
    """Test ultra-fast bulk operations targeting 10x performance"""
    print("=== Testing Ultra-Fast Bulk Operations ===")
    
    g = gr.Graph()
    
    # 1. Test ultra-fast bulk node addition with attributes
    print("\n1. Testing ultra_fast_add_nodes_with_attrs...")
    start = time.perf_counter()
    
    # Prepare bulk data
    nodes_data = []
    for i in range(1000):
        node_attrs = {
            'salary': json.dumps(50000 + i * 1000),
            'role': json.dumps(f"role_{i%4}"),
            'active': json.dumps(i % 2 == 0),
            'level': json.dumps(i % 5 + 1)
        }
        nodes_data.append((f"n{i}", node_attrs))
    
    g.ultra_fast_add_nodes_with_attrs(nodes_data)
    ultra_add_time = time.perf_counter() - start
    print(f"   Ultra-fast added 1000 nodes with 4 attrs each in {ultra_add_time:.6f}s")
    print(f"   Rate: {1000/ultra_add_time:.0f} nodes/sec, {4000/ultra_add_time:.0f} attrs/sec")
    
    # 2. Test ultra-fast vectorized attribute setting
    print("\n2. Testing ultra_fast_set_attrs_vectorized...")
    start = time.perf_counter()
    
    # Prepare vectorized data
    vector_data = [(f"n{i}", json.dumps(f"department_{i%3}")) for i in range(500)]
    g.ultra_fast_set_attrs_vectorized("department", vector_data)
    ultra_vector_time = time.perf_counter() - start
    print(f"   Ultra-fast vectorized set 500 attributes in {ultra_vector_time:.6f}s")
    print(f"   Rate: {500/ultra_vector_time:.0f} attrs/sec")
    
    # Check memory usage
    print("\n3. Memory efficiency:")
    info = g.info()
    fast_core_memory = float(info["attributes"].get("memory_fast_core_mb", "0"))
    total_entities = int(info["attributes"].get("fast_core_nodes", "0"))
    
    if total_entities > 0:
        bytes_per_entity = (fast_core_memory * 1024 * 1024) / total_entities
        print(f"   FastCore memory: {fast_core_memory:.2f} MB")
        print(f"   Total entities: {total_entities}")
        print(f"   Memory per entity: {bytes_per_entity:.1f} bytes")
    
    total_ultra_time = ultra_add_time + ultra_vector_time
    print(f"\n🚀 Total ultra-fast operations time: {total_ultra_time:.6f}s")
    
    return {
        'ultra_add_time': ultra_add_time,
        'ultra_vector_time': ultra_vector_time,
        'total_time': total_ultra_time,
        'memory_mb': fast_core_memory,
        'bytes_per_entity': bytes_per_entity if total_entities > 0 else 0
    }

def test_performance_progression():
    """Compare regular vs fast vs ultra-fast operations"""
    print("\n=== Performance Progression Test ===")
    
    test_size = 1000
    
    # 1. Regular operations baseline
    print(f"\n1. Regular operations ({test_size} nodes with attrs)...")
    g1 = gr.Graph()
    start = time.perf_counter()
    nodes_data = [{'id': f"n{i}", 'salary': 50000 + i * 1000, 'role': f'role_{i%4}'} for i in range(test_size)]
    g1.nodes.add(nodes_data)
    regular_time = time.perf_counter() - start
    print(f"   Regular: {regular_time:.6f}s ({test_size/regular_time:.0f} nodes/sec)")
    
    # 2. Fast operations
    print(f"\n2. Fast operations ({test_size} nodes + attrs separately)...")
    g2 = gr.Graph()
    start = time.perf_counter()
    nodes = [f"n{i}" for i in range(test_size)]
    g2.fast_add_nodes(nodes)
    batch_data = {f"n{i}": json.dumps(50000 + i * 1000) for i in range(test_size)}
    g2.fast_set_node_attrs_batch("salary", batch_data)
    fast_time = time.perf_counter() - start
    print(f"   Fast: {fast_time:.6f}s ({test_size/fast_time:.0f} nodes/sec)")
    
    # 3. Ultra-fast operations
    print(f"\n3. Ultra-fast operations ({test_size} nodes with attrs atomically)...")
    g3 = gr.Graph()
    start = time.perf_counter()
    ultra_data = [(f"n{i}", {'salary': json.dumps(50000 + i * 1000), 'role': json.dumps(f'role_{i%4}')}) for i in range(test_size)]
    g3.ultra_fast_add_nodes_with_attrs(ultra_data)
    ultra_time = time.perf_counter() - start
    print(f"   Ultra-fast: {ultra_time:.6f}s ({test_size/ultra_time:.0f} nodes/sec)")
    
    # Calculate speedups
    fast_speedup = regular_time / fast_time if fast_time > 0 else float('inf')
    ultra_speedup = regular_time / ultra_time if ultra_time > 0 else float('inf')
    ultra_vs_fast = fast_time / ultra_time if ultra_time > 0 else float('inf')
    
    print(f"\n🎯 Performance Progression Results:")
    print(f"   Fast vs Regular: {fast_speedup:.2f}x speedup")
    print(f"   Ultra-fast vs Regular: {ultra_speedup:.2f}x speedup")
    print(f"   Ultra-fast vs Fast: {ultra_vs_fast:.2f}x speedup")
    
    return {
        'regular_time': regular_time,
        'fast_time': fast_time,
        'ultra_time': ultra_time,
        'fast_speedup': fast_speedup,
        'ultra_speedup': ultra_speedup
    }

def test_scalability():
    """Test scalability with larger datasets"""
    print("\n=== Scalability Test ===")
    
    sizes = [1000, 5000, 10000]
    results = {}
    
    for size in sizes:
        print(f"\n📊 Testing with {size:,} nodes...")
        
        g = gr.Graph()
        start = time.perf_counter()
        
        # Ultra-fast bulk operation
        ultra_data = [(f"n{i}", {
            'salary': json.dumps(50000 + i * 1000),
            'role': json.dumps(f'role_{i%4}'),
            'level': json.dumps(i % 5 + 1)
        }) for i in range(size)]
        
        g.ultra_fast_add_nodes_with_attrs(ultra_data)
        elapsed = time.perf_counter() - start
        
        rate = size / elapsed if elapsed > 0 else float('inf')
        print(f"   {size:,} nodes in {elapsed:.6f}s ({rate:.0f} nodes/sec)")
        
        # Memory check
        info = g.info()
        memory_mb = float(info["attributes"].get("memory_fast_core_mb", "0"))
        print(f"   Memory: {memory_mb:.2f} MB ({memory_mb*1024/size:.1f} KB/node)")
        
        results[size] = {'time': elapsed, 'rate': rate, 'memory_mb': memory_mb}
    
    return results

if __name__ == "__main__":
    print("🚀 Ultra-Fast Groggy Optimization Test")
    print("=" * 60)
    
    # Test ultra-fast operations
    ultra_results = test_ultra_fast_operations()
    
    # Test performance progression
    progression_results = test_performance_progression()
    
    # Test scalability
    scalability_results = test_scalability()
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"   Ultra-fast speedup vs regular: {progression_results['ultra_speedup']:.2f}x")
    print(f"   Memory efficiency: {ultra_results['bytes_per_entity']:.1f} bytes/entity")
    print(f"   Peak rate: {max(r['rate'] for r in scalability_results.values()):.0f} nodes/sec")
    
    # Check if we hit our 10x target
    target = 10.0
    if progression_results['ultra_speedup'] >= target:
        print(f"✅ SUCCESS: Achieved {progression_results['ultra_speedup']:.2f}x speedup (target: {target}x)")
    elif progression_results['ultra_speedup'] >= 5.0:
        print(f"🎯 GOOD PROGRESS: Achieved {progression_results['ultra_speedup']:.2f}x speedup (target: {target}x)")
    else:
        print(f"⚡ EARLY STAGE: Achieved {progression_results['ultra_speedup']:.2f}x speedup (target: {target}x)")
    
    print("\n🔬 Next optimization targets:")
    if progression_results['ultra_speedup'] < 10:
        remaining = target / progression_results['ultra_speedup']
        print(f"   Need {remaining:.1f}x more improvement for 10x goal")
        print(f"   Consider: SIMD operations, lock-free data structures, memory layout optimization")