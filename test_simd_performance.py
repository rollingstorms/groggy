#!/usr/bin/env python3
"""
Test SIMD Performance Improvements
Compare regular vs SIMD-optimized NumArray operations
"""

import time
import statistics
import groggy as gr

def benchmark_comparison(operation_name: str, regular_func, simd_func, data_size: int, iterations: int = 100):
    """Compare regular vs SIMD performance"""
    
    # Generate test data
    test_data = [i * 0.1 for i in range(data_size)]
    
    # Create NumArray (this should be f64 array internally)
    num_array = gr.num_array(test_data)
    
    print(f"\n🔧 Benchmarking {operation_name} (size: {data_size})")
    
    # Benchmark regular implementation
    regular_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        regular_result = regular_func(num_array)
        end = time.perf_counter()
        regular_times.append(end - start)
    
    regular_avg = statistics.mean(regular_times) * 1_000_000  # Convert to microseconds
    
    # Check if SIMD methods are available
    try:
        # Test SIMD implementation  
        simd_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            simd_result = simd_func(num_array)
            end = time.perf_counter()
            simd_times.append(end - start)
        
        simd_avg = statistics.mean(simd_times) * 1_000_000
        
        # Calculate speedup
        speedup = regular_avg / simd_avg
        
        print(f"  Regular : {regular_avg:8.2f}µs")
        print(f"  SIMD    : {simd_avg:8.2f}µs") 
        print(f"  Speedup : {speedup:.2f}x {'🚀' if speedup > 1.1 else '📊'}")
        
        # Verify results are similar
        if regular_result is not None and simd_result is not None:
            if isinstance(regular_result, (int, float)) and isinstance(simd_result, (int, float)):
                diff = abs(regular_result - simd_result)
                if diff > 1e-10:
                    print(f"  ⚠️  Results differ: {regular_result} vs {simd_result}")
                else:
                    print(f"  ✓ Results match: {regular_result}")
        
        return speedup
        
    except AttributeError as e:
        print(f"  SIMD method not available: {e}")
        print(f"  Regular : {regular_avg:8.2f}µs")
        return 0.0

def test_simd_optimizations():
    """Test all SIMD-optimized operations"""
    print("🚀 Testing SIMD Performance Optimizations")
    print("=" * 60)
    
    # Test different array sizes
    sizes = [100, 1000, 10000, 50000]
    
    speedups = []
    
    for size in sizes:
        print(f"\n📊 Array Size: {size}")
        print("-" * 30)
        
        # Test available operations
        operations = {
            'sum': (
                lambda arr: arr.sum(),
                lambda arr: arr.sum_simd() if hasattr(arr, 'sum_simd') else arr.sum()
            ),
            'mean': (
                lambda arr: arr.mean(),
                lambda arr: arr.mean_simd() if hasattr(arr, 'mean_simd') else arr.mean()
            ),
            'min': (
                lambda arr: arr.min(),
                lambda arr: arr.min_simd() if hasattr(arr, 'min_simd') else arr.min()
            ),
            'max': (
                lambda arr: arr.max(),
                lambda arr: arr.max_simd() if hasattr(arr, 'max_simd') else arr.max()
            ),
            'median': (
                lambda arr: arr.median(),
                lambda arr: arr.median_optimized() if hasattr(arr, 'median_optimized') else arr.median()
            ),
        }
        
        size_speedups = []
        
        for op_name, (regular_func, simd_func) in operations.items():
            speedup = benchmark_comparison(op_name, regular_func, simd_func, size)
            if speedup > 0:
                size_speedups.append(speedup)
        
        if size_speedups:
            avg_speedup = statistics.mean(size_speedups)
            print(f"\n  📈 Average speedup for size {size}: {avg_speedup:.2f}x")
            speedups.extend(size_speedups)
    
    # Overall summary
    if speedups:
        print(f"\n🎯 OVERALL PERFORMANCE SUMMARY:")
        print(f"  • Total tests run: {len(speedups)}")
        print(f"  • Average speedup: {statistics.mean(speedups):.2f}x")
        print(f"  • Best speedup: {max(speedups):.2f}x")
        print(f"  • Speedups > 1.5x: {sum(1 for s in speedups if s > 1.5)}/{len(speedups)}")
        
        if statistics.mean(speedups) > 1.2:
            print("  ✅ SIMD optimizations are working effectively!")
        else:
            print("  📊 SIMD optimizations provide modest improvements")
    else:
        print("  ⚠️ SIMD methods may not be exposed to Python yet")

def test_auto_selection():
    """Test auto-selection between regular and SIMD"""
    print(f"\n🤖 Testing Auto-Selection Algorithms")
    print("-" * 40)
    
    # Test auto methods if they exist
    test_data = [i * 0.1 for i in range(10000)]
    num_array = gr.num_array(test_data)
    
    auto_methods = ['sum_auto', 'mean_auto', 'median_auto']
    
    for method_name in auto_methods:
        if hasattr(num_array, method_name):
            method = getattr(num_array, method_name)
            
            start = time.perf_counter()
            for _ in range(100):
                result = method()
            duration = time.perf_counter() - start
            
            print(f"  {method_name:12}: {duration * 10_000:.2f}µs avg, result: {result}")
        else:
            print(f"  {method_name:12}: Not available")

def main():
    """Main test function"""
    print("🧪 SIMD Performance Test Suite")
    print("Testing Phase 2.3+ SIMD optimizations")
    print("=" * 70)
    
    try:
        print("✓ Successfully imported groggy")
        
        # Test SIMD optimizations
        test_simd_optimizations()
        
        # Test auto-selection methods
        test_auto_selection()
        
        print("\n🎉 SIMD performance testing completed!")
        print("\n💡 Next Steps:")
        print("- SIMD methods may need FFI bindings for Python access")
        print("- Consider making SIMD the default for larger arrays")
        print("- Add more SIMD operations (std_dev, variance, correlation)")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()