#!/usr/bin/env python3
"""
Performance profiling for Builder IR system.

Micro-benchmarks to measure:
- FFI crossing overhead
- Operation-level performance
- Memory allocation patterns
- Compilation time vs execution time
- Fusion opportunities
"""

import time
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import groggy as gr
from groggy.builder import AlgorithmBuilder, algorithm


def timer(func):
    """Simple timing decorator."""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        return result, (end - start) * 1000  # ms
    return wrapper


class BuilderProfiler:
    """Profile Builder IR performance and optimization opportunities."""
    
    def __init__(self):
        self.results = {}
        self.graph = None
        
    def setup_graph(self, n_nodes=1000, n_edges=5000):
        """Create test graph."""
        print(f"Setting up test graph: {n_nodes} nodes, {n_edges} edges")
        self.graph = gr.Graph()
        
        # Add nodes
        for i in range(n_nodes):
            self.graph.add_node(i)
        
        # Add edges (random-ish but deterministic)
        import random
        random.seed(42)
        edges_added = 0
        while edges_added < n_edges:
            src = random.randint(0, n_nodes - 1)
            dst = random.randint(0, n_nodes - 1)
            if src != dst:
                self.graph.add_edge(src, dst)
                edges_added += 1
        
        print(f"Graph created: {self.graph.node_count()} nodes, {self.graph.edge_count()} edges")
        
    def profile_ffi_overhead(self):
        """Measure raw FFI crossing cost."""
        print("\n=== FFI Overhead Profiling ===")
        
        # 1. Empty operation (baseline)
        @timer
        def baseline():
            return self.graph.node_count()
        
        _, time_baseline = baseline()
        print(f"Baseline (node_count call): {time_baseline:.4f} ms")
        
        # 2. Multiple independent FFI calls
        @timer
        def multiple_ffi():
            n = self.graph.node_count()
            m = self.graph.edge_count()
            return n, m
        
        _, time_multi = multiple_ffi()
        print(f"Two FFI calls: {time_multi:.4f} ms ({time_multi / time_baseline:.2f}x)")
        
        # 3. Measure per-call overhead
        n_calls = 100
        @timer
        def many_calls():
            for _ in range(n_calls):
                self.graph.node_count()
        
        _, time_many = many_calls()
        per_call = time_many / n_calls
        print(f"{n_calls} calls: {time_many:.4f} ms total, {per_call:.4f} ms per call")
        
        self.results['ffi_overhead_ms'] = per_call
        return per_call
        
    def profile_builder_operations(self):
        """Profile individual builder operations."""
        print("\n=== Builder Operation Profiling ===")
        
        # Test each primitive operation via algorithm decorator
        ops = {}
        
        # Simple node initialization
        @algorithm("test_init")
        def test_init(sG):
            return sG.nodes(1.0)
        
        @timer
        def run_init():
            sG = self.graph.to_subgraph()
            spec = test_init()
            result = sG.apply(spec)
        
        _, init_time = run_init()
        ops['init_nodes'] = init_time
        print(f"  {'init_nodes':20s}: {init_time:8.4f} ms")
        
        # Scalar multiplication
        @algorithm("test_mul")
        def test_mul(sG):
            x = sG.nodes(1.0)
            return x * 2.0
        
        @timer
        def run_mul():
            sG = self.graph.to_subgraph()
            spec = test_mul()
            result = sG.apply(spec)
        
        _, mul_time = run_mul()
        ops['scalar_mul'] = mul_time
        print(f"  {'scalar_mul':20s}: {mul_time:8.4f} ms")
        
        # Degrees
        @algorithm("test_degrees")
        def test_degrees(sG):
            x = sG.nodes(1.0)
            return x.degrees()
        
        @timer
        def run_degrees():
            sG = self.graph.to_subgraph()
            spec = test_degrees()
            result = sG.apply(spec)
        
        _, deg_time = run_degrees()
        ops['degrees'] = deg_time
        print(f"  {'degrees':20s}: {deg_time:8.4f} ms")
        
        self.results['operation_times'] = ops
        return ops
        
    def profile_compilation_overhead(self):
        """Measure IR construction vs execution time."""
        print("\n=== Compilation Overhead ===")
        
        # Build a medium-complexity algorithm
        @algorithm("complex_test")
        def complex_algo(sG):
            ranks = sG.nodes(1.0)
            deg = ranks.degrees()
            inv_deg = 1.0 / (deg + 1e-9)
            
            # Some arithmetic
            x = ranks * inv_deg
            y = x + 0.15
            z = y * 0.85
            
            return z
        
        @timer
        def build_spec():
            return complex_algo()
        
        spec, build_time = build_spec()
        print(f"IR construction: {build_time:.4f} ms")
        
        @timer
        def execute():
            sG = self.graph.to_subgraph()
            result = sG.apply(spec)
        
        _, exec_time = execute()
        print(f"Execution:       {exec_time:.4f} ms")
        print(f"Ratio (build/exec): {build_time / exec_time:.2f}x")
        
        self.results['build_time_ms'] = build_time
        self.results['exec_time_ms'] = exec_time
        
        return build_time, exec_time
        
    def profile_fusion_opportunities(self):
        """Identify potential fusion opportunities in real algorithms."""
        print("\n=== Fusion Opportunity Analysis ===")
        
        # Create a PageRank-like pattern and analyze its IR
        @algorithm("pagerank_pattern")
        def pagerank_pattern(sG):
            n = sG.N
            ranks = sG.nodes(1.0)
            inv_n = 1.0 / (n + 1e-9)
            
            deg = ranks.degrees()
            inv_deg = 1.0 / (deg + 1e-9)
            is_sink = (deg == 0.0)
            
            # Fusable chain: ranks * inv_deg
            contrib = ranks * inv_deg
            
            # Fusable with where: where(is_sink, 0.0, contrib)
            contrib = is_sink.where(0.0, contrib)
            
            # Neighbor aggregation (could fuse with above)
            neighbor_sum = sG @ contrib
            
            # More fusable arithmetic: 0.85 * neighbor_sum + 0.15 / n
            damped = neighbor_sum * 0.85
            teleport_val = 0.15 / (n + 1e-9)
            ranks_new = damped + teleport_val
            
            return ranks_new
        
        # Build the spec to analyze IR
        spec = pagerank_pattern()
        
        # Get the builder's IR from the spec (we need to access internal state)
        # For now, just count steps
        if hasattr(spec, 'steps'):
            total_ops = len(spec['steps'])
            print(f"Total operations: {total_ops}")
            
            # Manually identify fusable patterns by analyzing step types
            fusable_patterns = [
                ('mul', 'where'),  # ranks * inv_deg then where
                ('mul', 'add'),    # 0.85 * x + 0.15 * y
            ]
            
            # Count potential fusion opportunities
            fusion_count = 0
            for i in range(len(spec['steps']) - 1):
                step1 = spec['steps'][i]['type']
                step2 = spec['steps'][i+1]['type']
                if any(step1 == p[0] and step2 == p[1] for p in fusable_patterns):
                    fusion_count += 1
            
            print(f"Potential fusion opportunities: {fusion_count}")
            print(f"Operations could be reduced by: {fusion_count} ({fusion_count/total_ops*100:.1f}%)")
            
            theoretical_speedup = total_ops / (total_ops - fusion_count) if fusion_count > 0 else 1.0
            print(f"Theoretical speedup: {theoretical_speedup:.2f}x")
            
            self.results['fusion_opportunities'] = {
                'total_ops': total_ops,
                'fusion_chains': fusion_count,
                'ops_saved': fusion_count,
                'theoretical_speedup': theoretical_speedup
            }
        
        return None
        
    def profile_loop_overhead(self):
        """Measure overhead of loop constructs."""
        print("\n=== Loop Overhead ===")
        
        # Simple loop pattern
        @algorithm("loop_test")
        def loop_test(sG):
            x = sG.nodes(1.0)
            with sG.iterate(10):
                x = x * 0.9
                x = x + 0.1
                x = sG.var("x", x)
            return x
        
        @timer
        def with_loop():
            sG = self.graph.to_subgraph()
            spec = loop_test()
            result = sG.apply(spec)
        
        _, loop_time = with_loop()
        print(f"10 iterations (looped): {loop_time:.4f} ms")
        
        # Unrolled version
        @algorithm("unrolled_test")
        def unrolled_test(sG):
            x = sG.nodes(1.0)
            for _ in range(10):
                x = x * 0.9
                x = x + 0.1
            return x
        
        @timer
        def unrolled():
            sG = self.graph.to_subgraph()
            spec = unrolled_test()
            result = sG.apply(spec)
        
        _, unroll_time = unrolled()
        print(f"10 iterations (unrolled): {unroll_time:.4f} ms")
        print(f"Loop overhead: {(loop_time - unroll_time):.4f} ms")
        
        self.results['loop_overhead_ms'] = loop_time - unroll_time
        
        return loop_time, unroll_time
        
    def compare_with_native(self):
        """Compare builder performance with native implementation."""
        print("\n=== Builder vs Native Performance ===")
        
        # Builder version using the decorator - simplified
        @algorithm("pagerank_builder_simple")
        def pagerank_builder(sG, damping=0.85, max_iter=10):
            ranks = sG.nodes(1.0)
            deg = ranks.degrees()
            
            with sG.iterate(max_iter):
                # Simple neighbor aggregation
                neighbor_sum = sG @ ranks
                
                # Simplified PageRank update
                ranks = sG.var("ranks", neighbor_sum * damping + (1.0 - damping))
            
            return ranks.normalize()
        
        @timer
        def run_builder():
            sG = self.graph.to_subgraph()
            spec = pagerank_builder(damping=0.85, max_iter=10)
            result = sG.apply(spec)
        
        _, builder_time = run_builder()
        print(f"Builder PageRank (10 iter): {builder_time:.4f} ms")
        
        # Native version (if available)
        try:
            @timer
            def native_pagerank():
                sG = self.graph.to_subgraph()
                # Check if there's a native pagerank method
                if hasattr(sG, 'pagerank'):
                    sG.pagerank(damping=0.85, max_iter=10, tol=1e-6)
                else:
                    # Use graph-level if available
                    self.graph.pagerank(damping=0.85, max_iter=10, tol=1e-6)
            
            _, native_time = native_pagerank()
            print(f"Native PageRank (10 iter):  {native_time:.4f} ms")
            print(f"Overhead: {(builder_time / native_time):.2f}x")
            
            self.results['pagerank_builder_ms'] = builder_time
            self.results['pagerank_native_ms'] = native_time
            self.results['overhead_ratio'] = builder_time / native_time
            
            return builder_time, native_time
        except Exception as e:
            print(f"Native PageRank not available: {e}")
            print(f"Builder PageRank time: {builder_time:.4f} ms (no native comparison)")
            self.results['pagerank_builder_ms'] = builder_time
            return builder_time, None
        
    def generate_report(self):
        """Generate comprehensive performance report."""
        print("\n" + "="*60)
        print("BUILDER IR PERFORMANCE BASELINE REPORT")
        print("="*60)
        
        if 'ffi_overhead_ms' in self.results:
            print(f"\n📊 FFI Overhead: {self.results['ffi_overhead_ms']:.4f} ms per call")
        
        if 'fusion_opportunities' in self.results:
            fo = self.results['fusion_opportunities']
            print(f"\n🔥 Fusion Opportunities:")
            print(f"   - Operations saved: {fo['ops_saved']} ({fo['ops_saved']/fo['total_ops']*100:.1f}%)")
            print(f"   - Theoretical speedup: {fo['theoretical_speedup']:.2f}x")
        
        if 'overhead_ratio' in self.results:
            print(f"\n⚡ Builder vs Native:")
            print(f"   - Builder: {self.results['pagerank_builder_ms']:.4f} ms")
            print(f"   - Native:  {self.results['pagerank_native_ms']:.4f} ms")
            print(f"   - Overhead: {self.results['overhead_ratio']:.2f}x")
        
        if 'build_time_ms' in self.results and 'exec_time_ms' in self.results:
            ratio = self.results['build_time_ms'] / self.results['exec_time_ms']
            print(f"\n🏗️  Compilation Overhead:")
            print(f"   - Build time: {self.results['build_time_ms']:.4f} ms")
            print(f"   - Exec time:  {self.results['exec_time_ms']:.4f} ms")
            print(f"   - Ratio:      {ratio:.2f}x")
        
        print("\n" + "="*60)
        print("Next Steps:")
        print("1. Implement arithmetic fusion (target: 50% ops reduction)")
        print("2. Add neighbor aggregation fusion")
        print("3. Implement loop hoisting for invariants")
        print("4. Target overhead < 2x native performance")
        print("="*60)
        
        return self.results


def main():
    """Run full profiling suite."""
    profiler = BuilderProfiler()
    
    # Setup
    profiler.setup_graph(n_nodes=1000, n_edges=5000)
    
    # Run all profiling tasks
    profiler.profile_ffi_overhead()
    profiler.profile_builder_operations()
    profiler.profile_compilation_overhead()
    profiler.profile_fusion_opportunities()
    profiler.profile_loop_overhead()
    profiler.compare_with_native()
    
    # Generate report
    results = profiler.generate_report()
    
    # Save results
    output_file = Path(__file__).parent / "builder_ir_baseline.json"
    import json
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to {output_file}")


if __name__ == "__main__":
    main()
