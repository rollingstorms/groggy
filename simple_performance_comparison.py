#!/usr/bin/env python3
"""
Simple side-by-side performance comparison using existing stress tests
"""

import subprocess
import time
import sys
import statistics
from typing import Dict, List, Tuple

def run_rust_stress_test(size_args: str) -> Dict[str, float]:
    """Run the existing Rust stress test and parse output"""
    print(f"🦀 Running Rust stress test: {size_args}")
    
    # Use our existing stress_test_rust.rs
    try:
        result = subprocess.run([
            'cargo', 'run', '--release', '--bin=stress_test_rust'
        ], capture_output=True, text=True, cwd='/Users/michaelroth/Documents/Code/groggy', timeout=60)
        
        if result.returncode != 0:
            print(f"❌ Rust test failed: {result.stderr}")
            return {}
            
        # Parse timing info from output (take only first test run for simplicity)
        lines = result.stdout.split('\n')
        metrics = {}
        parsing_first_test = False
        
        for line in lines:
            # Start parsing when we see the first test
            if 'Testing 10000 nodes, 5000 edges' in line:
                parsing_first_test = True
                continue
            # Stop parsing when we see the second test
            elif 'Testing 50000 nodes' in line:
                parsing_first_test = False
                break
                
            if not parsing_first_test:
                continue
        
        def parse_time_to_ms(time_str):
            """Parse time string with different units to milliseconds"""
            time_str = time_str.strip()
            
            # Use regex to extract just the numeric value and unit
            import re
            match = re.search(r'(\d+\.?\d*)\s*([a-zA-Zµμ]+)', time_str)
            if not match:
                # Try just numeric value
                numeric_match = re.search(r'(\d+\.?\d*)', time_str)
                if numeric_match:
                    return float(numeric_match.group(1))
                return 0.0
                
            value = float(match.group(1))
            unit = match.group(2).lower()
            
            if unit == 'ms':
                return value
            elif unit in ['µs', 'μs', 'us']:  # Handle both variants of microsecond symbol
                return value / 1000.0
            elif unit == 's':
                return value * 1000.0
            elif unit == 'ns':
                return value / 1000000.0
            else:
                # Default to assuming milliseconds
                return value

        for line in lines:
            if 'Graph creation:' in line:
                # Extract time string after the colon
                parts = line.split(':', 1)
                if len(parts) >= 2:
                    time_str = parts[1].strip()
                    metrics['creation_time_ms'] = parse_time_to_ms(time_str)
                    
            elif 'Single attribute:' in line:
                # Extract time between colon and opening parenthesis
                parts = line.split(':', 1)
                if len(parts) >= 2:
                    time_part = parts[1].split('(')[0].strip()
                    metrics['single_filter_time_ms'] = parse_time_to_ms(time_part)
                    
            elif 'Complex AND:' in line:
                parts = line.split(':', 1)
                if len(parts) >= 2:
                    time_part = parts[1].split('(')[0].strip()
                    metrics['complex_filter_time_ms'] = parse_time_to_ms(time_part)
                    
            elif 'Connected components:' in line:
                parts = line.split(':', 1)
                if len(parts) >= 2:
                    time_part = parts[1].split('(')[0].strip()
                    metrics['connected_components_time_ms'] = parse_time_to_ms(time_part)
                    
            elif 'BFS traversal:' in line:
                parts = line.split(':', 1)
                if len(parts) >= 2:
                    time_part = parts[1].split('(')[0].strip()
                    metrics['bfs_time_ms'] = parse_time_to_ms(time_part)
        
        return metrics
        
    except subprocess.TimeoutExpired:
        print("⏰ Rust test timed out")
        return {}
    except Exception as e:
        print(f"❌ Rust test error: {e}")
        return {}

def run_python_stress_test() -> Dict[str, float]:
    """Run the existing Python stress test and parse output"""
    print(f"🐍 Running Python stress test (OPTIMIZED)")
    
    try:
        result = subprocess.run([
            'python', 'stress_test_python.py'
        ], capture_output=True, text=True, cwd='/Users/michaelroth/Documents/Code/groggy', timeout=120)
        
        if result.returncode != 0:
            print(f"❌ Python test failed: {result.stderr}")
            print(f"stdout: {result.stdout}")
            return {}
            
        # Parse timing info from output (take only first test run for simplicity)
        lines = result.stdout.split('\n')
        metrics = {}
        parsing_first_test = False
        
        for line in lines:
            # Start parsing when we see the first test
            if 'Testing 10000 nodes, 5000 edges' in line:
                parsing_first_test = True
                continue
            # Stop parsing when we see the second test
            elif 'Testing 50000 nodes' in line:
                parsing_first_test = False
                break
                
            if not parsing_first_test:
                continue
        
        for line in lines:
            if 'Graph creation:' in line:
                # Extract time in seconds, convert to ms
                parts = line.split(':', 1)
                if len(parts) >= 2:
                    time_str = parts[1].strip()
                    if 's' in time_str:
                        metrics['creation_time_ms'] = float(time_str.replace('s', '').strip()) * 1000
                    
            elif 'Single attribute:' in line:
                parts = line.split(':', 1)
                if len(parts) >= 2:
                    time_part = parts[1].split('(')[0].strip()
                    if 's' in time_part:
                        metrics['single_filter_time_ms'] = float(time_part.replace('s', '').strip()) * 1000
                    
            elif 'Complex AND:' in line:
                parts = line.split(':', 1)
                if len(parts) >= 2:
                    time_part = parts[1].split('(')[0].strip()
                    if 's' in time_part:
                        metrics['complex_filter_time_ms'] = float(time_part.replace('s', '').strip()) * 1000
                    
            elif 'Connected components:' in line:
                parts = line.split(':', 1)
                if len(parts) >= 2:
                    time_part = parts[1].split('(')[0].strip()
                    if 's' in time_part:
                        metrics['connected_components_time_ms'] = float(time_part.replace('s', '').strip()) * 1000
                    
            elif 'BFS traversal:' in line:
                parts = line.split(':', 1)
                if len(parts) >= 2:
                    time_part = parts[1].split('(')[0].strip()
                    if 's' in time_part:
                        metrics['bfs_time_ms'] = float(time_part.replace('s', '').strip()) * 1000
        
        return metrics
        
    except subprocess.TimeoutExpired:
        print("⏰ Python test timed out")
        return {}
    except Exception as e:
        print(f"❌ Python test error: {e}")
        return {}

def compare_results(rust_metrics: Dict[str, float], python_metrics: Dict[str, float]):
    """Compare and analyze the results"""
    print(f"\\n🔬 PERFORMANCE COMPARISON RESULTS")
    print("=" * 60)
    
    if not rust_metrics or not python_metrics:
        print("❌ Missing benchmark data")
        return
        
    # Compare each available metric
    comparisons = []
    print(f"{'Operation':<30} {'Rust (ms)':<12} {'Python (ms)':<12} {'Overhead':<10} {'Status'}")
    print("-" * 65)
    
    for metric in rust_metrics:
        if metric in python_metrics:
            rust_time = rust_metrics[metric]
            python_time = python_metrics[metric]
            
            if rust_time > 0:
                overhead = python_time / rust_time
                # Handle edge case where python_time is 0 but rust_time > 0
                if python_time == 0:
                    overhead = 0.1  # Treat as very fast (0.1x overhead)
                comparisons.append(overhead)
                
                # Format operation name
                op_name = metric.replace('_time_ms', '').replace('_', ' ').title()
                
                # Status indicator
                if overhead > 20:
                    status = "🚨 CRITICAL"
                elif overhead > 10:
                    status = "⚠️  HIGH"  
                elif overhead > 5:
                    status = "⚠️  MEDIUM"
                else:
                    status = "✅ GOOD"
                    
                print(f"{op_name:<30} {rust_time:<12.2f} {python_time:<12.2f} {overhead:<10.1f}x {status}")
    
    # Summary statistics
    if comparisons:
        print(f"\\n📊 SUMMARY STATISTICS")
        print("-" * 30)
        print(f"Operations tested: {len(comparisons)}")
        print(f"Average overhead: {statistics.mean(comparisons):.1f}x")
        print(f"Median overhead: {statistics.median(comparisons):.1f}x")
        print(f"Geometric mean: {statistics.geometric_mean(comparisons):.1f}x")
        print(f"Range: {min(comparisons):.1f}x - {max(comparisons):.1f}x")
        
        # Identify bottlenecks
        critical = [c for c in comparisons if c > 20]
        high = [c for c in comparisons if 10 < c <= 20]
        
        if critical:
            print(f"\\n🚨 CRITICAL BOTTLENECKS: {len(critical)} operations >20x slower")
        if high:
            print(f"⚠️  HIGH BOTTLENECKS: {len(high)} operations 10-20x slower")
            
        # Overall assessment
        geomean = statistics.geometric_mean(comparisons)
        if geomean > 10:
            print(f"\\n🎯 PRIORITY: HIGH - Geometric mean {geomean:.1f}x indicates major bottlenecks")
        elif geomean > 5:
            print(f"\\n🎯 PRIORITY: MEDIUM - Geometric mean {geomean:.1f}x shows room for improvement")
        else:
            print(f"\\n🎯 PRIORITY: LOW - Geometric mean {geomean:.1f}x shows good performance")

def main():
    print("🚀 SIMPLE SIDE-BY-SIDE PERFORMANCE COMPARISON")
    print("=" * 50)
    print("Using existing stress tests to compare Rust vs Python performance\\n")
    
    try:
        # Run both benchmarks
        rust_metrics = run_rust_stress_test("")
        python_metrics = run_python_stress_test()
        
        # Analyze results
        compare_results(rust_metrics, python_metrics)
        
        # Add analysis of bottlenecks
        print("\\n🔍 BOTTLENECK ANALYSIS")
        print("-" * 30)
        if rust_metrics.get('creation_time_ms', 0) > 0:
            creation_overhead = python_metrics.get('creation_time_ms', 0) / rust_metrics.get('creation_time_ms', 1)
            print(f"Graph creation overhead: {creation_overhead:.1f}x")
            if creation_overhead > 5:
                print("🎯 ROOT CAUSE: Individual PyAttrValue conversions (50K objects for 10K nodes)")
                print("💡 SOLUTION: ✅ IMPLEMENTED - Bulk columnar attribute operations")
                print("📊 STATUS: Partially optimized, more improvements needed")
            else:
                print("🎯 STATUS: ✅ MAJOR IMPROVEMENT - Bulk API working well!")
                print("💡 NEXT: Further optimizations for remaining overhead")
        
        if rust_metrics.get('connected_components_time_ms', 0) > 0:
            cc_overhead = python_metrics.get('connected_components_time_ms', 0) / rust_metrics.get('connected_components_time_ms', 1)
            print(f"Connected components overhead: {cc_overhead:.1f}x")
            if cc_overhead > 5:
                print("🎯 ROOT CAUSE: Inefficient BFS-per-component + Python API overhead")
                print("💡 SOLUTION: Fix algorithm + reduce Python conversion overhead")
            else:
                print("🎯 STATUS: ✅ GOOD - Overhead within acceptable range")
        
        print("\\n🏁 Comparison complete!")
        
    except KeyboardInterrupt:
        print("\\n⏹️  Interrupted by user")
    except Exception as e:
        print(f"\\n❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()