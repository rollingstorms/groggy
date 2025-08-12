#!/usr/bin/env python3
"""
Comprehensive Performance Comparison: Native Rust vs Python API
Identifies remaining bottlenecks and measures actual overhead
"""

import subprocess
import json
import time
import sys
import os
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import statistics

@dataclass
class BenchmarkResult:
    name: str
    rust_time_ms: float
    python_time_ms: float
    overhead_factor: float
    rust_throughput: float
    python_throughput: float
    efficiency_ratio: float

@dataclass 
class PerformanceReport:
    test_name: str
    results: List[BenchmarkResult]
    summary: Dict[str, Any]

class PerformanceComparator:
    def __init__(self):
        self.results = []
        
    def run_rust_benchmark(self, test_name: str, graph_size: Tuple[int, int]) -> Dict[str, float]:
        """Run Rust native benchmark and parse results"""
        nodes, edges = graph_size
        print(f"🦀 Running Rust benchmark: {test_name} ({nodes} nodes, {edges} edges)")
        
        # Create temporary Rust benchmark file
        rust_code = '''
use groggy::{{Graph, AttrValue}};
use groggy::core::query::NodeFilter;
use groggy::core::traversal::TraversalOptions;
use std::time::Instant;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {{
    let start_total = Instant::now();
    
    // === GRAPH CREATION ===
    let start = Instant::now();
    let mut graph = Graph::new();
    let nodes = graph.add_nodes(''' + str(nodes) + ''');
    
    // Set attributes in bulk
    let mut bulk_attrs = HashMap::new();
    let mut type_data = Vec::new();
    let mut dept_data = Vec::new();
    let mut active_data = Vec::new();
    let mut age_data = Vec::new();
    let mut salary_data = Vec::new();
    
    for (i, &node_id) in nodes.iter().enumerate() {{
        type_data.push((node_id, AttrValue::Text("user".to_string())));
        dept_data.push((node_id, AttrValue::Text(match i % 6 {{
            0 => "Engineering", 1 => "Marketing", 2 => "Sales", 
            3 => "HR", 4 => "Finance", _ => "Operations"
        }}.to_string())));
        active_data.push((node_id, AttrValue::Bool(i % 3 != 0)));
        age_data.push((node_id, AttrValue::Int(25 + (i % 40) as i64)));
        salary_data.push((node_id, AttrValue::Int(50000 + (i % 100000) as i64)));
    }}
    
    bulk_attrs.insert("type".to_string(), type_data);
    bulk_attrs.insert("department".to_string(), dept_data);
    bulk_attrs.insert("active".to_string(), active_data);
    bulk_attrs.insert("age".to_string(), age_data);
    bulk_attrs.insert("salary".to_string(), salary_data);
    graph.set_node_attrs(bulk_attrs)?;
    
    // Create edges
    let mut edge_specs = Vec::new();
    for i in 0..''' + str(edges) + ''' {{
        let from_idx = fastrand::usize(..''' + str(nodes) + ''');
        let to_idx = fastrand::usize(..''' + str(nodes) + ''');
        if from_idx != to_idx {{
            edge_specs.push((nodes[from_idx], nodes[to_idx]));
        }}
    }}
    graph.add_edges(&edge_specs);
    let creation_time = start.elapsed();
    
    // === FILTERING ===
    let start = Instant::now();
    let type_filter = NodeFilter::AttributeEquals {{
        name: "type".to_string(),
        value: AttrValue::Text("user".to_string()),
    }};
    let type_results = graph.find_nodes(type_filter)?;
    let single_filter_time = start.elapsed();
    
    let start = Instant::now();
    let complex_filter = NodeFilter::And(vec![
        NodeFilter::AttributeEquals {{
            name: "type".to_string(),
            value: AttrValue::Text("user".to_string()),
        }},
        NodeFilter::AttributeEquals {{
            name: "department".to_string(),
            value: AttrValue::Text("Engineering".to_string()),
        }},
        NodeFilter::AttributeEquals {{
            name: "active".to_string(),
            value: AttrValue::Bool(true),
        }}
    ]);
    let complex_results = graph.find_nodes(complex_filter)?;
    let complex_filter_time = start.elapsed();
    
    // === TRAVERSAL ===
    let start = Instant::now();
    let components = graph.connected_components(TraversalOptions::default())?;
    let connected_components_time = start.elapsed();
    
    let start = Instant::now();
    if !nodes.is_empty() {{
        let start_node = nodes[fastrand::usize(..nodes.len())];
        let bfs_result = graph.bfs(start_node, TraversalOptions::default())?;
        let _ = bfs_result.nodes.len(); // Use result
    }}
    let bfs_time = start.elapsed();
    
    // === NEIGHBORS AND TOPOLOGY ===
    let start = Instant::now();
    let mut total_neighbors = 0;
    for &node in &nodes[..std::cmp::min(1000, nodes.len())] {{
        let neighbors = graph.neighbors(node)?;
        total_neighbors += neighbors.len();
    }}
    let neighbors_time = start.elapsed();
    
    // === ATTRIBUTE ACCESS ===
    let start = Instant::now();
    let mut attribute_count = 0;
    for &node in &nodes[..std::cmp::min(1000, nodes.len())] {{
        if let Ok(Some(_)) = graph.get_node_attr(node, "age") {{
            attribute_count += 1;
        }}
    }}
    let attribute_time = start.elapsed();
    
    let total_time = start_total.elapsed();
    
    // Output JSON results for parsing
    println!("{{");
    println!("  \\"creation_time_ms\\": {},", creation_time.as_millis());
    println!("  \\"single_filter_time_ms\\": {},", single_filter_time.as_millis());
    println!("  \\"complex_filter_time_ms\\": {},", complex_filter_time.as_millis());
    println!("  \\"connected_components_time_ms\\": {},", connected_components_time.as_millis());
    println!("  \\"bfs_time_ms\\": {},", bfs_time.as_millis());
    println!("  \\"neighbors_time_ms\\": {},", neighbors_time.as_millis());
    println!("  \\"attribute_time_ms\\": {},", attribute_time.as_millis());
    println!("  \\"total_time_ms\\": {},", total_time.as_millis());
    println!("  \\"type_results_count\\": {},", type_results.len());
    println!("  \\"complex_results_count\\": {},", complex_results.len());
    println!("  \\"components_count\\": {},", components.total_components);
    println!("  \\"total_neighbors\\": {},", total_neighbors);
    println!("  \\"attribute_count\\": {}", attribute_count);
    println!("}}");
    
    Ok(())
}}
        '''
        
        # Write and compile Rust benchmark
        with open('temp_benchmark.rs', 'w') as f:
            f.write(rust_code)
            
        try:
            # Compile and run
            compile_result = subprocess.run([
                'rustc', 'temp_benchmark.rs', 
                '--extern', 'groggy=/Users/michaelroth/Documents/Code/groggy/target/debug/deps/libgroggy.rlib',
                '-L', '/Users/michaelroth/Documents/Code/groggy/target/debug/deps',
                '-o', 'temp_benchmark'
            ], capture_output=True, text=True, cwd='/Users/michaelroth/Documents/Code/groggy')
            
            if compile_result.returncode != 0:
                print(f"Rust compilation failed: {compile_result.stderr}")
                return {}
                
            # Run benchmark
            run_result = subprocess.run(['./temp_benchmark'], 
                                      capture_output=True, text=True,
                                      cwd='/Users/michaelroth/Documents/Code/groggy')
            
            if run_result.returncode != 0:
                print(f"Rust benchmark failed: {run_result.stderr}")
                return {}
                
            # Parse JSON output
            try:
                return json.loads(run_result.stdout)
            except json.JSONDecodeError:
                print(f"Failed to parse Rust output: {run_result.stdout}")
                return {}
                
        finally:
            # Cleanup
            for file in ['temp_benchmark.rs', 'temp_benchmark']:
                if os.path.exists(f'/Users/michaelroth/Documents/Code/groggy/{file}'):
                    os.remove(f'/Users/michaelroth/Documents/Code/groggy/{file}')
    
    def run_python_benchmark(self, test_name: str, graph_size: Tuple[int, int]) -> Dict[str, float]:
        """Run Python benchmark with equivalent operations"""
        nodes, edges = graph_size
        print(f"🐍 Running Python benchmark: {test_name} ({nodes} nodes, {edges} edges)")
        
        try:
            import groggy as gr
        except ImportError:
            print("❌ Python groggy module not available")
            return {}
            
        start_total = time.time()
        
        # === GRAPH CREATION ===
        start = time.time()
        graph = gr.Graph()
        node_list = graph.add_nodes(nodes)
        
        # Set attributes in bulk  
        attrs_dict = {
            "type": [(node_id, gr.AttrValue("user")) for node_id in node_list],
            "department": [(node_id, gr.AttrValue(["Engineering", "Marketing", "Sales", "HR", "Finance", "Operations"][i % 6])) 
                          for i, node_id in enumerate(node_list)],
            "active": [(node_id, gr.AttrValue(i % 3 != 0)) for i, node_id in enumerate(node_list)],
            "age": [(node_id, gr.AttrValue(25 + (i % 40))) for i, node_id in enumerate(node_list)],
            "salary": [(node_id, gr.AttrValue(50000 + (i % 100000))) for i, node_id in enumerate(node_list)]
        }
        graph.set_node_attributes(attrs_dict)
        
        # Create edges
        edge_specs = []
        for _ in range(edges):
            from_idx = __import__('random').randint(0, nodes - 1)
            to_idx = __import__('random').randint(0, nodes - 1) 
            if from_idx != to_idx:
                edge_specs.append((node_list[from_idx], node_list[to_idx]))
        if edge_specs:
            graph.add_edges(edge_specs)
        creation_time = (time.time() - start) * 1000
        
        # === FILTERING ===
        start = time.time()
        type_filter = gr.NodeFilter.attribute_equals("type", gr.AttrValue("user"))
        type_results = graph.filter_nodes(type_filter)
        single_filter_time = (time.time() - start) * 1000
        
        start = time.time()
        complex_filter = gr.NodeFilter.and_filters([
            gr.NodeFilter.attribute_equals("type", gr.AttrValue("user")),
            gr.NodeFilter.attribute_equals("department", gr.AttrValue("Engineering")),
            gr.NodeFilter.attribute_equals("active", gr.AttrValue(True))
        ])
        complex_results = graph.filter_nodes(complex_filter)
        complex_filter_time = (time.time() - start) * 1000
        
        # === TRAVERSAL ===
        start = time.time()
        components = graph.find_connected_components()
        connected_components_time = (time.time() - start) * 1000
        
        start = time.time()
        if node_list:
            start_node = __import__('random').choice(node_list)
            bfs_result = graph.traverse_bfs(start_node, max_depth=None)
            _ = len(bfs_result.nodes)  # Use result
        bfs_time = (time.time() - start) * 1000
        
        # === NEIGHBORS AND TOPOLOGY ===
        start = time.time()
        total_neighbors = 0
        for node in node_list[:min(1000, len(node_list))]:
            neighbors = graph.neighbors(node)
            total_neighbors += len(neighbors)
        neighbors_time = (time.time() - start) * 1000
        
        # === ATTRIBUTE ACCESS ===
        start = time.time()
        attribute_count = 0
        for node in node_list[:min(1000, len(node_list))]:
            age_attr = graph.get_node_attribute(node, "age")
            if age_attr:
                attribute_count += 1
        attribute_time = (time.time() - start) * 1000
        
        total_time = (time.time() - start_total) * 1000
        
        return {
            "creation_time_ms": creation_time,
            "single_filter_time_ms": single_filter_time,
            "complex_filter_time_ms": complex_filter_time,
            "connected_components_time_ms": connected_components_time,
            "bfs_time_ms": bfs_time,
            "neighbors_time_ms": neighbors_time,
            "attribute_time_ms": attribute_time,
            "total_time_ms": total_time,
            "type_results_count": len(type_results) if hasattr(type_results, '__len__') else len(type_results.nodes) if hasattr(type_results, 'nodes') else 0,
            "complex_results_count": len(complex_results) if hasattr(complex_results, '__len__') else len(complex_results.nodes) if hasattr(complex_results, 'nodes') else 0,
            "components_count": len(components),
            "total_neighbors": total_neighbors,
            "attribute_count": attribute_count
        }
        
    def compare_performance(self, test_name: str, graph_sizes: List[Tuple[int, int]]) -> PerformanceReport:
        """Run comprehensive comparison across graph sizes"""
        print(f"\\n🔬 COMPREHENSIVE PERFORMANCE COMPARISON: {test_name}")
        print("=" * 60)
        
        all_results = []
        
        for size in graph_sizes:
            nodes, edges = size
            print(f"\\n📊 Testing with {nodes:,} nodes, {edges:,} edges")
            
            # Run both benchmarks
            rust_metrics = self.run_rust_benchmark(test_name, size)
            python_metrics = self.run_python_benchmark(test_name, size)
            
            if not rust_metrics or not python_metrics:
                print(f"⚠️  Skipping {size} due to benchmark failure")
                continue
                
            # Compare each metric
            for metric in ["creation_time_ms", "single_filter_time_ms", "complex_filter_time_ms", 
                          "connected_components_time_ms", "bfs_time_ms", "neighbors_time_ms", "attribute_time_ms"]:
                rust_time = rust_metrics.get(metric, 0)
                python_time = python_metrics.get(metric, 0)
                
                if rust_time > 0:
                    overhead = python_time / rust_time
                    rust_throughput = nodes / rust_time if rust_time > 0 else 0
                    python_throughput = nodes / python_time if python_time > 0 else 0
                    efficiency = python_throughput / rust_throughput if rust_throughput > 0 else 0
                    
                    result = BenchmarkResult(
                        name=f"{metric.replace('_time_ms', '')} ({nodes:,} nodes)",
                        rust_time_ms=rust_time,
                        python_time_ms=python_time,
                        overhead_factor=overhead,
                        rust_throughput=rust_throughput,
                        python_throughput=python_throughput,
                        efficiency_ratio=efficiency
                    )
                    all_results.append(result)
        
        # Calculate summary statistics
        if all_results:
            overheads = [r.overhead_factor for r in all_results if r.overhead_factor > 0]
            summary = {
                "total_tests": len(all_results),
                "avg_overhead": statistics.mean(overheads) if overheads else 0,
                "median_overhead": statistics.median(overheads) if overheads else 0,
                "max_overhead": max(overheads) if overheads else 0,
                "min_overhead": min(overheads) if overheads else 0,
                "geometric_mean_overhead": statistics.geometric_mean(overheads) if overheads else 0,
            }
        else:
            summary = {}
            
        return PerformanceReport(test_name, all_results, summary)
    
    def print_detailed_report(self, report: PerformanceReport):
        """Print comprehensive analysis report"""
        print(f"\\n📋 DETAILED PERFORMANCE REPORT: {report.test_name}")
        print("=" * 70)
        
        if not report.results:
            print("❌ No results to report")
            return
            
        # Performance table
        print(f"{'Operation':<35} {'Rust (ms)':<12} {'Python (ms)':<12} {'Overhead':<10} {'Status'}")
        print("-" * 70)
        
        for result in report.results:
            status = "🚨" if result.overhead_factor > 10 else "⚠️" if result.overhead_factor > 5 else "✅"
            print(f"{result.name:<35} {result.rust_time_ms:<12.2f} {result.python_time_ms:<12.2f} "
                  f"{result.overhead_factor:<10.1f}x {status}")
        
        # Summary statistics
        if report.summary:
            print(f"\\n📊 SUMMARY STATISTICS")
            print("-" * 30)
            print(f"Total Tests: {report.summary['total_tests']}")
            print(f"Average Overhead: {report.summary['avg_overhead']:.1f}x")
            print(f"Median Overhead: {report.summary['median_overhead']:.1f}x")
            print(f"Geometric Mean: {report.summary['geometric_mean_overhead']:.1f}x")
            print(f"Range: {report.summary['min_overhead']:.1f}x - {report.summary['max_overhead']:.1f}x")
        
        # Bottleneck identification
        high_overhead = [r for r in report.results if r.overhead_factor > 10]
        if high_overhead:
            print(f"\\n🚨 CRITICAL BOTTLENECKS (>10x overhead):")
            for result in sorted(high_overhead, key=lambda x: x.overhead_factor, reverse=True):
                print(f"   • {result.name}: {result.overhead_factor:.1f}x slower")
        
        medium_overhead = [r for r in report.results if 5 < r.overhead_factor <= 10]
        if medium_overhead:
            print(f"\\n⚠️  MODERATE BOTTLENECKS (5-10x overhead):")
            for result in medium_overhead:
                print(f"   • {result.name}: {result.overhead_factor:.1f}x slower")
        
        good_performance = [r for r in report.results if r.overhead_factor <= 5]
        if good_performance:
            print(f"\\n✅ GOOD PERFORMANCE (≤5x overhead):")
            for result in good_performance:
                print(f"   • {result.name}: {result.overhead_factor:.1f}x slower")

def main():
    print("🚀 COMPREHENSIVE NATIVE vs PYTHON PERFORMANCE ANALYSIS")
    print("=" * 60)
    print("This script runs identical operations in both Rust and Python")
    print("to identify remaining performance bottlenecks and overhead patterns.")
    print()
    
    comparator = PerformanceComparator()
    
    # Test different graph sizes to identify scaling behavior
    graph_sizes = [
        (1000, 2000),    # Small
        (5000, 10000),   # Medium  
        (10000, 20000),  # Large
    ]
    
    try:
        report = comparator.compare_performance("GraphOperations", graph_sizes)
        comparator.print_detailed_report(report)
        
        print(f"\\n🎯 OPTIMIZATION PRIORITIES")
        print("=" * 30)
        
        if report.summary and report.summary.get('geometric_mean_overhead', 0) > 10:
            print("🚨 HIGH PRIORITY: Geometric mean overhead >10x")
        elif report.summary and report.summary.get('geometric_mean_overhead', 0) > 5:
            print("⚠️  MEDIUM PRIORITY: Geometric mean overhead 5-10x") 
        else:
            print("✅ LOW PRIORITY: Geometric mean overhead <5x")
            
        # Specific recommendations
        if report.results:
            worst_result = max(report.results, key=lambda x: x.overhead_factor)
            print(f"\\n🎯 Focus on optimizing: {worst_result.name}")
            print(f"   Current overhead: {worst_result.overhead_factor:.1f}x")
            
        print(f"\\n🏁 Analysis complete! Check results above for optimization targets.")
        
    except KeyboardInterrupt:
        print("\\n⏹️  Benchmark interrupted by user")
    except Exception as e:
        print(f"\\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()