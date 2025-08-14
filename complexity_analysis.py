#!/usr/bin/env python3
"""
Computational Complexity Analysis for Groggy Graph Filtering

This script analyzes the per-item computational cost of filtering operations
to understand true algorithmic performance beyond dataset size effects.
"""

import time
import statistics
import numpy as np
from typing import List, Dict, Tuple
import sys
import os

# Add the python-groggy package to path
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python-groggy/python')
import groggy as gr

def create_test_graph(num_nodes: int, num_edges: int) -> Tuple[gr.Graph, Dict]:
    """Create a test graph with specified number of nodes and edges"""
    g = gr.Graph()
    
    # Create nodes with attributes
    node_data = []
    for i in range(num_nodes):
        node_data.append({
            "id": f"node_{i}",
            "department": f"dept_{i % 5}",  # 5 departments
            "salary": 50000 + (i * 1000) % 100000,  # salary range 50k-150k
            "active": i % 3 == 0,  # every 3rd node is active
            "performance": (i * 7) % 100 / 100.0  # performance 0.0-1.0
        })
    
    node_mapping = g.add_nodes(node_data, uid_key="id")
    
    # Create edges with attributes
    edge_data = []
    nodes_list = list(range(num_nodes))
    np.random.seed(42)  # Reproducible results
    
    for i in range(num_edges):
        source = np.random.choice(nodes_list)
        target = np.random.choice(nodes_list)
        if source != target:  # Avoid self-loops
            edge_data.append({
                "source": f"node_{source}",
                "target": f"node_{target}",
                "relationship": f"rel_{i % 3}",  # 3 relationship types
                "weight": np.random.random()  # weight 0.0-1.0
            })
    
    if edge_data:
        g.add_edges(edge_data, node_mapping=node_mapping)
    
    return g, node_mapping

def measure_filter_performance(g: gr.Graph, filter_type: str, filter_expr: str, 
                             dataset_size: int, iterations: int = 10) -> Dict:
    """Measure filtering performance with multiple iterations"""
    times = []
    
    for _ in range(iterations):
        start_time = time.perf_counter()
        
        if filter_type == "nodes":
            results = g.filter_nodes(filter_expr)
        else:  # edges
            results = g.filter_edges(filter_expr)
            
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    # Calculate statistics
    avg_time = statistics.mean(times)
    min_time = min(times)
    max_time = max(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0
    
    # Per-item metrics
    per_item_time = avg_time / dataset_size if dataset_size > 0 else 0
    per_item_time_ns = per_item_time * 1_000_000_000  # Convert to nanoseconds
    
    return {
        "avg_time_ms": avg_time * 1000,
        "min_time_ms": min_time * 1000,
        "max_time_ms": max_time * 1000,
        "std_time_ms": std_time * 1000,
        "per_item_time_ns": per_item_time_ns,
        "dataset_size": dataset_size,
        "iterations": iterations
    }

def analyze_complexity_by_scale():
    """Analyze how filtering performance scales with dataset size"""
    print("=== COMPUTATIONAL COMPLEXITY ANALYSIS ===\n")
    
    # Test different scales
    test_scales = [
        (1000, 500),    # Small: 1K nodes, 500 edges
        (5000, 2500),   # Medium: 5K nodes, 2.5K edges  
        (10000, 5000),  # Large: 10K nodes, 5K edges
        (25000, 12500), # XL: 25K nodes, 12.5K edges
        (50000, 25000), # XXL: 50K nodes, 25K edges (original benchmark size)
    ]
    
    # Test filters
    node_filters = [
        ('salary > 75000', 'Numeric comparison'),
        ('department == "dept_1"', 'String equality'), 
        ('active == true', 'Boolean equality'),
        ('performance > 0.5', 'Float comparison')
    ]
    
    edge_filters = [
        ('weight > 0.5', 'Numeric comparison'),
        ('relationship == "rel_1"', 'String equality')
    ]
    
    results = []
    
    for num_nodes, num_edges in test_scales:
        print(f"Testing scale: {num_nodes:,} nodes, {num_edges:,} edges")
        
        # Create test graph
        g, _ = create_test_graph(num_nodes, num_edges)
        
        print(f"  Actual graph: {len(g.nodes):,} nodes, {len(g.edges):,} edges")
        
        # Test node filters
        for filter_expr, filter_desc in node_filters:
            perf = measure_filter_performance(g, "nodes", filter_expr, len(g.nodes))
            results.append({
                "scale": f"{num_nodes//1000}K",
                "type": "nodes",
                "filter": filter_desc,
                "dataset_size": len(g.nodes),
                **perf
            })
            
        # Test edge filters
        for filter_expr, filter_desc in edge_filters:
            perf = measure_filter_performance(g, "edges", filter_expr, len(g.edges))
            results.append({
                "scale": f"{num_nodes//1000}K", 
                "type": "edges",
                "filter": filter_desc,
                "dataset_size": len(g.edges),
                **perf
            })
        
        print(f"  ✓ Completed scale {num_nodes//1000}K\n")
    
    return results

def print_complexity_analysis(results: List[Dict]):
    """Print detailed complexity analysis"""
    print("\n=== PER-ITEM PERFORMANCE ANALYSIS ===")
    print("(Lower nanoseconds per item = better algorithm efficiency)\n")
    
    # Group by filter type and scale
    node_results = [r for r in results if r["type"] == "nodes"]
    edge_results = [r for r in results if r["type"] == "edges"]
    
    # Node filtering analysis
    print("📊 NODE FILTERING PERFORMANCE:")
    print("Scale   | Filter Type       | Dataset Size | Avg Time (ms) | Per Item (ns) | Items/sec")
    print("--------|-------------------|--------------|---------------|---------------|----------")
    
    for result in node_results:
        items_per_sec = 1_000_000_000 / result["per_item_time_ns"] if result["per_item_time_ns"] > 0 else 0
        print(f"{result['scale']:>7} | {result['filter']:<17} | {result['dataset_size']:>12,} | "
              f"{result['avg_time_ms']:>11.3f} | {result['per_item_time_ns']:>11.1f} | {items_per_sec:>9,.0f}")
    
    print("\n📊 EDGE FILTERING PERFORMANCE:")
    print("Scale   | Filter Type       | Dataset Size | Avg Time (ms) | Per Item (ns) | Items/sec")
    print("--------|-------------------|--------------|---------------|---------------|----------")
    
    for result in edge_results:
        items_per_sec = 1_000_000_000 / result["per_item_time_ns"] if result["per_item_time_ns"] > 0 else 0
        print(f"{result['scale']:>7} | {result['filter']:<17} | {result['dataset_size']:>12,} | "
              f"{result['avg_time_ms']:>11.3f} | {result['per_item_time_ns']:>11.1f} | {items_per_sec:>9,.0f}")

def analyze_algorithmic_efficiency(results: List[Dict]):
    """Analyze algorithmic efficiency independent of dataset size"""
    print("\n\n=== ALGORITHMIC EFFICIENCY ANALYSIS ===")
    
    # Calculate average per-item times by filter type
    node_perf = {}
    edge_perf = {}
    
    for result in results:
        filter_type = result["filter"]
        per_item_ns = result["per_item_time_ns"]
        
        if result["type"] == "nodes":
            if filter_type not in node_perf:
                node_perf[filter_type] = []
            node_perf[filter_type].append(per_item_ns)
        else:
            if filter_type not in edge_perf:
                edge_perf[filter_type] = []
            edge_perf[filter_type].append(per_item_ns)
    
    print("🔍 AVERAGE PER-ITEM PROCESSING TIME (across all scales):")
    print("\nNode Filters:")
    for filter_type, times in node_perf.items():
        avg_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0
        print(f"  {filter_type:<20}: {avg_time:>8.1f} ± {std_time:>5.1f} ns/item")
    
    print("\nEdge Filters:")  
    for filter_type, times in edge_perf.items():
        avg_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0
        print(f"  {filter_type:<20}: {avg_time:>8.1f} ± {std_time:>5.1f} ns/item")
    
    # Compare node vs edge efficiency
    print("\n🎯 EFFICIENCY COMPARISON:")
    
    # Find common filter types
    common_filters = set(node_perf.keys()) & set(edge_perf.keys())
    
    for filter_type in common_filters:
        node_avg = statistics.mean(node_perf[filter_type])
        edge_avg = statistics.mean(edge_perf[filter_type])
        
        if node_avg > edge_avg:
            ratio = node_avg / edge_avg
            faster = "EDGES"
            print(f"  {filter_type:<20}: {faster} {ratio:.1f}x faster ({edge_avg:.1f} vs {node_avg:.1f} ns/item)")
        else:
            ratio = edge_avg / node_avg
            faster = "NODES" 
            print(f"  {filter_type:<20}: {faster} {ratio:.1f}x faster ({node_avg:.1f} vs {edge_avg:.1f} ns/item)")

def analyze_scaling_behavior(results: List[Dict]):
    """Analyze how per-item time changes with scale (should be constant for O(n) algorithms)"""
    print("\n\n=== SCALING BEHAVIOR ANALYSIS ===")
    print("(Per-item time should remain constant for truly O(n) algorithms)")
    
    # Group results by type and filter
    groups = {}
    for result in results:
        key = (result["type"], result["filter"])
        if key not in groups:
            groups[key] = []
        groups[key].append((result["dataset_size"], result["per_item_time_ns"]))
    
    print("\n📈 SCALING ANALYSIS:")
    print("Type/Filter                          | Small→Large Per-Item Time Change")
    print("-------------------------------------|----------------------------------")
    
    for (result_type, filter_name), data_points in groups.items():
        # Sort by dataset size
        data_points.sort(key=lambda x: x[0])
        
        if len(data_points) >= 2:
            smallest_per_item = data_points[0][1]
            largest_per_item = data_points[-1][1]
            
            if smallest_per_item > 0:
                change_ratio = largest_per_item / smallest_per_item
                change_pct = (change_ratio - 1) * 100
                
                # Determine if scaling is good (< 20% increase) or bad (> 50% increase)
                if change_pct < 20:
                    status = "✅ GOOD"
                elif change_pct < 50:
                    status = "⚠️  FAIR"
                else:
                    status = "❌ POOR"
                
                print(f"{result_type.upper()}/{filter_name:<25} | {smallest_per_item:>6.1f}→{largest_per_item:>6.1f} ns "
                      f"({change_pct:>+5.1f}%) {status}")

def main():
    print("🚀 Starting Computational Complexity Analysis for Groggy Graph Filtering\n")
    
    # Run the analysis
    results = analyze_complexity_by_scale()
    
    # Print detailed analysis
    print_complexity_analysis(results)
    analyze_algorithmic_efficiency(results)
    analyze_scaling_behavior(results)
    
    print(f"\n✅ Analysis complete! Tested {len(results)} filter configurations across multiple scales.")
    print("\n🎯 KEY INSIGHTS:")
    print("   • Per-item processing time reveals true algorithmic efficiency")
    print("   • Dataset size effects are normalized out")
    print("   • Scaling behavior shows algorithm complexity characteristics")
    print("   • Edge filtering may be faster simply due to smaller datasets")

if __name__ == "__main__":
    main()
