#!/usr/bin/env python3
"""
Benchmark comparison: Groggy vs other high-performance graph libraries
Tests filtering performance on large graphs
"""

import sys
import os
import time
import random
import gc
from typing import Dict, List, Any

# Remove any existing groggy from the module cache
modules_to_remove = [mod for mod in sys.modules.keys() if mod.startswith('groggy')]
for mod in modules_to_remove:
    del sys.modules[mod]

# Remove any paths containing 'groggy' from sys.path (pip-installed versions)
sys.path = [p for p in sys.path if 'groggy' not in p.lower()]

# Add only our local development version
local_groggy_path = '/Users/michaelroth/Documents/Code/groggy/python'
sys.path.insert(0, local_groggy_path)

# Import libraries
import groggy as gr

# Try to import other graph libraries
libraries_available = {'groggy': True}

try:
    import networkx as nx
    libraries_available['networkx'] = True
    print("✅ NetworkX available")
except ImportError:
    libraries_available['networkx'] = False
    print("❌ NetworkX not available")

try:
    import igraph as ig
    libraries_available['igraph'] = True
    print("✅ igraph available")
except ImportError:
    libraries_available['igraph'] = False
    print("❌ igraph not available")

try:
    import graph_tool.all as gt
    libraries_available['graph_tool'] = True
    print("✅ graph-tool available")
except ImportError:
    libraries_available['graph_tool'] = False
    print("❌ graph-tool not available")

try:
    import networkit as nk
    libraries_available['networkit'] = True
    print("✅ NetworKit available")
except ImportError:
    libraries_available['networkit'] = False
    print("❌ NetworKit not available")

def create_test_data(num_nodes: int, num_edges: int):
    """Create consistent test data for all libraries"""
    print(f"Generating test data: {num_nodes:,} nodes, {num_edges:,} edges...")
    
    # Generate nodes with attributes
    nodes_data = []
    for i in range(num_nodes):
        nodes_data.append({
            'id': f"n{i}",
            'role': random.choice(['engineer', 'manager', 'designer', 'analyst']),
            'department': random.choice(['AI', 'Web', 'Mobile', 'Data']),
            'salary': random.randint(50000, 150000),
            'active': random.choice([True, False]),
            'level': random.randint(1, 5),
            'value': random.randint(1, 1000)
        })
    
    # Generate edges with attributes
    edges_data = []
    for i in range(num_edges):
        src = random.randint(0, num_nodes-1)
        tgt = random.randint(0, num_nodes-1)
        if src != tgt:
            edges_data.append({
                'source': f"n{src}",
                'target': f"n{tgt}",
                'relationship': random.choice(['reports_to', 'collaborates', 'manages']),
                'strength': random.uniform(0.1, 1.0),
                'weight': random.uniform(0.1, 1.0)
            })
    
    return nodes_data, edges_data

class GroggyBenchmark:
    def __init__(self, nodes_data, edges_data):
        print("🔧 Creating Groggy graph...")
        start = time.time()
        self.graph = gr.Graph()
        
        # Use optimized bulk operations - pass dictionaries directly
        self.graph.add_nodes(nodes_data)
        self.graph.add_edges(edges_data)
        
        self.creation_time = time.time() - start
        print(f"   Created in {self.creation_time:.3f}s")
    
    def filter_nodes_by_role(self):
        """Filter nodes by role (dictionary - optimized path)"""
        start = time.time()
        result = self.graph.filter_nodes({'role': 'engineer'})
        return time.time() - start, len(result)
    
    def filter_nodes_by_salary(self):
        """Filter nodes by salary > 100000 (numeric comparison)"""
        start = time.time()
        result = self.graph.filter_nodes({'salary': ('>', 100000)})
        return time.time() - start, len(result)
    
    def filter_nodes_complex(self):
        """Complex multi-attribute filter - using optimized sparse method"""
        start = time.time()
        result = self.graph.filter_nodes({
            'role': 'engineer',
            'active': True,
            'salary': ('>', 80000)
        })
        return time.time() - start, len(result)
    
    def filter_edges_by_relationship(self):
        """Filter edges by relationship (dictionary - optimized path)"""
        start = time.time()
        result = self.graph.filter_edges({'relationship': 'reports_to'})
        return time.time() - start, len(result)
    
    def filter_edges_by_strength(self):
        """Filter edges by strength > 0.7 (numeric comparison)"""
        start = time.time()
        result = self.graph.filter_edges({'strength': ('>', 0.7)})
        return time.time() - start, len(result)

class NetworkXBenchmark:
    def __init__(self, nodes_data, edges_data):
        print("🔧 Creating NetworkX graph...")
        start = time.time()
        self.graph = nx.DiGraph()
        
        # Add nodes with attributes
        for node in nodes_data:
            node_id = node['id']
            attrs = {k: v for k, v in node.items() if k != 'id'}
            self.graph.add_node(node_id, **attrs)
        
        # Add edges with attributes
        for edge in edges_data:
            attrs = {k: v for k, v in edge.items() if k not in ['source', 'target']}
            self.graph.add_edge(edge['source'], edge['target'], **attrs)
        
        self.creation_time = time.time() - start
        print(f"   Created in {self.creation_time:.3f}s")
    
    def filter_nodes_by_role(self):
        """Filter nodes by role"""
        start = time.time()
        result = [n for n, d in self.graph.nodes(data=True) if d.get('role') == 'engineer']
        return time.time() - start, len(result)
    
    def filter_nodes_by_salary(self):
        """Filter nodes by salary > 100000"""
        start = time.time()
        result = [n for n, d in self.graph.nodes(data=True) if d.get('salary', 0) > 100000]
        return time.time() - start, len(result)
    
    def filter_nodes_complex(self):
        """Complex multi-attribute filter"""
        start = time.time()
        result = [n for n, d in self.graph.nodes(data=True) 
                 if d.get('role') == 'engineer' and 
                    d.get('salary', 0) > 80000 and
                    d.get('active', False)]
        return time.time() - start, len(result)
    
    def filter_edges_by_relationship(self):
        """Filter edges by relationship"""
        start = time.time()
        result = [(u, v) for u, v, d in self.graph.edges(data=True) if d.get('relationship') == 'reports_to']
        return time.time() - start, len(result)
    
    def filter_edges_by_strength(self):
        """Filter edges by strength > 0.7"""
        start = time.time()
        result = [(u, v) for u, v, d in self.graph.edges(data=True) if d.get('strength', 0) > 0.7]
        return time.time() - start, len(result)

class IGraphBenchmark:
    def __init__(self, nodes_data, edges_data):
        print("🔧 Creating igraph graph...")
        start = time.time()
        
        # Create vertex list and edge list
        vertices = [node['id'] for node in nodes_data]
        edges = [(edge['source'], edge['target']) for edge in edges_data]
        
        # Create graph
        self.graph = ig.Graph.TupleList(edges, directed=True)
        
        # Add vertex attributes
        for i, node in enumerate(nodes_data):
            for attr, value in node.items():
                if attr != 'id':
                    if attr not in self.graph.vs.attributes():
                        self.graph.vs[attr] = [None] * len(self.graph.vs)
                    # Find vertex by name
                    vertex_idx = self.graph.vs.find(name=node['id']).index
                    self.graph.vs[vertex_idx][attr] = value
        
        # Add edge attributes
        for i, edge in enumerate(edges_data):
            for attr, value in edge.items():
                if attr not in ['source', 'target']:
                    if attr not in self.graph.es.attributes():
                        self.graph.es[attr] = [None] * len(self.graph.es)
                    if i < len(self.graph.es):
                        self.graph.es[i][attr] = value
        
        self.creation_time = time.time() - start
        print(f"   Created in {self.creation_time:.3f}s")
    
    def filter_nodes_by_role(self):
        """Filter nodes by role"""
        start = time.time()
        result = self.graph.vs.select(role_eq='engineer')
        return time.time() - start, len(result)
    
    def filter_nodes_by_salary(self):
        """Filter nodes by salary > 100000"""
        start = time.time()
        result = self.graph.vs.select(salary_gt=100000)
        return time.time() - start, len(result)
    
    def filter_nodes_complex(self):
        """Complex multi-attribute filter"""
        start = time.time()
        result = self.graph.vs.select(lambda v: 
            v['role'] == 'engineer' and 
            v.get('salary', 0) > 80000 and
            v.get('active', False)
        )
        return time.time() - start, len(result)
    
    def filter_edges_by_relationship(self):
        """Filter edges by relationship"""
        start = time.time()
        result = self.graph.es.select(relationship_eq='reports_to')
        return time.time() - start, len(result)
    
    def filter_edges_by_strength(self):
        """Filter edges by strength > 0.7"""
        start = time.time()
        result = self.graph.es.select(strength_gt=0.7)
        return time.time() - start, len(result)

def run_benchmark_suite(benchmark, name):
    """Run all benchmark tests for a library"""
    print(f"\n🧪 Testing {name}...")
    results = {}
    
    try:
        # Node filtering tests
        results['node_role'], node_role_count = benchmark.filter_nodes_by_role()
        print(f"   Node role filter: {results['node_role']:.4f}s ({node_role_count:,} results)")
        
        results['node_salary'], node_salary_count = benchmark.filter_nodes_by_salary()
        print(f"   Node salary filter: {results['node_salary']:.4f}s ({node_salary_count:,} results)")
        
        results['node_complex'], node_complex_count = benchmark.filter_nodes_complex()
        print(f"   Node complex filter: {results['node_complex']:.4f}s ({node_complex_count:,} results)")
        
        # Edge filtering tests
        results['edge_relationship'], edge_rel_count = benchmark.filter_edges_by_relationship()
        print(f"   Edge relationship filter: {results['edge_relationship']:.4f}s ({edge_rel_count:,} results)")
        
        results['edge_strength'], edge_str_count = benchmark.filter_edges_by_strength()
        print(f"   Edge strength filter: {results['edge_strength']:.4f}s ({edge_str_count:,} results)")
        
        results['creation_time'] = benchmark.creation_time
        
    except Exception as e:
        print(f"   ❌ Error in {name}: {e}")
        results = None
    
    return results

def main():
    print("=" * 70)
    print("🚀 GRAPH LIBRARY FILTERING PERFORMANCE BENCHMARK")
    print("=" * 70)
    print(f"Using groggy from: {gr.__file__}")
    print(f"Libraries available: {[k for k, v in libraries_available.items() if v]}")
    
    # Test sizes
    test_sizes = [
        (10_000, 5_000),    # Medium
        (50_000, 25_000),   # Large
    ]
    
    for num_nodes, num_edges in test_sizes:
        print(f"\n{'=' * 70}")
        print(f"📊 BENCHMARK: {num_nodes:,} nodes, {num_edges:,} edges")
        print(f"{'=' * 70}")
        
        # Generate test data
        nodes_data, edges_data = create_test_data(num_nodes, num_edges)
        
        # Store results for comparison
        all_results = {}
        
        # Test Groggy (our optimized version)
        if libraries_available['groggy']:
            groggy_bench = GroggyBenchmark(nodes_data, edges_data)
            all_results['Groggy (Optimized)'] = run_benchmark_suite(groggy_bench, "Groggy")
            del groggy_bench
            gc.collect()
        
        # Test NetworkX
        if libraries_available['networkx']:
            try:
                nx_bench = NetworkXBenchmark(nodes_data, edges_data)
                all_results['NetworkX'] = run_benchmark_suite(nx_bench, "NetworkX")
                del nx_bench
                gc.collect()
            except Exception as e:
                print(f"❌ NetworkX failed: {e}")
        
        # Test igraph
        if libraries_available['igraph']:
            try:
                ig_bench = IGraphBenchmark(nodes_data, edges_data)
                all_results['igraph'] = run_benchmark_suite(ig_bench, "igraph")
                del ig_bench
                gc.collect()
            except Exception as e:
                print(f"❌ igraph failed: {e}")
        
        # Performance comparison table
        print(f"\n📈 PERFORMANCE COMPARISON ({num_nodes:,} nodes)")
        print("=" * 90)
        
        operations = [
            ('Graph Creation', 'creation_time'),
            ('Node Role Filter', 'node_role'),
            ('Node Salary Filter', 'node_salary'),
            ('Node Complex Filter', 'node_complex'),
            ('Edge Relationship Filter', 'edge_relationship'),
            ('Edge Strength Filter', 'edge_strength')
        ]
        
        # Print header
        libraries = [lib for lib in all_results.keys() if all_results[lib] is not None]
        header = f"{'Operation':<25}"
        for lib in libraries:
            header += f"{lib:<20}"
        print(header)
        print("-" * 90)
        
        # Print results
        for op_name, op_key in operations:
            row = f"{op_name:<25}"
            times = []
            for lib in libraries:
                if all_results[lib] and op_key in all_results[lib]:
                    time_val = all_results[lib][op_key]
                    row += f"{time_val:<20.4f}"
                    times.append(time_val)
                else:
                    row += f"{'N/A':<20}"
            print(row)
        
        # Performance insights
        if 'Groggy (Optimized)' in all_results and all_results['Groggy (Optimized)']:
            print(f"\n🎯 Performance Insights for {num_nodes:,} nodes:")
            groggy_results = all_results['Groggy (Optimized)']
            
            for lib in libraries:
                if lib != 'Groggy (Optimized)' and all_results[lib]:
                    lib_results = all_results[lib]
                    
                    # Compare key operations
                    for op_name, op_key in operations[1:]:  # Skip creation time
                        if op_key in groggy_results and op_key in lib_results:
                            groggy_time = groggy_results[op_key]
                            lib_time = lib_results[op_key]
                            if lib_time > 0 and groggy_time > 0:
                                speedup = lib_time / groggy_time
                                if speedup > 1:
                                    print(f"   🚀 Groggy is {speedup:.1f}x faster than {lib} for {op_name}")
                                elif speedup < 1:
                                    print(f"   📊 {lib} is {1/speedup:.1f}x faster than Groggy for {op_name}")
        
        print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
