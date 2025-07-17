#!/usr/bin/env python3
"""
Benchmark: Native Implementation vs Original Implementation vs Other Libraries
"""

import sys
import os
import time
import random
import gc
import psutil
from typing import Dict, List, Any

# Add local development version
local_groggy_path = '/Users/michaelroth/Documents/Code/groggy/python'
sys.path.insert(0, local_groggy_path)

# Import libraries
libraries_available = {}

# Try to import native groggy
try:
    from groggy._core import NativeGraph
    libraries_available['native_groggy'] = True
    print("✅ Native Groggy available")
except ImportError:
    libraries_available['native_groggy'] = False
    print("❌ Native Groggy not available")

# Try to import original groggy
try:
    import groggy as gr
    libraries_available['original_groggy'] = True
    print("✅ Original Groggy available")
except ImportError:
    libraries_available['original_groggy'] = False
    print("❌ Original Groggy not available")

# Try to import other libraries
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

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def format_memory(mb):
    """Format memory in MB with appropriate units"""
    if mb < 1024:
        return f"{mb:.1f} MB"
    else:
        return f"{mb/1024:.2f} GB"

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

class NativeGroggyBenchmark:
    """Native Groggy benchmark using the new native API"""
    
    def __init__(self, nodes_data, edges_data):
        print("🔧 Creating Native Groggy graph...")
        
        # Measure memory before creation
        memory_before = get_memory_usage()
        start = time.time()
        
        # Create graph with native API
        self.graph = NativeGraph(directed=True)
        
        # Add nodes
        node_ids = [node['id'] for node in nodes_data]
        self.graph.add_nodes(node_ids)
        
        # Add edges
        edge_tuples = [(edge['source'], edge['target']) for edge in edges_data]
        self.graph.add_edges(edge_tuples)
        
        # Add node attributes (native Python objects - no JSON!)
        node_attrs = {}
        for node in nodes_data:
            node_id = node['id']
            attrs = {k: v for k, v in node.items() if k != 'id'}
            node_attrs[node_id] = attrs
        self.graph.set_node_attributes(node_attrs)
        
        # Add edge attributes
        edge_attrs = {}
        for edge in edges_data:
            edge_id = f"{edge['source']}->{edge['target']}"
            attrs = {k: v for k, v in edge.items() if k not in ['source', 'target']}
            edge_attrs[edge_id] = attrs
        self.graph.set_edge_attributes(edge_attrs)
        
        # Measure memory after creation
        memory_after = get_memory_usage()
        self.memory_usage = memory_after - memory_before
        self.creation_time = time.time() - start
        
        print(f"   Created in {self.creation_time:.3f}s")
        print(f"   Memory usage: {format_memory(self.memory_usage)}")
        
        # Get graph info
        info = self.graph.info()
        print(f"   Graph info: {info}")
    
    def filter_nodes_by_role(self):
        """Filter nodes by role - simplified for native implementation"""
        start = time.time()
        # Get all nodes and filter manually for now
        node_ids = self.graph.get_node_ids()
        result = []
        for node_id in node_ids:
            node = self.graph.get_node(node_id)
            if node:
                role = node.get_attr('role')
                if role == 'engineer':
                    result.append(node_id)
        return time.time() - start, len(result)
    
    def filter_nodes_by_salary(self):
        """Filter nodes by salary > 100000"""
        start = time.time()
        node_ids = self.graph.get_node_ids()
        result = []
        for node_id in node_ids:
            node = self.graph.get_node(node_id)
            if node:
                salary = node.get_attr('salary')
                if salary and salary > 100000:
                    result.append(node_id)
        return time.time() - start, len(result)
    
    def filter_nodes_complex(self):
        """Complex multi-attribute filter"""
        start = time.time()
        node_ids = self.graph.get_node_ids()
        result = []
        for node_id in node_ids:
            node = self.graph.get_node(node_id)
            if node:
                role = node.get_attr('role')
                salary = node.get_attr('salary')
                active = node.get_attr('active')
                if role == 'engineer' and salary and salary > 80000 and active:
                    result.append(node_id)
        return time.time() - start, len(result)
    
    def filter_edges_by_relationship(self):
        """Filter edges by relationship - simplified"""
        start = time.time()
        edges = self.graph.get_edges()
        result = []
        for source, target in edges:
            edge = self.graph.get_edge(source, target)
            if edge:
                relationship = edge.get_attr('relationship')
                if relationship == 'reports_to':
                    result.append((source, target))
        return time.time() - start, len(result)
    
    def filter_edges_by_strength(self):
        """Filter edges by strength > 0.7"""
        start = time.time()
        edges = self.graph.get_edges()
        result = []
        for source, target in edges:
            edge = self.graph.get_edge(source, target)
            if edge:
                strength = edge.get_attr('strength')
                if strength and strength > 0.7:
                    result.append((source, target))
        return time.time() - start, len(result)
    
    def get_stats(self):
        """Get graph statistics"""
        return {
            'nodes': self.graph.node_count(),
            'edges': self.graph.edge_count(),
            'node_ids_created': len(self.graph.get_node_ids()),
            'edge_ids_created': len(self.graph.get_edges())
        }

class OriginalGroggyBenchmark:
    """Original Groggy benchmark using the existing API"""
    
    def __init__(self, nodes_data, edges_data):
        print("🔧 Creating Original Groggy graph...")
        
        # Measure memory before creation
        memory_before = get_memory_usage()
        start = time.time()
        
        # Create graph with original API
        self.graph = gr.Graph()
        
        # Add nodes in batch
        self.graph.nodes.add(nodes_data)
        
        # Create and add edges
        self.graph.edges.add(edges_data)
        
        # Measure memory after creation
        memory_after = get_memory_usage()
        self.memory_usage = memory_after - memory_before
        self.creation_time = time.time() - start
        
        print(f"   Created in {self.creation_time:.3f}s")
        print(f"   Memory usage: {format_memory(self.memory_usage)}")
        
        # Get graph info
        info = self.graph.info()
        print(f"   Graph info: {info}")
    
    def filter_nodes_by_role(self):
        """Filter nodes by role"""
        start = time.time()
        try:
            result = self.graph.nodes.filter('role="engineer"')
            return time.time() - start, len(result)
        except:
            return time.time() - start, 0
    
    def filter_nodes_by_salary(self):
        """Filter nodes by salary > 100000"""
        start = time.time()
        try:
            result = self.graph.nodes.filter('salary > 100000')
            return time.time() - start, len(result)
        except:
            return time.time() - start, 0
    
    def filter_nodes_complex(self):
        """Complex multi-attribute filter"""
        start = time.time()
        try:
            result = self.graph.nodes.filter('role="engineer" & active & salary > 80000')
            return time.time() - start, len(result)
        except:
            return time.time() - start, 0
    
    def filter_edges_by_relationship(self):
        """Filter edges by relationship"""
        start = time.time()
        try:
            result = self.graph.edges.filter('relationship="reports_to"')
            return time.time() - start, len(result)
        except:
            return time.time() - start, 0
    
    def filter_edges_by_strength(self):
        """Filter edges by strength > 0.7"""
        start = time.time()
        try:
            result = self.graph.edges.filter('strength > 0.7')
            return time.time() - start, len(result)
        except:
            return time.time() - start, 0
    
    def get_stats(self):
        """Get graph statistics"""
        return {
            'nodes': len(self.graph.nodes),
            'edges': len(self.graph.edges),
            'node_ids_created': len(self.graph.nodes),
            'edge_ids_created': len(self.graph.edges)
        }

class NetworkXBenchmark:
    """NetworkX benchmark for comparison"""
    
    def __init__(self, nodes_data, edges_data):
        print("🔧 Creating NetworkX graph...")
        
        # Measure memory before creation
        memory_before = get_memory_usage()
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
        
        # Measure memory after creation
        memory_after = get_memory_usage()
        self.memory_usage = memory_after - memory_before
        self.creation_time = time.time() - start
        
        print(f"   Created in {self.creation_time:.3f}s")
        print(f"   Memory usage: {format_memory(self.memory_usage)}")
    
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

def benchmark_library(benchmark_class, nodes_data, edges_data):
    """Benchmark a single library"""
    try:
        # Create graph
        bench = benchmark_class(nodes_data, edges_data)
        
        # Run filtering operations
        results = {
            'creation_time': bench.creation_time,
            'memory_usage': getattr(bench, 'memory_usage', 0)
        }
        
        # Test filtering operations
        operations = [
            ('node_role', 'filter_nodes_by_role'),
            ('node_salary', 'filter_nodes_by_salary'),
            ('node_complex', 'filter_nodes_complex'),
            ('edge_relationship', 'filter_edges_by_relationship'),
            ('edge_strength', 'filter_edges_by_strength')
        ]
        
        for op_name, method_name in operations:
            try:
                method = getattr(bench, method_name)
                time_taken, count = method()
                results[op_name] = time_taken
                print(f"     {op_name}: {time_taken:.4f}s ({count} results)")
            except Exception as e:
                print(f"     ❌ {op_name} failed: {e}")
                results[op_name] = None
        
        # Show graph stats if available
        if hasattr(bench, 'get_stats'):
            stats = bench.get_stats()
            print(f"     Graph stats: {stats}")
        
        return results
        
    except Exception as e:
        print(f"   ❌ Failed to create {benchmark_class.__name__}: {e}")
        return None

def main():
    """Main benchmark runner"""
    print("🚀 Native vs Original Groggy Benchmark")
    print("=" * 70)
    
    # Test sizes
    test_sizes = [
        (5000, 5000),     # 5K nodes, 5K edges
        # (10000, 10000),   # 10K nodes, 10K edges
    ]
    
    for num_nodes, num_edges in test_sizes:
        print(f"\n🔥 BENCHMARK: {num_nodes:,} nodes, {num_edges:,} edges")
        print("=" * 70)
        
        # Generate test data
        nodes_data, edges_data = create_test_data(num_nodes, num_edges)
        
        # Benchmark each available library
        all_results = {}
        
        # Native Groggy
        if libraries_available['native_groggy']:
            print("📊 Testing Native Groggy...")
            all_results['Native Groggy'] = benchmark_library(NativeGroggyBenchmark, nodes_data, edges_data)
        
        # Original Groggy
        if libraries_available['original_groggy']:
            print("\n📊 Testing Original Groggy...")
            all_results['Original Groggy'] = benchmark_library(OriginalGroggyBenchmark, nodes_data, edges_data)
        
        # NetworkX
        if libraries_available['networkx']:
            print("\n📊 Testing NetworkX...")
            all_results['NetworkX'] = benchmark_library(NetworkXBenchmark, nodes_data, edges_data)
        
        # Clean up memory
        gc.collect()
        
        # Print comparison table
        print(f"\n📈 RESULTS COMPARISON ({num_nodes:,} nodes)")
        print("=" * 100)
        
        operations = [
            ('Graph Creation', 'creation_time'),
            ('Memory Usage (MB)', 'memory_usage'),
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
            header += f"{lib:<25}"
        print(header)
        print("-" * 100)
        
        # Print results
        for op_name, op_key in operations:
            row = f"{op_name:<25}"
            for lib in libraries:
                if all_results[lib] and op_key in all_results[lib] and all_results[lib][op_key] is not None:
                    value = all_results[lib][op_key]
                    if op_key == 'memory_usage':
                        row += f"{value:<25.1f}"
                    else:
                        row += f"{value:<25.4f}"
                else:
                    row += f"{'N/A':<25}"
            print(row)
        
        # Performance insights
        if 'Native Groggy' in all_results and all_results['Native Groggy']:
            print(f"\n🎯 Performance Insights for {num_nodes:,} nodes:")
            native_results = all_results['Native Groggy']
            
            for lib in libraries:
                if lib != 'Native Groggy' and all_results[lib]:
                    lib_results = all_results[lib]
                    
                    # Compare creation time
                    if 'creation_time' in native_results and 'creation_time' in lib_results:
                        native_time = native_results['creation_time']
                        lib_time = lib_results['creation_time']
                        if lib_time > 0 and native_time > 0:
                            speedup = lib_time / native_time
                            if speedup > 1:
                                print(f"   🚀 Native Groggy is {speedup:.1f}x faster than {lib} for graph creation")
                            elif speedup < 1:
                                print(f"   📊 {lib} is {1/speedup:.1f}x faster than Native Groggy for graph creation")
                    
                    # Compare memory usage
                    if 'memory_usage' in native_results and 'memory_usage' in lib_results:
                        native_mem = native_results['memory_usage']
                        lib_mem = lib_results['memory_usage']
                        if lib_mem > 0 and native_mem > 0:
                            mem_ratio = lib_mem / native_mem
                            if mem_ratio > 1:
                                print(f"   💾 Native Groggy uses {mem_ratio:.1f}x less memory than {lib} ({format_memory(native_mem)} vs {format_memory(lib_mem)})")
                            elif mem_ratio < 1:
                                print(f"   💾 {lib} uses {1/mem_ratio:.1f}x less memory than Native Groggy ({format_memory(lib_mem)} vs {format_memory(native_mem)})")
        
        print("\n" + "=" * 70)

if __name__ == "__main__":
    main()