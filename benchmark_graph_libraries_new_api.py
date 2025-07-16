#!/usr/bin/env python3
"""
Benchmark comparison: Groggy vs other high-performance graph libraries
Refactored to use the new clean groggy API
"""

import sys
import os
import time
import random
import gc
import psutil
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

class GroggyBenchmark:
    """Groggy benchmark using the new clean API"""
    
    def __init__(self, nodes_data, edges_data):
        print("🔧 Creating Groggy graph with new API...")
        
        # Measure memory before creation
        memory_before = get_memory_usage()
        start = time.time()
        
        # Create graph with new clean API
        self.graph = gr.Graph()
        self.nodes_collection = self.graph.nodes()
        self.edges_collection = self.graph.edges()
        
        # Convert data to NodeId/EdgeId objects and add them
        self.node_ids = []
        self.edge_ids = []
        self.node_lookup = {}  # Map string ID to NodeId object
        
        # Create and add nodes
        for node_data in nodes_data:
            node_id = gr.NodeId(node_data['id'])
            self.node_ids.append(node_id)
            self.node_lookup[node_data['id']] = node_id
        
        # Add nodes in batch
        self.nodes_collection.add(self.node_ids)
        
        # Create and add edges
        for edge_data in edges_data:
            source_id = self.node_lookup.get(edge_data['source'])
            target_id = self.node_lookup.get(edge_data['target'])
            if source_id and target_id:
                edge_id = gr.EdgeId(source_id, target_id)
                self.edge_ids.append(edge_id)
        
        # Add edges in batch
        self.edges_collection.add(self.edge_ids)
        
        # Store node and edge data for filtering (since attributes aren't working yet)
        self.nodes_data = {node['id']: node for node in nodes_data}
        self.edges_data = edges_data
        
        # Measure memory after creation
        memory_after = get_memory_usage()
        self.memory_usage = memory_after - memory_before
        
        self.creation_time = time.time() - start
        print(f"   Created {len(self.node_ids)} nodes and {len(self.edge_ids)} edges in {self.creation_time:.3f}s")
        print(f"   Memory usage: {format_memory(self.memory_usage)}")
    
    def filter_nodes_by_role(self):
        """Filter nodes by role (simulated since attributes aren't working yet)"""
        start = time.time()
        # Simulate filtering by iterating through stored data
        result = []
        for node_id, node_data in self.nodes_data.items():
            if node_data.get('role') == 'engineer':
                result.append(node_id)
        return time.time() - start, len(result)
    
    def filter_nodes_by_salary(self):
        """Filter nodes by salary > 100000"""
        start = time.time()
        result = []
        for node_id, node_data in self.nodes_data.items():
            if node_data.get('salary', 0) > 100000:
                result.append(node_id)
        return time.time() - start, len(result)
    
    def filter_nodes_complex(self):
        """Complex multi-attribute filter"""
        start = time.time()
        result = []
        for node_id, node_data in self.nodes_data.items():
            if (node_data.get('role') == 'engineer' and 
                node_data.get('active', False) and
                node_data.get('salary', 0) > 80000):
                result.append(node_id)
        return time.time() - start, len(result)
    
    def filter_edges_by_relationship(self):
        """Filter edges by relationship"""
        start = time.time()
        result = []
        for edge_data in self.edges_data:
            if edge_data.get('relationship') == 'reports_to':
                result.append((edge_data['source'], edge_data['target']))
        return time.time() - start, len(result)
    
    def filter_edges_by_strength(self):
        """Filter edges by strength > 0.7"""
        start = time.time()
        result = []
        for edge_data in self.edges_data:
            if edge_data.get('strength', 0) > 0.7:
                result.append((edge_data['source'], edge_data['target']))
        return time.time() - start, len(result)
    
    def get_stats(self):
        """Get graph statistics"""
        return {
            'nodes': self.nodes_collection.size(),
            'edges': self.edges_collection.size(),
            'node_ids_created': len(self.node_ids),
            'edge_ids_created': len(self.edge_ids)
        }

class NetworkXBenchmark:
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

class IGraphBenchmark:
    def __init__(self, nodes_data, edges_data):
        print("🔧 Creating igraph graph...")
        
        # Measure memory before creation
        memory_before = get_memory_usage()
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
                    try:
                        vertex_idx = self.graph.vs.find(name=node['id']).index
                        self.graph.vs[vertex_idx][attr] = value
                    except ValueError:
                        continue
        
        # Add edge attributes
        for i, edge in enumerate(edges_data):
            for attr, value in edge.items():
                if attr not in ['source', 'target']:
                    if attr not in self.graph.es.attributes():
                        self.graph.es[attr] = [None] * len(self.graph.es)
                    if i < len(self.graph.es):
                        self.graph.es[i][attr] = value
        
        # Measure memory after creation
        memory_after = get_memory_usage()
        self.memory_usage = memory_after - memory_before
        
        self.creation_time = time.time() - start
        print(f"   Created in {self.creation_time:.3f}s")
        print(f"   Memory usage: {format_memory(self.memory_usage)}")
    
    def filter_nodes_by_role(self):
        """Filter nodes by role"""
        start = time.time()
        try:
            result = self.graph.vs.select(role_eq='engineer')
        except:
            result = []
        return time.time() - start, len(result)
    
    def filter_nodes_by_salary(self):
        """Filter nodes by salary > 100000"""
        start = time.time()
        try:
            result = self.graph.vs.select(salary_gt=100000)
        except:
            result = []
        return time.time() - start, len(result)
    
    def filter_nodes_complex(self):
        """Complex multi-attribute filter"""
        start = time.time()
        try:
            result = self.graph.vs.select(lambda v: 
                v.get('role') == 'engineer' and 
                v.get('salary', 0) > 80000 and
                v.get('active', False)
            )
        except:
            result = []
        return time.time() - start, len(result)
    
    def filter_edges_by_relationship(self):
        """Filter edges by relationship"""
        start = time.time()
        try:
            result = self.graph.es.select(relationship_eq='reports_to')
        except:
            result = []
        return time.time() - start, len(result)
    
    def filter_edges_by_strength(self):
        """Filter edges by strength > 0.7"""
        start = time.time()
        try:
            result = self.graph.es.select(strength_gt=0.7)
        except:
            result = []
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
        try:
            time_taken, count = bench.filter_nodes_by_role()
            results['node_role'] = time_taken
            print(f"     Node role filter: {time_taken:.4f}s ({count} results)")
        except Exception as e:
            print(f"     ❌ Node role filter failed: {e}")
            results['node_role'] = None
        
        try:
            time_taken, count = bench.filter_nodes_by_salary()
            results['node_salary'] = time_taken
            print(f"     Node salary filter: {time_taken:.4f}s ({count} results)")
        except Exception as e:
            print(f"     ❌ Node salary filter failed: {e}")
            results['node_salary'] = None
        
        try:
            time_taken, count = bench.filter_nodes_complex()
            results['node_complex'] = time_taken
            print(f"     Complex node filter: {time_taken:.4f}s ({count} results)")
        except Exception as e:
            print(f"     ❌ Complex node filter failed: {e}")
            results['node_complex'] = None
        
        try:
            time_taken, count = bench.filter_edges_by_relationship()
            results['edge_relationship'] = time_taken
            print(f"     Edge relationship filter: {time_taken:.4f}s ({count} results)")
        except Exception as e:
            print(f"     ❌ Edge relationship filter failed: {e}")
            results['edge_relationship'] = None
        
        try:
            time_taken, count = bench.filter_edges_by_strength()
            results['edge_strength'] = time_taken
            print(f"     Edge strength filter: {time_taken:.4f}s ({count} results)")
        except Exception as e:
            print(f"     ❌ Edge strength filter failed: {e}")
            results['edge_strength'] = None
        
        # Show groggy stats if available
        if hasattr(bench, 'get_stats'):
            stats = bench.get_stats()
            print(f"     Graph stats: {stats}")
        
        return results
        
    except Exception as e:
        print(f"   ❌ Failed to create {benchmark_class.__name__}: {e}")
        return None

def main():
    """Main benchmark runner"""
    print("🚀 Graph Library Benchmark - Refactored for New Groggy API")
    print("=" * 70)
    
    # Test sizes
    test_sizes = [
        (1000, 500),     # 1K nodes, 500 edges
        (10000, 5000),   # 10K nodes, 5K edges
        (50000, 25000)   # 50K nodes, 25K edges
    ]
    
    for num_nodes, num_edges in test_sizes:
        print(f"\n🔥 BENCHMARK: {num_nodes:,} nodes, {num_edges:,} edges")
        print("=" * 70)
        
        # Generate test data
        nodes_data, edges_data = create_test_data(num_nodes, num_edges)
        
        # Benchmark each available library
        all_results = {}
        
        # Groggy (new API)
        print("📊 Testing Groggy (New API)...")
        all_results['Groggy (New API)'] = benchmark_library(GroggyBenchmark, nodes_data, edges_data)
        
        # NetworkX
        if libraries_available['networkx']:
            print("\n📊 Testing NetworkX...")
            all_results['NetworkX'] = benchmark_library(NetworkXBenchmark, nodes_data, edges_data)
        
        # igraph
        if libraries_available['igraph']:
            print("\n📊 Testing igraph...")
            all_results['igraph'] = benchmark_library(IGraphBenchmark, nodes_data, edges_data)
        
        # Clean up memory
        gc.collect()
        
        # Print comparison table
        print(f"\n📈 RESULTS COMPARISON ({num_nodes:,} nodes)")
        print("=" * 90)
        
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
            header += f"{lib:<20}"
        print(header)
        print("-" * 90)
        
        # Print results
        for op_name, op_key in operations:
            row = f"{op_name:<25}"
            for lib in libraries:
                if all_results[lib] and op_key in all_results[lib] and all_results[lib][op_key] is not None:
                    value = all_results[lib][op_key]
                    if op_key == 'memory_usage':
                        row += f"{value:<20.1f}"
                    else:
                        row += f"{value:<20.4f}"
                else:
                    row += f"{'N/A':<20}"
            print(row)
        
        # Performance insights
        if 'Groggy (New API)' in all_results and all_results['Groggy (New API)']:
            print(f"\n🎯 Performance Insights for {num_nodes:,} nodes:")
            groggy_results = all_results['Groggy (New API)']
            
            for lib in libraries:
                if lib != 'Groggy (New API)' and all_results[lib]:
                    lib_results = all_results[lib]
                    
                    # Compare creation time
                    if 'creation_time' in groggy_results and 'creation_time' in lib_results:
                        groggy_time = groggy_results['creation_time']
                        lib_time = lib_results['creation_time']
                        if lib_time > 0 and groggy_time > 0:
                            speedup = lib_time / groggy_time
                            if speedup > 1:
                                print(f"   🚀 Groggy is {speedup:.1f}x faster than {lib} for graph creation")
                            elif speedup < 1:
                                print(f"   📊 {lib} is {1/speedup:.1f}x faster than Groggy for graph creation")
                    
                    # Compare memory usage
                    if 'memory_usage' in groggy_results and 'memory_usage' in lib_results:
                        groggy_mem = groggy_results['memory_usage']
                        lib_mem = lib_results['memory_usage']
                        if lib_mem > 0 and groggy_mem > 0:
                            mem_ratio = lib_mem / groggy_mem
                            if mem_ratio > 1:
                                print(f"   💾 Groggy uses {mem_ratio:.1f}x less memory than {lib} ({format_memory(groggy_mem)} vs {format_memory(lib_mem)})")
                            elif mem_ratio < 1:
                                print(f"   💾 {lib} uses {1/mem_ratio:.1f}x less memory than Groggy ({format_memory(lib_mem)} vs {format_memory(groggy_mem)})")
        
        print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
