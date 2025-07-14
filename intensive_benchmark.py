#!/usr/bin/env python3
"""
🚀 COMPREHENSIVE GROGGY VS NETWORKX BENCHMARK SUITE
===================================================

Intensive testing of:
- Graph creation (various sizes and structures)
- Numerical analysis and statistics
- Basic graph operations
- Advanced filtering and queries
- Graph traversals and algorithms
- Memory usage and scalability

This benchmark pushes both libraries to their limits.
"""

import time
import random
import statistics
import gc
import psutil
import os
import sys
import traceback
import tracemalloc
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict

# Add groggy to path
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python')

import groggy as gr

# Try to import NetworkX
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("❌ NetworkX not available")

# Try to import NumPy for numerical analysis
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("❌ NumPy not available - some numerical tests will be skipped")


@dataclass
class BenchmarkResult:
    """Store benchmark results with timing and memory info"""
    operation: str
    library: str
    time_ms: float
    memory_mb: float
    result_size: int
    success: bool = True
    error: str = ""


class MemoryTracker:
    """Track memory usage during operations"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_memory = 0
        
    def start(self):
        gc.collect()  # Clean up before measurement
        self.start_memory = self.process.memory_info().rss / 1024 / 1024
        
    def stop(self):
        gc.collect()
        end_memory = self.process.memory_info().rss / 1024 / 1024
        return end_memory - self.start_memory


class ComprehensiveBenchmark:
    """Main benchmark suite"""
    
    def __init__(self):
        self.results = []
        self.memory_tracker = MemoryTracker()
        
    def time_operation(self, func, *args, **kwargs):
        """Time an operation and track memory"""
        self.memory_tracker.start()
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = (time.time() - start_time) * 1000  # Convert to ms
            memory_used = self.memory_tracker.stop()
            return elapsed, memory_used, result, True, ""
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            memory_used = self.memory_tracker.stop()
            return elapsed, memory_used, None, False, str(e)
    
    def run_benchmark(self, operation_name: str, groggy_func, networkx_func=None, 
                     *args, **kwargs):
        """Run a benchmark comparing Groggy and NetworkX"""
        
        print(f"  🔄 {operation_name}...")
        
        # Test Groggy
        elapsed, memory, result, success, error = self.time_operation(groggy_func, *args, **kwargs)
        result_size = len(result) if hasattr(result, '__len__') else (1 if result is not None else 0)
        
        self.results.append(BenchmarkResult(
            operation=operation_name,
            library="Groggy",
            time_ms=elapsed,
            memory_mb=memory,
            result_size=result_size,
            success=success,
            error=error
        ))
        
        # Test NetworkX if available and provided
        if NETWORKX_AVAILABLE and networkx_func:
            elapsed, memory, result, success, error = self.time_operation(networkx_func, *args, **kwargs)
            result_size = len(result) if hasattr(result, '__len__') else (1 if result is not None else 0)
            
            self.results.append(BenchmarkResult(
                operation=operation_name,
                library="NetworkX",
                time_ms=elapsed,
                memory_mb=memory,
                result_size=result_size,
                success=success,
                error=error
            ))
    
    def create_test_data(self, n_nodes: int, n_edges: int, structured: bool = False):
        """Create test data for benchmarks"""
        nodes_data = []
        edges_data = []
        
        # Create nodes with rich attributes
        for i in range(n_nodes):
            node_data = {
                'id': f'n{i}',
                'salary': random.randint(40000, 200000),
                'age': random.randint(22, 65),
                'experience': random.randint(0, 40),
                'rating': round(random.uniform(1.0, 5.0), 2),
                'role': random.choice(['engineer', 'manager', 'analyst', 'director', 'senior_engineer']),
                'department': random.choice(['Engineering', 'Sales', 'Marketing', 'Finance', 'HR', 'Operations']),
                'location': random.choice(['San Francisco', 'New York', 'Austin', 'Seattle', 'Boston']),
                'skills': random.choice(['Python', 'JavaScript', 'Java', 'C++', 'Go', 'Rust']),
                'active': random.choice([True, False]),
                'hire_date': f"202{random.randint(0, 3)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"
            }
            nodes_data.append(node_data)
        
        # Create edges with attributes
        if structured:
            # Create a more structured graph (small-world-like)
            for i in range(n_nodes):
                # Connect to a few nearby nodes
                for j in range(min(3, n_nodes - i - 1)):
                    target = (i + j + 1) % n_nodes
                    edge_data = {
                        'source': f'n{i}',
                        'target': f'n{target}',
                        'weight': round(random.uniform(0.1, 1.0), 3),
                        'relationship': random.choice(['reports_to', 'collaborates', 'mentors', 'friend']),
                        'frequency': random.choice(['daily', 'weekly', 'monthly', 'rare']),
                        'strength': random.uniform(0.1, 1.0),
                        'created_date': f"202{random.randint(1, 3)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"
                    }
                    edges_data.append(edge_data)
                    if len(edges_data) >= n_edges:
                        break
                if len(edges_data) >= n_edges:
                    break
        else:
            # Random edges
            for _ in range(n_edges):
                source = random.randint(0, n_nodes - 1)
                target = random.randint(0, n_nodes - 1)
                if source != target:  # Avoid self-loops
                    edge_data = {
                        'source': f'n{source}',
                        'target': f'n{target}',
                        'weight': round(random.uniform(0.1, 1.0), 3),
                        'relationship': random.choice(['reports_to', 'collaborates', 'mentors', 'friend']),
                        'frequency': random.choice(['daily', 'weekly', 'monthly', 'rare']),
                        'strength': random.uniform(0.1, 1.0),
                        'created_date': f"202{random.randint(1, 3)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"
                    }
                    edges_data.append(edge_data)
        
        return nodes_data[:n_nodes], edges_data[:n_edges]
    
    def benchmark_graph_creation(self):
        """Benchmark graph creation with various sizes"""
        print("\n🏗️ GRAPH CREATION BENCHMARKS")
        print("=" * 50)
        
        test_sizes = [
            (500, 250),
            (1000, 500), 
            (2500, 1250),
            (5000, 2500)
        ]
        
        for n_nodes, n_edges in test_sizes:
            print(f"\n📊 Testing {n_nodes:,} nodes, {n_edges:,} edges")
            
            nodes_data, edges_data = self.create_test_data(n_nodes, n_edges)
            
            # Groggy creation
            def create_groggy():
                graph = gr.Graph(backend='rust')
                graph.add_nodes([{k: v for k, v in node.items()} for node in nodes_data])
                graph.add_edges(edges_data)
                return graph
            
            # NetworkX creation using batch operations
            def create_networkx():
                if not NETWORKX_AVAILABLE:
                    return None
                graph = nx.DiGraph()
                # Add nodes with attributes (batch)
                nodes_for_nx = [(node['id'], {k: v for k, v in node.items() if k != 'id'}) for node in nodes_data]
                graph.add_nodes_from(nodes_for_nx)
                # Add edges with attributes (batch)
                edges_for_nx = [(edge['source'], edge['target'], {k: v for k, v in edge.items() if k not in ['source', 'target']}) for edge in edges_data]
                graph.add_edges_from(edges_for_nx)
                return graph
            
            self.run_benchmark(f"Create graph ({n_nodes:,} nodes)", create_groggy, create_networkx)
    
    def benchmark_basic_operations(self):
        """Benchmark basic graph operations"""
        print("\n🔧 BASIC OPERATIONS BENCHMARKS")
        print("=" * 50)
        
        # Create test graphs
        nodes_data, edges_data = self.create_test_data(5000, 2500)
        
        print("Creating test graphs...")
        groggy_graph = gr.Graph(backend='rust')
        groggy_graph.add_nodes(nodes_data)
        groggy_graph.add_edges(edges_data)
        
        nx_graph = None
        if NETWORKX_AVAILABLE:
            nx_graph = nx.DiGraph()
            # Add nodes with attributes (batch)
            nodes_for_nx = [(node['id'], {k: v for k, v in node.items() if k != 'id'}) for node in nodes_data]
            nx_graph.add_nodes_from(nodes_for_nx)
            # Add edges with attributes (batch)
            edges_for_nx = [(edge['source'], edge['target'], {k: v for k, v in edge.items() if k not in ['source', 'target']}) for edge in edges_data]
            nx_graph.add_edges_from(edges_for_nx)
        
        # Test basic operations
        operations = [
            ("Node count", lambda: groggy_graph.node_count(), lambda: nx_graph.number_of_nodes() if nx_graph else 0),
            ("Edge count", lambda: groggy_graph.edge_count(), lambda: nx_graph.number_of_edges() if nx_graph else 0),
            ("Get all node IDs", lambda: groggy_graph.get_node_ids(), lambda: list(nx_graph.nodes()) if nx_graph else []),
            ("Check node exists", lambda: groggy_graph.has_node('n100'), lambda: nx_graph.has_node('n100') if nx_graph else False),
            ("Check edge exists", lambda: groggy_graph.has_edge('n100', 'n200'), lambda: nx_graph.has_edge('n100', 'n200') if nx_graph else False),
            ("Get node neighbors", lambda: groggy_graph.get_neighbors('n100'), lambda: list(nx_graph.neighbors('n100')) if nx_graph else []),
        ]
        
        for op_name, groggy_func, nx_func in operations:
            self.run_benchmark(op_name, groggy_func, nx_func)
    
    def benchmark_filtering_operations(self):
        """Benchmark filtering and query operations"""
        print("\n🔍 FILTERING & QUERY BENCHMARKS")
        print("=" * 50)
        
        # Create test graphs with rich data
        nodes_data, edges_data = self.create_test_data(10000, 5000)
        
        print("Creating test graphs...")
        groggy_graph = gr.Graph(backend='rust')
        groggy_graph.add_nodes(nodes_data)
        groggy_graph.add_edges(edges_data)
        
        nx_graph = None
        if NETWORKX_AVAILABLE:
            nx_graph = nx.DiGraph()
            # Add nodes with attributes (batch)
            nodes_for_nx = [(node['id'], {k: v for k, v in node.items() if k != 'id'}) for node in nodes_data]
            nx_graph.add_nodes_from(nodes_for_nx)
            # Add edges with attributes (batch)
            edges_for_nx = [(edge['source'], edge['target'], {k: v for k, v in edge.items() if k not in ['source', 'target']}) for edge in edges_data]
            nx_graph.add_edges_from(edges_for_nx)
        
        # Node filtering operations
        node_filters = [
            ("Exact role filter", {'role': 'engineer'}),
            ("Salary range filter", {'salary': ('>', 80000)}),
            ("Age range filter", {'age': ('>', 30)}),
            ("Complex numeric filter", {'salary': ('>', 80000), 'age': ('>', 30)}),
            ("Multi-criteria filter", {'role': 'engineer', 'salary': ('>', 100000), 'experience': ('>=', 5)}),
            ("Department filter", {'department': 'Engineering'}),
            ("Location + role filter", {'location': 'San Francisco', 'role': 'engineer'}),
            ("Rating filter", {'rating': ('>', 4.0)}),
            ("String contains filter", {'skills': ('contains', 'Python')}),
            ("Boolean filter", {'active': True}),
        ]
        
        for filter_name, filter_dict in node_filters:
            # Groggy filtering
            groggy_func = lambda f=filter_dict: groggy_graph.filter_nodes(f)
            
            # NetworkX filtering (approximate)
            def create_nx_func(f):
                def nx_func():
                    if not NETWORKX_AVAILABLE or not nx_graph:
                        return []
                    
                    def matches_filter(node_id, attrs):
                        for attr_name, condition in f.items():
                            if attr_name not in attrs:
                                return False
                            
                            value = attrs[attr_name]
                            if isinstance(condition, tuple):
                                op, target = condition
                                if op == '>':
                                    if not (value > target):
                                        return False
                                elif op == '>=':
                                    if not (value >= target):
                                        return False
                                elif op == '<':
                                    if not (value < target):
                                        return False
                                elif op == '<=':
                                    if not (value <= target):
                                        return False
                                elif op == 'contains':
                                    if target not in str(value):
                                        return False
                            else:
                                if value != condition:
                                    return False
                        return True
                    
                    return [node for node, attrs in nx_graph.nodes(data=True) if matches_filter(node, attrs)]
                return nx_func
            
            nx_func = create_nx_func(filter_dict)
            self.run_benchmark(f"Node filter: {filter_name}", groggy_func, nx_func)
        
        # Edge filtering operations
        edge_filters = [
            ("Relationship filter", {'relationship': 'collaborates'}),
            ("Weight range filter", {'weight': ('>', 0.5)}),
            ("Frequency filter", {'frequency': 'daily'}),
            ("Strength filter", {'strength': ('>', 0.7)}),
            ("Complex edge filter", {'relationship': 'collaborates', 'weight': ('>', 0.5)}),
        ]
        
        for filter_name, filter_dict in edge_filters:
            # Groggy edge filtering
            groggy_func = lambda f=filter_dict: groggy_graph.filter_edges(f)
            
            # NetworkX edge filtering
            def create_nx_edge_func(f):
                def nx_func():
                    if not NETWORKX_AVAILABLE or not nx_graph:
                        return []
                    
                    def matches_filter(attrs):
                        for attr_name, condition in f.items():
                            if attr_name not in attrs:
                                return False
                            
                            value = attrs[attr_name]
                            if isinstance(condition, tuple):
                                op, target = condition
                                if op == '>':
                                    if not (value > target):
                                        return False
                                elif op == '>=':
                                    if not (value >= target):
                                        return False
                            else:
                                if value != condition:
                                    return False
                        return True
                    
                    return [(u, v) for u, v, attrs in nx_graph.edges(data=True) if matches_filter(attrs)]
                return nx_func
            
            nx_func = create_nx_edge_func(filter_dict)
            self.run_benchmark(f"Edge filter: {filter_name}", groggy_func, nx_func)
    
    def benchmark_numerical_analysis(self):
        """Benchmark numerical analysis operations"""
        print("\n📊 NUMERICAL ANALYSIS BENCHMARKS")
        print("=" * 50)
        
        if not NUMPY_AVAILABLE:
            print("⚠️ NumPy not available - skipping numerical analysis")
            return
        
        # Create test graphs
        nodes_data, edges_data = self.create_test_data(7500, 3750)
        
        print("Creating test graphs...")
        groggy_graph = gr.Graph(backend='rust')
        groggy_graph.add_nodes(nodes_data)
        groggy_graph.add_edges(edges_data)
        
        nx_graph = None
        if NETWORKX_AVAILABLE:
            nx_graph = nx.DiGraph()
            # Add nodes with attributes (batch)
            nodes_for_nx = [(node['id'], {k: v for k, v in node.items() if k != 'id'}) for node in nodes_data]
            nx_graph.add_nodes_from(nodes_for_nx)
            # Add edges with attributes (batch)
            edges_for_nx = [(edge['source'], edge['target'], {k: v for k, v in edge.items() if k not in ['source', 'target']}) for edge in edges_data]
            nx_graph.add_edges_from(edges_for_nx)
        
        # Numerical operations on node attributes
        def groggy_salary_stats():
            # Use the new efficient bulk attribute retrieval method
            all_salaries = groggy_graph.get_all_nodes_attribute('salary')
            salaries = list(all_salaries.values())
            return {
                'mean': np.mean(salaries),
                'median': np.median(salaries),
                'std': np.std(salaries),
                'min': np.min(salaries),
                'max': np.max(salaries),
                'count': len(salaries)
            }
        
        def nx_salary_stats():
            if not NETWORKX_AVAILABLE or nx_graph is None:
                return {}
            salaries = [attrs.get('salary', 0) for _, attrs in nx_graph.nodes(data=True) if 'salary' in attrs]
            return {
                'mean': np.mean(salaries),
                'median': np.median(salaries),
                'std': np.std(salaries),
                'min': np.min(salaries),
                'max': np.max(salaries),
                'count': len(salaries)
            }
        
        # Edge weight analysis using batch methods
        def groggy_edge_weight_stats():
            # Use the new efficient bulk edge attribute retrieval method
            all_weights = groggy_graph.get_all_edges_attribute('weight')
            weights = list(all_weights.values())
            return {
                'mean': np.mean(weights),
                'median': np.median(weights),
                'std': np.std(weights),
                'min': np.min(weights),
                'max': np.max(weights),
                'count': len(weights)
            }
        
        def nx_edge_weight_stats():
            if not NETWORKX_AVAILABLE or nx_graph is None:
                return {}
            weights = [attrs.get('weight', 0) for _, _, attrs in nx_graph.edges(data=True) if 'weight' in attrs]
            return {
                'mean': np.mean(weights),
                'median': np.median(weights),
                'std': np.std(weights),
                'min': np.min(weights),
                'max': np.max(weights),
                'count': len(weights)
            }
        
        # Multi-attribute analysis using batch methods
        def groggy_multi_attribute_analysis():
            # Get multiple attributes at once for efficiency
            node_ids = list(groggy_graph.get_node_ids())[:1000]  # Sample subset for performance
            multi_attrs = groggy_graph.get_nodes_attributes(node_ids)  # Fixed: removed the second parameter
            
            # Calculate correlations and statistics
            salaries = [attrs.get('salary', 0) for attrs in multi_attrs.values()]
            ages = [attrs.get('age', 0) for attrs in multi_attrs.values()]
            experiences = [attrs.get('experience', 0) for attrs in multi_attrs.values()]
            
            return {
                'salary_age_corr': np.corrcoef(salaries, ages)[0, 1] if len(salaries) > 1 else 0,
                'salary_exp_corr': np.corrcoef(salaries, experiences)[0, 1] if len(salaries) > 1 else 0,
                'count': len(multi_attrs)
            }
        
        def nx_multi_attribute_analysis():
            if not NETWORKX_AVAILABLE or nx_graph is None:
                return {}
            
            # Sample subset for performance
            nodes_sample = list(nx_graph.nodes())[:1000]
            salaries = []
            ages = []
            experiences = []
            
            for node in nodes_sample:
                attrs = nx_graph.nodes[node]
                salaries.append(attrs.get('salary', 0))
                ages.append(attrs.get('age', 0))
                experiences.append(attrs.get('experience', 0))
            
            return {
                'salary_age_corr': np.corrcoef(salaries, ages)[0, 1] if len(salaries) > 1 else 0,
                'salary_exp_corr': np.corrcoef(salaries, experiences)[0, 1] if len(salaries) > 1 else 0,
                'count': len(salaries)
            }
        
        # Specific node subset analysis
        def groggy_subset_analysis():
            # Get engineers' salaries using batch method
            engineer_nodes = groggy_graph.filter_nodes({'role': 'engineer'})
            if engineer_nodes:
                engineer_salaries = groggy_graph.get_nodes_attribute(engineer_nodes, 'salary')
                salaries = list(engineer_salaries.values())
                return {
                    'engineer_avg_salary': np.mean(salaries) if salaries else 0,
                    'engineer_count': len(salaries)
                }
            return {'engineer_avg_salary': 0, 'engineer_count': 0}
        
        def nx_subset_analysis():
            if not NETWORKX_AVAILABLE or nx_graph is None:
                return {}
            
            engineer_salaries = []
            for node, attrs in nx_graph.nodes(data=True):
                if attrs.get('role') == 'engineer':
                    engineer_salaries.append(attrs.get('salary', 0))
            
            return {
                'engineer_avg_salary': np.mean(engineer_salaries) if engineer_salaries else 0,
                'engineer_count': len(engineer_salaries)
            }

        # More numerical operations
        numerical_ops = [
            ("Salary statistics", groggy_salary_stats, nx_salary_stats),
            ("Edge weight statistics", groggy_edge_weight_stats, nx_edge_weight_stats),
            ("Multi-attribute correlation analysis", groggy_multi_attribute_analysis, nx_multi_attribute_analysis),
            ("Engineer subset analysis", groggy_subset_analysis, nx_subset_analysis),
            ("High salary filter + stats", 
             lambda: len(groggy_graph.filter_nodes({'salary': ('>', 150000)})),
             lambda: len([n for n, attrs in nx_graph.nodes(data=True) if attrs.get('salary', 0) > 150000]) if NETWORKX_AVAILABLE and nx_graph else 0),
            ("Age distribution analysis",
             lambda: len(groggy_graph.filter_nodes({'age': ('>', 40)})),
             lambda: len([n for n, attrs in nx_graph.nodes(data=True) if attrs.get('age', 0) > 40]) if NETWORKX_AVAILABLE and nx_graph else 0),
            ("Experience vs Salary correlation",
             lambda: len(groggy_graph.filter_nodes({'experience': ('>', 10), 'salary': ('>', 120000)})),
             lambda: len([n for n, attrs in nx_graph.nodes(data=True) 
                         if attrs.get('experience', 0) > 10 and attrs.get('salary', 0) > 120000]) if NETWORKX_AVAILABLE and nx_graph else 0),
        ]
        
        for op_name, groggy_func, nx_func in numerical_ops:
            self.run_benchmark(op_name, groggy_func, nx_func)
    
    def benchmark_batch_attribute_operations(self):
        """Benchmark comprehensive batch attribute operations"""
        print("\n⚡ BATCH ATTRIBUTE OPERATIONS BENCHMARKS")
        print("=" * 50)
        
        # Create test graphs
        nodes_data, edges_data = self.create_test_data(8000, 4000)
        
        print("Creating test graphs...")
        groggy_graph = gr.Graph(backend='rust')
        groggy_graph.add_nodes(nodes_data)
        groggy_graph.add_edges(edges_data)
        
        nx_graph = None
        if NETWORKX_AVAILABLE:
            nx_graph = nx.DiGraph()
            # Add nodes with attributes (batch)
            nodes_for_nx = [(node['id'], {k: v for k, v in node.items() if k != 'id'}) for node in nodes_data]
            nx_graph.add_nodes_from(nodes_for_nx)
            # Add edges with attributes (batch)
            edges_for_nx = [(edge['source'], edge['target'], {k: v for k, v in edge.items() if k not in ['source', 'target']}) for edge in edges_data]
            nx_graph.add_edges_from(edges_for_nx)
        
        # Test all batch attribute methods
        batch_ops = [
            # Single attribute, all nodes
            ("Get all node salaries", 
             lambda: groggy_graph.get_all_nodes_attribute('salary'),
             lambda: {n: attrs.get('salary', 0) for n, attrs in nx_graph.nodes(data=True)} if nx_graph else {}),
            
            # Single attribute, all edges  
            ("Get all edge weights",
             lambda: groggy_graph.get_all_edges_attribute('weight'),
             lambda: {(u, v): attrs.get('weight', 0) for u, v, attrs in nx_graph.edges(data=True)} if nx_graph else {}),
            
            # Single attribute, subset of nodes
            ("Get subset node ages",
             lambda: groggy_graph.get_nodes_attribute(list(groggy_graph.get_node_ids())[:1000], 'age'),
             lambda: {n: nx_graph.nodes[n].get('age', 0) for n in list(nx_graph.nodes())[:1000]} if nx_graph else {}),
            
            # Single attribute, subset of edges
            ("Get subset edge relationships",
             lambda: groggy_graph.get_edges_attribute([tuple(edge_id.split('->')) for edge_id in list(groggy_graph.get_edge_ids())[:1000]], 'relationship'),  # Fixed: convert edge IDs to tuples
             lambda: {(u, v): nx_graph.edges[u, v].get('relationship', '') for u, v in list(nx_graph.edges())[:1000]} if nx_graph else {}),
            
            # Multiple attributes, subset of nodes
            ("Get multi-attributes for nodes",
             lambda: groggy_graph.get_nodes_attributes(list(groggy_graph.get_node_ids())[:500]),  # Fixed: removed the second parameter
             lambda: {n: {attr: nx_graph.nodes[n].get(attr, 0) for attr in ['salary', 'age', 'experience', 'rating']} 
                     for n in list(nx_graph.nodes())[:500]} if nx_graph else {}),
            
            # Large batch operations
            ("Bulk salary analysis (5000 nodes)",
             lambda: len([v for v in groggy_graph.get_nodes_attribute(list(groggy_graph.get_node_ids())[:5000], 'salary').values() if v > 100000]),
             lambda: len([attrs.get('salary', 0) for n, attrs in list(nx_graph.nodes(data=True))[:5000] if attrs.get('salary', 0) > 100000]) if nx_graph else 0),
             
            # Complex batch filtering with attributes
            ("Engineers with high salaries (batch)",
             lambda: len([node_id for node_id, attrs in groggy_graph.get_nodes_attributes(
                 groggy_graph.filter_nodes({'role': 'engineer'})).items()  # Fixed: removed the second parameter
                 if attrs.get('salary', 0) > 120000]),
             lambda: len([n for n, attrs in nx_graph.nodes(data=True) 
                         if attrs.get('role') == 'engineer' and attrs.get('salary', 0) > 120000]) if nx_graph else 0),
        ]
        
        for op_name, groggy_func, nx_func in batch_ops:
            self.run_benchmark(op_name, groggy_func, nx_func)
    
    def benchmark_traversal_operations(self):
        """Benchmark graph traversal operations"""
        print("\n🗺️ TRAVERSAL & ALGORITHM BENCHMARKS")
        print("=" * 50)
        
        # Create a structured graph for traversals
        nodes_data, edges_data = self.create_test_data(2500, 7500, structured=True)
        
        print("Creating test graphs...")
        groggy_graph = gr.Graph(backend='rust')
        groggy_graph.add_nodes(nodes_data)
        groggy_graph.add_edges(edges_data)
        
        nx_graph = None
        if NETWORKX_AVAILABLE:
            nx_graph = nx.DiGraph()
            # Add nodes with attributes (batch)
            nodes_for_nx = [(node['id'], {k: v for k, v in node.items() if k != 'id'}) for node in nodes_data]
            nx_graph.add_nodes_from(nodes_for_nx)
            # Add edges with attributes (batch)
            edges_for_nx = [(edge['source'], edge['target'], {k: v for k, v in edge.items() if k not in ['source', 'target']}) for edge in edges_data]
            nx_graph.add_edges_from(edges_for_nx)
        
        # Test traversal operations
        start_node = 'n0'
        
        traversal_ops = [
            ("Get neighbors", 
             lambda: groggy_graph.get_neighbors(start_node),
             lambda: list(nx_graph.neighbors(start_node)) if NETWORKX_AVAILABLE and nx_graph else []),
            ("Get outgoing neighbors",
             lambda: groggy_graph.get_outgoing_neighbors(start_node),
             lambda: list(nx_graph.successors(start_node)) if NETWORKX_AVAILABLE and nx_graph else []),
            ("Get incoming neighbors", 
             lambda: groggy_graph.get_incoming_neighbors(start_node),
             lambda: list(nx_graph.predecessors(start_node)) if NETWORKX_AVAILABLE and nx_graph else []),
            ("Multi-hop neighbor analysis",
             lambda: len(set().union(*[groggy_graph.get_neighbors(n) for n in groggy_graph.get_neighbors(start_node)])),
             lambda: len(set().union(*[list(nx_graph.neighbors(n)) for n in nx_graph.neighbors(start_node)])) if NETWORKX_AVAILABLE and nx_graph else 0),
        ]
        
        for op_name, groggy_func, nx_func in traversal_ops:
            self.run_benchmark(op_name, groggy_func, nx_func)
    
    def benchmark_memory_scalability(self):
        """Test memory usage and scalability"""
        print("\n💾 MEMORY & SCALABILITY BENCHMARKS")
        print("=" * 50)
        
        sizes = [(5000, 2500), (10000, 5000), (25000, 12500)]
        
        for n_nodes, n_edges in sizes:
            print(f"\n📈 Memory test: {n_nodes:,} nodes, {n_edges:,} edges")
            
            nodes_data, edges_data = self.create_test_data(n_nodes, n_edges)
            
            # Test memory usage for graph creation
            def create_and_measure():
                graph = gr.Graph(backend='rust')
                graph.add_nodes(nodes_data)
                graph.add_edges(edges_data)
                
                # Perform some operations to stress test
                graph.filter_nodes({'role': 'engineer'})
                graph.filter_nodes({'salary': ('>', 80000)})
                graph.filter_edges({'relationship': 'collaborates'})
                
                return graph
            
            self.run_benchmark(f"Memory test ({n_nodes:,} nodes)", create_and_measure)
    
    def print_results(self):
        """Print comprehensive results"""
        print("\n" + "=" * 80)
        print("🏆 COMPREHENSIVE BENCHMARK RESULTS")
        print("=" * 80)
        
        # Group results by operation
        results_by_op = defaultdict(list)
        for result in self.results:
            results_by_op[result.operation].append(result)
        
        # Print detailed results
        for operation, results in results_by_op.items():
            print(f"\n📊 {operation}")
            print("-" * 60)
            
            groggy_result = None
            nx_result = None
            
            for result in results:
                if result.library == "Groggy":
                    groggy_result = result
                elif result.library == "NetworkX":
                    nx_result = result
            
            if groggy_result:
                status = "✅" if groggy_result.success else "❌"
                print(f"  Groggy:    {status} {groggy_result.time_ms:8.2f}ms  "
                      f"{groggy_result.memory_mb:6.1f}MB  {groggy_result.result_size:,} results")
                if not groggy_result.success:
                    print(f"    Error: {groggy_result.error}")
            
            if nx_result:
                status = "✅" if nx_result.success else "❌"
                print(f"  NetworkX:  {status} {nx_result.time_ms:8.2f}ms  "
                      f"{nx_result.memory_mb:6.1f}MB  {nx_result.result_size:,} results")
                if not nx_result.success:
                    print(f"    Error: {nx_result.error}")
            
            # Show performance comparison
            if groggy_result and nx_result and groggy_result.success and nx_result.success:
                if nx_result.time_ms > 0:
                    speedup = nx_result.time_ms / groggy_result.time_ms
                    if speedup > 1.1:
                        print(f"  🚀 Groggy is {speedup:.1f}x faster")
                    elif speedup < 0.9:
                        print(f"  📊 NetworkX is {1/speedup:.1f}x faster")
                    else:
                        print(f"  ⚖️ Similar performance ({speedup:.1f}x)")
        
        # Summary statistics
        print("\n" + "=" * 80)
        print("📈 SUMMARY STATISTICS")
        print("=" * 80)
        
        groggy_times = [r.time_ms for r in self.results if r.library == "Groggy" and r.success]
        nx_times = [r.time_ms for r in self.results if r.library == "NetworkX" and r.success]
        
        if groggy_times:
            print(f"Groggy - Operations: {len(groggy_times)}, "
                  f"Avg time: {statistics.mean(groggy_times):.2f}ms, "
                  f"Median: {statistics.median(groggy_times):.2f}ms")
        
        if nx_times:
            print(f"NetworkX - Operations: {len(nx_times)}, "
                  f"Avg time: {statistics.mean(nx_times):.2f}ms, "
                  f"Median: {statistics.median(nx_times):.2f}ms")
        
        # Count wins
        wins = {"Groggy": 0, "NetworkX": 0, "Tie": 0}
        for operation, results in results_by_op.items():
            groggy_result = next((r for r in results if r.library == "Groggy"), None)
            nx_result = next((r for r in results if r.library == "NetworkX"), None)
            
            if groggy_result and nx_result and groggy_result.success and nx_result.success:
                if groggy_result.time_ms < nx_result.time_ms * 0.9:
                    wins["Groggy"] += 1
                elif nx_result.time_ms < groggy_result.time_ms * 0.9:
                    wins["NetworkX"] += 1
                else:
                    wins["Tie"] += 1
        
        print(f"\n🏆 Performance Wins: Groggy: {wins['Groggy']}, "
              f"NetworkX: {wins['NetworkX']}, Ties: {wins['Tie']}")


def main():
    """Run the comprehensive benchmark suite"""
    print("🚀 COMPREHENSIVE GROGGY VS NETWORKX BENCHMARK SUITE")
    print("=" * 80)
    print(f"Python version: {sys.version}")
    print(f"NetworkX available: {'✅' if NETWORKX_AVAILABLE else '❌'}")
    print(f"NumPy available: {'✅' if NUMPY_AVAILABLE else '❌'}")
    print(f"Process ID: {os.getpid()}")
    print(f"Available memory: {psutil.virtual_memory().available / 1024 / 1024 / 1024:.1f} GB")
    
    benchmark = ComprehensiveBenchmark()
    
    # Run all benchmark suites
    try:
        benchmark.benchmark_graph_creation()
        benchmark.benchmark_basic_operations()
        benchmark.benchmark_filtering_operations()
        benchmark.benchmark_numerical_analysis()
        benchmark.benchmark_batch_attribute_operations()
        benchmark.benchmark_traversal_operations()
        benchmark.benchmark_memory_scalability()
        
        # Print comprehensive results
        benchmark.print_results()
        
    except KeyboardInterrupt:
        print("\n⚠️ Benchmark interrupted by user")
        benchmark.print_results()
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        traceback.print_exc()
        benchmark.print_results()


if __name__ == "__main__":
    main()
