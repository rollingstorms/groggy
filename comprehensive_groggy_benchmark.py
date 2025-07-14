#!/usr/bin/env python3
"""
🚀 COMPREHENSIVE GROGGY FUNCTIONALITY BENCHMARK
===============================================

Extensive testing of Groggy's full feature set:
- Graph creation (nodes, edges, batch operations)
- Attribute management (set, get, update, batch)
- Advanced filtering (single, multi-criteria, complex queries)
- Graph traversal and navigation
- Subgraph operations
- Batch operations and context managers
- Memory efficiency and scalability
- Complex real-world scenarios

This benchmark focuses on Groggy's unique capabilities and performance.
"""

import time
import random
import statistics
import gc
import psutil
import os
import sys
import traceback
from typing import Dict, List, Any
from dataclasses import dataclass
from collections import defaultdict

# Add groggy to path
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python')

import groggy as gr

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("❌ NumPy not available - some numerical tests will be skipped")


@dataclass
class OperationResult:
    """Store operation results with timing info"""
    operation: str
    time_ms: float
    memory_mb: float
    result_size: int
    success: bool = True
    error: str = ""
    details: str = ""


class GroggyBenchmark:
    """Comprehensive Groggy functionality benchmark"""
    
    def __init__(self):
        self.results = []
        self.process = psutil.Process()
        
    def measure_operation(self, func, description, expected_size=None):
        """Time and measure an operation"""
        print(f"  🔄 {description}...", end=" ")
        
        # Memory before
        gc.collect()
        mem_before = self.process.memory_info().rss / 1024 / 1024
        
        start_time = time.time()
        try:
            result = func()
            elapsed = (time.time() - start_time) * 1000
            
            # Memory after
            gc.collect()
            mem_after = self.process.memory_info().rss / 1024 / 1024
            memory_used = mem_after - mem_before
            
            result_size = len(result) if hasattr(result, '__len__') else (1 if result is not None else 0)
            
            # Success message
            status = "✅"
            details = f"{elapsed:.2f}ms, {memory_used:+.1f}MB, {result_size:,} results"
            if expected_size and result_size != expected_size:
                details += f" (expected {expected_size:,})"
            
            print(f"{status} {details}")
            
            self.results.append(OperationResult(
                operation=description,
                time_ms=elapsed,
                memory_mb=memory_used,
                result_size=result_size,
                success=True,
                details=details
            ))
            
            return result, elapsed, memory_used
            
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            mem_after = self.process.memory_info().rss / 1024 / 1024
            memory_used = mem_after - mem_before
            
            print(f"❌ {elapsed:.2f}ms - Error: {str(e)}")
            
            self.results.append(OperationResult(
                operation=description,
                time_ms=elapsed,
                memory_mb=memory_used,
                result_size=0,
                success=False,
                error=str(e)
            ))
            
            return None, elapsed, memory_used
    
    def create_rich_test_data(self, n_nodes: int, n_edges: int):
        """Create rich test data with diverse attributes"""
        print(f"Creating rich test data: {n_nodes:,} nodes, {n_edges:,} edges")
        
        # Rich node data
        departments = ['Engineering', 'Sales', 'Marketing', 'Finance', 'HR', 'Operations', 'Research', 'Support']
        roles = ['engineer', 'senior_engineer', 'lead_engineer', 'manager', 'director', 'analyst', 'specialist', 'coordinator']
        locations = ['San Francisco', 'New York', 'Austin', 'Seattle', 'Boston', 'Denver', 'Remote', 'London']
        skills = ['Python', 'JavaScript', 'Java', 'C++', 'Go', 'Rust', 'SQL', 'Machine Learning', 'DevOps', 'Design']
        projects = ['ProjectAlpha', 'ProjectBeta', 'ProjectGamma', 'ProjectDelta', 'ProjectOmega']
        
        nodes_data = []
        for i in range(n_nodes):
            node_data = {
                'id': f'emp_{i:06d}',
                'name': f'Employee {i}',
                'salary': random.randint(45000, 250000),
                'age': random.randint(22, 65),
                'experience': random.randint(0, 40),
                'rating': round(random.uniform(1.0, 5.0), 2),
                'role': random.choice(roles),
                'department': random.choice(departments),
                'location': random.choice(locations),
                'skills': random.choice(skills),
                'active': random.choice([True, False]),
                'hire_date': f"20{random.randint(15, 24)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
                'project': random.choice(projects),
                'performance_score': random.randint(60, 100),
                'team_size': random.randint(1, 15),
                'remote_work': random.choice([True, False]),
                'certification_count': random.randint(0, 10)
            }
            nodes_data.append(node_data)
        
        # Rich edge data
        relationships = ['reports_to', 'collaborates', 'mentors', 'manages', 'works_with', 'coordinates_with']
        frequencies = ['daily', 'weekly', 'monthly', 'quarterly', 'rarely']
        
        edges_data = []
        for _ in range(n_edges):
            source = random.randint(0, n_nodes - 1)
            target = random.randint(0, n_nodes - 1)
            if source != target:
                edge_data = {
                    'source': f'emp_{source:06d}',
                    'target': f'emp_{target:06d}',
                    'relationship': random.choice(relationships),
                    'weight': round(random.uniform(0.1, 1.0), 3),
                    'frequency': random.choice(frequencies),
                    'strength': round(random.uniform(0.1, 1.0), 2),
                    'duration_months': random.randint(1, 60),
                    'project_shared': random.choice([True, False]),
                    'communication_score': random.randint(1, 10),
                    'created_date': f"20{random.randint(20, 24)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"
                }
                edges_data.append(edge_data)
        
        return nodes_data, edges_data
    
    def benchmark_graph_creation(self):
        """Test graph creation with various methods"""
        print("\n🏗️ GRAPH CREATION & INITIALIZATION")
        print("=" * 60)
        
        # Test different backends
        print("\n🔧 Backend Initialization:")
        self.measure_operation(lambda: gr.Graph(backend='rust'), "Create empty Rust graph")
        self.measure_operation(lambda: gr.Graph(backend='python'), "Create empty Python graph")
        
        # Test graph creation sizes
        sizes = [(1000, 500), (5000, 2500), (10000, 5000), (25000, 12500)]
        
        for n_nodes, n_edges in sizes:
            print(f"\n📊 Graph Creation - {n_nodes:,} nodes, {n_edges:,} edges:")
            
            nodes_data, edges_data = self.create_rich_test_data(n_nodes, n_edges)
            
            # Individual node/edge addition
            def create_individual():
                graph = gr.Graph(backend='rust')
                for node in nodes_data[:100]:  # Only test with 100 for speed
                    graph.add_node(node['id'], **{k: v for k, v in node.items() if k != 'id'})
                for edge in edges_data[:50]:
                    attrs = {k: v for k, v in edge.items() if k not in ['source', 'target']}
                    graph.add_edge(edge['source'], edge['target'], **attrs)
                return graph
            
            self.measure_operation(create_individual, f"Individual adds (100 nodes)")
            
            # Batch operations
            def create_batch():
                graph = gr.Graph(backend='rust')
                graph.add_nodes(nodes_data)
                graph.add_edges(edges_data)
                return graph
            
            graph, time_ms, memory_mb = self.measure_operation(create_batch, f"Batch creation ({n_nodes:,} nodes)")
            
            if graph:
                # Verify graph
                actual_nodes = graph.node_count()
                actual_edges = graph.edge_count()
                print(f"    📋 Verification: {actual_nodes:,} nodes, {actual_edges:,} edges")
                
                if n_nodes <= 10000:  # Store for later tests
                    self.test_graph = graph
                    self.test_nodes = nodes_data
                    self.test_edges = edges_data
    
    def benchmark_attribute_operations(self):
        """Test attribute management operations"""
        print("\n🏷️ ATTRIBUTE OPERATIONS")
        print("=" * 60)
        
        if not hasattr(self, 'test_graph'):
            print("⚠️ No test graph available, creating one...")
            nodes_data, edges_data = self.create_rich_test_data(5000, 2500)
            self.test_graph = gr.Graph(backend='rust')
            self.test_graph.add_nodes(nodes_data)
            self.test_graph.add_edges(edges_data)
        
        graph = self.test_graph
        
        print("\n🔹 Single Attribute Operations:")
        
        # Test getting node attributes
        test_node = 'emp_000100'
        self.measure_operation(lambda: graph.get_node(test_node), f"Get node {test_node}")
        
        # Test setting single attribute
        self.measure_operation(
            lambda: graph.set_node_attribute(test_node, 'bonus', 5000),
            f"Set single node attribute"
        )
        
        # Test getting edge attributes
        edges = graph.get_edge_ids()[:10] if hasattr(graph, 'get_edge_ids') else []
        if edges:
            test_edge = edges[0].split('->')
            self.measure_operation(
                lambda: graph.get_edge(test_edge[0], test_edge[1]),
                f"Get edge attributes"
            )
        
        print("\n🔹 Batch Attribute Operations:")
        
        # Test batch node attribute updates
        node_ids = graph.get_node_ids()[:100]
        update_data = {node_id: {'updated': True, 'batch_bonus': 1000} for node_id in node_ids}
        
        self.measure_operation(
            lambda: graph.update_nodes(update_data),
            f"Batch update 100 nodes"
        )
        
        # Test bulk attribute setting
        if hasattr(graph, 'set_nodes_attributes_batch'):
            bulk_updates = {node_id: {'bulk_flag': True} for node_id in node_ids[:50]}
            self.measure_operation(
                lambda: graph.set_nodes_attributes_batch(bulk_updates),
                f"Bulk set attributes (50 nodes)"
            )
    
    def benchmark_filtering_operations(self):
        """Test comprehensive filtering capabilities"""
        print("\n🔍 FILTERING & QUERY OPERATIONS")
        print("=" * 60)
        
        graph = self.test_graph
        
        print("\n🔹 Node Filtering:")
        
        # Simple exact match filters
        simple_filters = [
            ({'role': 'engineer'}, "Exact role match"),
            ({'department': 'Engineering'}, "Department match"),
            ({'active': True}, "Boolean filter"),
            ({'location': 'San Francisco'}, "Location filter"),
        ]
        
        for filter_dict, description in simple_filters:
            self.measure_operation(
                lambda f=filter_dict: graph.filter_nodes(f),
                f"Node filter: {description}"
            )
        
        # Numeric comparison filters
        numeric_filters = [
            ({'salary': ('>', 100000)}, "Salary > 100K"),
            ({'age': ('>=', 30)}, "Age >= 30"),
            ({'experience': ('>', 5)}, "Experience > 5 years"),
            ({'rating': ('>', 4.0)}, "High rating"),
            ({'performance_score': ('>=', 85)}, "High performers"),
        ]
        
        for filter_dict, description in numeric_filters:
            self.measure_operation(
                lambda f=filter_dict: graph.filter_nodes(f),
                f"Node filter: {description}"
            )
        
        # Complex multi-criteria filters
        complex_filters = [
            ({'role': 'engineer', 'salary': ('>', 120000)}, "Senior engineers"),
            ({'department': 'Engineering', 'experience': ('>', 5), 'rating': ('>', 4.0)}, "Top engineers"),
            ({'active': True, 'remote_work': True, 'salary': ('>', 80000)}, "Remote high earners"),
            ({'age': ('>', 35), 'team_size': ('>', 5), 'role': 'manager'}, "Senior managers"),
        ]
        
        for filter_dict, description in complex_filters:
            self.measure_operation(
                lambda f=filter_dict: graph.filter_nodes(f),
                f"Complex filter: {description}"
            )
        
        # String operations (if supported)
        if hasattr(graph, 'filter_nodes'):
            try:
                self.measure_operation(
                    lambda: graph.filter_nodes({'skills': ('contains', 'Python')}),
                    "String contains filter"
                )
            except:
                print("    ⚠️ String contains not supported")
        
        print("\n🔹 Edge Filtering:")
        
        # Edge filters
        edge_filters = [
            ({'relationship': 'reports_to'}, "Reports relationship"),
            ({'frequency': 'daily'}, "Daily interaction"),
            ({'weight': ('>', 0.7)}, "Strong connections"),
            ({'strength': ('>', 0.8)}, "High strength"),
            ({'project_shared': True}, "Shared projects"),
            ({'communication_score': ('>', 8)}, "High communication"),
        ]
        
        for filter_dict, description in edge_filters:
            self.measure_operation(
                lambda f=filter_dict: graph.filter_edges(f),
                f"Edge filter: {description}"
            )
        
        # Complex edge filters
        complex_edge_filters = [
            ({'relationship': 'collaborates', 'weight': ('>', 0.5)}, "Strong collaborations"),
            ({'frequency': 'daily', 'strength': ('>', 0.7)}, "Daily strong connections"),
        ]
        
        for filter_dict, description in complex_edge_filters:
            self.measure_operation(
                lambda f=filter_dict: graph.filter_edges(f),
                f"Complex edge filter: {description}"
            )
    
    def benchmark_traversal_operations(self):
        """Test graph traversal and navigation"""
        print("\n🗺️ TRAVERSAL & NAVIGATION")
        print("=" * 60)
        
        graph = self.test_graph
        
        # Test with multiple nodes
        test_nodes = graph.get_node_ids()[:10]
        
        print("\n🔹 Basic Traversal:")
        
        for test_node in test_nodes[:3]:
            # Basic neighbor operations
            self.measure_operation(
                lambda n=test_node: graph.get_neighbors(n),
                f"Get neighbors of {test_node}"
            )
            
            if hasattr(graph, 'get_outgoing_neighbors'):
                self.measure_operation(
                    lambda n=test_node: graph.get_outgoing_neighbors(n),
                    f"Get outgoing neighbors of {test_node}"
                )
            
            if hasattr(graph, 'get_incoming_neighbors'):
                self.measure_operation(
                    lambda n=test_node: graph.get_incoming_neighbors(n),
                    f"Get incoming neighbors of {test_node}"
                )
        
        print("\n🔹 Advanced Navigation:")
        
        # Multi-hop analysis
        def analyze_2hop(node):
            neighbors = graph.get_neighbors(node)
            second_hop = set()
            for neighbor in neighbors:
                second_hop.update(graph.get_neighbors(neighbor))
            return list(second_hop)
        
        self.measure_operation(
            lambda: analyze_2hop(test_nodes[0]),
            f"2-hop neighbor analysis"
        )
        
        # Degree analysis
        def degree_analysis():
            degrees = []
            for node in test_nodes:
                degree = len(graph.get_neighbors(node))
                degrees.append((node, degree))
            return sorted(degrees, key=lambda x: x[1], reverse=True)
        
        self.measure_operation(degree_analysis, "Degree analysis (10 nodes)")
        
        # Path existence (if supported)
        if hasattr(graph, 'has_path'):
            self.measure_operation(
                lambda: graph.has_path(test_nodes[0], test_nodes[1]),
                f"Path existence check"
            )
    
    def benchmark_subgraph_operations(self):
        """Test subgraph creation and manipulation"""
        print("\n🔗 SUBGRAPH OPERATIONS")
        print("=" * 60)
        
        graph = self.test_graph
        
        # Filter-based subgraphs
        print("\n🔹 Filter-based Subgraphs:")
        
        # Create subgraph of engineers
        engineer_nodes = graph.filter_nodes({'role': 'engineer'})
        if engineer_nodes:
            def create_engineer_subgraph():
                if hasattr(graph, 'create_subgraph'):
                    return graph.create_subgraph(engineer_nodes[:100])
                else:
                    # Manual subgraph creation
                    subgraph = gr.Graph(backend='rust')
                    for node_id in engineer_nodes[:100]:
                        node = graph.get_node(node_id)
                        if node:
                            subgraph.add_node(node_id, **node.attributes)
                    return subgraph
            
            self.measure_operation(create_engineer_subgraph, "Engineer subgraph (100 nodes)")
        
        # Department-based subgraph
        eng_dept_nodes = graph.filter_nodes({'department': 'Engineering'})
        if eng_dept_nodes:
            def create_dept_subgraph():
                if hasattr(graph, 'create_subgraph'):
                    return graph.create_subgraph(eng_dept_nodes[:200])
                else:
                    subgraph = gr.Graph(backend='rust')
                    for node_id in eng_dept_nodes[:200]:
                        node = graph.get_node(node_id)
                        if node:
                            subgraph.add_node(node_id, **node.attributes)
                    return subgraph
            
            self.measure_operation(create_dept_subgraph, "Engineering dept subgraph")
    
    def benchmark_batch_operations(self):
        """Test batch operations and context managers"""
        print("\n📦 BATCH OPERATIONS & CONTEXT MANAGERS")
        print("=" * 60)
        
        print("\n🔹 Batch Context Manager:")
        
        # Test batch context if available
        if hasattr(gr.Graph, 'batch_operations'):
            def test_batch_context():
                graph = gr.Graph(backend='rust')
                with graph.batch_operations() as batch:
                    for i in range(500):
                        batch.add_node(f'batch_{i}', value=i, category=f'cat_{i%10}')
                    for i in range(250):
                        batch.add_edge(f'batch_{i}', f'batch_{(i+1)%500}', weight=0.5)
                return graph
            
            self.measure_operation(test_batch_context, "Batch context manager (500 nodes)")
        
        # Large batch operations
        print("\n🔹 Large Batch Operations:")
        
        # Create large datasets
        large_nodes = []
        for i in range(5000):
            large_nodes.append({
                'id': f'large_{i}',
                'value': random.randint(1, 1000),
                'category': f'cat_{i%20}',
                'active': random.choice([True, False])
            })
        
        large_edges = []
        for i in range(2500):
            large_edges.append({
                'source': f'large_{random.randint(0, 4999)}',
                'target': f'large_{random.randint(0, 4999)}',
                'weight': random.uniform(0.1, 1.0)
            })
        
        def create_large_batch():
            graph = gr.Graph(backend='rust')
            graph.add_nodes(large_nodes)
            graph.add_edges(large_edges)
            return graph
        
        self.measure_operation(create_large_batch, "Large batch creation (5K nodes, 2.5K edges)")
    
    def benchmark_memory_efficiency(self):
        """Test memory efficiency and scalability"""
        print("\n💾 MEMORY EFFICIENCY & SCALABILITY")
        print("=" * 60)
        
        print("\n🔹 Memory Scaling:")
        
        sizes = [(10000, 5000), (25000, 12500), (50000, 25000)]
        
        for n_nodes, n_edges in sizes:
            print(f"\n📈 Testing {n_nodes:,} nodes, {n_edges:,} edges:")
            
            nodes_data, edges_data = self.create_rich_test_data(n_nodes, n_edges)
            
            def create_and_analyze():
                graph = gr.Graph(backend='rust')
                graph.add_nodes(nodes_data)
                graph.add_edges(edges_data)
                
                # Perform operations to test memory under load
                engineers = graph.filter_nodes({'role': 'engineer'})
                high_earners = graph.filter_nodes({'salary': ('>', 120000)})
                collaborations = graph.filter_edges({'relationship': 'collaborates'})
                
                return {
                    'nodes': graph.node_count(),
                    'edges': graph.edge_count(),
                    'engineers': len(engineers),
                    'high_earners': len(high_earners),
                    'collaborations': len(collaborations)
                }
            
            result, time_ms, memory_mb = self.measure_operation(
                create_and_analyze, 
                f"Full analysis ({n_nodes:,} nodes)"
            )
            
            if result:
                print(f"    📊 Results: {result}")
                
                # Calculate efficiency metrics
                if memory_mb > 0:
                    nodes_per_mb = n_nodes / memory_mb
                    print(f"    💡 Efficiency: {nodes_per_mb:.0f} nodes/MB")
    
    def benchmark_real_world_scenarios(self):
        """Test real-world usage scenarios"""
        print("\n🌍 REAL-WORLD SCENARIOS")
        print("=" * 60)
        
        graph = self.test_graph
        
        print("\n🔹 Organizational Analysis:")
        
        # Management hierarchy analysis
        def analyze_management():
            managers = graph.filter_nodes({'role': 'manager'})
            reports = graph.filter_edges({'relationship': 'reports_to'})
            return {'managers': len(managers), 'reporting_relationships': len(reports)}
        
        self.measure_operation(analyze_management, "Management hierarchy analysis")
        
        # Team collaboration analysis
        def analyze_collaboration():
            collaborations = graph.filter_edges({'relationship': 'collaborates', 'frequency': 'daily'})
            strong_collab = graph.filter_edges({'relationship': 'collaborates', 'strength': ('>', 0.8)})
            return {'daily_collaborations': len(collaborations), 'strong_collaborations': len(strong_collab)}
        
        self.measure_operation(analyze_collaboration, "Team collaboration analysis")
        
        # Talent pipeline analysis
        def analyze_talent():
            senior_engineers = graph.filter_nodes({
                'role': 'senior_engineer', 
                'experience': ('>', 5), 
                'rating': ('>', 4.0)
            })
            high_potential = graph.filter_nodes({
                'performance_score': ('>', 90),
                'age': ('<', 35)
            })
            return {'senior_engineers': len(senior_engineers), 'high_potential': len(high_potential)}
        
        self.measure_operation(analyze_talent, "Talent pipeline analysis")
        
        print("\n🔹 Network Analysis:")
        
        # Communication network analysis
        def analyze_communication():
            high_comm = graph.filter_edges({'communication_score': ('>', 8)})
            cross_dept = []
            
            # Find cross-department edges (simplified)
            for edge_id in graph.get_edge_ids()[:1000]:  # Sample for performance
                try:
                    parts = edge_id.split('->')
                    if len(parts) == 2:
                        source_node = graph.get_node(parts[0])
                        target_node = graph.get_node(parts[1])
                        if (source_node and target_node and 
                            source_node.attributes.get('department') != target_node.attributes.get('department')):
                            cross_dept.append(edge_id)
                except:
                    continue
            
            return {'high_communication': len(high_comm), 'cross_department': len(cross_dept)}
        
        self.measure_operation(analyze_communication, "Communication network analysis")
    
    def print_comprehensive_results(self):
        """Print detailed benchmark results"""
        print("\n" + "=" * 80)
        print("🏆 COMPREHENSIVE GROGGY BENCHMARK RESULTS")
        print("=" * 80)
        
        # Group results by category
        categories = defaultdict(list)
        for result in self.results:
            if 'Creation' in result.operation or 'create' in result.operation.lower():
                categories['Graph Creation'].append(result)
            elif 'Attribute' in result.operation or 'attribute' in result.operation.lower():
                categories['Attribute Operations'].append(result)
            elif 'filter' in result.operation.lower() or 'Filter' in result.operation:
                categories['Filtering Operations'].append(result)
            elif 'neighbor' in result.operation.lower() or 'Traversal' in result.operation:
                categories['Traversal Operations'].append(result)
            elif 'Batch' in result.operation or 'batch' in result.operation.lower():
                categories['Batch Operations'].append(result)
            elif 'Memory' in result.operation or 'memory' in result.operation.lower():
                categories['Memory & Scalability'].append(result)
            elif 'Subgraph' in result.operation or 'subgraph' in result.operation.lower():
                categories['Subgraph Operations'].append(result)
            else:
                categories['Advanced Operations'].append(result)
        
        # Print results by category
        for category, results in categories.items():
            print(f"\n📊 {category.upper()}")
            print("-" * 60)
            
            successful_results = [r for r in results if r.success]
            failed_results = [r for r in results if not r.success]
            
            for result in successful_results:
                print(f"  ✅ {result.operation:.<50} {result.details}")
            
            for result in failed_results:
                print(f"  ❌ {result.operation:.<50} {result.error}")
            
            if successful_results:
                times = [r.time_ms for r in successful_results]
                memories = [r.memory_mb for r in successful_results]
                
                print(f"  📈 {len(successful_results)} operations: "
                      f"avg {statistics.mean(times):.2f}ms, "
                      f"total memory {sum(memories):+.1f}MB")
        
        # Overall statistics
        print("\n" + "=" * 80)
        print("📈 OVERALL PERFORMANCE STATISTICS")
        print("=" * 80)
        
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]
        
        print(f"Total Operations: {len(self.results)}")
        print(f"Successful: {len(successful)} ✅")
        print(f"Failed: {len(failed)} ❌")
        print(f"Success Rate: {len(successful)/len(self.results)*100:.1f}%")
        
        if successful:
            times = [r.time_ms for r in successful]
            memories = [r.memory_mb for r in successful]
            
            print(f"\nTiming Statistics:")
            print(f"  Average: {statistics.mean(times):.2f}ms")
            print(f"  Median: {statistics.median(times):.2f}ms")
            print(f"  Min: {min(times):.2f}ms")
            print(f"  Max: {max(times):.2f}ms")
            
            print(f"\nMemory Statistics:")
            print(f"  Total Memory Used: {sum(memories):+.1f}MB")
            print(f"  Average per Operation: {statistics.mean(memories):+.1f}MB")
        
        # Performance categories
        fast_ops = [r for r in successful if r.time_ms < 1.0]
        medium_ops = [r for r in successful if 1.0 <= r.time_ms < 10.0]
        slow_ops = [r for r in successful if r.time_ms >= 10.0]
        
        print(f"\nPerformance Distribution:")
        print(f"  🚀 Fast (<1ms): {len(fast_ops)} operations")
        print(f"  ⚡ Medium (1-10ms): {len(medium_ops)} operations")
        print(f"  🐌 Slow (>10ms): {len(slow_ops)} operations")


def main():
    """Run the comprehensive Groggy benchmark"""
    print("🚀 COMPREHENSIVE GROGGY FUNCTIONALITY BENCHMARK")
    print("=" * 80)
    print(f"Python version: {sys.version}")
    print(f"NumPy available: {'✅' if NUMPY_AVAILABLE else '❌'}")
    print(f"Process ID: {os.getpid()}")
    print(f"Available memory: {psutil.virtual_memory().available / 1024 / 1024 / 1024:.1f} GB")
    
    benchmark = GroggyBenchmark()
    
    try:
        benchmark.benchmark_graph_creation()
        benchmark.benchmark_attribute_operations()
        benchmark.benchmark_filtering_operations()
        benchmark.benchmark_traversal_operations()
        benchmark.benchmark_subgraph_operations()
        benchmark.benchmark_batch_operations()
        benchmark.benchmark_memory_efficiency()
        benchmark.benchmark_real_world_scenarios()
        
        benchmark.print_comprehensive_results()
        
        print("\n✅ Comprehensive Groggy benchmark completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Benchmark interrupted by user")
        benchmark.print_comprehensive_results()
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        traceback.print_exc()
        benchmark.print_comprehensive_results()


if __name__ == "__main__":
    main()
