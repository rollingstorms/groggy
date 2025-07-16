#!/usr/bin/env python3
"""
Working groggy benchmark focused on operations that currently work.
This demonstrates the clean API and excellent performance.
"""

import sys
import os
import time
import random
from typing import List, Dict, Any

# Remove any existing groggy from the module cache
modules_to_remove = [mod for mod in sys.modules.keys() if mod.startswith('groggy')]
for mod in modules_to_remove:
    del sys.modules[mod]

# Add local development version
local_groggy_path = '/Users/michaelroth/Documents/Code/groggy/python'
sys.path.insert(0, local_groggy_path)

import groggy as gr


class GroggyBenchmarkBasic:
    """Basic benchmark class for groggy focusing on working operations."""
    
    def __init__(self):
        self.graph = gr.Graph()
        self.node_ids = []
        self.edge_ids = []
    
    def create_nodes(self, count: int) -> float:
        """Create nodes and measure time."""
        start_time = time.time()
        
        # Create NodeId objects
        self.node_ids = [gr.NodeId(f"node_{i}") for i in range(count)]
        
        # Add nodes to the graph
        nodes = self.graph.nodes()
        nodes.add(self.node_ids)
        
        end_time = time.time()
        return end_time - start_time
    
    def create_edges(self, count: int) -> float:
        """Create edges between random nodes and measure time."""
        if len(self.node_ids) < 2:
            raise ValueError("Need at least 2 nodes to create edges")
        
        start_time = time.time()
        
        # Create EdgeId objects with random node connections
        self.edge_ids = []
        edges = self.graph.edges()
        
        for i in range(count):
            source = random.choice(self.node_ids)
            target = random.choice(self.node_ids)
            edge_id = gr.EdgeId(source, target)
            self.edge_ids.append(edge_id)
        
        # Add edges to the graph
        edges.add(self.edge_ids)
        
        end_time = time.time()
        return end_time - start_time
    
    def batch_create_nodes(self, count: int) -> float:
        """Create and add nodes in a single batch operation."""
        start_time = time.time()
        
        # Create all NodeId objects
        batch_nodes = [gr.NodeId(f"batch_node_{i}") for i in range(count)]
        
        # Add all nodes in one operation
        nodes = self.graph.nodes()
        nodes.add(batch_nodes)
        
        self.node_ids.extend(batch_nodes)
        
        end_time = time.time()
        return end_time - start_time
    
    def batch_create_edges(self, count: int) -> float:
        """Create and add edges in a single batch operation."""
        if len(self.node_ids) < 2:
            raise ValueError("Need at least 2 nodes to create edges")
        
        start_time = time.time()
        
        # Create all EdgeId objects
        batch_edges = []
        for i in range(count):
            source = random.choice(self.node_ids)
            target = random.choice(self.node_ids)
            edge_id = gr.EdgeId(source, target)
            batch_edges.append(edge_id)
        
        # Add all edges in one operation
        edges = self.graph.edges()
        edges.add(batch_edges)
        
        self.edge_ids.extend(batch_edges)
        
        end_time = time.time()
        return end_time - start_time
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        nodes = self.graph.nodes()
        edges = self.graph.edges()
        
        return {
            "node_count": nodes.size(),
            "edge_count": edges.size(),
            "node_ids_created": len(self.node_ids),
            "edge_ids_created": len(self.edge_ids),
        }


def run_performance_test():
    """Run performance tests focusing on working operations."""
    print("🚀 Groggy Graph Library Performance Test")
    print("🎯 Testing core operations: node/edge creation and batch operations")
    print("=" * 60)
    
    # Test different sizes
    sizes = [1000, 10000, 50000, 100000]
    
    for size in sizes:
        print(f"\n📊 Performance test with {size} nodes...")
        
        benchmark = GroggyBenchmarkBasic()
        
        try:
            # Test individual node creation
            node_time = benchmark.create_nodes(size)
            nodes_per_sec = size / node_time if node_time > 0 else float('inf')
            print(f"  🏗️  Individual nodes: {size} nodes in {node_time:.4f}s ({nodes_per_sec:.0f} nodes/sec)")
            
            # Test individual edge creation
            edge_count = size // 2
            edge_time = benchmark.create_edges(edge_count)
            edges_per_sec = edge_count / edge_time if edge_time > 0 else float('inf')
            print(f"  🔗 Individual edges: {edge_count} edges in {edge_time:.4f}s ({edges_per_sec:.0f} edges/sec)")
            
            # Test batch node creation
            batch_size = size // 4
            batch_node_time = benchmark.batch_create_nodes(batch_size)
            batch_nodes_per_sec = batch_size / batch_node_time if batch_node_time > 0 else float('inf')
            print(f"  📦 Batch nodes: {batch_size} nodes in {batch_node_time:.4f}s ({batch_nodes_per_sec:.0f} nodes/sec)")
            
            # Test batch edge creation
            batch_edge_count = batch_size // 2
            batch_edge_time = benchmark.batch_create_edges(batch_edge_count)
            batch_edges_per_sec = batch_edge_count / batch_edge_time if batch_edge_time > 0 else float('inf')
            print(f"  🔗📦 Batch edges: {batch_edge_count} edges in {batch_edge_time:.4f}s ({batch_edges_per_sec:.0f} edges/sec)")
            
            # Get final statistics
            stats = benchmark.get_statistics()
            print(f"  📈 Final stats: {stats}")
            
            # Calculate totals
            total_time = node_time + edge_time + batch_node_time + batch_edge_time
            total_operations = size + edge_count + batch_size + batch_edge_count
            ops_per_sec = total_operations / total_time if total_time > 0 else float('inf')
            print(f"  ⚡ Overall: {total_operations} operations in {total_time:.4f}s ({ops_per_sec:.0f} ops/sec)")
            
            # Memory efficiency estimation
            memory_per_node = 64  # Rough estimate in bytes
            estimated_memory = (stats['node_count'] + stats['edge_count']) * memory_per_node
            print(f"  💾 Estimated memory: {estimated_memory / 1024 / 1024:.2f} MB")
            
        except Exception as e:
            print(f"  ❌ Error during test: {e}")
            import traceback
            traceback.print_exc()


def run_api_showcase():
    """Showcase the clean API design."""
    print("\n" + "=" * 60)
    print("🎨 API Showcase: Clean and Pythonic Design")
    print("=" * 60)
    
    # Create graph
    print("1️⃣  Creating graph:")
    print("   graph = gr.Graph()")
    graph = gr.Graph()
    
    # Create nodes
    print("\n2️⃣  Creating nodes:")
    print("   node1 = gr.NodeId('user_123')")
    print("   node2 = gr.NodeId('user_456')")
    node1 = gr.NodeId('user_123')
    node2 = gr.NodeId('user_456')
    node3 = gr.NodeId('user_789')
    
    print("   graph.nodes().add([node1, node2, node3])")
    graph.nodes().add([node1, node2, node3])
    
    # Create edges
    print("\n3️⃣  Creating edges:")
    print("   edge1 = gr.EdgeId(node1, node2)  # user_123 -> user_456")
    print("   edge2 = gr.EdgeId(node2, node3)  # user_456 -> user_789")
    edge1 = gr.EdgeId(node1, node2)
    edge2 = gr.EdgeId(node2, node3)
    
    print("   graph.edges().add([edge1, edge2])")
    graph.edges().add([edge1, edge2])
    
    # Show statistics
    print("\n4️⃣  Graph statistics:")
    nodes = graph.nodes()
    edges = graph.edges()
    print(f"   Nodes: {nodes.size()}")
    print(f"   Edges: {edges.size()}")
    
    print("\n✨ Key API Features:")
    print("   • Clean imports: import groggy as gr")
    print("   • Simple constructor: gr.Graph()")
    print("   • Strongly typed IDs: gr.NodeId(), gr.EdgeId()")
    print("   • Collection-based operations: graph.nodes().add()")
    print("   • Automatic JSON serialization for attributes")
    print("   • Batch operations for performance")


if __name__ == "__main__":
    run_api_showcase()
    run_performance_test()
