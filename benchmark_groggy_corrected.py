#!/usr/bin/env python3
"""
Corrected benchmark script for groggy with proper API usage.
Based on benchmark_graph_libraries.py but with updated groggy API calls.
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


class GroggyBenchmark:
    """Benchmark class for the groggy graph library with corrected API."""
    
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
    
    def set_node_attributes(self, count: int) -> float:
        """Set attributes on nodes and measure time."""
        if len(self.node_ids) == 0:
            raise ValueError("No nodes available for setting attributes")
        
        start_time = time.time()
        
        nodes = self.graph.nodes()
        
        # Set attributes on random nodes
        for i in range(count):
            node_id = random.choice(self.node_ids)
            node_proxy = nodes.get(node_id)
            if node_proxy:
                # Test different attribute types
                node_proxy.set_attr("name", f"Node_{i}")
                node_proxy.set_attr("value", random.randint(1, 1000))
                node_proxy.set_attr("metadata", {
                    "type": "test",
                    "score": random.random(),
                    "active": True
                })
        
        end_time = time.time()
        return end_time - start_time
    
    def set_edge_attributes(self, count: int) -> float:
        """Set attributes on edges and measure time."""
        if len(self.edge_ids) == 0:
            raise ValueError("No edges available for setting attributes")
        
        start_time = time.time()
        
        edges = self.graph.edges()
        
        # Set attributes on random edges
        for i in range(count):
            edge_id = random.choice(self.edge_ids)
            edge_proxy = edges.get(edge_id)
            if edge_proxy:
                # Test different attribute types
                edge_proxy.set_attr("weight", random.random())
                edge_proxy.set_attr("type", random.choice(["friend", "colleague", "family"]))
                edge_proxy.set_attr("properties", {
                    "strength": random.random(),
                    "directed": random.choice([True, False])
                })
        
        end_time = time.time()
        return end_time - start_time
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        nodes = self.graph.nodes()
        edges = self.graph.edges()
        
        return {
            "node_count": nodes.size(),
            "edge_count": edges.size(),
            "memory_usage": "N/A",  # Would need memory profiling
        }


def run_benchmark():
    """Run the groggy benchmark with various sizes."""
    print("🚀 Running Groggy Graph Library Benchmark")
    print("=" * 50)
    
    # Test different sizes
    sizes = [100, 1000, 5000]
    
    for size in sizes:
        print(f"\n📊 Testing with {size} nodes...")
        
        benchmark = GroggyBenchmark()
        
        try:
            # Create nodes
            node_time = benchmark.create_nodes(size)
            print(f"  ✅ Created {size} nodes in {node_time:.4f} seconds")
            
            # Create edges (roughly size/2 edges)
            edge_count = size // 2
            edge_time = benchmark.create_edges(edge_count)
            print(f"  ✅ Created {edge_count} edges in {edge_time:.4f} seconds")
            
            # Set node attributes (on 50% of nodes)
            attr_count = size // 2
            node_attr_time = benchmark.set_node_attributes(attr_count)
            print(f"  ✅ Set attributes on {attr_count} nodes in {node_attr_time:.4f} seconds")
            
            # Set edge attributes (on 50% of edges)
            edge_attr_count = edge_count // 2
            edge_attr_time = benchmark.set_edge_attributes(edge_attr_count)
            print(f"  ✅ Set attributes on {edge_attr_count} edges in {edge_attr_time:.4f} seconds")
            
            # Get statistics
            stats = benchmark.get_statistics()
            print(f"  📈 Final stats: {stats}")
            
            # Calculate rates
            total_time = node_time + edge_time + node_attr_time + edge_attr_time
            print(f"  ⏱️  Total time: {total_time:.4f} seconds")
            print(f"  🏃 Nodes/second: {size / node_time:.0f}")
            print(f"  🏃 Edges/second: {edge_count / edge_time:.0f}")
            
        except Exception as e:
            print(f"  ❌ Error during benchmark: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    run_benchmark()
