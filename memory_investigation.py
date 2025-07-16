#!/usr/bin/env python3
"""
Memory Investigation Script for Groggy

Investigates where memory is being used and why there's a discrepancy
between Rust-reported memory usage and Python-measured memory usage.
"""

import sys
import os
import gc
import psutil
import time

# Add local groggy to path
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python')
import groggy as gr

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def format_memory(mb):
    """Format memory in MB"""
    return f"{mb:.2f} MB"

def detailed_memory_analysis():
    """Perform detailed memory analysis"""
    print("🔍 Memory Investigation for Groggy")
    print("=" * 60)
    
    # Baseline memory
    gc.collect()  # Force garbage collection
    baseline_memory = get_memory_usage()
    print(f"Baseline memory: {format_memory(baseline_memory)}")
    
    # Create empty graph
    print("\n📊 Creating empty graph...")
    g = gr.Graph()
    empty_graph_memory = get_memory_usage()
    print(f"Empty graph memory: {format_memory(empty_graph_memory)}")
    print(f"Empty graph overhead: {format_memory(empty_graph_memory - baseline_memory)}")
    print(f"Empty graph info(): {g.info()}")
    
    # Test different sizes to see memory scaling
    test_sizes = [1000, 10000, 100000]
    
    for num_nodes in test_sizes:
        print(f"\n📈 Testing {num_nodes:,} nodes:")
        
        # Create fresh graph
        g = gr.Graph()
        before_nodes = get_memory_usage()
        
        # Add nodes
        node_ids = [gr.NodeId(f'n{i}') for i in range(num_nodes)]
        g.nodes.add(node_ids)
        after_nodes = get_memory_usage()
        
        # Add edges (half the nodes)
        num_edges = num_nodes // 2
        edge_ids = []
        for i in range(num_edges):
            src = f'n{i}'
            tgt = f'n{(i + 1) % num_nodes}'
            edge_ids.append(gr.EdgeId(gr.NodeId(src), gr.NodeId(tgt)))
        
        g.edges.add(edge_ids)
        after_edges = get_memory_usage()
        
        # Add some attributes (skip for now due to backend issue)
        print("   Skipping attributes due to backend issue...")
        # for i in range(min(1000, num_nodes)):  # Add attributes to first 1000 nodes
        #     try:
        #         node_proxy = g.nodes().get(gr.NodeId(f'n{i}'))
        #         if node_proxy:
        #             node_proxy.set_attr('role', 'engineer')
        #             node_proxy.set_attr('salary', 100000)
        #             node_proxy.set_attr('active', True)
        #     except Exception as e:
        #         print(f"   Warning: Failed to set attributes for n{i}: {e}")
        #         break
        
        after_attrs = after_edges  # No attributes added due to backend issue
        
        # Get memory breakdown
        info = g.info()
        rust_graph_store = float(info['attributes'].get('memory_graph_store_mb', 0))
        rust_content_pool = float(info['attributes'].get('memory_content_pool_mb', 0))
        rust_columnar = float(info['attributes'].get('memory_columnar_store_mb', 0))
        rust_total = rust_graph_store + rust_content_pool + rust_columnar
        
        # Calculate Python overhead
        python_total = after_attrs - baseline_memory
        python_overhead = python_total - rust_total
        
        # Results
        print(f"   Nodes added: {g.info()['node_count']:,}")
        print(f"   Edges added: {g.info()['edge_count']:,}")
        print(f"   ")
        print(f"   Memory after nodes: {format_memory(after_nodes)} (+{format_memory(after_nodes - before_nodes)})")
        print(f"   Memory after edges: {format_memory(after_edges)} (+{format_memory(after_edges - after_nodes)})")
        print(f"   Memory after attrs: {format_memory(after_attrs)} (+{format_memory(after_attrs - after_edges)})")
        print(f"   ")
        print(f"   📊 Rust Memory Breakdown:")
        print(f"     Graph Store:    {rust_graph_store:>8.2f} MB")
        print(f"     Content Pool:   {rust_content_pool:>8.2f} MB")
        print(f"     Columnar Store: {rust_columnar:>8.2f} MB")
        print(f"     Rust Total:     {rust_total:>8.2f} MB")
        print(f"   ")
        print(f"   🐍 Python Memory Analysis:")
        print(f"     Python Total:   {python_total:>8.2f} MB")
        print(f"     Python Overhead:{python_overhead:>8.2f} MB ({python_overhead/python_total*100:.1f}%)")
        
        if rust_columnar == 0:
            print(f"   ⚠️  WARNING: Columnar store shows 0.00 MB - attributes may not be stored!")
        
        # Test if attributes are actually stored
        try:
            test_node = g.nodes().get(gr.NodeId('n0'))
            if test_node:
                role = test_node.get_attr('role')
                print(f"   🔍 Attribute test - role: {role}")
            else:
                print(f"   ❌ Could not retrieve node n0")
        except Exception as e:
            print(f"   ❌ Attribute retrieval failed: {e}")
        
        print("   " + "-" * 50)

def investigate_columnar_store():
    """Specifically investigate the columnar store"""
    print("\n🔬 Columnar Store Investigation")
    print("=" * 60)
    
    g = gr.Graph()
    
    # Add some nodes
    print("Adding 1000 nodes...")
    node_ids = [gr.NodeId(f'n{i}') for i in range(1000)]
    g.nodes.add(node_ids)
    
    print(f"Before attributes: {g.info()}")
    
    # Try different ways to add attributes
    print("\nMethod 1: Using node proxy...")
    try:
        node = g.nodes.get(gr.NodeId('n0'))
        if node:
            node.set_attr('test', 'value')
            print(f"After proxy attribute: {g.info()}")
        else:
            print("Could not get node proxy")
    except Exception as e:
        print(f"Proxy method failed: {e}")
    
    # Check if we can access the attribute manager directly
    print("\nMethod 2: Checking Rust backend directly...")
    try:
        rust_graph = g._rust
        print(f"Rust graph type: {type(rust_graph)}")
        print(f"Rust graph attributes: {[attr for attr in dir(rust_graph) if not attr.startswith('_')]}")
        
        # Check attribute manager
        if hasattr(rust_graph, 'attribute_manager'):
            attr_manager = rust_graph.attribute_manager
            print(f"Attribute manager type: {type(attr_manager)}")
            print(f"Attribute manager methods: {[attr for attr in dir(attr_manager) if not attr.startswith('_')]}")
            
            # Try to get memory usage
            if hasattr(attr_manager, 'memory_usage_bytes'):
                bytes_used = attr_manager.memory_usage_bytes()
                mb_used = bytes_used / 1024 / 1024
                print(f"Attribute manager memory: {mb_used:.2f} MB ({bytes_used} bytes)")
    except Exception as e:
        print(f"Direct backend access failed: {e}")

def main():
    """Main investigation"""
    detailed_memory_analysis()
    investigate_columnar_store()

if __name__ == "__main__":
    main()
