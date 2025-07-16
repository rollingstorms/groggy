#!/usr/bin/env python3
"""
Simple test script to understand the groggy Python API
"""

import sys
import os

# Remove any existing groggy from the module cache
modules_to_remove = [mod for mod in sys.modules.keys() if mod.startswith('groggy')]
for mod in modules_to_remove:
    del sys.modules[mod]

# Add local development version
local_groggy_path = '/Users/michaelroth/Documents/Code/groggy/python'
sys.path.insert(0, local_groggy_path)

import groggy as gr

def test_clean_api():
    """Test the clean, user-friendly API"""
    print("🔧 Testing clean groggy API...")
    
    # Create a graph with clean API
    graph = gr.Graph()
    print(f"Created graph: {graph}")
    
    # Get node and edge collections
    nodes = graph.nodes
    edges = graph.edges
    print(f"Nodes collection: {nodes}")
    print(f"Edges collection: {edges}")
    
    # Test creating NodeId objects
    print("\n📝 Testing NodeId creation...")
    try:
        node1 = gr.NodeId("node1")
        node2 = gr.NodeId("node2") 
        node3 = gr.NodeId("node3")
        print(f"✅ Created NodeIds: {node1}, {node2}, {node3}")
        
        # Test adding nodes
        nodes.add([node1, node2, node3])
        print("✅ Added nodes")
        print(f"Node count: {nodes.size()}")
        
    except Exception as e:
        print(f"❌ Failed to create/add NodeIds: {e}")
    
    # Test creating EdgeId objects and adding edges
    print("\n📝 Testing EdgeId creation...")
    try:
        edge1 = gr.EdgeId(node1, node2)
        edge2 = gr.EdgeId(node2, node3)
        print(f"✅ Created EdgeIds: {edge1}, {edge2}")
        
        # Test adding edges
        edges.add([edge1, edge2])
        print("✅ Added edges")
        print(f"Edge count: {edges.size()}")
        
    except Exception as e:
        print(f"❌ Failed with EdgeIds: {e}")
    
    # Test node attributes with clean API (no JSON needed!)
    print("\n📝 Testing node attributes with clean API...")
    try:
        # Get a specific node to test attributes
        node_proxy = nodes.get(node1)
        if node_proxy:
            print(f"Got node proxy: {node_proxy}")
            
            # Test setting attributes - no JSON needed!
            node_proxy.set_attr("name", "Node One")
            node_proxy.set_attr("value", 42)
            node_proxy.set_attr("metadata", {"type": "important", "score": 0.95})
            
            # Test getting attributes - returns Python objects!
            name = node_proxy.get_attr("name")
            value = node_proxy.get_attr("value")
            metadata = node_proxy.get_attr("metadata")
            print(f"✅ Node attributes:")
            print(f"  name: {name} (type: {type(name)})")
            print(f"  value: {value} (type: {type(value)})")
            print(f"  metadata: {metadata} (type: {type(metadata)})")
                
        else:
            print("❌ Could not get node proxy")
    except Exception as e:
        print(f"❌ Failed with node attributes: {e}")
    
    # Print memory usage after setting node attributes
    info = graph.info()
    print(f"\n📊 Graph info: {info}")

    # Force columnar type for 'value' attribute before stress test
    print("\n🔧 Forcing columnar type for 'value' attribute...")
    graph.attributes.set_type("value", int, True)

    # Stress test: set 'value' for 1000 nodes and print memory usage breakdown
    print("\n🧪 Stress test: setting 'value' for 1000 nodes...")
    for i in range(1000):
        node_id = gr.NodeId(f"node{i}")
        nodes.add([node_id])
        proxy = nodes.get(node_id)
        if proxy:
            proxy.set_attr("value", i)

    # Print memory usage breakdown if available
    print("\n🔍 Direct access test: graph.attributes")
    print("type:", type(graph.attributes))
    print("dir:", dir(graph.attributes))
    try:
        breakdown = graph.attributes.memory_usage_breakdown()
        print("\n📊 Columnar memory usage breakdown:")
        for k, v in breakdown.items():
            print(f"{k}: {v} bytes")
        print("\nNode attribute names:", graph.attributes.node_attr_names())
        print("Total columnar memory usage (bytes):", graph.attributes.memory_usage_bytes())
    except Exception as e:
        print(f"Error calling memory_usage_breakdown: {e}")
    
    # Test edge attributes with clean API
    print("\n📝 Testing edge attributes with clean API...")
    try:
        # Get a specific edge to test attributes
        edge_proxy = edges.get(edge1)
        if edge_proxy:
            print(f"Got edge proxy: {edge_proxy}")
            
            # Test setting attributes - no JSON needed!
            edge_proxy.set_attr("weight", 1.5)
            edge_proxy.set_attr("type", "connection")
            edge_proxy.set_attr("properties", {"directed": True, "color": "blue"})
            
            # Test getting attributes - returns Python objects!
            weight = edge_proxy.get_attr("weight")
            edge_type = edge_proxy.get_attr("type")
            properties = edge_proxy.get_attr("properties")
            print(f"✅ Edge attributes:")
            print(f"  weight: {weight} (type: {type(weight)})")
            print(f"  type: {edge_type} (type: {type(edge_type)})")
            print(f"  properties: {properties} (type: {type(properties)})")
                
        else:
            print("❌ Could not get edge proxy")
    except Exception as e:
        print(f"❌ Failed with edge attributes: {e}")

if __name__ == "__main__":
    test_clean_api()
