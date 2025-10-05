#!/usr/bin/env python3
"""
Test viz.server() and create a graph with complex attributes to debug.
"""

import sys
import os
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python-groggy/python')

import groggy as gr
import time

def create_test_graph():
    """Create a test graph with various attribute types."""
    print("🧪 Creating test graph with complex attributes...")

    g = gr.Graph()

    # Node 1: Simple attributes
    node1 = g.add_node(
        label="Simple Node",
        age=25,
        score=99.5,
        active=True,
        department="Engineering"
    )

    # Node 2: Vector attributes (these might cause [object Object])
    node2 = g.add_node(
        label="Vector Node",
        tags=["important", "user", "verified"],  # TextVec
        coords=[1.0, 2.0, 3.0],  # FloatVec
        counts=[10, 20, 30, 40, 50],  # IntVec
        flags=[True, False, True]  # BoolVec
    )

    # Node 3: Large vectors (should be summarized)
    node3 = g.add_node(
        label="Large Vector Node",
        embedding=list(range(100)),  # Large IntVec
        large_coords=[float(i) * 0.1 for i in range(50)],  # Large FloatVec
        many_tags=[f"tag_{i}" for i in range(20)]  # Large TextVec
    )

    # Node 4: Mixed attributes
    node4 = g.add_node(
        label="Mixed Node",
        simple_text="hello world",
        simple_number=42,
        vector_data=[1.1, 2.2, 3.3, 4.4],
        category="test"
    )

    # Add edges with attributes
    edge1 = g.add_edge(node1, node2, weight=0.8, relationship="colleague")
    edge2 = g.add_edge(node2, node3, weight=0.6, relationship="friend", strength=0.9)
    edge3 = g.add_edge(node3, node4, weight=0.4, relationship="acquaintance")

    print(f"✅ Created graph with {g.node_count()} nodes and {g.edge_count()} edges")

    # Print what we see directly from the graph
    print("\n🔍 Node attributes as seen from graph:")
    for node_id in [node1, node2, node3, node4]:
        print(f"\nNode {node_id}:")
        for attr_name in g.all_node_attribute_names():
            try:
                value = g.get_node_attr(node_id, attr_name)
                print(f"  {attr_name}: {repr(value)} (type: {type(value).__name__})")
            except:
                pass

    print("\n🔍 Edge attributes as seen from graph:")
    for edge_id in [edge1, edge2, edge3]:
        print(f"\nEdge {edge_id}:")
        for attr_name in g.all_edge_attribute_names():
            try:
                value = g.get_edge_attr(edge_id, attr_name)
                print(f"  {attr_name}: {repr(value)} (type: {type(value).__name__})")
            except:
                pass

    return g

def start_debug_server(g):
    """Start the viz server for debugging."""
    print("\n🚀 Starting viz server...")

    viz = g.viz
    print(f"VizAccessor: {viz}")

    try:
        # Try viz.server() method without port argument
        server_result = viz.server()
        print(f"Server result: {server_result}")
        print(f"Server type: {type(server_result)}")

        # Check what methods the server has
        if server_result:
            server_methods = [m for m in dir(server_result) if not m.startswith('_')]
            print(f"Server methods: {server_methods}")

        return server_result

    except Exception as e:
        print(f"❌ Error starting server: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_server_requests():
    """Test making requests to the server to debug attributes."""
    print("\n🔍 Testing server API endpoints...")

    import requests
    import json

    base_url = "http://localhost:8080"

    try:
        # Test basic connection
        response = requests.get(f"{base_url}/", timeout=5)
        print(f"✅ Server responding: {response.status_code}")

        # Try to find debug endpoints
        # Common debug endpoints we might want to add:
        debug_endpoints = [
            "/debug/nodes",
            "/debug/edges",
            "/debug/snapshot",
            "/api/nodes",
            "/api/edges",
            "/data/nodes",
            "/data/edges"
        ]

        for endpoint in debug_endpoints:
            try:
                response = requests.get(f"{base_url}{endpoint}", timeout=2)
                if response.status_code == 200:
                    print(f"✅ Found endpoint {endpoint}: {response.status_code}")
                    data = response.json()
                    print(f"   Response type: {type(data)}")
                    if isinstance(data, list) and len(data) > 0:
                        print(f"   First item: {data[0]}")
                elif response.status_code == 404:
                    print(f"❌ Not found: {endpoint}")
                else:
                    print(f"🔍 {endpoint}: {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"❌ Error with {endpoint}: {e}")

    except Exception as e:
        print(f"❌ Error testing server: {e}")

if __name__ == "__main__":
    print("🔧 Testing viz.server() and debugging attributes")
    print("=" * 60)

    # Create test graph
    g = create_test_graph()

    # Start server
    server = start_debug_server(g)

    if server:
        print("✅ Server started successfully")
        print("🌐 Server should be available at http://localhost:8080")

        # Wait a moment for server to fully start
        time.sleep(2)

        # Test server endpoints
        test_server_requests()

        print("\n🎯 Manual testing:")
        print("1. Visit http://localhost:8080 in your browser")
        print("2. Click on nodes to see their attributes")
        print("3. Look for [object Object] vs readable values")
        print("4. Check browser console for any errors")

        try:
            input("\nPress Enter when done testing...")
        except KeyboardInterrupt:
            pass

    else:
        print("❌ Failed to start server")