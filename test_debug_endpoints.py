#!/usr/bin/env python3
"""
Test the debug endpoints to see the actual attribute data.
"""

import sys
import os
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python-groggy/python')

import groggy as gr
import requests
import json
import time

def test_debug_endpoints():
    """Test the debug endpoints to inspect attributes."""
    print("🔧 Testing debug endpoints for attribute inspection...")

    # Create a simple test graph
    g = gr.Graph()

    # Add nodes with different attribute types
    node1 = g.add_node(
        label="Simple Node",
        age=25,
        score=99.5,
        active=True
    )

    node2 = g.add_node(
        label="Vector Node",
        tags=["tag1", "tag2", "tag3"],  # TextVec - might cause [object Object]
        coords=[1.0, 2.0, 3.0],  # FloatVec - might cause [object Object]
        counts=[10, 20, 30]  # IntVec - might cause [object Object]
    )

    # Add an edge
    g.add_edge(node1, node2, weight=0.8, relationship="test")

    print(f"Created graph with {g.node_count()} nodes and {g.edge_count()} edges")

    # Start the viz server
    try:
        viz = g.viz
        server = viz.server()  # This should start on port 8083
        print("✅ Server started")

        # Wait for server to fully start
        time.sleep(2)

        # Test debug endpoints
        base_url = "http://localhost:8083"

        print(f"\n🔍 Testing debug endpoints at {base_url}...")

        # Test /debug/nodes
        try:
            response = requests.get(f"{base_url}/debug/nodes", timeout=5)
            print(f"\n📊 /debug/nodes response (status: {response.status_code}):")
            if response.status_code == 200:
                nodes_data = response.json()
                print(json.dumps(nodes_data, indent=2))

                # Check for [object Object] issues
                nodes_str = response.text
                if "[object Object]" in nodes_str:
                    print("❌ Found [object Object] in nodes response!")
                else:
                    print("✅ No [object Object] found in nodes response")

            else:
                print(f"❌ Error: {response.text}")

        except Exception as e:
            print(f"❌ Error with /debug/nodes: {e}")

        # Test /debug/edges
        try:
            response = requests.get(f"{base_url}/debug/edges", timeout=5)
            print(f"\n📊 /debug/edges response (status: {response.status_code}):")
            if response.status_code == 200:
                edges_data = response.json()
                print(json.dumps(edges_data, indent=2))

                # Check for [object Object] issues
                edges_str = response.text
                if "[object Object]" in edges_str:
                    print("❌ Found [object Object] in edges response!")
                else:
                    print("✅ No [object Object] found in edges response")

            else:
                print(f"❌ Error: {response.text}")

        except Exception as e:
            print(f"❌ Error with /debug/edges: {e}")

        # Test /debug/snapshot
        try:
            response = requests.get(f"{base_url}/debug/snapshot", timeout=5)
            print(f"\n📊 /debug/snapshot response (status: {response.status_code}):")
            if response.status_code == 200:
                snapshot_data = response.json()
                print(f"Nodes count: {snapshot_data.get('nodes_count', 0)}")
                print(f"Edges count: {snapshot_data.get('edges_count', 0)}")

                # Show first node in detail
                if snapshot_data.get('nodes'):
                    first_node = snapshot_data['nodes'][0]
                    print(f"\nFirst node details:")
                    print(f"  ID: {first_node.get('id')}")
                    print(f"  Attributes:")
                    for key, value in first_node.get('attributes', {}).items():
                        print(f"    {key}: {value} (type: {type(value).__name__})")

                # Check for [object Object] issues
                snapshot_str = response.text
                if "[object Object]" in snapshot_str:
                    print("❌ Found [object Object] in snapshot response!")
                else:
                    print("✅ No [object Object] found in snapshot response")

            else:
                print(f"❌ Error: {response.text}")

        except Exception as e:
            print(f"❌ Error with /debug/snapshot: {e}")

        print(f"\n🌐 Server is running at {base_url}")
        print("You can also test manually:")
        print(f"  - {base_url}/debug/nodes")
        print(f"  - {base_url}/debug/edges")
        print(f"  - {base_url}/debug/snapshot")
        print(f"  - {base_url}/ (main UI)")

        return True

    except Exception as e:
        print(f"❌ Error starting server: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 Testing debug endpoints for attribute inspection")
    print("=" * 60)

    success = test_debug_endpoints()

    if success:
        try:
            print("\n⏳ Press Enter to stop...")
            input()
        except KeyboardInterrupt:
            pass
    else:
        print("❌ Test failed")