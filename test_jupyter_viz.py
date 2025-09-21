#!/usr/bin/env python3
"""
Test the Jupyter-safe graph visualization
"""

import sys
import os

# Add python-groggy to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python-groggy/python'))

# Load the jupyter graph viz functions
exec(open('jupyter_graph_viz.py').read())

def test_jupyter_functions():
    print("🧪 Testing Jupyter Graph Visualization Functions")
    print("=" * 60)

    # Test 1: Quick test graph
    print("\n📊 Test 1: Quick test graph")
    try:
        g, iframe = quick_test_graph()
        print(f"✅ Quick test successful: {len(iframe)} character iframe")
        show_server_status()
    except Exception as e:
        print(f"❌ Quick test failed: {e}")

    # Test 2: Custom data
    print("\n📊 Test 2: Custom graph data")
    try:
        nodes = [
            {'id': 'Alice', 'name': 'Alice Smith', 'role': 'manager'},
            {'id': 'Bob', 'name': 'Bob Jones', 'role': 'developer'},
            {'id': 'Carol', 'name': 'Carol Brown', 'role': 'designer'}
        ]
        edges = [
            {'source': 'Alice', 'target': 'Bob'},
            {'source': 'Alice', 'target': 'Carol'}
        ]

        g2, iframe2 = jupyter_graph_viz_with_data(nodes, edges)
        print(f"✅ Custom data successful: {len(iframe2)} character iframe")
        show_server_status()
    except Exception as e:
        print(f"❌ Custom data failed: {e}")

    # Test 3: Existing graph
    print("\n📊 Test 3: Existing graph")
    try:
        import groggy as gr
        g3 = gr.Graph()

        # Create a simple network
        center = g3.add_node()
        g3.set_node_attr(center, "name", "Center")

        for i in range(4):
            node = g3.add_node()
            g3.set_node_attr(node, "name", f"Satellite_{i}")
            g3.add_edge(center, node)

        iframe3 = jupyter_graph_viz(g3)
        print(f"✅ Existing graph successful: {len(iframe3)} character iframe")
        show_server_status()
    except Exception as e:
        print(f"❌ Existing graph failed: {e}")

    # Test cleanup
    print("\n🧹 Testing cleanup")
    try:
        cleanup_servers(keep_last=1)
        show_server_status()
        print("✅ Cleanup successful")
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")

    print("\n" + "=" * 60)
    print("🎉 Jupyter visualization testing complete!")
    print("📱 In Jupyter, you would use: display(HTML(iframe))")

if __name__ == "__main__":
    test_jupyter_functions()