#!/usr/bin/env python3
"""
Test node dragging functionality in the graph visualization
"""

import sys
import os
import webbrowser
import time

# Add python-groggy to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python-groggy/python'))

import groggy as gr

def test_node_dragging():
    print("🎮 Testing Node Dragging Functionality")
    print("=" * 50)

    # Create a test graph with multiple nodes
    g = gr.Graph()

    # Create a star pattern - one center node connected to satellite nodes
    center = g.add_node()
    g.set_node_attr(center, "name", "Center")
    g.set_node_attr(center, "type", "hub")

    satellites = []
    for i in range(6):
        node = g.add_node()
        g.set_node_attr(node, "name", f"Node_{i}")
        g.set_node_attr(node, "type", "satellite")
        g.add_edge(center, node)
        satellites.append(node)

    # Add a few connections between satellites to make it more interesting
    g.add_edge(satellites[0], satellites[2])
    g.add_edge(satellites[1], satellites[4])
    g.add_edge(satellites[3], satellites[5])

    print(f"✅ Created test graph: {g.node_count()} nodes, {g.edge_count()} edges")

    # Get the visualization
    try:
        iframe_html = g.graph_viz()
        print("✅ Graph visualization generated successfully")

        # Extract the URL
        import re
        url_match = re.search(r'src="([^"]*)"', iframe_html)
        if url_match:
            url = url_match.group(1)
            print(f"🌐 Opening: {url}")

            # Open in browser
            webbrowser.open(url)

            print("\n🎮 DRAGGING TEST INSTRUCTIONS:")
            print("=" * 40)
            print("1. ✅ You should see a star-shaped graph with 7 nodes")
            print("2. 🖱️  Click on ANY node to select it")
            print("3. 🖱️  Drag that node to move it around")
            print("4. 🖱️  Click on empty space and drag to pan the view")
            print("5. 🔄 Try dragging different nodes")
            print("6. ⚡ Zoom with mouse wheel")
            print()
            print("Expected behavior:")
            print("- Clicking a node + dragging = moves that node")
            print("- Clicking empty space + dragging = pans the viewport")
            print("- Node positions should update in real-time")
            print("- Released nodes should stay in their new positions")

            return True
        else:
            print("❌ Could not extract URL from iframe")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_node_dragging()

    print("\n" + "=" * 50)
    if success:
        print("🎯 Node dragging test launched!")
        print("🔍 Check the browser window to test dragging")
    else:
        print("❌ Test setup failed")
    print("=" * 50)