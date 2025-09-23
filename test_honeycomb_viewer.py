#!/usr/bin/env python3
"""Test script to check honeycomb layout in web viewer"""

import groggy as gr
import time

def test_honeycomb_in_viewer():
    """Test that honeycomb appears as a layout option in the viewer"""
    print("🧪 Testing honeycomb layout in web viewer...")

    # Create a test graph
    g = gr.generators.karate_club()
    print(f"📊 Created karate club graph with {g.node_count()} nodes and {g.edge_count()} edges")

    # Get some nodes to visualize
    nodes = g.nodes[g.degree() > 2]
    print(f"📊 Selected {len(nodes)} nodes with degree > 2")

    # Launch the interactive viewer
    print("🌐 Launching interactive viewer...")
    print("👀 Look for 'Honeycomb' option in the Layout dropdown!")
    print("🍯 You should now see 'Honeycomb' as an option alongside Force, Circular, Grid, and Tree")

    # Start the visualization
    result = nodes.viz.show()

    print("✅ Viewer launched successfully")
    print("📋 Layout options should include:")
    print("   • Force")
    print("   • Circular")
    print("   • Grid")
    print("   • Tree")
    print("   • Honeycomb 🍯 <- NEW!")

    return result

if __name__ == "__main__":
    test_honeycomb_in_viewer()