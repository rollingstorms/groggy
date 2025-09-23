#!/usr/bin/env python3
"""Test script for honeycomb layout functionality"""

import groggy as gr

def test_honeycomb_layout():
    """Test honeycomb layout on a small graph"""
    print("🧪 Testing honeycomb layout functionality...")

    # Create a simple test graph
    g = gr.generators.karate_club()
    print(f"📊 Created karate club graph with {g.node_count()} nodes and {g.edge_count()} edges")

    # Test accessing viz accessor
    print("🔍 Testing viz accessor...")
    nodes = g.nodes[g.degree() > 2]
    print(f"📊 Selected {len(nodes)} nodes with degree > 2")

    # Test if honeycomb method exists
    if hasattr(nodes.viz, 'honeycomb'):
        print("✅ honeycomb method found in viz accessor")

        # Test honeycomb with default parameters
        print("🍯 Testing honeycomb layout with default parameters...")
        try:
            result = nodes.viz.honeycomb()
            print("✅ honeycomb() call successful")
        except Exception as e:
            print(f"❌ honeycomb() call failed: {e}")

        # Test honeycomb with custom parameters
        print("🍯 Testing honeycomb layout with custom parameters...")
        try:
            result = nodes.viz.honeycomb(cell_size=50.0, energy_optimization=False, iterations=50)
            print("✅ honeycomb(cell_size=50.0, energy_optimization=False, iterations=50) call successful")
        except Exception as e:
            print(f"❌ honeycomb() with parameters failed: {e}")
    else:
        print("❌ honeycomb method not found in viz accessor")

    # Test if show method still works
    print("🖼️  Testing regular show method...")
    try:
        result = nodes.viz.show()
        print("✅ show() call successful")
    except Exception as e:
        print(f"❌ show() call failed: {e}")

if __name__ == "__main__":
    test_honeycomb_layout()