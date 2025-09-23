#!/usr/bin/env python3
"""Test script to verify debug output from honeycomb layout"""

import groggy as gr
import subprocess
import sys

def test_debug_output():
    """Test that debug output appears in honeycomb calls"""
    print("🧪 Testing debug output from honeycomb layout...")

    # Create a simple test graph
    g = gr.generators.karate_club()
    print(f"📊 Created karate club graph with {g.node_count()} nodes and {g.edge_count()} edges")

    # Test accessing viz accessor
    print("🔍 Testing viz accessor...")
    nodes = g.nodes[g.degree() > 2]
    print(f"📊 Selected {len(nodes)} nodes with degree > 2")

    print("🍯 Calling honeycomb layout - look for debug output below:")
    print("-" * 50)
    
    try:
        # Capture the output from the honeycomb call
        result = nodes.viz.honeycomb()
        print("-" * 50)
        print("✅ honeycomb() call completed successfully")
        print("📝 If you see debug prints above, the new code is working!")
        
        # The debug output should appear in stdout before this point
        
    except Exception as e:
        print(f"❌ honeycomb() call failed: {e}")

if __name__ == "__main__":
    test_debug_output()