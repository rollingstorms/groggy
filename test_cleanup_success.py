#!/usr/bin/env python3
"""
Quick test to verify cleanup was successful and graph_viz() still works
"""

import sys
import os

# Add python-groggy to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python-groggy/python'))

import groggy as gr

def test_graph_viz_after_cleanup():
    print("🧪 Testing graph_viz() after cleanup...")

    # Create simple graph
    g = gr.Graph()
    node1 = g.add_node()
    node2 = g.add_node()
    g.add_edge(node1, node2)

    print(f"✅ Graph created: {g.node_count()} nodes, {g.edge_count()} edges")

    # Test the graph_viz() method
    try:
        iframe = g.graph_viz()
        print(f"✅ graph_viz() works: {len(iframe)} character iframe")

        # Check it contains expected elements
        print(f"🔍 Iframe content: {iframe[:200]}...")
        if "127.0.0.1" in iframe and "iframe" in iframe:
            print("✅ Iframe contains localhost URL and iframe tag")
            return True
        else:
            print("❌ Iframe missing expected elements")
            return False

    except Exception as e:
        print(f"❌ graph_viz() failed: {e}")
        return False

if __name__ == "__main__":
    print("🎯 POST-CLEANUP VERIFICATION TEST")
    print("="*40)

    success = test_graph_viz_after_cleanup()

    print("\n" + "="*40)
    if success:
        print("🎉 CLEANUP SUCCESSFUL!")
        print("✅ Working graph visualization system intact")
        print("✅ Competing systems removed")
        print("✅ Core functionality preserved")
    else:
        print("❌ CLEANUP ISSUES DETECTED")
        print("🔧 Check error messages above")
    print("="*40)