#!/usr/bin/env python3
"""
Test script for honeycomb n-dimensional rotation controls
"""

import groggy

# Create a sample graph
g = groggy.Graph()

# Add some nodes and edges to create a more interesting structure
nodes = []
for i in range(20):
    node = g.add_node()
    nodes.append(node)
    if i > 0:
        # Create a connected structure
        g.add_edge(nodes[i-1], nodes[i])
    if i > 5:
        # Add some cross connections
        g.add_edge(nodes[i-5], nodes[i])

print("🍯 Testing Honeycomb N-Dimensional Controls")
print(f"Graph created with {len(nodes)} nodes")
print()
print("Available visualization methods:")
print("1. g.viz.show() - Regular visualization")
print("2. g.viz.show_honeycomb() - Honeycomb with n-dimensional controls")
print("3. g.viz.honeycomb() - Honeycomb layout with parameters")
print()

# Test the new honeycomb controls
print("🎯 Testing honeycomb visualization with n-dimensional rotation controls...")
try:
    g.viz.show_honeycomb()
    print("✅ Honeycomb controls displayed successfully!")
    print()
    print("Expected controls:")
    print("- Left Mouse + Drag: Rotate in dimensions 0-1")
    print("- Left + Ctrl + Drag: Rotate in higher dimensions (2-3)")
    print("- Right Mouse + Drag: Multi-dimensional rotation (0-2, 1-3)")
    print("- Middle Mouse + Drag: Rotate across all dimension pairs")
    print("- Node Dragging: Move individual points in n-dimensional space")
    print("- Momentum Rotation: Smooth rotation continues after release")
    print("- Real-time Updates: 60 FPS with WebSocket streaming")

except Exception as e:
    print(f"❌ Error testing honeycomb controls: {e}")

print()
print("🔧 Testing honeycomb with custom parameters...")
try:
    g.viz.honeycomb(cell_size=50.0, energy_optimization=True, iterations=150)
    print("✅ Custom honeycomb parameters applied successfully!")
except Exception as e:
    print(f"❌ Error with custom honeycomb: {e}")

print()
print("📊 Testing regular visualization for comparison...")
try:
    g.viz.show()
    print("✅ Regular visualization works!")
except Exception as e:
    print(f"❌ Error with regular viz: {e}")