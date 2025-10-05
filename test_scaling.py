#!/usr/bin/env python3
import groggy as gr

# Create graph with values ranging from 5 to 100
g = gr.Graph()
g.add_node(label="A", val=5)
g.add_node(label="B", val=25)
g.add_node(label="C", val=50)
g.add_node(label="D", val=75)
g.add_node(label="E", val=100)

g.add_edge(0, 1, weight=1)
g.add_edge(1, 2, weight=5)
g.add_edge(2, 3, weight=10)
g.add_edge(3, 4, weight=15)

print("=" * 60)
print("Testing VizConfig Scaling")
print("=" * 60)
print("\nNode values: 5, 25, 50, 75, 100")
print("Edge weights: 1, 5, 10, 15")
print("\n1. WITHOUT scaling (raw values):")
print("   - Node sizes will be: 5, 25, 50, 75, 100")
print("\n2. WITH node_size_range=(10, 50):")
print("   - Smallest node (val=5) → size=10")
print("   - Largest node (val=100) → size=50")
print("   - Middle nodes scaled proportionally")
print("\n3. WITH edge_width_range=(1, 10):")
print("   - Thinnest edge (weight=1) → width=1")
print("   - Thickest edge (weight=15) → width=10")
print("=" * 60)

print("\nStarting visualization with scaling...")
print("Open http://127.0.0.1:8080/ to see the result")

g.viz.show(
    layout='circular',
    node_size='val',
    node_size_range=(10, 50),
    node_color=['red', 'orange', 'yellow', 'green', 'blue'],
    edge_width='weight',
    edge_width_range=(1, 10)
)
