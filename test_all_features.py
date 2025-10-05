#!/usr/bin/env python3
"""
Complete test of VizConfig features:
- Node size scaling with range
- Node colors (array)
- Edge selection and hover
- Edge info popup
- Node selection and hover
"""
import groggy as gr

print("=" * 70)
print("VizConfig Feature Test")
print("=" * 70)

# Create a small graph
g = gr.Graph()
n0 = g.add_node(label="Server", val=100, type="infrastructure")
n1 = g.add_node(label="Database", val=75, type="infrastructure")
n2 = g.add_node(label="API", val=50, type="service")
n3 = g.add_node(label="Frontend", val=25, type="service")
n4 = g.add_node(label="Cache", val=10, type="infrastructure")

e0 = g.add_edge(n0, n1, weight=10, connection="primary")
e1 = g.add_edge(n0, n2, weight=5, connection="secondary")
e2 = g.add_edge(n1, n2, weight=8, connection="primary")
e3 = g.add_edge(n2, n3, weight=3, connection="secondary")
e4 = g.add_edge(n2, n4, weight=6, connection="primary")

print("\nGraph structure:")
print(f"  Nodes: {g.node_count()}")
print(f"  Edges: {g.edge_count()}")

print("\nFeatures to test:")
print("  ✓ Node sizes scaled to range (5-30) based on 'val' column")
print("  ✓ Node colors: infrastructure=blue, service=orange")
print("  ✓ Edge widths scaled to range (1-5) based on 'weight'")
print("  ✓ Hover effects: Orange highlight on hover")
print("  ✓ Selection: Red highlight when clicked")
print("  ✓ Edge popup: Click edge to see source/target/attributes")
print("  ✓ Node popup: Click node to see attributes")

print("\n" + "=" * 70)
print("Starting visualization...")
print("Open http://127.0.0.1:8080/")
print("=" * 70)
print("\nTry:")
print("  1. Hover over nodes and edges (orange highlight)")
print("  2. Click nodes to see attributes")
print("  3. Click edges to see connection info")
print("  4. Notice different sizes based on 'val'")
print("  5. Notice edge thickness based on 'weight'")

g.viz.show(
    layout='force_directed',
    node_size='val',
    node_size_range=(5, 30),
    node_color=['#3498db', '#3498db', '#e67e22', '#e67e22', '#3498db'],
    edge_width='weight',
    edge_width_range=(1, 5)
)
