#!/usr/bin/env python3
"""Interactive viz test with blocking server"""
import groggy as gr

g = gr.Graph()
g.add_node(label="Server", val=100, type="infra")
g.add_node(label="Database", val=75, type="infra")
g.add_node(label="API", val=50, type="service")
g.add_node(label="Frontend", val=25, type="service")
g.add_node(label="Cache", val=10, type="infra")

g.add_edge(0, 1, weight=10)
g.add_edge(0, 2, weight=5)
g.add_edge(1, 2, weight=8)
g.add_edge(2, 3, weight=3)
g.add_edge(2, 4, weight=6)

print("Starting viz on http://127.0.0.1:8080/")
print("Features:")
print("  - Hover: Orange highlight")
print("  - Click: Red selection + info panel")
print("  - Node sizes: 5-30 (scaled from val)")
print("  - Edge widths: 1-5 (scaled from weight)")

g.viz.show(
    layout='force_directed',
    node_size='val',
    node_size_range=(5, 30),
    node_color=['#3498db', '#3498db', '#e67e22', '#e67e22', '#3498db'],
    edge_width='weight',
    edge_width_range=(1, 5)
)

# Keep server running
import time
print("\nServer running. Press Ctrl+C to stop.")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nStopping...")
