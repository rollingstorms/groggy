#!/usr/bin/env python3
import groggy as gr

# Create graph with VERY different sizes
g = gr.Graph()
g.add_node(label="Tiny", val=2)
g.add_node(label="Small", val=10)
g.add_node(label="Medium", val=25)
g.add_node(label="Large", val=50)
g.add_node(label="Huge", val=100)

g.add_edge(0, 1)
g.add_edge(1, 2)
g.add_edge(2, 3)
g.add_edge(3, 4)

print("Starting viz with dramatic size differences...")
print("Node sizes: 2, 10, 25, 50, 100")
print("\nOpen http://127.0.0.1:8080/ - you should see VERY different node sizes!")

g.viz.show(
    layout='circular',
    node_size='val',
    node_color=['red', 'orange', 'yellow', 'green', 'blue']
)
