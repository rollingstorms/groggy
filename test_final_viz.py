#!/usr/bin/env python3
import groggy as gr
import time

# Create simple graph with different size values
g = gr.Graph()
g.add_node(label="Small", size_val=5)
g.add_node(label="Medium", size_val=15)
g.add_node(label="Large", size_val=30)
g.add_node(label="Huge", size_val=50)

g.add_edge(0, 1)
g.add_edge(1, 2)
g.add_edge(2, 3)
g.add_edge(3, 0)

print("Starting viz with node_size='size_val'...")
print("Nodes should have sizes: 5, 15, 30, 50")
g.viz.show(layout='circular', node_size='size_val')

time.sleep(10)
print("\nCheck http://127.0.0.1:8080/ to see different node sizes!")
