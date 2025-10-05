#!/usr/bin/env python3
import groggy as gr
import requests
import time

g = gr.Graph()
n1 = g.add_node(label="A", score=10)
n2 = g.add_node(label="B", score=20)
g.add_edge(n1, n2, weight=1.0)

print("Starting server with styling...")
g.viz.show(
    layout="circular",
    node_color=["red", "blue"],
    node_size=[10.0, 15.0],
    node_shape=["circle", "square"],
    label=["Node A", "Node B"]
)

time.sleep(3)

print("\nQuerying /debug/snapshot...")
response = requests.get("http://127.0.0.1:8080/debug/snapshot")
data = response.json()

print("\nNode 0:")
print(data['nodes'][0])

print("\nChecking for styling fields...")
node = data['nodes'][0]
if 'color' in node:
    print(f"✅ color: {node['color']}")
else:
    print("❌ color field missing!")

if 'size' in node:
    print(f"✅ size: {node['size']}")
else:
    print("❌ size field missing!")

if 'shape' in node:
    print(f"✅ shape: {node['shape']}")
else:
    print("❌ shape field missing!")

if 'label' in node:
    print(f"✅ label: {node['label']}")
else:
    print("❌ label field missing!")

g.viz.stop()
