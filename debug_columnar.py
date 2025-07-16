#!/usr/bin/env python3
"""
Debug script to check if columnar storage is being used
"""

import groggy as gr

print("🔍 Debugging Columnar Storage Usage")
print("=" * 50)

# Create a graph
graph = gr.Graph()
nodes = graph.nodes

# Create a few nodes
node_ids = []
for i in range(10):
    node_id = gr.NodeId(f"node_{i}")
    node_ids.append(node_id)

nodes.add(node_ids)

print("📊 Initial graph info:")
info = graph.info()
print("   ", info)

print("\n🔧 Setting simple integer attribute...")
node = nodes.get(node_ids[0])
node.set_attr("age", 25)

print("📊 After setting one integer:")
info = graph.info()
print("   ", info)

print("\n🔧 Setting many integer attributes...")
for i, node_id in enumerate(node_ids):
    node = nodes.get(node_id)
    node.set_attr("age", i + 20)

print("📊 After setting many integers:")
info = graph.info()
print("   ", info)

print("\n🔧 Setting many different types...")
for i, node_id in enumerate(node_ids):
    node = nodes.get(node_id)
    node.set_attr("score", float(i) + 0.5)
    node.set_attr("active", i % 2 == 0)
    node.set_attr("name", f"user_{i}")

print("📊 After setting different types:")
info = graph.info()
print("   ", info)

# Test retrieving attributes
print("\n🔍 Testing attribute retrieval:")
node = nodes.get(node_ids[5])
print(f"   Age: {node.get_attr('age')}")
print(f"   Score: {node.get_attr('score')}")
print(f"   Active: {node.get_attr('active')}")
print(f"   Name: {node.get_attr('name')}")

print("\n✅ Done!")
