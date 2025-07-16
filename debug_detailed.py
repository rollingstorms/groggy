#!/usr/bin/env python3
"""
Debug script to print detailed columnar storage info
"""

import groggy as gr

print("🔍 Detailed Columnar Storage Debug")
print("=" * 50)

# Create a graph
graph = gr.Graph()
nodes = graph.nodes

# Create a few nodes
node_ids = []
for i in range(5):
    node_id = gr.NodeId(f"node_{i}")
    node_ids.append(node_id)

nodes.add(node_ids)

print("📊 Before setting attributes:")
info = graph.info()
print("   ", info)

print("\n🔧 Setting one integer attribute...")
node = nodes.get(node_ids[0])
print(f"   Setting age=25 on {node_ids[0]}")
node.set_attr("age", 25)

print("📊 After setting one integer:")
info = graph.info()
print("   ", info)

print("\n🔧 Getting the attribute back...")
age = node.get_attr("age")
print(f"   Retrieved age: {age} (type: {type(age)})")

print("\n🔧 Setting multiple integer attributes...")
for i, node_id in enumerate(node_ids):
    node = nodes.get(node_id)
    node.set_attr("age", i + 20)
    print(f"   Set age={i + 20} on {node_id}")

print("📊 After setting multiple integers:")
info = graph.info()
print("   ", info)

print("\n🔧 Testing all types...")
for i, node_id in enumerate(node_ids):
    node = nodes.get(node_id)
    node.set_attr("score", float(i) + 0.5)      # Float
    node.set_attr("active", i % 2 == 0)         # Bool
    node.set_attr("name", f"user_{i}")          # String
    node.set_attr("metadata", {"id": i})        # Complex object
    print(f"   Set all types on {node_id}")

print("📊 Final state:")
info = graph.info()
print("   ", info)

# Test retrieving all types
print("\n🔍 Testing attribute retrieval for node_2:")
node = nodes.get(node_ids[2])
print(f"   Age (int): {node.get_attr('age')} (type: {type(node.get_attr('age'))})")
print(f"   Score (float): {node.get_attr('score')} (type: {type(node.get_attr('score'))})")
print(f"   Active (bool): {node.get_attr('active')} (type: {type(node.get_attr('active'))})")
print(f"   Name (str): {node.get_attr('name')} (type: {type(node.get_attr('name'))})")
print(f"   Metadata (dict): {node.get_attr('metadata')} (type: {type(node.get_attr('metadata'))})")

print("\n✅ Done!")
