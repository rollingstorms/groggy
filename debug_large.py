#!/usr/bin/env python3
"""
Debug script with larger dataset to see memory usage
"""

import groggy as gr

print("🔍 Large Dataset Memory Test")
print("=" * 50)

# Create a graph
graph = gr.Graph()
nodes = graph.nodes

# Create many nodes to generate significant memory usage
node_ids = []
print("📝 Creating 10,000 nodes...")
for i in range(10000):
    node_id = gr.NodeId(f"node_{i}")
    node_ids.append(node_id)

nodes.add(node_ids)

print("📊 After creating nodes:")
info = graph.info()
print("   ", info)

print("\n🔧 Setting integer attributes on all nodes...")
for i, node_id in enumerate(node_ids):
    node = nodes.get(node_id)
    node.set_attr("age", i + 20)
    if i % 1000 == 0:
        print(f"   Processed {i} nodes...")

print("📊 After setting 10,000 integers:")
info = graph.info()
print("   ", info)

print("\n🔧 Setting multiple types on first 1000 nodes...")
for i, node_id in enumerate(node_ids[:1000]):
    node = nodes.get(node_id)
    node.set_attr("score", float(i) + 0.5)      # Float
    node.set_attr("active", i % 2 == 0)         # Bool
    node.set_attr("name", f"user_{i}")          # String
    if i % 100 == 0:
        print(f"   Processed {i} multi-type nodes...")

print("📊 Final state with mixed types:")
info = graph.info()
print("   ", info)

# Calculate expected memory usage
int_memory = 10000 * 8  # 10,000 i64 values = 80,000 bytes
float_memory = 1000 * 8  # 1,000 f64 values = 8,000 bytes  
bool_memory = 1000 * 1   # 1,000 bool values = 1,000 bytes
string_memory = 1000 * 10  # Estimate 10 bytes per string = 10,000 bytes
total_expected = int_memory + float_memory + bool_memory + string_memory

print(f"\n📊 Expected columnar memory: ~{total_expected / (1024*1024):.2f} MB")
print("✅ Done!")
