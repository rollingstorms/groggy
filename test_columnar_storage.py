#!/usr/bin/env python3
"""
Test script to verify columnar storage optimization is working
"""

import time
import groggy as gr

# Test with simple types that should use columnar storage
print("🧪 Testing Columnar Storage Optimization")
print("=" * 50)

# Create a graph
graph = gr.Graph()

print("📊 Initial state:")
print("   Graph info:", graph.info())

# Create some nodes
print("\n📝 Creating nodes...")
nodes = graph.nodes
node_ids = []
for i in range(1000):
    node_id = gr.NodeId(f"node_{i}")
    node_ids.append(node_id)

nodes.add(node_ids)

print("📊 After adding nodes:")
print("   Graph info:", graph.info())

# Now set attributes of different types
print("\n🔧 Setting integer attributes...")
for i, node_id in enumerate(node_ids[:100]):
    node = nodes.get(node_id)
    node.set_attr("age", i + 20)  # Integer - should go to columnar

print("🔧 Setting float attributes...")
for i, node_id in enumerate(node_ids[100:200]):
    node = nodes.get(node_id)
    node.set_attr("score", float(i) + 0.5)  # Float - should go to columnar

print("🔧 Setting boolean attributes...")
for i, node_id in enumerate(node_ids[200:300]):
    node = nodes.get(node_id)
    node.set_attr("active", i % 2 == 0)  # Boolean - should go to columnar

print("🔧 Setting string attributes...")
for i, node_id in enumerate(node_ids[300:400]):
    node = nodes.get(node_id)
    node.set_attr("name", f"user_{i}")  # String - should go to columnar

print("🔧 Setting complex attributes...")
for i, node_id in enumerate(node_ids[400:500]):
    node = nodes.get(node_id)
    node.set_attr("metadata", {"level": i, "tags": ["a", "b"]})  # Complex - should go to sparse

print("\n📊 After setting attributes:")
info = graph.info()
print("   Graph info:", info)

# Let's see the memory breakdown
attrs = info.get('attributes', {})
columnar_mb = float(attrs.get('memory_columnar_store_mb', '0'))
graph_store_mb = float(attrs.get('memory_graph_store_mb', '0'))
content_pool_mb = float(attrs.get('memory_content_pool_mb', '0'))

print(f"\n💾 Memory breakdown:")
print(f"   Columnar Store: {columnar_mb:.2f} MB")
print(f"   Graph Store: {graph_store_mb:.2f} MB")
print(f"   Content Pool: {content_pool_mb:.2f} MB")

if columnar_mb > 0:
    print("✅ SUCCESS: Columnar storage is being used!")
else:
    print("❌ PROBLEM: Columnar storage shows 0 MB")

# Test reading the attributes back
print("\n🔍 Testing attribute retrieval...")
node = nodes.get(node_ids[0])
age = node.get_attr("age")
print(f"   Age attribute: {age} (type: {type(age)})")

node = nodes.get(node_ids[150])
score = node.get_attr("score")
print(f"   Score attribute: {score} (type: {type(score)})")

node = nodes.get(node_ids[250])
active = node.get_attr("active")
print(f"   Active attribute: {active} (type: {type(active)})")

node = nodes.get(node_ids[350])
name = node.get_attr("name")
print(f"   Name attribute: {name} (type: {type(name)})")

node = nodes.get(node_ids[450])
metadata = node.get_attr("metadata")
print(f"   Metadata attribute: {metadata} (type: {type(metadata)})")

print("\n🎯 Test complete!")
