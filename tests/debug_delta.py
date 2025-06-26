#!/usr/bin/env python3
"""
Debug the delta initialization issue
"""

from gli import Graph, GraphStore, create_random_graph

# Create a simple test
store = GraphStore()
graph = create_random_graph(10, 0.1)
store.update_graph(graph, "initial")

print("Graph created and stored")
print(f"Graph has {len(graph.nodes)} nodes")

# Get current graph
current = store.get_current_graph()
print(f"Retrieved graph has {len(current.nodes)} nodes")
print(f"Current graph _pending_delta: {current._pending_delta}")
print(f"Current graph _is_modified: {current._is_modified}")

# Try to add multiple nodes like in the benchmark
try:
    current = store.get_current_graph()
    for i in range(5):
        print(f"Adding node {i}")
        current = current.add_node(f"new_node_{i}", iteration=i)
        store.update_graph(current, f"add_node_{i}")
        print(f"Successfully added node {i}")
    print("All nodes added successfully")
except Exception as e:
    print(f"Error adding nodes: {e}")
    import traceback
    traceback.print_exc()
