import groggy as gr
import json

# Create a simple graph with labels
g = gr.Graph()
a = g.add_node(name='Alice', age=29)
b = g.add_node(name='Bob', age=35)
c = g.add_node(name='Carol', age=31)

e1 = g.add_edge(a, b, relationship='friend', weight=5)
e2 = g.add_edge(b, c, relationship='colleague', weight=3)
e3 = g.add_edge(a, c, relationship='mentor', weight=2)

print("Graph created with:")
print(f"  Nodes: {g.node_count()}")
print(f"  Edges: {g.edge_count()}")
print(f"  Node columns: {g.nodes.table().column_names}")
print(f"  Edge columns: {g.edges.table().column_names}")

# Test with node labels
print("\n" + "="*60)
print("Testing node_label='name'")
print("="*60)
result = g.viz.debug(node_label='name', verbose=3)
debug_data = json.loads(result)

print(f"\nNodes in debug output: {len(debug_data['nodes'])}")
for i, node in enumerate(debug_data['nodes'][:3]):
    print(f"  Node {i}: id={node.get('id')}, label={node.get('label')}")

# Test with edge labels
print("\n" + "="*60)
print("Testing edge_label='relationship'")
print("="*60)
result = g.viz.debug(edge_label='relationship', verbose=3)
debug_data = json.loads(result)

print(f"\nEdges in debug output: {len(debug_data['edges'])}")
for i, edge in enumerate(debug_data['edges']):
    print(f"  Edge {i}: source={edge.get('source')}, target={edge.get('target')}, label={edge.get('label')}")

# Test with both
print("\n" + "="*60)
print("Testing both node_label='name' and edge_label='relationship'")
print("="*60)
result = g.viz.debug(node_label='name', edge_label='relationship', verbose=3)
debug_data = json.loads(result)

print(f"\nNodes: {len(debug_data['nodes'])}")
for node in debug_data['nodes']:
    print(f"  Node id={node.get('id')}, label={node.get('label')}")

print(f"\nEdges: {len(debug_data['edges'])}")
for edge in debug_data['edges']:
    print(f"  Edge {edge.get('source')} -> {edge.get('target')}, label={edge.get('label')}")
