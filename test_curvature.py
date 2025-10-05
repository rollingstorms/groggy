"""Test edge curvature with multi-edges"""
import groggy as gr

# Create a graph with multiple edges between same node pairs
g = gr.Graph()

# Add nodes
a = g.add_node(label="A")
b = g.add_node(label="B")
c = g.add_node(label="C")

# Add multiple edges between A and B
e1 = g.add_edge(a, b, type="type1", weight=1.0)
e2 = g.add_edge(a, b, type="type2", weight=2.0)
e3 = g.add_edge(a, b, type="type3", weight=3.0)

# Add single edge from B to C
e4 = g.add_edge(b, c, type="single", weight=1.5)

print(f"Created graph with {g.node_count()} nodes and {g.edge_count()} edges")
print(f"Multi-edges between nodes A and B: {e1}, {e2}, {e3}")

# Start the visualization server
print("\nStarting visualization server...")
print("The curvature slider is in the 'Style' tab of the settings panel")
print("Multi-edges should automatically have different curvatures")
print("Try adjusting the 'Edge Curvature' slider to change the curve strength")

g.viz.server()
