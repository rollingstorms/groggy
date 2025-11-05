"""Test the simplest possible builder algorithm - just degree centrality."""
from groggy import Graph
from groggy.builder import algorithm

@algorithm("simple_degree")
def simple_degree(sG):
    """Just compute node degrees."""
    values = sG.nodes(1.0)
    degrees = values.degrees()
    return degrees

# Create a small test graph
g = Graph()
nodes = [g.add_node() for _ in range(10)]
g.add_edges([(nodes[i], nodes[i+1]) for i in range(9)])

# Test the algorithm
sg = g.view()
algo = simple_degree()  # Call it to build the algorithm

print("Testing simple degree algorithm...")
result = sg.apply(algo)

print("\nNode degrees:")
for node in result.nodes:
    print(f"  Node {node.id}: {node.simple_degree}")

print("\n✅ Simple algorithm works!")
