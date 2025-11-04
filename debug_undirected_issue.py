"""Check if we're creating graphs correctly."""
from groggy import Graph

# Test 1: Undirected graph (default)
print("=" * 60)
print("Test 1: Undirected graph (default)")
print("=" * 60)
g1 = Graph()
n0 = g1.add_node()
n1 = g1.add_node()
g1.add_edge(n0, n1)

sg1 = g1.view()
print(f"Is directed: {g1.is_directed}")
print(f"Node {n0} degree: {sg1.degree(n0)}")
print(f"Node {n1} degree: {sg1.degree(n1)}")
print(f"Total edges: {len(list(sg1.edges))}")

# Test 2: Undirected graph with both directions added
print("\n" + "=" * 60)
print("Test 2: Undirected graph with BOTH directions")
print("=" * 60)
g2 = Graph()
n0 = g2.add_node()
n1 = g2.add_node()
g2.add_edge(n0, n1)
g2.add_edge(n1, n0)  # Add reverse too

sg2 = g2.view()
print(f"Is directed: {g2.is_directed}")
print(f"Node {n0} degree: {sg2.degree(n0)}")
print(f"Node {n1} degree: {sg2.degree(n1)}")
print(f"Total edges: {len(list(sg2.edges))}")

# Test 3: Directed graph
print("\n" + "=" * 60)
print("Test 3: Directed graph")
print("=" * 60)
g3 = Graph(directed=True)
n0 = g3.add_node()
n1 = g3.add_node()
g3.add_edge(n0, n1)

sg3 = g3.view()
print(f"Is directed: {g3.is_directed}")
print(f"Node {n0} out-degree: {sg3.out_degree(n0)}")
print(f"Node {n0} in-degree: {sg3.in_degree(n0)}")
print(f"Node {n1} out-degree: {sg3.out_degree(n1)}")
print(f"Node {n1} in-degree: {sg3.in_degree(n1)}")
print(f"Total edges: {len(list(sg3.edges))}")

# Test 4: Directed graph with both directions
print("\n" + "=" * 60)
print("Test 4: Directed graph with BOTH directions")
print("=" * 60)
g4 = Graph(directed=True)
n0 = g4.add_node()
n1 = g4.add_node()
g4.add_edge(n0, n1)
g4.add_edge(n1, n0)

sg4 = g4.view()
print(f"Is directed: {g4.is_directed}")
print(f"Node {n0} out-degree: {sg4.out_degree(n0)}")
print(f"Node {n0} in-degree: {sg4.in_degree(n0)}")
print(f"Node {n1} out-degree: {sg4.out_degree(n1)}")
print(f"Node {n1} in-degree: {sg4.in_degree(n1)}")
print(f"Total edges: {len(list(sg4.edges))}")
