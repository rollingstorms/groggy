# Groggy Python: algorithms.py

Implements core graph algorithms with backend delegation for performance.

- **bfs, dfs, shortest_path**: Traversal and pathfinding.
- **connected_components, clustering_coefficient**: Structural analysis.
- **pagerank, betweenness_centrality, label_propagation_algorithm, louvain, modularity**: Ranking and community detection.

## Example
```python
from groggy import Graph
G = Graph()
G.nodes.add(['a', 'b', 'c'])
G.edges.add([('a', 'b'), ('b', 'c')])
paths = G.shortest_path('a', 'c')
pagerank = G.pagerank()
```
