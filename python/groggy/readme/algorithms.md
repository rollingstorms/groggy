# Groggy Python: algorithms.py

Implements core graph algorithms with backend delegation for performance.

- **bfs, dfs, shortest_path**: Traversal and pathfinding.
- **connected_components, clustering_coefficient**: Structural analysis.
- **pagerank, betweenness_centrality, label_propagation_algorithm, louvain, modularity**: Ranking and community detection.

## Example
> **Note:** All algorithms are functions in `groggy.algorithms`, not methods on `Graph`.

```python
from groggy import Graph
from groggy.algorithms import bfs, dfs, shortest_path, pagerank

G = Graph()
G.nodes.add(['a', 'b', 'c'])
G.edges.add([('a', 'b'), ('b', 'c')])

# Traversals
bfs_result = bfs(G, 'a')
dfs_result = dfs(G, 'a')

# Pathfinding
path = shortest_path(G, 'a', 'c')

# Ranking
pr = pagerank(G)
```
