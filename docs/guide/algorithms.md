# Graph Algorithms

Groggy ships 11 native algorithms implemented in Rust, exposed through the Python API. Everything runs on subgraphs (views of your graph), writes results back as attributes, and works with the Batch Executor for the heavy iterative cases.

- Centrality: PageRank, Betweenness, Closeness
- Community: Label Propagation (LPA), Louvain, Leiden, Connected Components
- Pathfinding/Traversal: BFS, DFS, Dijkstra, A*

## How to Run

Use the algorithm handles from `groggy.algorithms.*` and apply them to a graph view or subgraph:

```python
from groggy.algorithms import centrality, community, pathfinding

sg = graph.view()  # or any Subgraph
result = sg.apply(centrality.pagerank(damping=0.9, max_iter=50))

# Results are written as node attributes on the returned Subgraph
scores = result.nodes["pagerank"]
```

Patterns:
- One-off run: `subgraph.apply(algo_handle)`
- Pipelines: `graph.pipeline([algo1, algo2])(subgraph)`
- Builder DSL: use `builder.iterate()` for custom loops; native algorithms already use the Batch Executor when applicable.

## Performance Snapshot

| Algorithm | Notes | Batch Executor effect |
|-----------|-------|-----------------------|
| PageRank | Iterative centrality | ~100x faster (1000 nodes, 100 iters) |
| LPA | Iterative community | ~40x faster (10000 nodes) |
| Louvain / Leiden | Iterative refinement | Benefit from batched steps |
| BFS / DFS | Single-pass traversal | Linear in V+E |
| Dijkstra | Weighted shortest paths | O((V+E) log V) |
| A* | Goal-directed shortest paths | Faster than Dijkstra with a good heuristic |

Known drift: PageRank can show ~5–6% numerical drift after 100+ iterations; practical usage (10–20 iterations) is stable.

## When to Use What

- PageRank: importance ranking on directed or undirected graphs; personalize with a node attribute when needed.
- Betweenness: highlight bridge nodes that sit on many shortest paths.
- Closeness: favor nodes that are on average near all others (per connected component).
- LPA: fast, parameter-light community detection; good for exploratory clustering.
- Louvain: modularity-optimizing communities; balanced speed and quality.
- Leiden: higher-quality, well-connected communities; better for production clustering.
- Connected Components: quickly partition or label disconnected regions (undirected, weak, or strong).
- BFS / DFS: traversals and unweighted distances; BFS for shortest paths, DFS for deep exploration/cycle detection.
- Dijkstra: weighted single-source shortest paths; use when you have edge weights.
- A*: weighted paths with a heuristic and explicit goals; use when you know sources and goals.

---

## Centrality

### PageRank {#pagerank}

Importance based on incoming links with damping.

**Usage**
```python
from groggy.algorithms import centrality

sg = graph.view()
result = sg.apply(
    centrality.pagerank(
        damping=0.85,
        max_iter=100,
        tolerance=1e-6,
        output_attr="pagerank",
        # personalization_attr="traffic"  # optional
    )
)
scores = result.nodes["pagerank"]
```

**Parameters**
- `damping` (float, default 0.85): random jump probability.
- `max_iter` (int, default 100): iteration cap.
- `tolerance` (float, default 1e-6): convergence threshold.
- `personalization_attr` (str | None): node attribute for personalized PageRank.
- `output_attr` (str, default "pagerank"): node attribute to store scores.

**Returns**: Subgraph with `output_attr` on nodes.

**Use cases**: ranking pages or users, surfacing authorities, ordering recommendations.

**Notes**: Batch Executor accelerates iterations; small drift may appear beyond 100 iterations.

### Betweenness {#betweenness}

Shortest-path load centrality.

**Usage**
```python
from groggy.algorithms import centrality

result = graph.view().apply(
    centrality.betweenness(
        normalized=True,
        # weight_attr="weight",  # optional
        output_attr="betweenness",
    )
)
scores = result.nodes["betweenness"]
```

**Parameters**
- `normalized` (bool, default True): normalize scores by graph size.
- `weight_attr` (str | None): edge weight attribute for weighted paths.
- `output_attr` (str, default "betweenness"): node attribute for scores.

**Returns**: Subgraph with `output_attr` on nodes.

**Use cases**: bridge detection, choke point analysis, routing resilience.

### Closeness {#closeness}

Average distance centrality.

**Usage**
```python
from groggy.algorithms import centrality

result = graph.view().apply(
    centrality.closeness(
        # weight_attr="latency",  # optional
        output_attr="closeness",
    )
)
scores = result.nodes["closeness"]
```

**Parameters**
- `weight_attr` (str | None): edge weight attribute.
- `output_attr` (str, default "closeness"): node attribute for scores.

**Returns**: Subgraph with `output_attr` on nodes.

**Use cases**: finding hubs with short average reach, selecting facility locations.

---

## Communities

### Label Propagation (LPA) {#label-propagation-lpa}

Fast, parameter-free-ish community labels via neighbor majority voting.

**Usage**
```python
from groggy.algorithms import community

result = graph.view().apply(
    community.lpa(
        max_iter=100,
        # seed=42,  # optional
        output_attr="community",
    )
)
labels = result.nodes["community"]
```

**Parameters**
- `max_iter` (int, default 100): iteration cap.
- `seed` (int | None): random seed.
- `output_attr` (str, default "community"): node label attribute.

**Returns**: Subgraph with community labels.

**Use cases**: quick segmentation, exploratory clustering, warm-starting Louvain/Leiden.

### Louvain {#louvain}

Modularity-maximizing communities with iterative refinement.

**Usage**
```python
from groggy.algorithms import community

result = graph.view().apply(
    community.louvain(
        resolution=1.0,
        max_iter=100,
        output_attr="community",
    )
)
labels = result.nodes["community"]
```

**Parameters**
- `resolution` (float, default 1.0): controls community granularity.
- `max_iter` (int, default 100): iteration cap.
- `output_attr` (str, default "community"): node label attribute.

**Returns**: Subgraph with community labels.

**Use cases**: balanced quality/speed modularity clustering; baseline for Leiden.

### Leiden {#leiden}

Leiden refinement for well-connected, higher-quality communities.

**Usage**
```python
from groggy.algorithms import community

result = graph.view().apply(
    community.leiden(
        resolution=1.0,
        max_iter=20,
        max_phases=10,
        # seed=123,  # optional
        output_attr="community",
    )
)
labels = result.nodes["community"]
```

**Parameters**
- `resolution` (float, default 1.0): modularity granularity.
- `max_iter` (int, default 20): node-move iterations per phase.
- `max_phases` (int, default 10): refinement phases.
- `seed` (int | None): random seed.
- `output_attr` (str, default "community"): node label attribute.

**Returns**: Subgraph with community labels.

**Use cases**: production-grade community detection, better-connected clusters than Louvain.

### Connected Components {#connected-components}

Partition nodes by reachability.

**Usage**
```python
from groggy.algorithms import community

result = graph.view().apply(
    community.connected_components(
        mode="undirected",  # "undirected" | "weak" | "strong"
        output_attr="component",
    )
)
components = result.nodes["component"]
```

**Parameters**
- `mode` (str, default "undirected"): undirected, weak (ignore direction), or strong.
- `output_attr` (str, default "component"): node component ID attribute.

**Returns**: Subgraph with component IDs; also useful as a mask for further analysis.

**Use cases**: graph cleaning, fragmentation analysis, per-component metrics.

---

## Pathfinding and Traversal

### BFS (unweighted shortest paths) {#bfs-unweighted-shortest-paths}

Layered traversal for hop distances.

**Usage**
```python
from groggy.algorithms import pathfinding

# Mark sources
graph.nodes["is_source"] = graph.nodes.ids() == 0

result = graph.view().apply(
    pathfinding.bfs(
        start_attr="is_source",
        output_attr="distance",
    )
)
distances = result.nodes["distance"]
```

**Parameters**
- `start_attr` (str): node attribute flagging sources (truthy values).
- `output_attr` (str, default "distance"): hop distance attribute.

**Returns**: Subgraph with hop distances; unreachable nodes may remain unset/null.

**Use cases**: unweighted shortest paths, reachability, level-order exploration.

### DFS (depth-first traversal) {#dfs-depth-first-traversal}

Depth-first discovery order.

**Usage**
```python
from groggy.algorithms import pathfinding

graph.nodes["is_root"] = graph.nodes.ids() == 0
result = graph.view().apply(
    pathfinding.dfs(
        start_attr="is_root",
        output_attr="discovery_time",
    )
)
order = result.nodes["discovery_time"]
```

**Parameters**
- `start_attr` (str): node attribute flagging roots.
- `output_attr` (str, default "discovery_time"): discovery order attribute.

**Returns**: Subgraph with discovery times.

**Use cases**: cycle detection, topological-like traversals, exhaustive exploration.

### Dijkstra (weighted shortest paths) {#dijkstra-weighted-shortest-paths}

Single-source weighted shortest paths.

**Usage**
```python
from groggy.algorithms import pathfinding

graph.nodes["is_source"] = graph.nodes.ids() == 0
result = graph.view().apply(
    pathfinding.dijkstra(
        start_attr="is_source",
        weight_attr="weight",  # optional for weighted graphs
        output_attr="distance",
    )
)
distances = result.nodes["distance"]
```

**Parameters**
- `start_attr` (str): node attribute flagging sources.
- `weight_attr` (str | None): edge weight attribute; omit for unweighted cost 1.
- `output_attr` (str, default "distance"): distance attribute.

**Returns**: Subgraph with distances.

**Use cases**: routing with weights (latency, cost), baseline for A*.

### A* (goal-directed shortest paths) {#a-goal-directed-shortest-paths}

Heuristic-guided shortest paths to explicit goals.

**Usage**
```python
from groggy.algorithms import pathfinding

graph.nodes["is_start"] = graph.nodes.ids() == 0
graph.nodes["is_goal"] = graph.nodes.ids() == 9

result = graph.view().apply(
    pathfinding.astar(
        start_attr="is_start",
        goal_attr="is_goal",
        heuristic_attr="h_score",  # optional, per-node heuristic
        weight_attr="weight",      # optional
        output_attr="distance",
    )
)
distances = result.nodes["distance"]
```

**Parameters**
- `start_attr` (str): node attribute flagging sources.
- `goal_attr` (str): node attribute flagging targets.
- `heuristic_attr` (str | None): node heuristic estimates to goal.
- `weight_attr` (str | None): edge weights.
- `output_attr` (str, default "distance"): distance attribute.

**Returns**: Subgraph with distances (and traversal state encoded in attributes).

**Use cases**: navigation with known goals, faster-than-Dijkstra routing when a heuristic is available.

---

## Quick Selection Guide

- Need rankings: PageRank for link-based importance; Betweenness for bridges; Closeness for global proximity.
- Need communities: LPA for speed; Louvain for modularity with good speed; Leiden for highest quality/connectedness.
- Need reachability: Connected Components for partitioning; BFS/DFS for traversal; Dijkstra/A* for weighted routes.
- Iterative workloads: prefer native algorithms or Builder DSL with `builder.iterate()` to leverage the Batch Executor.

## See Also

- `docs/guide/builder.md` for custom pipelines and the Batch Executor.
- `docs/guide/graph-core.md` for graph operations prior to running algorithms.
- `docs/guide/performance.md` for performance tuning tips.
