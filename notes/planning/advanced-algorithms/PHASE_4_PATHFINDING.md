## Phase 4 – Pathfinding Algorithms (Traversal & Distance)

**Timeline**: 4-5 weeks  
**Dependencies**: Phase 1 (builder primitives)

### Objectives

Extend pathfinding beyond basic Dijkstra/BFS to include advanced shortest path algorithms,
k-shortest paths, all-pairs algorithms, and specialized traversals.

### Current State (v0.5.0)

- ✅ Dijkstra's algorithm
- ✅ BFS (breadth-first search)
- ✅ DFS (depth-first search)
- ✅ A* pathfinding

### Planned Additions

#### 4.1 Bellman-Ford Algorithm
**Priority**: High (handles negative weights)

- [ ] Implementation: `src/algorithms/pathfinding/bellman_ford.rs`
- [ ] Single-source shortest paths with negative weights
- [ ] Negative cycle detection
- [ ] Parameters: `source`, `weight_attr`
- [ ] Return: distances + predecessor map
- [ ] Python factory: `groggy.algorithms.pathfinding.bellman_ford()`

**Notes**: O(VE) complexity. Use when negative weights possible. Detects negative cycles.

#### 4.2 Floyd-Warshall Algorithm
**Priority**: Medium (all-pairs shortest paths)

- [ ] Implementation: `src/algorithms/pathfinding/floyd_warshall.rs`
- [ ] All-pairs shortest path distances
- [ ] O(n³) complexity—document limitations
- [ ] Handle negative weights
- [ ] Parameters: `weight_attr`
- [ ] Return: distance matrix
- [ ] Python factory: `groggy.algorithms.pathfinding.floyd_warshall()`

**Notes**: Memory-intensive (O(n²)). Recommend for <5K nodes. Consider sparse output format.

#### 4.3 Johnson's Algorithm
**Priority**: Medium (efficient all-pairs)

- [ ] Implementation: `src/algorithms/pathfinding/johnson.rs`
- [ ] All-pairs using Bellman-Ford + Dijkstra
- [ ] Reweighting technique for negative weights
- [ ] O(V²log V + VE) complexity
- [ ] Parameters: `weight_attr`
- [ ] Return: distance matrix (sparse)
- [ ] Python factory: `groggy.algorithms.pathfinding.johnson()`

**Notes**: Faster than Floyd-Warshall for sparse graphs. Handles negative weights.

#### 4.4 Yen's K-Shortest Paths
**Priority**: High (k-shortest paths)

- [ ] Implementation: `src/algorithms/pathfinding/yen.rs`
- [ ] Find k simple shortest paths
- [ ] Iterative deviation from previous paths
- [ ] Parameters: `source`, `target`, `k`, `weight_attr`
- [ ] Return: list of paths with costs
- [ ] Python factory: `groggy.algorithms.pathfinding.yen_ksp()`

**Notes**: Expensive (O(kn(m + n log n))). Practical for small k (<10).

#### 4.5 Bidirectional Search
**Priority**: Medium (optimization for single queries)

- [ ] Implementation: `src/algorithms/pathfinding/bidirectional.rs`
- [ ] Search from both source and target
- [ ] Meet-in-the-middle
- [ ] Parameters: `source`, `target`, `weight_attr`
- [ ] Return: shortest path
- [ ] Python factory: `groggy.algorithms.pathfinding.bidirectional()`

**Notes**: Roughly 2x faster than unidirectional Dijkstra for single pairs.

#### 4.6 Shortest Paths All-Pairs (APSP variants)
**Priority**: Low (wrapper)

- [ ] Wrapper selecting best algorithm based on graph properties
- [ ] Dispatch to Floyd-Warshall, Johnson, or repeated Dijkstra
- [ ] Parameters: `weight_attr`, `algorithm` (auto|floyd|johnson|dijkstra)
- [ ] Python factory: `groggy.algorithms.pathfinding.all_pairs_shortest_paths()`

#### 4.7 Landmark-Based Shortest Paths
**Priority**: Low (approximation)

- [ ] Implementation: `src/algorithms/pathfinding/landmark.rs`
- [ ] Precompute distances to landmarks
- [ ] Use triangle inequality for estimates
- [ ] Parameters: `landmarks`, `weight_attr`
- [ ] Python factory: `groggy.algorithms.pathfinding.landmark_paths()`

**Notes**: Approximate distances. Fast queries. Useful for large graphs.

#### 4.8 Constrained Shortest Path
**Priority**: Low (specialized)

- [ ] Implementation: `src/algorithms/pathfinding/constrained.rs`
- [ ] Shortest path with resource constraints
- [ ] Example: shortest path with hop limit
- [ ] Parameters: `source`, `target`, `constraints`
- [ ] Python factory: `groggy.algorithms.pathfinding.constrained_shortest_path()`

#### 4.9 Random Walk (enhanced)
**Priority**: Medium (sampling, embedding)

- [ ] Implementation: `src/algorithms/pathfinding/random_walk.rs`
- [ ] Generate random walk sequences
- [ ] Support restart probability (PageRank-style)
- [ ] Weighted transition probabilities
- [ ] Parameters: `start_nodes`, `length`, `restart_prob`, `weight_attr`, `seed`
- [ ] Return: walk sequences
- [ ] Python factory: `groggy.algorithms.pathfinding.random_walk()`

**Notes**: Foundation for Node2Vec, DeepWalk. Streaming-friendly.

#### 4.10 Monte Carlo Path Sampling
**Priority**: Low (stochastic)

- [ ] Implementation: `src/algorithms/pathfinding/monte_carlo.rs`
- [ ] Sample paths probabilistically
- [ ] Estimate path distributions
- [ ] Parameters: `source`, `target`, `num_samples`, `seed`
- [ ] Python factory: `groggy.algorithms.pathfinding.monte_carlo_paths()`

### Shared Infrastructure

#### Path Representation

Unified path type across algorithms:

```rust
// src/algorithms/pathfinding/path.rs
pub struct Path {
    pub nodes: Vec<NodeId>,
    pub edges: Vec<EdgeId>,
    pub cost: f64,
}

pub struct PathSet {
    pub paths: Vec<Path>,
    pub source: NodeId,
    pub target: Option<NodeId>,
}
```

#### Distance Matrix

Sparse and dense variants for all-pairs results:

```rust
pub enum DistanceMatrix {
    Dense(Vec<Vec<f64>>),              // O(n²) storage
    Sparse(HashMap<(NodeId, NodeId), f64>),  // Sparse storage
}

impl DistanceMatrix {
    pub fn get(&self, src: NodeId, dst: NodeId) -> Option<f64>;
    pub fn to_dataframe(&self) -> DataFrame;  // For Python export
}
```

### Testing Strategy

**Correctness:**
- Known shortest paths on small graphs
- Compare against NetworkX/igraph
- Verify path validity (edges exist, no cycles for simple paths)

**Edge Cases:**
- Negative weights (Bellman-Ford, Johnson)
- Disconnected graphs (infinite distances)
- Self-loops, multi-edges
- Source = target

**Performance:**
- SSSP algorithms: 10K, 100K, 1M nodes
- APSP algorithms: 100, 1K, 5K nodes (memory constraints)
- k-shortest: varying k (1, 5, 10, 50)

### Success Metrics

- SSSP algorithms <1s for 100K nodes on sparse graphs
- APSP algorithms feasible for 5K nodes
- k-shortest paths returns exact results (not approximations)

### Python API Examples

```python
from groggy.algorithms.pathfinding import (
    bellman_ford, yen_ksp, bidirectional, random_walk
)

# Bellman-Ford with negative weights
paths = sg.apply(bellman_ford(source=0, weight_attr="cost"))

# K-shortest paths
k_paths = sg.apply(yen_ksp(source=0, target=100, k=5))
for path in k_paths.paths:
    print(f"Cost: {path.cost}, Path: {path.nodes}")

# Bidirectional search
path = sg.apply(bidirectional(source=0, target=100))

# Random walks for embedding
walks = sg.apply(random_walk(
    start_nodes=list(range(100)),
    length=80,
    restart_prob=0.15,
    seed=42
))
```

---

