## Phase 3 – Centrality Algorithms (Node/Edge Importance)

**Timeline**: 4-6 weeks  
**Dependencies**: Phase 1 (builder primitives)

### Objectives

Comprehensive centrality measures covering degree-based, distance-based, spectral, and flow-based
importance metrics. Support both node and edge centrality where applicable.

### Current State (v0.5.0)

- ✅ PageRank
- ✅ Betweenness centrality (weighted variant)
- ✅ Closeness centrality (weighted variant, harmonic)

### Planned Additions

#### 3.1 Degree Centrality
**Priority**: High (baseline metric)

- [ ] Implementation: `src/algorithms/centrality/degree.rs`
- [ ] Variants: in-degree, out-degree, total (for directed graphs)
- [ ] Weighted version (strength centrality)
- [ ] Normalization options
- [ ] Parameters: `mode` (in|out|total), `normalized`, `weight_attr`
- [ ] Python factory: `groggy.algorithms.centrality.degree()`

**Notes**: Trivial computation, mostly builder sugar over existing degree operations.

#### 3.2 Eigenvector Centrality
**Priority**: High (importance via connections)

- [ ] Implementation: `src/algorithms/centrality/eigenvector.rs`
- [ ] Power iteration method
- [ ] Convergence tolerance and max iterations
- [ ] Handle disconnected graphs (per-component)
- [ ] Parameters: `max_iter`, `tolerance`, `weight_attr`
- [ ] Python factory: `groggy.algorithms.centrality.eigenvector()`

**Notes**: Similar to PageRank but without damping. Depends on dominant eigenvector.

#### 3.3 Katz Centrality
**Priority**: Medium (attenuation-based)

- [ ] Implementation: `src/algorithms/centrality/katz.rs`
- [ ] Matrix inversion or iterative solver
- [ ] Attenuation factor (alpha) parameter
- [ ] Support exogenous node importance (beta)
- [ ] Parameters: `alpha`, `beta`, `max_iter`, `tolerance`
- [ ] Python factory: `groggy.algorithms.centrality.katz()`

**Notes**: Like eigenvector but attenuates distant connections. Alpha must be < 1/lambda_max.

#### 3.4 Harmonic Centrality
**Priority**: Medium (closeness variant)

- [ ] Implementation: `src/algorithms/centrality/harmonic.rs`
- [ ] Sum of inverse distances
- [ ] Handle disconnected graphs (no infinities)
- [ ] Weighted variant
- [ ] Parameters: `weight_attr`, `normalized`
- [ ] Python factory: `groggy.algorithms.centrality.harmonic()`

**Notes**: Robust closeness variant for disconnected graphs. May already exist as closeness option—verify.

#### 3.5 Load Centrality
**Priority**: Low (betweenness variant)

- [ ] Implementation: `src/algorithms/centrality/load.rs`
- [ ] Fraction of shortest paths through node
- [ ] Similar to betweenness but counts paths differently
- [ ] Parameters: `weight_attr`, `normalized`
- [ ] Python factory: `groggy.algorithms.centrality.load()`

#### 3.6 Subgraph Centrality
**Priority**: Low (spectral method)

- [ ] Implementation: `src/algorithms/centrality/subgraph.rs`
- [ ] Based on graph spectrum (closed walks)
- [ ] Requires eigendecomposition
- [ ] Parameters: `use_laplacian`
- [ ] Python factory: `groggy.algorithms.centrality.subgraph()`

**Notes**: Depends on Phase 5A (decomposition module).

#### 3.7 Communicability Centrality
**Priority**: Low (alternative spectral)

- [ ] Implementation: `src/algorithms/centrality/communicability.rs`
- [ ] Matrix exponential approach
- [ ] Counts weighted walks of all lengths
- [ ] Parameters: none (fully determined by structure)
- [ ] Python factory: `groggy.algorithms.centrality.communicability()`

**Notes**: Expensive (matrix exponential). Consider approximations.

#### 3.8 Flow Betweenness
**Priority**: Low (flow-based)

- [ ] Implementation: `src/algorithms/centrality/flow_betweenness.rs`
- [ ] Maximum flow through nodes
- [ ] Requires flow primitives from Phase 1
- [ ] Parameters: `capacity_attr`
- [ ] Python factory: `groggy.algorithms.centrality.flow_betweenness()`

#### 3.9 Percolation Centrality
**Priority**: Low (specialized)

- [ ] Implementation: `src/algorithms/centrality/percolation.rs`
- [ ] Node importance in network robustness
- [ ] Percolation state parameter
- [ ] Parameters: `percolation_state`
- [ ] Python factory: `groggy.algorithms.centrality.percolation()`

#### 3.10 Variants & Extensions

- [ ] **Weighted PageRank** (if not already supported)
- [ ] **Closeness Harmonic** (may be merged with harmonic centrality)
- [ ] **Edge betweenness** (edge-focused variant)

### Shared Infrastructure

#### Normalization Helper

Centrality scores often need normalization:

```rust
// src/algorithms/centrality/normalize.rs
pub fn normalize_centrality(
    scores: &mut HashMap<NodeId, f64>,
    mode: NormalizationMode,
) -> GraphResult<()>;

pub enum NormalizationMode {
    Sum,      // Sum to 1.0
    Max,      // Max value = 1.0
    MinMax,   // Scale to [0, 1]
    None,     // No normalization
}
```

#### Convergence Checking

Iterative algorithms share convergence logic:

```rust
pub struct ConvergenceChecker {
    tolerance: f64,
    max_iter: usize,
    history: Vec<f64>,  // Recent error values
}

impl ConvergenceChecker {
    pub fn check(&mut self, current: &HashMap<NodeId, f64>, prev: &HashMap<NodeId, f64>) -> bool;
    pub fn iterations(&self) -> usize;
}
```

### Testing Strategy

**Correctness Tests:**
- Compare against NetworkX/igraph on small graphs
- Verify rank ordering (not just absolute values)
- Test on known structures (star, path, cycle, grid)

**Edge Cases:**
- Disconnected graphs
- Self-loops and multi-edges
- Negative weights (where applicable)
- Single node, empty graph

**Performance Benchmarks:**
- Scalability: 1K, 10K, 100K, 1M nodes
- Sparse vs dense graphs
- Convergence iteration counts

### Success Metrics

- All algorithms match reference implementations within 0.01 correlation
- PageRank-style algorithms converge in <50 iterations typically
- <5s for 100K node graphs (distance-based may be slower)

### Python API Examples

```python
from groggy.algorithms.centrality import (
    degree, eigenvector, katz, harmonic
)

# Degree centrality
deg = sg.apply(degree(mode="total", normalized=True))

# Eigenvector centrality
eig = sg.apply(eigenvector(max_iter=100, tolerance=1e-6))

# Katz centrality with custom alpha
katz_scores = sg.apply(katz(alpha=0.1, beta=1.0))

# Compare centralities
pr = sg.apply(pagerank())
eig = sg.apply(eigenvector())
correlation = compute_rank_correlation(pr, eig)
```

---

