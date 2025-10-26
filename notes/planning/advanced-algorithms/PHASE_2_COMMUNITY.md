## Phase 2 – Community Algorithms (Structural Grouping)

**Timeline**: 6-8 weeks  
**Dependencies**: Phase 1 (builder primitives)

### Objectives

Expand community detection capabilities to cover major algorithm families: modularity-based,
information-theoretic, hierarchical, overlapping, and statistical models. Each algorithm
must support weighted graphs, provide quality metrics, and compose with pipelines.

### Current State (v0.5.0)

- ✅ Label Propagation (LPA)
- ✅ Louvain method

### Planned Additions

#### 2.1 Leiden Algorithm
**Priority**: High (Louvain improvement)  
**File**: `src/algorithms/community/leiden.rs`  
**Status**: ✅ Implementation complete, benchmarking pending

Leiden improves on Louvain by guaranteeing connected communities and faster convergence.

- ✅ Rust core implementation in `src/algorithms/community/leiden.rs` (444 lines)
- ✅ Move proposal phase with quality function
- ✅ Refinement phase for connectivity using `find_connected_components`
- ✅ Aggregation phase with hierarchical merging
- ✅ Parameters: `resolution`, `max_iter`, `max_phases`, `seed`, `output_attr`
- ✅ Integration with existing modularity helpers (`ModularityData`)
- ✅ Algorithm factory registered in `community::register_algorithms`
- ⏸️ Benchmark against Louvain (speed, quality)

**Algorithm Notes:**
- Use same modularity optimization as Louvain
- Add node movement constraints for connectivity
- Support weighted and directed graphs
- Typical iterations: 10-20 for convergence

#### 2.2 Infomap
**Priority**: High (information-theoretic approach)  
**File**: `src/algorithms/community/infomap.rs`  
**Status**: ✅ Implementation complete, benchmarking pending

Random-walk based community detection using information theory.

- ✅ Rust implementation in `src/algorithms/community/infomap.rs` (591 lines)
- ✅ Random walk simulation with transition probabilities (PageRank-style)
- ✅ Code length computation (map equation)
- ✅ Two-level partitioning via node-move optimization
- ⏸️ Hierarchical extension (multi-level)
- ✅ Parameters: `teleportation`, `num_trials`, `max_iter`, `seed`, `output_attr`
- ✅ Support weighted graphs
- ✅ Algorithm factory registered in `community::register_algorithms`
- ⏸️ Benchmark against Louvain/Leiden (speed, quality)

**Algorithm Notes:**
- Minimize description length of random walks
- Natural handling of directed graphs
- Computationally expensive (O(m log n) iterations)
- Excellent for flow-based communities

#### 2.3 Girvan-Newman
**Priority**: Medium (edge betweenness based)  
**File**: `src/algorithms/community/girvan_newman.rs`  
**Status**: ✅ Implementation complete, benchmarking pending

Hierarchical community detection via iterative edge removal.

- ✅ Rust implementation in `src/algorithms/community/girvan_newman.rs` (656 lines)
- ✅ Edge betweenness computation using Brandes algorithm (BFS/Dijkstra variants)
- ✅ Iterative edge removal with modularity tracking at each step
- ✅ Modularity-based stopping criterion (best partition selection)
- ✅ Union-Find for efficient component detection after edge removal
- ✅ Parameters: `num_levels`, `modularity_threshold`, `weight_attr`, `output_attr`
- ✅ Support for weighted graphs
- ✅ Algorithm factory registered in `community::register_algorithms`
- ✅ Tests: small graph (two triangles), Karate club network
- ⏸️ Benchmark against other hierarchical methods

**Algorithm Notes:**
- O(m²n) complexity (expensive!)
- Produces hierarchical clustering (dendrogram via best modularity)
- Good for small graphs (<10K edges)
- Returns best partition based on modularity peaks
- Edge betweenness recomputed after each removal

#### 2.4 Connected Components
**Priority**: Low (already exists implicitly)  
**File**: `src/algorithms/community/components.rs`  
**Status**: ✅ Complete implementation

Formalize connected component detection as algorithm.

- ✅ Builder-friendly algorithm in `src/algorithms/community/components.rs` (409 lines)
- ✅ Undirected: Union-Find implementation with path compression and union by rank
- ✅ Directed: Tarjan's algorithm for strongly connected components
- ✅ Directed: Weak connectivity (ignores direction, uses Union-Find)
- ✅ Parameters: `mode` (strong|weak|undirected), `output_attr`
- ✅ Algorithm factory registered in `community::register_algorithms`
- ✅ Comprehensive tests (undirected, strong, weak vs strong comparisons)
- ⏸️ Support incremental updates (future enhancement)
- ⏸️ Temporal analysis integration (component evolution tracking)

**Algorithm Notes:**
- O(m α(n)) with union-find for undirected/weak (nearly linear)
- O(m + n) for strong connectivity via Tarjan's algorithm
- `UnionFind` helper in `src/algorithms/community/utils.rs`
- Baseline for multi-component graphs
- Uses bulk attribute operations (`set_node_attrs`) for efficiency

#### 2.5 Hierarchical Agglomerative Clustering
**Priority**: Medium (dendrogram support)

Bottom-up clustering with configurable linkage.

- [ ] Rust implementation in `src/algorithms/community/hierarchical.rs`
- [ ] Distance matrix computation from graph
- [ ] Linkage methods: single, complete, average, Ward
- [ ] Dendrogram construction
- [ ] Cut tree at height/count threshold
- [ ] Parameters: `linkage`, `num_clusters`, `distance_metric`
- [ ] Python factory: `groggy.algorithms.community.hierarchical()`

**Algorithm Notes:**
- O(n²) space for distance matrix (memory concern)
- Consider sparse variants for large graphs
- Dendrogram useful for visualization

#### 2.6 Overlapping Clique Detection
**Priority**: Low (specialized)

Find maximal cliques, allow overlapping communities.

- [ ] Rust implementation in `src/algorithms/community/overlapping_clique.rs`
- [ ] Bron-Kerbosch algorithm for maximal cliques
- [ ] Clique percolation method (k-clique communities)
- [ ] Parameters: `k` (clique size), `overlap_threshold`
- [ ] Return overlapping community assignments
- [ ] Python factory: `groggy.algorithms.community.overlapping_clique()`

**Algorithm Notes:**
- Exponential worst-case complexity
- Practical for sparse graphs with small cliques
- Overlapping structure: nodes can belong to multiple communities

#### 2.7 Stochastic Block Model (SBM)
**Priority**: Medium (statistical foundation)

Statistical model for community structure.

- [ ] Rust implementation in `src/algorithms/community/sbm.rs`
- [ ] EM algorithm for parameter estimation
- [ ] Degree-corrected variant (DC-SBM)
- [ ] Model selection (number of blocks)
- [ ] Parameters: `num_blocks`, `iterations`, `seed`
- [ ] Return block assignments and likelihood
- [ ] Python factory: `groggy.algorithms.community.sbm()`

**Algorithm Notes:**
- Generative model (can sample graphs)
- Handles assortative and disassortative structure
- Computationally intensive for large graphs

#### 2.8 Spectral Clustering
**Priority**: High (foundation for other algorithms)

Use graph Laplacian eigenvectors for clustering.

- [ ] Rust implementation in `src/algorithms/community/spectral.rs`
- [ ] Laplacian construction (normalized, unnormalized)
- [ ] Eigenvalue decomposition (use decomposition module from Phase 5A)
- [ ] K-means on eigenvectors
- [ ] Parameters: `num_clusters`, `laplacian_type`, `kmeans_iterations`
- [ ] Python factory: `groggy.algorithms.community.spectral()`

**Algorithm Notes:**
- Requires decomposition module (Phase 5A dependency)
- O(n³) for dense eigendecomposition (use sparse solvers)
- Excellent for balanced clusters

#### 2.9 Core-Periphery Detection
**Priority**: Low (structural pattern)

Identify core (densely connected) and periphery (sparse) nodes.

- [ ] Rust implementation in `src/algorithms/community/core_periphery.rs`
- [ ] Borgatti-Everett algorithm
- [ ] Iterative correlation maximization
- [ ] Parameters: `iterations`, `seed`
- [ ] Return core/periphery labels + correlation score
- [ ] Python factory: `groggy.algorithms.community.core_periphery()`

**Algorithm Notes:**
- Captures hierarchical structure (different from k-core)
- Useful for social networks, infrastructure graphs
- Convergence typically fast (<10 iterations)

#### 2.10 Additional Algorithms (Lower Priority)

- [ ] **Modularity Maximization** (greedy variant)
- [ ] **Edge Betweenness Clustering** (simplified Girvan-Newman)
- [ ] **BigCLAM** (overlapping communities via matrix factorization)

### Shared Infrastructure

All community algorithms share common patterns:

#### Quality Metrics

- [ ] Modularity computation (Q-score)
- [ ] Coverage and performance metrics
- [ ] Conductance per community
- [ ] Normalized mutual information (NMI) for comparison

Implementation in `src/algorithms/community/metrics.rs`:
```rust
pub fn modularity(subgraph: &Subgraph, communities: &[CommunityId]) -> f64;
pub fn coverage(subgraph: &Subgraph, communities: &[CommunityId]) -> f64;
pub fn conductance(subgraph: &Subgraph, community_nodes: &[NodeId]) -> f64;
pub fn nmi(partition1: &[CommunityId], partition2: &[CommunityId]) -> f64;
```

#### Dendrogram Support

Hierarchical algorithms produce tree structures:

- [ ] `Dendrogram` type in `src/algorithms/community/dendrogram.rs`
- [ ] Cut tree at height/count
- [ ] Export to Newick format
- [ ] Visualization hooks

#### Overlapping Communities

Algorithms supporting overlaps need specialized storage:

- [ ] `OverlappingPartition` type
- [ ] Membership strength scores
- [ ] Community overlap metrics

### Testing Strategy

**Unit Tests:**
- Correctness on synthetic graphs (known communities)
- Edge cases: disconnected graphs, single node, empty
- Parameter validation

**Integration Tests:**
- Compare against known implementations (NetworkX, igraph)
- Quality metric regression (modularity, NMI)
- Performance on standard benchmarks (LFR graphs)

**Benchmarks:**
- Scaling with graph size (nodes: 1K, 10K, 100K, 1M)
- Sparse vs dense graphs
- Weighted vs unweighted

### Success Metrics

- All algorithms tested on LFR benchmark (standard synthetic communities)
- Modularity within 5% of reference implementations
- <10s execution on 100K node graphs (where applicable)
- Clear documentation of complexity and limitations

### Python API Examples

```python
from groggy.algorithms.community import leiden, infomap, spectral

# Leiden with custom resolution
communities = sg.apply(leiden(resolution=1.5, iterations=20))

# Infomap with teleportation
communities = sg.apply(infomap(teleportation=0.15, num_trials=10))

# Spectral clustering
communities = sg.apply(spectral(num_clusters=5, laplacian_type="normalized"))

# Compare algorithms
leiden_result = sg.apply(leiden())
louvain_result = sg.apply(louvain())
nmi_score = compare_communities(leiden_result, louvain_result)  # Uses NMI
```

