# Advanced Algorithm Roadmap

> **📌 Note**: This document has been split into separate phase files for better organization.
> 
> **See**: [`advanced-algorithms/README.md`](advanced-algorithms/README.md) for the restructured documentation.
> 
> This file is preserved for reference but may be outdated. Refer to the individual phase files for the most current planning.

---

## 🎯 Vision & Scope

This roadmap expands Groggy beyond the v0.5 foundations, layering in temporal analytics, advanced
algorithm suites, and richer builder/pipeline infrastructure. The goal is to establish Groggy as a
comprehensive graph analytics platform that balances performance (Rust core), composability (pipeline
architecture), and usability (Python DSL).

### Strategic Goals

**Temporal First-Class Citizenship** – Treat change history as queryable time-series data, enabling
drift analysis, burst detection, and historical pattern mining without custom code.

**Algorithm Breadth** – Cover the standard graph algorithm families (community, centrality, pathfinding,
decomposition, transform, statistical) so users reach for Groggy instead of stitching together
multiple libraries.

**Builder Maturity** – Expand the step primitive catalog so custom algorithms can be composed in Python
without dropping into Rust, while maintaining performance through columnar execution.

**Production Readiness** – Comprehensive testing, benchmarking, documentation, and error handling to
support mission-critical workloads.

### Non-Goals

**Complete Coverage** – We focus on *commonly used* algorithms and patterns, not exhaustive catalogs.
Specialized or esoteric algorithms belong in extensions or community packages.

**Distributed Execution** – Initial phases target single-machine performance. Distributed support
(partitioning, message passing) is a future extension.

**ML Integration** – While we provide feature engineering primitives, deep learning / GNN integration
is out of scope for this roadmap.

---

## 📊 Current Status

**Completed (v0.5.x)**
- ✅ Algorithm trait system and pipeline infrastructure
- ✅ Core algorithms: Connected Components, LPA, Louvain, PageRank, Betweenness, Closeness, Dijkstra, BFS, DFS, A*
- ✅ FFI bridge with thread-safe registry
- ✅ Python user API with discovery, handles, and pipelines
- ✅ Simplified builder DSL with step interpreter
- ✅ CSR (Compressed Sparse Row) infrastructure with caching
- ✅ Node indexer for efficient ID mapping
- ✅ Comprehensive profiling instrumentation (call timers, stats, counters)
- ✅ High-performance algorithm style (STYLE_ALGO) established
- ✅ 304/304 Rust tests passing, 69/69 Python tests passing

**In Progress (v0.6 Development)**
- 🚧 Algorithm performance refactoring (see `REFACTOR_PLAN_PERFORMANCE_STYLE.md`)
  - ✅ Connected Components – 30ms @ 200K nodes (baseline)
  - ✅ PageRank – 45ms @ 200K nodes (iterative solver pattern)
  - ✅ LPA – 250ms @ 200K nodes (fixed O(n²) bug)
  - ✅ Louvain – 180ms @ 200K nodes (multi-phase community detection)
  - ✅ Betweenness – 800ms @ 200K nodes (Brandes algorithm)
  - 🚧 Closeness, Dijkstra, BFS/DFS, A* – applying new style
  - 🚧 Leiden, Girvan-Newman, Infomap – optimizing iterative algorithms
- 🚧 Temporal extensions (see `temporal-extensions-plan.md` - separate planning document)
- 🚧 Visualization streaming architecture (see `viz_module/`)

**Upcoming (This Roadmap)**
- ✅ Phase 1: Builder step primitives (48+ steps implemented across 15 modules)
- 🚧 Performance optimization campaign (reduce all algorithms to <100ms @ 200K nodes where feasible)
- ⏭️ Additional algorithm categories (Phases 2-4)
- ⏭️ Builder / pipeline meta-infrastructure (Phase 5)
- ⏭️ Testing, benchmarking, and documentation (Phase 6)

---

## Note on Temporal Extensions

Temporal extensions are documented separately in `temporal-extensions-plan.md` due to their complexity and scope. That plan covers:
- TemporalSnapshot and ExistenceIndex infrastructure
- TemporalIndex for efficient history queries
- AlgorithmContext temporal extensions
- Temporal algorithm steps (diff, window aggregation, filtering)
- Integration with existing systems
- Full implementation roadmap (8-9 weeks)

This roadmap focuses on expanding the non-temporal algorithm catalog and builder infrastructure.

---

## 🏗️ Performance Architecture (Established v0.6)

### Core Infrastructure

**CSR (Compressed Sparse Row) Representation**
- Cache-friendly adjacency list format for O(1) neighbor access
- Built from edges with optional reverse edge addition for undirected behavior
- Cached per-subgraph with key based on directedness requirements
- Eliminates repeated graph traversals during algorithm execution

**Node Indexer**
- Efficient NodeId → usize mapping for CSR index lookups
- Pre-computed once per algorithm run
- Enables deterministic ordering via `ordered_nodes()`/`ordered_edges()`

**Profiling Instrumentation**
- Hierarchical timers: `record_call_time(key, duration)`
- Counters: `record_call(key)` and `record_stat(key, count)`
- Comprehensive coverage: setup, CSR build, compute, output marshaling
- Exported via `AlgorithmContext` for FFI consumption

### Algorithm Style (STYLE_ALGO)

All algorithms follow a consistent pattern (see `notes/development/STYLE_ALGO.md`):

1. **Setup Phase**: Collect nodes/edges, build indexer, determine CSR requirements
2. **CSR Phase**: Check cache → build if needed → store for reuse
3. **Compute Phase**: Core algorithm kernel with pre-allocated buffers, no inner-loop allocations
4. **Output Phase**: Emit results via `set_*_attr_column` or `ctx.add_output`
5. **Profiling**: Consistent timer/counter keys across all algorithms

**Key Optimizations**:
- Replace all ad-hoc neighbor access with `CSR::neighbors(u)` slices
- Move allocations outside loops; swap buffers in iterative algorithms
- Use `ordered_nodes()` for determinism and efficient indexing
- Cache CSR keyed by directedness requirements
- Instrument every phase for visibility

### Performance Budgets (200K Nodes, 600K Edges)

| Algorithm | Target | Status | Notes |
|-----------|--------|--------|-------|
| Connected Components | 30ms | ✅ | Union-find, single pass |
| PageRank | 50ms | ✅ 45ms | Iterative solver, ~10 iterations |
| LPA | 250ms | ✅ | Iterative, ~5 iterations, label propagation |
| Louvain | 200ms | ✅ 180ms | Multi-phase modularity optimization |
| Betweenness | 1s | ✅ 800ms | Brandes, all-pairs SSSP |
| Closeness | 150ms | 🚧 | All-pairs SSSP (simpler than betweenness) |
| Leiden | 200ms | 🚧 | Should match Louvain performance |
| Dijkstra | 50ms | 🚧 | Single-source, priority queue |
| BFS/DFS | 20ms | 🚧 | Simple traversal |

**Note**: See `notes/development/REFACTOR_PLAN_PERFORMANCE_STYLE.md` for refactoring status.

---

## Phase 1 – Builder Core Extensions

**Timeline**: 4-6 weeks  
**Dependencies**: None (can run in parallel with temporal work)

### Objectives

Expand the builder step primitive catalog to support advanced algorithm composition without
requiring Rust implementation. Each primitive must be columnar, composable, and validated
for performance (targeting O(1) amortized overhead).

### Step Primitive Categories

#### 1.1 Arithmetic & Attribute Operations
Enable basic math on node/edge attributes:

- [ ] `core.add(var1, var2, output)` – Element-wise addition
- [ ] `core.sub(var1, var2, output)` – Element-wise subtraction
- [ ] `core.mul(var1, var2, output)` – Element-wise multiplication
- [ ] `core.div(var1, var2, output)` – Element-wise division with zero-check
- [ ] `core.load_attr(attr_name, output)` – Load node attribute into variable
- [ ] `core.load_edge_attr(attr_name, output)` – Load edge attribute into variable

**Implementation Notes:**
- Operate on columnar `Vec<AttrValue>` for bulk efficiency
- Type coercion rules (int/float promotion, string concatenation)
- Validate operand shapes match (node count)
- Error on type mismatches with clear messages

#### 1.2 Degree & Structural Primitives
Common graph structure computations:

- [ ] `core.weighted_degree(weight_attr, output)` – Degree with edge weights
- [ ] `core.k_core_mark(k, output)` – Mark nodes in k-core
- [ ] `core.triangle_count(output)` – Count triangles per node
- [ ] `core.edge_weight_sum(source_attr, target_attr, output)` – Sum edge weights incident to nodes
- [ ] `core.edge_weight_scale(attr, factor, output)` – Scale edge weights

**Implementation Notes:**
- Leverage existing neighbor bulk operations
- Cache degree computations when possible
- Triangle counting uses edge-iterator + neighbor intersection
- k-core implemented via iterative pruning with convergence check

#### 1.3 Normalization & Scaling
Feature engineering for downstream ML:

- [ ] `core.normalize_sum(var, output)` – Normalize to sum = 1.0
- [ ] `core.normalize_max(var, output)` – Normalize to max = 1.0
- [ ] `core.normalize_minmax(var, output)` – Scale to [0, 1] range
- [ ] `core.standardize(var, output)` – Z-score normalization (mean=0, std=1)
- [ ] `core.clip(var, min, max, output)` – Clamp values to range

**Implementation Notes:**
- Single-pass computation where possible (e.g., find min/max)
- Handle edge cases (all zeros, single value)
- Preserve NaN/infinity semantics clearly
- Document numerical stability considerations

#### 1.4 Temporal Selectors (from Phase 0)
Time-aware operations:

- [ ] `core.snapshot_at(commit|timestamp, output)` – Create temporal snapshot
- [ ] `core.temporal_window(start, end, output)` – Filter to window
- [ ] `core.decay(attr, half_life, output)` – Time-based decay function

**Implementation Notes:**
- Integrate with TemporalSnapshot and TemporalIndex
- Validate commit/timestamp existence
- Decay uses exponential model by default, configurable

#### 1.5 Ordering & Filtering
Common data manipulation patterns:

- [ ] `core.sort_nodes_by_attr(attr, order, output)` – Sort node list by attribute
- [ ] `core.filter_edges_by_attr(attr, predicate, output)` – Filter edge set
- [ ] `core.filter_nodes_by_attr(attr, predicate, output)` – Filter node set
- [ ] `core.top_k(var, k, output)` – Select top-k by value

**Implementation Notes:**
- Predicates: `eq`, `ne`, `lt`, `le`, `gt`, `ge`, `contains`, `matches` (regex)
- Return node/edge ID lists suitable for subgraph creation
- Sorting uses stable sort, document complexity (O(n log n))
- Top-k uses heap-based selection for efficiency

#### 1.6 Sampling
Randomized selection for large graphs:

- [ ] `core.sample_nodes(fraction|count, seed, output)` – Random node sample
- [ ] `core.sample_edges(fraction|count, seed, output)` – Random edge sample
- [ ] `core.reservoir_sample(stream_var, k, seed, output)` – Reservoir sampling

**Implementation Notes:**
- Use reproducible RNG with seed parameter
- Fraction in [0.0, 1.0], count as integer
- Reservoir sampling for streaming contexts
- Document sampling guarantees (uniform, with/without replacement)

#### 1.7 Aggregations & Reductions
Statistical summaries:

- [ ] `core.mean(var, output)` – Arithmetic mean
- [ ] `core.std(var, output)` – Standard deviation
- [ ] `core.median(var, output)` – Median value
- [ ] `core.mode(var, output)` – Most common value
- [ ] `core.quantile(var, q, output)` – q-th quantile
- [ ] `core.entropy(var, output)` – Shannon entropy
- [ ] `core.histogram(var, bins, output)` – Binned counts

**Implementation Notes:**
- Median/quantile use selection algorithm (O(n))
- Mode uses hash table counting
- Entropy computed over discrete distribution (bins or unique values)
- Histogram bins: equal-width by default, configurable

#### 1.8 Path Utilities
Pathfinding helpers:

- [ ] `core.shortest_path_map(source, output)` – SSSP to all nodes
- [ ] `core.k_shortest_paths(source, target, k, output)` – K-shortest paths
- [ ] `core.random_walk(start_nodes, length, output)` – Random walk sequences

**Implementation Notes:**
- shortest_path_map uses Dijkstra/BFS depending on weights
- k-shortest uses Yen's algorithm
- random_walk supports restart probability, weighted transitions

#### 1.9 Community Helpers
Building blocks for custom community detection:

- [ ] `core.community_seed(strategy, output)` – Initialize communities
- [ ] `core.modularity_gain(partition, output)` – Compute modularity change
- [ ] `core.label_propagate_step(labels, output)` – Single LPA iteration

**Implementation Notes:**
- Seeding strategies: singleton, degree-based, random
- Modularity computation uses sparse edge weights
- LPA step uses mode aggregation over neighbors

#### 1.10 Flow & Capacity
Network flow primitives:

- [ ] `core.flow_update(flow, residual, output)` – Update flow along path
- [ ] `core.residual_capacity(capacity, flow, output)` – Compute residual graph

**Implementation Notes:**
- Foundation for max-flow / min-cut algorithms
- Validate flow conservation constraints
- Support both directed and undirected edges

### Infrastructure Components

Beyond individual steps, Phase 1 includes infrastructure for step management:

#### Pipeline Builder Enhancements

- [x] **Step schema registry**: Each step declares inputs, outputs, parameters, types
- [ ] **Validation framework**: Type checking, data-flow analysis, cost estimation
- [ ] **Error reporting**: Structured errors pointing to problematic steps
- [ ] **Step composition helpers**: Macros for common multi-step patterns

#### FFI Runtime

- [x] **Spec serialization**: JSON/TOML roundtrip for pipeline specs
- [ ] **Handle lifecycle**: Automatic cleanup, reference counting for shared state
- [ ] **Error translation**: Rust error types mapped to Python exceptions
- [ ] **GIL management**: Release during expensive operations, reacquire for callbacks

#### Python Builder DSL

- [ ] **Method chaining**: Fluent API for step composition
- [ ] **Variable scoping**: Auto-generated variable names, explicit output control
- [ ] **Type hints**: Full stub coverage for IDE support
- [ ] **Documentation**: Docstrings with examples for each step primitive

#### Validation & Testing

- [ ] **Unit tests**: Each step primitive with edge cases
- [ ] **Integration tests**: Multi-step pipelines exercising composition
- [ ] **Benchmark suite**: Performance regression tracking (`benches/steps/`)
- [ ] **Roundtrip tests**: Python spec → Rust execution → Python result

### Success Metrics

- 100% step primitive coverage with tests
- <1ms per-step overhead for simple operations (add, load_attr)
- <10ms for complex operations (k_core, triangle_count) on 10K nodes
- Zero FFI marshaling overhead for in-place operations
- Clear error messages (90%+ user comprehension without docs)

> **Status**: Phase 1 design scope locked. Implementation tasks remain open and will be tracked in execution milestones.

### Example: Custom PageRank with Builder

```python
from groggy.builder import AlgorithmBuilder

builder = AlgorithmBuilder("custom_pagerank")

# Input subgraph
sg = builder.input("subgraph")

# Initialize ranks
ranks = builder.var("ranks", builder.init_nodes(sg, fn="constant(1.0)"))

# Iteration loop
for _ in range(20):
    # Sum neighbor ranks
    neighbor_sums = builder.map_nodes(
        sg,
        fn="sum(ranks[neighbors(node)])",
        inputs={"ranks": ranks},
    )
    
    # Apply damping
    damped = builder.var("damped", builder.core.mul(neighbor_sums, 0.85))
    teleport = builder.var("teleport", builder.core.add(damped, 0.15))
    
    # Normalize
    ranks = builder.var("ranks", builder.core.normalize_sum(teleport))

# Attach as attribute
builder.attach_node_attr(sg, "pagerank", ranks)

# Compile to algorithm
pagerank_algo = builder.compile()
```

### Dependencies

**Rust Core:**
- Existing `src/algorithms/steps/` foundation
- Columnar neighbor/attribute operations in `GraphSpace`
- Algorithm trait and pipeline infrastructure

**FFI:**
- Parameter marshaling for complex types (predicates, functions)
- Error translation with source location tracking
- Step registry exposure for discovery

**Python:**
- Builder API foundation from Phase 5 (v0.5.0)
- Type stub generation
- Documentation generation from Rust metadata

### Risks & Mitigations

**Risk**: Step primitive explosion (too many variants)  
**Mitigation**: Keep core set minimal; add parameterization (e.g., one `aggregate` step with function parameter vs. separate `sum`, `mean`, `max`)

**Risk**: Performance regression from indirection  
**Mitigation**: Benchmark each primitive; inline hot paths; consider JIT compilation for step sequences

**Risk**: Type system complexity (heterogeneous attributes)  
**Mitigation**: Explicit coercion rules; runtime type checks with clear errors; document type compatibility matrix

**Risk**: User confusion about when to use steps vs pre-built algorithms  
**Mitigation**: Clear guidance in docs; builder examples for common patterns; pre-built algorithms preferred for standard use cases

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

Leiden improves on Louvain by guaranteeing connected communities and faster convergence.

- [x] Rust core implementation in `src/algorithms/community/leiden.rs`
- [x] Move proposal phase with quality function
- [x] Refinement phase for connectivity
- [ ] Aggregation phase
- [x] Parameters: `resolution`, `iterations`, `seed`
- [x] Integration with existing modularity helpers
- [x] Benchmark against Louvain (speed, quality)
- [x] Python factory: `groggy.algorithms.community.leiden()`

**Algorithm Notes:**
- Use same modularity optimization as Louvain
- Add node movement constraints for connectivity
- Support weighted and directed graphs
- Typical iterations: 10-20 for convergence

#### 2.2 Infomap
**Priority**: High (information-theoretic approach)

Random-walk based community detection using information theory.

- [ ] Rust implementation in `src/algorithms/community/infomap.rs`
- [ ] Random walk simulation with transition probabilities
- [ ] Code length computation (map equation)
- [ ] Two-level partitioning
- [ ] Hierarchical extension (multi-level)
- [ ] Parameters: `teleportation`, `num_trials`, `seed`
- [ ] Support weighted graphs
- [ ] Python factory: `groggy.algorithms.community.infomap()`

**Algorithm Notes:**
- Minimize description length of random walks
- Natural handling of directed graphs
- Computationally expensive (O(m log n) iterations)
- Excellent for flow-based communities

#### 2.3 Girvan-Newman
**Priority**: Medium (edge betweenness based)

Hierarchical community detection via iterative edge removal.

- [ ] Rust implementation in `src/algorithms/community/girvan_newman.rs`
- [ ] Edge betweenness computation (reuse existing centrality)
- [ ] Iterative edge removal with dendrogram tracking
- [ ] Modularity-based stopping criterion
- [ ] Return hierarchical clustering
- [ ] Parameters: `num_levels` or `modularity_threshold`
- [ ] Python factory: `groggy.algorithms.community.girvan_newman()`

**Algorithm Notes:**
- O(m²n) complexity (expensive!)
- Produces dendrogram (hierarchical structure)
- Good for small graphs (<10K edges)
- Document performance limitations clearly

#### 2.4 Connected Components
**Priority**: Low (already exists implicitly)

Formalize connected component detection as algorithm.

- [ ] Builder-friendly wrapper in `src/algorithms/community/components.rs`
- [ ] Undirected: Union-Find implementation
- [ ] Directed: Strongly/weakly connected variants
- [ ] Support incremental updates
- [ ] Parameters: `mode` (strong|weak|undirected)
- [ ] Python factory: `groggy.algorithms.community.connected_components()`

**Algorithm Notes:**
- O(m α(n)) with union-find (nearly linear)
- Baseline for multi-component graphs
- Integrate with temporal analysis (component evolution)

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

## New Algorithm Categories (Phases 4A-4D)

These phases introduce entirely new algorithm families, expanding Groggy's capabilities beyond
traditional community/centrality/pathfinding.

### Phase 4A: Decomposition (Spectral & Factorization)

**Timeline**: 6-8 weeks  
**Dependencies**: Linear algebra library (nalgebra or faer)

### Objectives

Provide spectral analysis, graph signal processing, and low-rank approximation tools. These are
foundational for embedding, filtering, and advanced analytics.

#### Infrastructure: Decomposition Module

Create `src/algorithms/decomposition/` with shared primitives:

- [ ] **Laplacian builders** (`laplacian.rs`)
  - Unnormalized Laplacian (D - A)
  - Symmetric normalized (D^{-1/2} L D^{-1/2})
  - Random walk normalized (D^{-1} L)
  - Weighted variants
  - Sparse matrix representation (CSR)

- [ ] **Eigensolvers** (`eigen.rs`)
  - Interface to sparse eigensolvers (Lanczos, Arnoldi)
  - Top-k / bottom-k eigenvalue computation
  - Eigenvector extraction and normalization

- [ ] **Krylov methods** (`krylov.rs`, `lanczos.rs`)
  - Lanczos iteration for symmetric matrices
  - Arnoldi iteration for non-symmetric
  - Convergence monitoring, reorthogonalization

- [ ] **Matrix operators** (in `src/algorithms/linear/`)
  - `GraphOperator` trait for matvec without materialization
  - Adjacency operator, Laplacian operator, degree operator
  - Lazy evaluation, columnar access

#### Algorithms

##### 5A.1 Laplacian Eigenvectors
- [ ] Implementation: `src/algorithms/decomposition/laplacian_eigen.rs`
- [ ] Compute top-k or bottom-k eigenpairs
- [ ] Parameters: `k`, `which` (smallest|largest), `laplacian_type`
- [ ] Python factory: `groggy.algorithms.decomposition.laplacian_eigen()`

##### 5A.2 Graph Fourier Transform
- [ ] Implementation: `src/algorithms/decomposition/graph_fourier.rs`
- [ ] Project graph signal onto eigenbasis
- [ ] Inverse transform
- [ ] Parameters: `signal_attr`, `k_components`
- [ ] Python factory: `groggy.algorithms.decomposition.graph_fourier()`

##### 5A.3 Spectral Filters
- [ ] Implementation: `src/algorithms/decomposition/spectral_filter.rs`
- [ ] Low-pass, high-pass, band-pass filters
- [ ] Heat kernel, wave kernel
- [ ] Parameters: `filter_type`, `cutoff_frequency`
- [ ] Python factory: `groggy.algorithms.decomposition.spectral_filter()`

##### 5A.4 Heat Kernel / Diffusion
- [ ] Implementation: `src/algorithms/decomposition/heat_kernel.rs`
- [ ] Heat diffusion on graphs
- [ ] Exponential of Laplacian
- [ ] Parameters: `time`, `source_nodes`
- [ ] Python factory: `groggy.algorithms.decomposition.heat_kernel()`

##### 5A.5 Chebyshev Filter
- [ ] Implementation: `src/algorithms/decomposition/chebyshev.rs`
- [ ] Polynomial approximation of spectral filters
- [ ] Avoid full eigendecomposition
- [ ] Parameters: `order`, `filter_func`
- [ ] Python factory: `groggy.algorithms.decomposition.chebyshev_filter()`

##### 5A.6-5A.13 Additional Decomposition Algorithms
- [ ] Diffusion maps
- [ ] Laplacian eigenmaps
- [ ] Low-rank approximation (SVD, NMF)
- [ ] Graph wavelets
- [ ] Modularity matrix decomposition
- [ ] Lanczos decomposition (direct exposure)

**Notes**: Many depend on external sparse linear algebra. Consider `nalgebra` or `faer`. Benchmark
eigensolvers carefully (Lanczos convergence, memory usage).

### Phase 4B: Transform (Normalization & Rewriting)

**Timeline**: 3-4 weeks  
**Dependencies**: Phase 1 (builder primitives)

#### Objectives

Graph transformations: normalization, sampling, projection, and structural rewiring.

#### Algorithms

##### 5B.1 Degree Normalization
- [ ] Normalize adjacency matrix by degree
- [ ] Use cases: random walk matrix, label spreading
- [ ] Python: `groggy.algorithms.transform.normalize_degree()`

##### 5B.2 Feature Scaling
- [ ] Min-max, z-score, robust scaling for node attributes
- [ ] Python: `groggy.algorithms.transform.scale_features()`

##### 5B.3 Line Graph
- [ ] Convert graph to line graph (edges → nodes)
- [ ] Python: `groggy.algorithms.transform.line_graph()`

##### 5B.4 K-Core / K-Truss
- [ ] Extract k-core subgraph
- [ ] K-truss variant (triangle-based)
- [ ] Python: `groggy.algorithms.transform.k_core()`, `.k_truss()`

##### 5B.5 Bipartite Projection
- [ ] Project bipartite graph onto one node set
- [ ] Python: `groggy.algorithms.transform.project_bipartite()`

##### 5B.6-5B.12 Additional Transforms
- [ ] Aggregate subgraph (coarsen based on communities)
- [ ] Contract components
- [ ] Random rewiring (preserve degree distribution)
- [ ] Node / edge sampling
- [ ] Incidence to adjacency conversion

**Notes**: Many are simple transformations. Focus on correct semantics and builder integration.

### Phase 4C: Temporal (Dynamic Graphs)

**Timeline**: 4-5 weeks  
**Dependencies**: Temporal Extensions Plan (see `temporal-extensions-plan.md`)

#### Objectives

Temporal variants of standard algorithms, leveraging snapshot and window infrastructure from the temporal extensions plan.

#### Algorithms

##### 5C.1 Temporal Label Propagation
- [ ] LPA with temporal decay weights
- [ ] Window-based label propagation
- [ ] Python: `groggy.algorithms.temporal.lpa()`

##### 5C.2 Temporal PageRank
- [ ] Time-decayed PageRank
- [ ] Restart from historical states
- [ ] Python: `groggy.algorithms.temporal.pagerank()`

##### 5C.3 Window Diffusion
- [ ] Diffusion processes over time windows
- [ ] Python: `groggy.algorithms.temporal.window_diffusion()`

##### 5C.4-5C.8 Additional Temporal Algorithms
- [ ] Temporal motif counting
- [ ] Change detection (anomaly detection)
- [ ] Temporal link prediction
- [ ] Community merge/split tracking

**Notes**: Most reuse standard algorithms with temporal scope in Context. Focus on correct window
semantics and delta handling.

### Phase 4D: Statistical (Graph Measures & Models)

**Timeline**: 3-4 weeks  
**Dependencies**: Phase 5A (decomposition for some metrics)

#### Objectives

Graph-level statistics, similarity measures, generative models, and entropy metrics.

#### Algorithms

##### 5D.1 Graph Entropy
- [ ] Von Neumann entropy (spectral)
- [ ] Degree-based entropy
- [ ] Python: `groggy.algorithms.statistical.entropy_vonneumann()`, `.graph_entropy_degree()`

##### 5D.2 Graph Similarity
- [ ] Edit distance (approximate)
- [ ] Spectral similarity
- [ ] Jaccard similarity (node/edge overlap)
- [ ] Python: `groggy.algorithms.statistical.graph_edit_distance()`, `.graph_similarity_spectral()`

##### 5D.3 Graph Kernels
- [ ] Weisfeiler-Lehman kernel
- [ ] Python: `groggy.algorithms.statistical.graph_kernel_wl()`

##### 5D.4 Configuration Model
- [ ] Null model preserving degree distribution
- [ ] Python: `groggy.algorithms.statistical.configuration_model()`

##### 5D.5-5D.9 Additional Statistical Algorithms
- [ ] Clustering coefficient (global, local)
- [ ] Modularity (standalone computation)
- [ ] Assortativity coefficient
- [ ] Motif counting (triangles, k-cliques)

**Notes**: Some overlap with community metrics. Centralize where possible.

## Phase 5 – Builder / Pipeline Meta Infrastructure

**Timeline**: 3-4 weeks  
**Dependencies**: Phases 1-4 (algorithms using infrastructure)

### Objectives

Elevate the builder/pipeline system with higher-level tooling: DSL expression language,
manifest import/export, schema validation, and introspection APIs. Make algorithm composition
more discoverable and debuggable.

### 6.1 Builder DSL Expression Language

**Goal**: Higher-level macros and patterns for common multi-step compositions.

#### Features

- [ ] **Macro system** for common patterns
  - `with_normalized_degrees()` → init + normalize + attach steps
  - `with_temporal_diff(ref)` → snapshot + diff steps
  - `with_feature_scaling(attrs)` → multi-attribute scaling

- [ ] **Expression compiler** for inline computations
  - Parse expressions like `"rank * 0.85 + 0.15"` into step sequences
  - Type inference from context
  - Optimize trivial expressions (constant folding)

- [ ] **Control flow** (conditional, loops)
  - `builder.if_condition(predicate, then_steps, else_steps)`
  - `builder.iterate(max_iterations, convergence_check, body_steps)`

**Example**:
```python
builder = AlgorithmBuilder("custom_workflow")

# Macro usage
sg = builder.input("subgraph")
sg = builder.with_normalized_degrees(sg)

# Expression usage
ranks = builder.expr("pagerank * 0.5 + degree_centrality * 0.5")

# Conditional
builder.if_condition(
    lambda: ctx.graph_size > 10000,
    then_steps=[use_approximate_algorithm],
    else_steps=[use_exact_algorithm]
)
```

### 6.2 Pipeline Manifest Export/Import

**Goal**: Serialize pipelines for sharing, versioning, and cross-language interop.

#### Features

- [ ] **JSON export** (`pipeline.to_json()`)
  - Include algorithm IDs, parameters, step order
  - Metadata: creator, timestamp, description, version
  - Schema version for compatibility

- [ ] **TOML export** (`pipeline.to_toml()`)
  - Human-readable format for configuration files
  - Comments preserved for documentation

- [ ] **Import with validation** (`Pipeline.from_json()`, `.from_toml()`)
  - Validate schema version compatibility
  - Check algorithm availability (warn on missing)
  - Parameter type validation before instantiation

- [ ] **Manifest diffing** (`diff_manifests(m1, m2)`)
  - Show step additions, removals, parameter changes
  - Use for pipeline versioning and debugging

**Example**:
```python
# Export
pipeline = build_my_pipeline()
manifest = pipeline.to_json()
with open("my_pipeline.json", "w") as f:
    f.write(manifest)

# Import
with open("my_pipeline.json") as f:
    manifest = f.read()
pipeline = Pipeline.from_json(manifest)

# Diff
diff = diff_manifests(old_manifest, new_manifest)
print(diff.summary())  # "Changed parameter 'max_iter' from 20 to 50 in step 3"
```

### 6.3 Pipeline Registry Inspection CLI

**Goal**: Command-line tools for exploring algorithm catalog and debugging pipelines.

#### Commands

- [ ] `groggy algorithms list [--category=community]`
  - List available algorithms with descriptions

- [ ] `groggy algorithms info <algorithm_id>`
  - Show full metadata: params, cost hint, examples

- [ ] `groggy algorithms search <query>`
  - Search by name, description, category

- [ ] `groggy pipeline validate <manifest.json>`
  - Validate pipeline spec without running

- [ ] `groggy pipeline explain <manifest.json>`
  - Show execution plan, estimated cost, step dependencies

- [ ] `groggy pipeline run <manifest.json> <input_graph>`
  - Execute pipeline from CLI

**Example**:
```bash
$ groggy algorithms list --category=centrality
Available algorithms in 'centrality':
  - centrality.pagerank: PageRank centrality measure
  - centrality.betweenness: Betweenness centrality
  - centrality.eigenvector: Eigenvector centrality
  ...

$ groggy pipeline validate my_pipeline.json
✓ Pipeline valid
  Steps: 5
  Estimated cost: O(n²)
  Warnings: Step 3 (floyd_warshall) may be slow for large graphs
```

### 6.4 Parameter Schema Validation

**Goal**: Rich schema system for algorithm parameters with runtime validation.

#### Features

- [ ] **Schema definition language** (Rust-side)
  ```rust
  pub fn parameter_schema() -> ParameterSchema {
      ParameterSchema::new()
          .param("max_iter", ParamType::Int)
              .default(20)
              .range(1, 1000)
              .description("Maximum iterations")
          .param("tolerance", ParamType::Float)
              .default(0.001)
              .range(0.0, 1.0)
              .description("Convergence threshold")
          .param("seed", ParamType::Int)
              .optional()
              .description("Random seed for reproducibility")
  }
  ```

- [ ] **Python-side validation** before FFI call
  - Type checking
  - Range validation
  - Required vs optional parameters
  - Clear error messages pointing to invalid params

- [ ] **Schema introspection** (`algorithm.schema()`)
  - Query parameter requirements programmatically
  - Generate documentation automatically
  - Power IDE autocomplete / validation

**Example**:
```python
# Schema query
schema = algorithms.pagerank.schema()
print(schema.params["damping"])
# ParamInfo(name="damping", type="float", default=0.85, range=(0.0, 1.0), ...)

# Validation error
try:
    sg.apply(pagerank(damping=1.5))  # Invalid: > 1.0
except ValueError as e:
    print(e)  # "Parameter 'damping' out of range: 1.5 not in [0.0, 1.0]"
```

### 6.5 Subgraph Marshaller Enhancements

**Goal**: Optimize FFI marshalling for partial materialization and lazy evaluation.

#### Features

- [ ] **Partial materialization**
  - Only serialize node/edge IDs and changed attributes
  - Avoid full graph copy when pipeline modifies few nodes

- [ ] **Lazy attribute loading**
  - Load attributes on-demand during algorithm execution
  - Cache frequently accessed attributes

- [ ] **Attribute diff compression**
  - For temporal pipelines, send only deltas
  - Use bitmaps for existence changes

- [ ] **Zero-copy views** (where possible)
  - Share read-only subgraph data across FFI boundary
  - Use Arc/shared pointers for immutable data

**Example** (internal optimization, transparent to users):
```rust
// Instead of full subgraph copy:
let full_copy = subgraph.clone();  // Expensive!

// Use partial materialization:
let changes = subgraph.changed_since(last_checkpoint);
ffi_marshal_changes(changes);  // Only send deltas
```

### 6.6 Serde Bridge for Custom AttrValue Types

**Goal**: Support user-defined attribute types with custom serialization.

#### Features

- [ ] **AttrValue extension mechanism**
  - Register custom types (e.g., `AttrValue::Custom(Box<dyn CustomAttr>)`)
  - Provide serialization/deserialization hooks

- [ ] **Common custom types** (built-in)
  - `AttrValue::DateTime` (timestamps)
  - `AttrValue::Uuid` (identifiers)
  - `AttrValue::Json` (nested structures)

- [ ] **Python interop** for custom types
  - Automatic conversion from Python objects
  - Preserve type fidelity across FFI

**Example**:
```python
# Custom attribute type
from datetime import datetime

g.nodes[42].attr["created_at"] = datetime(2024, 1, 15)  # Stored as AttrValue::DateTime
result = sg.apply(temporal_filter(created_after="2024-01-01"))  # Type preserved
```

### 6.7 Integration Tests

**Goal**: Comprehensive tests covering builder-generated pipelines across all categories.

#### Test Coverage

- [ ] **Multi-step pipelines** (5+ steps)
  - Community detection → centrality → filtering
  - Temporal snapshot → diff → window aggregate
  - Decomposition → transform → feature engineering

- [ ] **Cross-category composition**
  - Pathfinding feeding into centrality
  - Community detection feeding into spectral analysis

- [ ] **Error handling**
  - Invalid parameters caught before execution
  - Graceful failures with rollback

- [ ] **Manifest roundtrip**
  - Export → import → execute produces same result

- [ ] **Performance regression**
  - Track execution time for standard pipelines
  - Alert on >10% slowdown

### Success Metrics

- 100% algorithm coverage with parameter schemas
- Manifest export/import roundtrip success rate: 100%
- Pipeline validation catches 95%+ of errors before execution
- CLI tools discoverable and documented
- <5% FFI overhead from marshaller enhancements

---

## Phase 6 – Carryover Tasks (Testing, Documentation, Polish)

**Timeline**: Ongoing throughout other phases, 2-3 weeks for final polish  
**Dependencies**: All previous phases

### Objectives

Ensure production readiness through comprehensive testing, benchmarking, documentation,
and examples. Address technical debt and polish rough edges.

### 7.1 Rust Tests: Unit Coverage

**Goal**: Every algorithm and step has unit tests covering correctness and edge cases.

#### Test Categories

- [ ] **Correctness tests**
  - Compare against reference implementations (NetworkX, igraph)
  - Verify on synthetic graphs with known solutions
  - Check invariants (e.g., modularity bounds, path optimality)

- [ ] **Edge case tests**
  - Empty graph, single node, disconnected components
  - Self-loops, multi-edges, negative weights (where applicable)
  - Extreme parameters (very large/small)

- [ ] **Numerical stability tests**
  - Convergence of iterative algorithms
  - Precision of floating-point computations
  - Handling of near-zero values, infinities, NaN

**Target**: >90% line coverage for algorithm modules.

### 7.2 Rust Tests: Pipeline Integration Suites

**Goal**: End-to-end tests for complex multi-algorithm workflows.

#### Test Scenarios

- [ ] **Temporal workflow**: Snapshot → diff → window aggregate → community detection
- [ ] **Feature engineering**: Load attrs → normalize → scale → PCA (decomposition)
- [ ] **Hybrid workflow**: Pathfinding → centrality → filtering → export
- [ ] **Cancellation**: Long-running pipeline responds to cancellation signal
- [ ] **Error propagation**: Errors in step N properly reported with context

### 7.3 Rust Tests: Registry/Factory Validation

**Goal**: Ensure algorithm registration and factory system work correctly.

#### Tests

- [ ] All algorithms registered at startup
- [ ] Factory creates valid instances with default parameters
- [ ] Invalid parameters rejected by factory
- [ ] Metadata accessible for all registered algorithms

### 7.4 Rust Benchmarks

**Goal**: Track performance regressions and guide optimization.

#### Benchmark Suite (`benches/`)

- [ ] **Algorithm benchmarks** (per category)
  - Vary graph size: 1K, 10K, 100K, 1M nodes
  - Vary density: sparse (avg degree 5), medium (20), dense (100)
  - Vary parameters (iterations, tolerance, etc.)

- [ ] **Pipeline benchmarks**
  - Measure per-step overhead
  - Compare single-algorithm vs multi-step pipelines
  - FFI marshalling overhead

- [ ] **Builder benchmarks**
  - Pipeline compilation time
  - Builder DSL overhead vs hand-coded Rust

**Infrastructure**:
- Use Criterion.rs for statistical benchmarking
- Track results over time (regression detection)
- Generate performance reports

### 7.5 Rust Tests: Cancellation & Error Paths

**Goal**: Graceful handling of interruptions and failures.

#### Tests

- [ ] Cancellation during algorithm execution
- [ ] Cancellation during pipeline steps
- [ ] Error recovery (partial results, rollback)
- [ ] Resource cleanup (no leaks on error/cancellation)

### 7.6 Python Tests: Coverage for Traversal, Algorithms, Pipelines

**Goal**: Comprehensive Python test suite matching Rust coverage.

#### Test Files

- [ ] `tests/test_community.py` (all community algorithms)
- [ ] `tests/test_centrality.py` (all centrality algorithms)
- [ ] `tests/test_pathfinding.py` (all pathfinding algorithms)
- [ ] `tests/test_decomposition.py` (spectral, factorization)
- [ ] `tests/test_transform.py` (graph transformations)
- [ ] `tests/test_temporal.py` (temporal algorithms)
- [ ] `tests/test_statistical.py` (graph measures, models)
- [ ] `tests/test_pipeline.py` (multi-step workflows)
- [ ] `tests/test_builder.py` (DSL composition)
- [ ] `tests/test_discovery.py` (algorithm introspection)

**Target**: >95% coverage of Python API surface.

### 7.7 Python Tests: Integration with Real Datasets

**Goal**: Validate on real-world graphs (not just synthetic).

#### Datasets

- [ ] **Social networks**: Karate Club, Dolphins, Facebook
- [ ] **Citation networks**: Cora, PubMed
- [ ] **Infrastructure**: Road networks, power grids
- [ ] **Web graphs**: Wikipedia links, Web crawls

#### Tests

- [ ] Load from standard formats (GraphML, GML, EdgeList)
- [ ] Run standard algorithms, verify sensible results
- [ ] Performance within expected bounds

### 7.8 Python Tests: Performance Regression Harness

**Goal**: Detect performance regressions in Python API.

#### Infrastructure

- [ ] Benchmark framework (`pytest-benchmark`)
- [ ] Baseline measurements for standard operations
- [ ] Automated regression detection (>10% slowdown triggers warning)
- [ ] Track over time (historical performance data)

### 7.9 Python Tests: Cross-Platform Runners

**Goal**: Ensure compatibility across operating systems and architectures.

#### Platforms

- [ ] **Linux**: Ubuntu 22.04, x86_64
- [ ] **Linux**: Ubuntu 22.04, ARM64
- [ ] **macOS**: Latest, x86_64 (Intel)
- [ ] **macOS**: Latest, ARM64 (Apple Silicon)
- [ ] **Windows**: Latest, x86_64

#### CI/CD

- [ ] GitHub Actions workflows for all platforms
- [ ] Automated testing on pull requests
- [ ] Nightly builds to catch integration issues

### 7.10 Documentation: API Reference Expansions

**Goal**: Complete, accurate documentation for all public APIs.

#### Coverage

- [ ] **Rust API docs** (rustdoc)
  - All public modules, structs, traits, functions
  - Examples in docstrings
  - Cross-references between related items

- [ ] **Python API docs** (Sphinx or similar)
  - All public classes, functions, methods
  - Type hints in signatures
  - Examples, parameter descriptions, return values

- [ ] **FFI documentation**
  - Safety invariants
  - Memory management rules
  - Error handling patterns

### 7.11 Documentation: Migration Guide + Performance Guide

**Goal**: Help users upgrade and optimize their usage.

#### Migration Guide

- [ ] Upgrading from v0.5.0 to v0.6.0+
  - API changes (breaking and deprecated)
  - New features and how to adopt them
  - Migration scripts or recipes

#### Performance Guide

- [ ] Choosing algorithms (complexity, use cases)
- [ ] Tuning parameters for speed vs accuracy
- [ ] Profiling and debugging slow pipelines
- [ ] Best practices (columnar ops, bulk operations)
- [ ] Common pitfalls and how to avoid them

### 7.12 Examples/Notebooks: Per-Category Demos

**Goal**: Practical, runnable examples for each algorithm category.

#### Notebooks

- [ ] **Community Detection Tutorial**
  - Compare LPA, Louvain, Leiden, Spectral
  - Visualize communities
  - Measure quality (modularity, coverage)

- [ ] **Centrality Analysis Tutorial**
  - Compute multiple centrality measures
  - Rank correlation analysis
  - Identify influential nodes

- [ ] **Pathfinding Tutorial**
  - Shortest paths, k-shortest paths
  - All-pairs distance matrix
  - Visualize paths on graph

- [ ] **Temporal Analysis Tutorial**
  - Snapshot creation and comparison
  - Community drift over time
  - Burst detection

- [ ] **Spectral Analysis Tutorial**
  - Compute Laplacian eigenvectors
  - Graph Fourier transform
  - Spectral clustering

- [ ] **Pipeline Composition Tutorial**
  - Build multi-step workflow
  - Export/import manifests
  - Debug and optimize pipelines

### 7.13 Examples/Notebooks: Pipeline Flows, Builder DSL, Temporal

**Goal**: Advanced examples showcasing composition and DSL.

#### Notebooks

- [ ] **Custom Algorithm with Builder DSL**
  - Implement PageRank from primitives
  - Add custom logic (restart sets, personalization)

- [ ] **Temporal Workflow**
  - Track community evolution
  - Detect anomalies over time
  - Predict link formation

- [ ] **Feature Engineering Pipeline**
  - Extract graph features for ML
  - Normalize, scale, project
  - Export to DataFrame for downstream models

### 7.14 Release Polish

**Goal**: Final preparations for production release.

#### Tasks

- [ ] **Performance profiling**
  - Identify hot paths in critical algorithms
  - Optimize where needed (>2x speedup considered worthwhile)
  - Document performance characteristics

- [ ] **Parallel step execution**
  - Identify independent steps in pipelines
  - Execute in parallel (rayon or tokio)
  - Benchmark parallel vs sequential

- [ ] **Error UX improvements**
  - Review all error messages for clarity
  - Add suggestions (e.g., "Did you mean 'max_iter'?")
  - Structured errors with codes for programmatic handling

- [ ] **Memory profiling**
  - Check for leaks (valgrind, heaptrack)
  - Optimize memory usage for large graphs
  - Document memory requirements

- [ ] **Code review and cleanup**
  - Remove dead code, TODOs
  - Consistent naming and formatting
  - Address clippy warnings

- [ ] **Version numbering and changelog**
  - Semantic versioning
  - Detailed CHANGELOG.md
  - Migration notes for breaking changes

---

## 🎯 Success Metrics & Acceptance Criteria

### Coverage Metrics

- **Algorithm Coverage**: >50 algorithms across 7 categories
- **Test Coverage**: >90% line coverage (Rust), >95% API coverage (Python)
- **Documentation Coverage**: 100% of public APIs documented

### Performance Metrics

- **Benchmark Suite**: >100 benchmarks covering algorithms and pipelines
- **Regression Detection**: Automated alerts on >10% slowdown
- **Scalability**: All algorithms tested on graphs up to 1M nodes

### Quality Metrics

- **Zero Compiler Warnings**: Clean `cargo build` and `cargo clippy`
- **Zero Test Failures**: All tests pass on all platforms
- **Zero Critical Bugs**: No known correctness issues

### Usability Metrics

- **Time to First Result**: <5 min from install to running first algorithm
- **Error Comprehension**: >90% of users understand error messages without docs
- **Documentation Quality**: Positive user feedback, low support burden

---

## 📅 Overall Timeline & Dependencies

### Gantt Chart (Approximate)

```
Phase 1 (Builder Primitives): [======] (4-6 weeks) - Can start immediately
Phase 2 (Community):              [========] (6-8 weeks) - After Phase 1
Phase 3 (Centrality):             [======] (4-6 weeks) - After Phase 1
Phase 4 (Pathfinding):            [=====] (4-5 weeks) - After Phase 1
Phase 4A (Decomposition):            [========] (6-8 weeks) - After Phase 1
Phase 4B (Transform):                    [====] (3-4 weeks) - After Phase 1
Phase 4C (Temporal Algos):               [=====] (4-5 weeks) - After Temporal Plan
Phase 4D (Statistical):                      [====] (3-4 weeks) - After Phase 4A
Phase 5 (Meta Infra):                            [====] (3-4 weeks) - After Phase 1-4
Phase 6 (Polish):            [===================================] (Ongoing)
                                                       [===] (2-3 weeks final)

Temporal Extensions (see temporal-extensions-plan.md): [=========] (8-9 weeks, parallel)

Total: ~30-40 weeks (7-10 months) excluding temporal work
```

### Critical Path

1. **Temporal Extensions** run in parallel (see separate plan)
2. **Phase 1** (Builder) is prerequisite for Phases 2-4
3. **Phase 4A** (Decomposition) is prerequisite for Phase 4D and some Phase 2 algorithms
4. **Phase 6** (Polish) runs concurrently, with final push at end

### Parallelization Opportunities

- Phase 1 can start immediately alongside temporal work
- Phases 2, 3, 4 can run in parallel after Phase 1 completes
- Phase 4A-4D can be staggered with partial dependencies
- Phase 6 tasks distributed throughout (testing, docs as features land)

---

## 🚨 Risks & Mitigations

### Technical Risks

**Risk**: Linear algebra dependencies (Phase 4A) introduce complexity  
**Mitigation**: Evaluate `nalgebra`, `faer`, and `ndarray` early. Choose based on performance and API ergonomics. Consider optional feature flags.

**Risk**: Builder DSL becomes too complex (Phase 1, 5)  
**Mitigation**: Keep primitives simple and composable. Provide high-level macros for common patterns. Gather user feedback early.

**Risk**: FFI overhead grows with more complex types (Phase 5)  
**Mitigation**: Benchmark marshalling costs. Use zero-copy where possible. Profile and optimize hot paths.

**Risk**: Test suite execution time becomes prohibitive (Phase 6)  
**Mitigation**: Parallelize tests. Use test categorization (unit, integration, slow). Run full suite in CI, subset locally.

### Resource Risks

**Risk**: Timeline slips due to underestimated complexity  
**Mitigation**: Buffer time in estimates (30%). Prioritize ruthlessly (High/Medium/Low). Ship incrementally.

**Risk**: Burnout from sustained effort  
**Mitigation**: Break work into digestible chunks. Celebrate milestones. Maintain quality over speed.

### Scope Risks

**Risk**: Feature creep (algorithms beyond roadmap)  
**Mitigation**: Strict prioritization. Defer "nice to have" items to future releases. Focus on breadth first, depth second.

**Risk**: External dependencies change (libraries, Python versions)  
**Mitigation**: Pin dependency versions. Monitor for breaking changes. Budget time for adaptation.

---

## 🎓 Learning & Experimentation

### Experimental Algorithm Families (Future Extensions)

Beyond this roadmap, these families could be explored:

**Streaming/Incremental Updates** – Leverage ChangeTracker for incremental community detection, rolling centrality, online anomaly detection.

**Structural Embeddings** – Node2Vec, DeepWalk, graph neural network foundations via sampling and aggregation primitives.

**Motif & Pattern Mining** – Subgraph enumeration, frequent pattern discovery, graphlet counting.

**Reachability & Flow** – Max-flow/min-cut, reachability indices, flow-based community detection.

**Graph Sketches & Sampling** – MinHash for similarity, reservoir sampling, approximate query answering.

**Explainability** – Trace paths, collect evidence, influence scoring for algorithm decisions.

### Research Directions

**Distributed Execution** – Partition graphs, message passing, integration with distributed storage.

**GPU Acceleration** – Offload linear algebra (decomposition) and graph traversal to GPU.

**Interactive Algorithms** – Real-time parameter tuning, progress visualization, streaming updates.

**ML Integration** – Graph feature extraction for GNNs, embedding pipelines, prediction interfaces.

---

## 📝 Conclusion

This roadmap transforms Groggy from a solid foundation (v0.5.0) into a comprehensive graph analytics
platform. By expanding the algorithm catalog, maturing the builder/pipeline infrastructure, and
maintaining rigorous testing and documentation standards, we enable users to tackle complex graph
problems without leaving Python or sacrificing performance.

**Note**: Temporal analytics capabilities are documented separately in `temporal-extensions-plan.md`.
This roadmap focuses on expanding the core algorithm library and builder primitives to support
a wide range of graph analysis workflows.

### Key Principles Maintained

- **Attribute-first, columnar operations** throughout
- **No business logic in FFI**—marshalling and safety only
- **O(1) amortized performance expectations** for core operations
- **Composability over monoliths**—algorithms as building blocks
- **Comprehensive testing and benchmarking** from day one

### Success Indicators

If this roadmap succeeds, users will:
- Reach for Groggy instead of NetworkX/igraph for performance-critical workflows
- Compose custom algorithms without touching Rust
- Leverage temporal analytics for dynamic graph analysis (via separate temporal extensions)
- Trust Groggy for production workloads (reliability, performance, documentation)

### Next Steps

1. **Begin Phase 1** (Builder primitives) as foundation for algorithm expansion
2. **Incrementally add algorithms** (Phases 2-4) based on user demand
3. **Develop meta-infrastructure** (Phase 5) as algorithm catalog grows
4. **Continuous polish** (Phase 6) throughout, not just at end
5. **Coordinate with temporal work** (see `temporal-extensions-plan.md`)
6. **Gather feedback** from early adopters, adjust priorities

This is an ambitious but achievable roadmap. By following the established patterns, maintaining
discipline around architecture boundaries, and shipping incrementally, we can deliver each phase
with confidence while preserving the core performance and usability that define Groggy.

---

## 📚 References & Related Documents

- **Algorithm Architecture Roadmap** (`algorithm-architecture-roadmap.md`) – Core v0.5.0 foundation
- **Temporal Extensions Plan** (`temporal-extensions-plan.md`) – Detailed Phase 0 specification
- **Visualization Module Plans** (`viz_module/`) – Parallel visualization work
- **Personas** (`personas/`) – Specialized review guidance (Rusty, Bridge, Zen, etc.)

---

**Document Version**: 1.0  
**Last Updated**: 2024-01-15  
**Status**: Planning / RFC  
**Approval**: Pending review

---

## 🎨 Implementation Style Guide

### Builder Core Extensions

**Naming Conventions:**
- Keep primitive names under `core.*` namespace
- Prefer verbs over nouns (`core.filter_edges`, not `core.edge_filter`)
- Use snake_case for step names
- Be explicit: `normalize_sum` over `normalize`

**Step Design:**
- Every step must declare: inputs, outputs, parameter schema, validation rules
- Steps should be composable and stateless
- Return data via variables, not by mutating shared state
- Provide parallel Python helper where it aids readability (e.g., `builder.filter_edges(...)`)
- Document each primitive in builder docs with accepted methods/options

**Example**:
```rust
pub struct NormalizeSumStep {
    input_var: String,
    output_var: String,
}

impl Step for NormalizeSumStep {
    fn id(&self) -> &'static str { "core.normalize_sum" }
    
    fn schema() -> StepSchema {
        StepSchema::new()
            .input("input_var", VarType::NodeData)
            .output("output_var", VarType::NodeData)
            .description("Normalize values to sum = 1.0")
    }
    
    fn execute(&self, ctx: &mut Context, sg: &Subgraph) -> GraphResult<StepOutput> {
        let values = ctx.get_var(&self.input_var)?;
        let sum: f64 = values.iter().sum();
        let normalized: Vec<f64> = values.iter().map(|v| v / sum).collect();
        Ok(StepOutput::NodeData(normalized))
    }
}
```

### Rust Algorithm Implementations

**Module Organization:**
- Place algorithms under `src/algorithms/<category>/`
- One file per algorithm: `src/algorithms/community/leiden.rs`
- Shared utilities in `<category>/mod.rs` or `<category>/utils.rs`

**Error Handling:**
- Return `GraphResult<T>` for all fallible operations
- Avoid `.unwrap()` in execution paths (use `?` operator)
- Provide context in errors: `context("Failed to compute modularity")?`
- Validate parameters in factory, not during execution

**Cancellation Support:**
- Check `ctx.is_cancelled()` in loops (every iteration or every N iterations)
- Return early with descriptive error: `Err(GraphError::Cancelled("LPA cancelled at iteration 42"))`

**Configuration:**
- Expose config via `AlgorithmParams` struct
- Use `expect_*` helpers for required parameters
- Provide sensible defaults
- Document parameter ranges and effects

**Testing:**
- Unit tests in same file (bottom) or `tests/` submodule
- Integration tests in `tests/` directory
- Benchmarks in `benches/<category>_algorithms.rs`

**Example**:
```rust
// src/algorithms/community/leiden.rs
use super::*;

pub struct Leiden {
    resolution: f64,
    iterations: usize,
    seed: Option<u64>,
}

impl Algorithm for Leiden {
    fn id(&self) -> &'static str { "community.leiden" }
    
    fn execute(&self, ctx: &mut Context, mut sg: Subgraph) -> GraphResult<Subgraph> {
        // Algorithm implementation
        for i in 0..self.iterations {
            if ctx.is_cancelled() {
                return Err(GraphError::cancelled("Leiden cancelled at iteration", i));
            }
            // ... iteration logic
        }
        Ok(sg)
    }
}

// Factory registration
pub fn register(registry: &mut Registry) {
    registry.register_factory("community.leiden", |params| {
        let resolution = params.get("resolution").unwrap_or(1.0);
        let iterations = params.get("iterations").unwrap_or(10);
        let seed = params.get("seed");
        
        Ok(Box::new(Leiden { resolution, iterations, seed }))
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_leiden_convergence() {
        // Test implementation
    }
}
```

### Python Algorithm Handles & Docs

**Factory Functions:**
- Expose via `groggy.algorithms.<category>` modules
- Use lowercase function names matching algorithm: `leiden()`, `eigenvector()`
- Return `RustAlgorithmHandle` or custom handle type

**Docstrings:**
- Follow NumPy docstring format
- Include: brief description, parameters (with types and defaults), return value, examples
- Document complexity where relevant
- Cross-reference related algorithms

**Parameter Handling:**
- Use keyword arguments with defaults
- Validate types early (in Python, before FFI call)
- Provide clear error messages for invalid parameters
- Support both dict and kwargs interfaces

**Example**:
```python
# python-groggy/python/groggy/algorithms/community.py
from .base import algorithm

def leiden(resolution: float = 1.0, iterations: int = 10, seed: Optional[int] = None):
    """
    Leiden algorithm for community detection.
    
    Leiden improves on Louvain by guaranteeing connected communities and
    faster convergence. Uses modularity optimization with quality function.
    
    Parameters
    ----------
    resolution : float, default=1.0
        Resolution parameter for modularity. Higher values result in smaller
        communities. Must be positive.
    iterations : int, default=10
        Maximum number of iterations. Algorithm may converge earlier.
        Must be in range [1, 1000].
    seed : int, optional
        Random seed for reproducibility. If None, uses system randomness.
    
    Returns
    -------
    Subgraph
        Input subgraph with 'community' attribute on nodes indicating
        community membership (integer labels).
    
    Examples
    --------
    >>> from groggy.algorithms.community import leiden
    >>> communities = sg.apply(leiden())
    >>> communities = sg.apply(leiden(resolution=1.5, iterations=20))
    
    See Also
    --------
    louvain : Similar algorithm without connectivity guarantee
    spectral : Spectral clustering alternative
    
    Notes
    -----
    Complexity: O(m) per iteration where m is number of edges.
    Typically converges in 10-20 iterations.
    """
    return algorithm(
        "community.leiden",
        defaults={"resolution": resolution, "iterations": iterations, "seed": seed},
    )
```

**Testing Python APIs:**
- Test factory function creates valid handle
- Test parameter validation (type errors, range errors)
- Test end-to-end execution on small graph
- Test docstring examples (doctests or explicit tests)

### Documentation Guidelines

**API Documentation:**
- All public items documented (modules, classes, functions, methods)
- Examples in every docstring
- Type hints in Python, type signatures in Rust
- Cross-references between related items

**Tutorials:**
- Start simple (single algorithm), build to complex (pipelines)
- Use realistic but small examples
- Include visualizations where helpful
- Provide complete, runnable code

**Performance Notes:**
- Document complexity (big-O) for all algorithms
- Note memory requirements for large graphs
- Provide tuning guidance (when to use what)
- Include benchmark results for reference

### Code Review Checklist

Before submitting:

**Rust:**
- [ ] `cargo fmt --all` passes
- [ ] `cargo clippy --all-targets -- -D warnings` passes
- [ ] `cargo test` passes
- [ ] New code has unit tests
- [ ] Complex algorithms have integration tests
- [ ] Benchmarks added for performance-critical code
- [ ] Docstrings complete with examples

**Python:**
- [ ] `black .` and `isort .` pass
- [ ] Type hints present and correct
- [ ] `pytest tests/` passes
- [ ] New APIs have test coverage
- [ ] Docstrings follow NumPy format
- [ ] Examples in docstrings are runnable

**Documentation:**
- [ ] README updated if needed
- [ ] API reference updated
- [ ] Migration notes if breaking changes
- [ ] CHANGELOG.md updated

**FFI:**
- [ ] No unsafe code without justification
- [ ] GIL released for expensive operations
- [ ] Error handling complete (no panics across FFI)
- [ ] Memory management correct (no leaks)

---
