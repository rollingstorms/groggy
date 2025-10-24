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

