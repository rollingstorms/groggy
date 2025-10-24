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

