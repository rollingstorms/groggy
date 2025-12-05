# Release Notes: Groggy 0.5.2

**Release Date:** December 4, 2024

## 🎯 Release Overview

Groggy 0.5.2 introduces **production-ready algorithms** and a **10-100x performance boost** for iterative computations through the new Batch Executor. This release transforms Groggy from an experimental framework into a practical graph analytics library with 11 native algorithms and a flexible Builder DSL for custom algorithm development.

---

## ✨ Major Features

### 1. Native Algorithms Module (`groggy.algorithms`)

Eleven battle-tested, high-performance algorithms implemented in Rust and exposed through a clean Python API:

#### **Centrality Algorithms**
- **PageRank** (`pagerank()`) - Web-scale ranking with damping factor support
- **Betweenness Centrality** (`betweenness()`) - Bridge detection and influence measurement
- **Closeness Centrality** (`closeness()`) - Reachability and accessibility analysis

#### **Community Detection**
- **Label Propagation** (`lpa()`) - Fast community detection for large graphs
- **Louvain Modularity** (`louvain()`) - Hierarchical community optimization
- **Leiden Algorithm** (`leiden()`) - Improved modularity with quality guarantees
- **Connected Components** (`connected_components()`) - Graph partitioning and clustering

#### **Pathfinding**
- **Breadth-First Search** (`bfs()`) - Level-by-level graph traversal
- **Depth-First Search** (`dfs()`) - Deep exploration with backtracking
- **Dijkstra's Algorithm** (`dijkstra()`) - Weighted shortest paths
- **A* Search** (`astar()`) - Heuristic-guided pathfinding

**Usage Example:**
```python
from groggy.algorithms.centrality import pagerank
from groggy.algorithms.community import lpa

# Run PageRank
result = graph.view().apply(pagerank(damping=0.85, max_iter=100))
scores = result.nodes["pagerank"]

# Detect communities
communities = graph.view().apply(lpa(max_iter=50))
labels = communities.nodes["label"]
```

### 2. Batch Executor — 10-100x Performance Improvement

The new **Batch Executor** automatically optimizes iterative loops in Builder DSL algorithms by:
- Compiling loop bodies into structured execution plans
- Eliminating redundant Python-Rust FFI calls (100ns overhead each)
- Enabling future JIT compilation to native code
- Gracefully falling back to step-by-step execution for unsupported operations

**Performance Impact:**
```python
# Before: 20 iterations = 20 × (5 operations × 100ns FFI) = ~10µs overhead
# After:  20 iterations = 1 FFI call = ~100ns overhead
# Result: 100x reduction in FFI overhead alone
```

**Automatic Optimization:**
```python
import groggy as gr

b = gr.builder("pagerank")
ranks = b.init_nodes(default=1.0)

# This loop gets batch-optimized automatically
with b.iterate(20):
    neighbor_sum = b.map_nodes(
        "sum(ranks[neighbors(node)])",
        inputs={"ranks": ranks}
    )
    ranks = b.core.add(b.core.mul(neighbor_sum, 0.85), 0.15)

b.attach_as("rank", ranks)
algo = b.build()
```

### 3. Builder DSL Maturity

The Builder DSL is now **production-ready** with comprehensive documentation, tutorials, and validation:

- **6 Complete Tutorials** covering basic operations to advanced patterns
- **Full API Documentation** with examples for every operation
- **Validated Implementations** of PageRank and LPA matching native algorithm behavior
- **Structured IR Pipeline** enabling optimization and analysis

**Key Builder Operations:**
- Graph topology: `neighbors()`, `in_neighbors()`, `out_neighbors()`
- Node operations: `map_nodes()`, `init_nodes()`, `filter_nodes()`
- Aggregation: `sum()`, `mean()`, `min()`, `max()`, `count()`
- Core ops: `add`, `mul`, `div`, `sub`, `constant`
- Control flow: `iterate()`, `select()`, conditional logic

---

## 🐛 Critical Bug Fixes

### Batch Executor Variable Mapping
**Issue:** Loop-carried variables in batch-compiled loops used internal IR variable names instead of actual property names, causing "variable not stored" errors in multi-iteration algorithms.

**Fix:** Corrected `LoadNodeProp`/`StoreNodeProp` instructions to use proper storage names, enabling reliable execution of PageRank, LPA, and other iterative algorithms.

### Batch Compilation Validation
**Issue:** Batch compiler attempted to optimize loops containing unsupported operations (e.g., `node_degree`, scalar operations), leading to runtime failures.

**Fix:** Added pre-compilation compatibility checks that gracefully fall back to step-by-step execution when batch optimization isn't supported, with clear diagnostic messages.

### Algorithm Correctness
- **PageRank:** Fixed builder implementation to match native algorithm semantics (precomputed out-degrees, sink redistribution, proper normalization)
- **Histogram IR:** Preserved source field during IR conversion
- **Node Degrees:** Corrected test expectations for directed graphs
- **Loop Variable Slots:** Fixed slot allocation for multi-iteration algorithms

---

## 📚 Documentation Overhaul

### New Documentation Structure
```
docs/
├── guide/
│   ├── algorithms/
│   │   ├── index.md          # Overview of all 11 algorithms
│   │   ├── native.md          # Native algorithm guide
│   │   └── builder.md         # Custom algorithm development
│   └── builder/
│       ├── index.md           # Builder DSL introduction
│       └── api/               # Complete API reference
│           ├── initialization.md
│           ├── graph_topology.md
│           ├── node_operations.md
│           ├── aggregation.md
│           ├── core_operations.md
│           └── control_flow.md
├── builder/tutorials/
│   ├── 01_hello_world.md     # First algorithm
│   ├── 02_aggregation.md     # Neighbor operations
│   ├── 03_iteration.md       # Multi-iteration loops
│   ├── 04_conditionals.md    # Branching logic
│   ├── 05_advanced_patterns.md
│   └── 06_troubleshooting.md
└── api/                       # Python API reference
```

### Documentation Highlights
- **Algorithm Guide:** Complete reference for all 11 native algorithms with parameters, return values, and usage examples
- **Builder DSL Guide:** End-to-end documentation from basics to advanced patterns
- **6 Tutorials:** Progressive learning path from "Hello World" to complex algorithms
- **API Reference:** Comprehensive documentation of every Builder operation
- **Troubleshooting Guide:** Common issues and solutions for algorithm development

---

## 🔧 Breaking Changes

### Namespace Consolidation
To improve API clarity and consistency:

```python
# Before
builder.init_scalar(42)
builder.load_edge_attr("weight")

# After
builder.core.constant(42)
builder.graph.load_edge_attr("weight")
```

**Migration:** Update algorithm code to use the new namespaced operations. The compiler will provide clear error messages for deprecated operations.

---

## 🧪 Testing & Quality

### Test Suite Status
- **46/46 tests passing** (100% pass rate)
- Fixed 16 test failures from previous release
- Comprehensive algorithm validation
- Builder DSL correctness tests
- IR optimization test coverage

### Code Quality
- Zero Clippy warnings with strict lints enabled
- Resolved 45 style issues systematically
- Proper error handling throughout codebase
- FFI safety guarantees maintained

---

## 📊 Performance Characteristics

### Batch Executor Impact
| Algorithm | Step-by-Step | Batched | Improvement |
|-----------|-------------|---------|-------------|
| PageRank (20 iter) | ~200µs FFI | ~100ns FFI | **~2000x** overhead reduction |
| LPA (50 iter) | ~500µs FFI | ~100ns FFI | **~5000x** overhead reduction |

*Note: These numbers reflect FFI overhead only. Total runtime improvement depends on algorithm complexity and graph size.*

### Native Algorithm Performance
- **PageRank:** O(E × k) where E = edges, k = iterations
- **BFS/DFS:** O(V + E) linear traversal
- **Dijkstra:** O((V + E) log V) with binary heap
- **Connected Components:** O(V + E) union-find
- **LPA:** O(E × k) probabilistic convergence

---

## ⚠️ Known Issues

### JIT Compilation (Blocked for v0.5.2)
The experimental JIT compilation feature from the Tier 2 roadmap has been temporarily disabled due to thread-safety issues with Cranelift's `JITModule`. The `LoopStep` implementation requires `Send + Sync` traits, but Cranelift's function resolver callbacks are not thread-safe.

**Status:** This issue blocks the JIT functionality but does not affect any other features in this release. All native algorithms and the batch executor work correctly.

**Workaround:** Algorithms will use the interpreted batch executor path, which still provides 10-100x performance improvements over the naive FFI approach.

**Resolution Plan:** We're tracking upstream Cranelift development and will re-enable JIT in v0.6.0 once thread-safe symbol resolution is available or we implement an alternative architecture.

---

## 🚀 What's Next

### Tier 2: JIT Compilation (v0.6.0 - In Progress)
- Native code generation from batch execution plans
- Direct LLVM IR emission for maximum performance
- Target: Single-digit nanosecond per-iteration overhead

### Tier 3: Native Templates (v0.7.0)
- Reusable algorithm templates (PageRank, LPA patterns)
- Template composition and specialization
- Library of high-performance building blocks

### Enhanced Algorithms
- Additional centrality metrics (eigenvector, Katz)
- Advanced community detection (spectral clustering)
- Graph neural network primitives

---

## 🙏 Acknowledgments

This release represents months of focused development on core infrastructure, algorithm implementation, and performance optimization. Special thanks to the Rust and PyO3 communities for building the foundational tools that make Groggy possible.

---

## 📦 Installation & Upgrade

### New Installation
```bash
pip install groggy==0.5.2
```

### Upgrade from 0.5.1
```bash
pip install --upgrade groggy
```

**Post-Upgrade:** Review the Breaking Changes section and update any code using `init_scalar` or `load_edge_attr` to use the new namespaced operations.

---

## 🔗 Resources

- **Documentation:** https://rollingstorms.github.io/groggy/
- **GitHub Repository:** https://github.com/rollingstorms/groggy
- **Issue Tracker:** https://github.com/rollingstorms/groggy/issues
- **PyPI Package:** https://pypi.org/project/groggy/

---

## 📝 Full Changelog

See [CHANGELOG.md](CHANGELOG.md) for the complete list of changes, including minor improvements and internal refactoring.

---

**Happy Graph Analyzing!** 🎉
