# Development Notes & Guides

This directory contains development documentation, style guides, and performance tuning resources.

---

## 📚 Quick Navigation

### Core Style & Architecture

**[STYLE_ALGO.md](STYLE_ALGO.md)** – Canonical algorithm implementation pattern  
Pseudo-code style guide for all new algorithms. Covers CSR usage, profiling, buffer reuse, and the standard algorithm structure. **Start here** if implementing a new algorithm.

**[PERFORMANCE_TUNING_GUIDE.md](PERFORMANCE_TUNING_GUIDE.md)** – Performance optimization handbook  
Detailed guide for algorithm authors covering cache locality, allocation strategies, common patterns, and debugging tips. Reference this when optimizing existing code.

**[REFACTOR_PLAN_PERFORMANCE_STYLE.md](REFACTOR_PLAN_PERFORMANCE_STYLE.md)** – Refactoring roadmap  
Tracks the campaign to apply STYLE_ALGO across all algorithms. Includes status, priorities, checklists, anti-patterns, and case studies.

### Planning & Status

**[../planning/ALGORITHM_REFACTORING_SUMMARY.md](../planning/ALGORITHM_REFACTORING_SUMMARY.md)** – High-level summary  
Consolidated view of current state, performance budgets, refactoring priorities, and success metrics. Best single-page overview.

**[../planning/advanced-algorithm-roadmap.md](../planning/advanced-algorithm-roadmap.md)** – Strategic roadmap  
Long-term vision for algorithm expansion, phase-by-phase breakdown, and architectural foundations.

**Phase Files**:
- [PHASE_1_BUILDER_CORE.md](../planning/advanced-algorithms/PHASE_1_BUILDER_CORE.md) – Step primitives (✅ complete)
- [PHASE_2_COMMUNITY.md](../planning/advanced-algorithms/PHASE_2_COMMUNITY.md) – Community detection algorithms
- [PHASE_3_CENTRALITY.md](../planning/advanced-algorithms/PHASE_3_CENTRALITY.md) – Centrality measures
- [PHASE_4_PATHFINDING.md](../planning/advanced-algorithms/PHASE_4_PATHFINDING.md) – Shortest path algorithms

### Benchmarks & Validation

**[benchmark_algorithms_comparison.py](benchmark_algorithms_comparison.py)** – Performance comparison suite  
Benchmarks Groggy against NetworkX, igraph, and NetworKit on standard algorithms. Run after any optimization work.

---

## 🚀 Common Workflows

### Implementing a New Algorithm

1. Read **STYLE_ALGO.md** for the canonical pattern
2. Study a reference implementation (e.g., `src/algorithms/community/components.rs`)
3. Implement core algorithm following the pattern
4. Add comprehensive tests (unit + integration)
5. Run benchmark to validate performance
6. Check **PERFORMANCE_TUNING_GUIDE.md** if optimization needed
7. Submit PR with benchmark results

### Optimizing an Existing Algorithm

1. Read **REFACTOR_PLAN_PERFORMANCE_STYLE.md** for current status
2. Run benchmark to establish baseline
3. Apply STYLE_ALGO refactoring checklist
4. Re-run benchmark to verify speedup
5. Check profiling report for coverage
6. Update status in refactoring plan

### Understanding Performance Architecture

1. Start with **ALGORITHM_REFACTORING_SUMMARY.md** for overview
2. Read STYLE_ALGO.md for pattern details
3. Study reference implementations (connected_components, pagerank, betweenness)
4. Review **PERFORMANCE_TUNING_GUIDE.md** for specific patterns

---

## 📊 Current Performance (v0.6)

**Optimized Algorithms** (following STYLE_ALGO):
- Connected Components: 30ms @ 200K nodes
- PageRank: 45ms @ 200K nodes
- LPA: 250ms @ 200K nodes
- Louvain: 180ms @ 200K nodes
- Betweenness: 800ms @ 200K nodes

**In Progress** (need refactoring):
- Closeness, Dijkstra, BFS/DFS, A*, Leiden, Infomap, Girvan-Newman

See **ALGORITHM_REFACTORING_SUMMARY.md** for full details.

---

## 🎓 Key Concepts

**CSR (Compressed Sparse Row)**: Cache-friendly graph representation with O(1) neighbor access  
**Node Indexer**: Efficient NodeId → usize mapping for CSR indices  
**Buffer Swap Pattern**: Pre-allocate, swap pointers instead of reallocating  
**Profiling Instrumentation**: Hierarchical timers/counters for performance visibility  
**Deterministic Ordering**: `ordered_nodes()`/`ordered_edges()` for reproducibility

---

**Maintainer**: Performance & Architecture Team  
**Last Updated**: 2024  
**Questions?** See individual guides or ask in PR review.
