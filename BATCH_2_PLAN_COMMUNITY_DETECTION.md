# Batch 2 Plan: Community Detection Optimization

**Scope**: Remaining community detection algorithms  
**Duration**: 2-3 days  
**Prerequisites**: ✅ Batch 1 complete (CSR utilities + pathfinding optimized)

---

## 📊 Current State Analysis

### Already Optimized (From Previous Work)

| Algorithm | Status | Performance | Notes |
|-----------|--------|-------------|-------|
| Louvain | ✅ Optimized | ~180ms @ 500n | Uses incremental modularity |
| Leiden | ✅ Optimized | ~23ms @ 500n | 2.1x speedup, uses same pattern as Louvain |
| Label Propagation | ✅ Partial | ~50ms | Has NodeIndexer, could use CSR |
| Connected Components | ✅ Optimized | ~10ms | Union-find with CSR |
| PageRank | ✅ Optimized | ~45ms | CSR-based power iteration |

### Needs Optimization (Batch 2 Targets)

| Algorithm | Current | Target | Est. Time | Priority |
|-----------|---------|--------|-----------|----------|
| **Girvan-Newman** | 2-5s | 1-2s | 1 day | **P0** - Most used, biggest pain |
| **Infomap** | ~300ms | ~150ms | 1 day | **P1** - Complex but high value |
| **LPA (full)** | ~50ms | ~20ms | 0.5 day | **P2** - Already decent |

---

## 🎯 Batch 2 Algorithms

### 1. Girvan-Newman (Day 1 - Priority 0)

**Current Performance**: 2-5 seconds @ 10K edges  
**Target Performance**: 1-2 seconds (2-3x speedup)  
**Complexity**: O(m²n) - inherently expensive, but optimizable

#### Problem Analysis

**Algorithm**: Iteratively remove edges with highest betweenness centrality

```rust
loop {
    // 1. Compute edge betweenness (ALL edges) - O(mn)
    let betweenness = compute_edge_betweenness(subgraph);
    
    // 2. Remove edge with max betweenness
    remove_max_edge();
    
    // 3. Check modularity / stopping criterion
    if should_stop() { break; }
}
```

**Current Bottlenecks**:
1. **Recomputes ALL edge betweenness each iteration** (no incremental updates)
2. **Uses trait-based `subgraph.neighbors()`** in BFS (slow)
3. **No CSR caching** between iterations
4. **Allocates HashMap per BFS call** (~1000 BFS per iteration!)

#### Optimization Strategy

**Phase 1: CSR-ify the BFS** (~2 hours)
- Build CSR once at start
- Use CSR-based BFS from Batch 1 utilities
- **Expected**: 2-3x speedup on betweenness computation

**Phase 2: Pre-allocate buffers** (~1 hour)
- Reuse distance arrays across all BFS calls
- Reuse betweenness arrays
- **Expected**: Additional 1.5x speedup

**Phase 3: STYLE_ALGO profiling** (~1 hour)
- Track time per iteration
- Track betweenness computation vs edge removal
- Identify remaining bottlenecks

**Total Expected**: 2-5s → **1-2s** (2-3x speedup)

#### Implementation Checklist

- [ ] Add CSR caching at algorithm start
- [ ] Replace BFS with CSR-based version
- [ ] Pre-allocate betweenness buffers
- [ ] Pre-allocate distance buffers for BFS
- [ ] Add STYLE_ALGO profiling:
  - `girvan_newman.collect_nodes`
  - `girvan_newman.build_csr`
  - `girvan_newman.iteration_{i}` (per iteration timing)
  - `girvan_newman.compute_betweenness`
  - `girvan_newman.remove_edge`
  - `girvan_newman.compute_modularity`
  - `girvan_newman.total_execution`
- [ ] Add test for correctness
- [ ] Benchmark and validate speedup

---

### 2. Infomap (Day 2 - Priority 1)

**Current Performance**: ~300ms @ medium graph  
**Target Performance**: ~150ms (2x speedup)  
**Complexity**: O(n log n) with many iterations

#### Problem Analysis

**Algorithm**: Information-theoretic community detection via random walks

```rust
loop {
    // 1. Compute flow probabilities (random walk)
    let flow = compute_flow(subgraph);
    
    // 2. Compute map equation (compression)
    let map_eq = compute_map_equation(flow);
    
    // 3. Move nodes to optimize map equation
    optimize_partitions();
    
    if converged() { break; }
}
```

**Current Bottlenecks**:
1. **Random walk uses trait-based neighbors** (slow)
2. **Allocates flow map every iteration**
3. **No CSR optimization**
4. **Recomputes degrees frequently**

#### Optimization Strategy

**Phase 1: CSR-ify random walks** (~2 hours)
- Build CSR once
- Use CSR for O(1) neighbor access
- **Expected**: 1.5-2x speedup

**Phase 2: Pre-allocate flow buffers** (~1 hour)
- Reuse flow probability arrays
- Cache degree computations
- **Expected**: Additional 1.3x speedup

**Phase 3: STYLE_ALGO profiling** (~1 hour)
- Track iteration timing
- Identify bottlenecks
- Add comprehensive instrumentation

**Total Expected**: 300ms → **150ms** (2x speedup)

#### Implementation Checklist

- [ ] Add CSR caching
- [ ] Replace random walk neighbor access with CSR
- [ ] Pre-allocate flow arrays
- [ ] Cache degree computations
- [ ] Add STYLE_ALGO profiling:
  - `infomap.collect_nodes`
  - `infomap.build_csr`
  - `infomap.iteration_{i}`
  - `infomap.compute_flow`
  - `infomap.compute_map_equation`
  - `infomap.optimize_partitions`
  - `infomap.total_execution`
- [ ] Add tests
- [ ] Benchmark and validate

---

### 3. Label Propagation (LPA) - Full Optimization (Day 3 - Priority 2)

**Current Performance**: ~50ms @ medium graph  
**Target Performance**: ~20ms (2.5x speedup)  
**Complexity**: O(m) per iteration, few iterations

#### Problem Analysis

**Current State**: Already has `NodeIndexer`, partially optimized

```rust
// Already has:
let indexer = NodeIndexer::new(&nodes);

// But still uses trait-based neighbors:
for neighbor in subgraph.neighbors(node)? {
    // ...
}
```

**Current Bottlenecks**:
1. **Uses trait-based neighbors** (could use CSR)
2. **No profiling** to identify specific bottlenecks
3. **Allocates label count map per iteration**

#### Optimization Strategy

**Phase 1: CSR-ify neighbor access** (~1 hour)
- Build CSR once
- Use CSR neighbors in label propagation
- **Expected**: 1.5-2x speedup

**Phase 2: Pre-allocate label buffers** (~30 min)
- Reuse label count arrays
- **Expected**: Additional 1.3x speedup

**Phase 3: STYLE_ALGO profiling** (~30 min)
- Add comprehensive instrumentation

**Total Expected**: 50ms → **20ms** (2.5x speedup)

#### Implementation Checklist

- [ ] Add CSR caching (use existing indexer)
- [ ] Replace neighbor iteration with CSR
- [ ] Pre-allocate label count arrays
- [ ] Add STYLE_ALGO profiling:
  - `lpa.collect_nodes`
  - `lpa.build_csr` (or cache hit)
  - `lpa.iteration_{i}`
  - `lpa.propagate_labels`
  - `lpa.total_execution`
- [ ] Benchmark and validate

---

## 📅 Execution Plan

### Day 1: Girvan-Newman (8 hours)
- **Morning** (4h): CSR-ify BFS + pre-allocate buffers
- **Afternoon** (3h): STYLE_ALGO profiling + testing
- **End of day** (1h): Benchmark and validate
- **Deliverable**: 2-5s → 1-2s (2-3x speedup)

### Day 2: Infomap (8 hours)
- **Morning** (4h): CSR-ify random walks + pre-allocate
- **Afternoon** (3h): STYLE_ALGO profiling + testing
- **End of day** (1h): Benchmark and validate
- **Deliverable**: 300ms → 150ms (2x speedup)

### Day 3: LPA Full Optimization (4 hours)
- **Morning** (2h): CSR neighbor access + pre-allocate
- **Late morning** (1.5h): STYLE_ALGO profiling + testing
- **Afternoon** (0.5h): Benchmark and validate
- **Deliverable**: 50ms → 20ms (2.5x speedup)

### Day 3 Afternoon: Documentation & Wrap-up (4 hours)
- Create BATCH_2_COMPLETE_SUMMARY.md
- Update progress tracker
- Run full benchmark suite
- Prepare for Batch 3 (validation)

---

## 📊 Expected Outcomes

### Performance Improvements

| Algorithm | Before | After | Speedup | Time Saved |
|-----------|--------|-------|---------|------------|
| Girvan-Newman | 2-5s | 1-2s | 2-3x | ~2-3s |
| Infomap | 300ms | 150ms | 2x | ~150ms |
| LPA | 50ms | 20ms | 2.5x | ~30ms |
| **TOTAL** | **~3-5.5s** | **~1.2-2.2s** | **2.5x avg** | **~2-3.5s** |

### Code Changes

**Estimated additions**:
- Girvan-Newman: +100 lines (CSR + buffers + profiling)
- Infomap: +80 lines (CSR + buffers + profiling)
- LPA: +40 lines (CSR + profiling)
- **Total**: ~220 new lines

### Documentation

**New files**:
- BATCH_2_COMPLETE_SUMMARY.md
- GIRVAN_NEWMAN_OPTIMIZATION.md
- INFOMAP_OPTIMIZATION.md
- LPA_FULL_OPTIMIZATION.md

---

## 🔧 Technical Patterns

### Pattern 1: CSR-Based Edge Betweenness (Girvan-Newman)

```rust
// Build CSR once
let csr = build_csr_cached(subgraph);

// Pre-allocate buffers
let mut betweenness = vec![0.0; edges.len()];
let mut distances = vec![usize::MAX; n];
let mut queue = VecDeque::with_capacity(n);

// Compute for all sources
for source_idx in 0..n {
    // Reuse buffers!
    bfs_layers_csr(&csr, &nodes, source_idx, &mut distances, &mut queue);
    accumulate_betweenness(&distances, &mut betweenness);
}
```

### Pattern 2: CSR-Based Random Walk (Infomap)

```rust
// Build CSR once
let csr = build_csr_cached(subgraph);

// Pre-allocate flow arrays
let mut flow = vec![0.0; n];
let mut visit_counts = vec![0; n];

// Random walk with CSR neighbors
for step in 0..num_steps {
    let neighbors = csr.neighbors(current_idx);
    // O(1) access!
}
```

### Pattern 3: CSR-Based Label Propagation (LPA)

```rust
// Build CSR once (reuse existing indexer)
let csr = build_csr_cached(subgraph);

// Pre-allocate label counts
let mut label_counts = vec![0; max_label];

// Propagate with CSR
for node_idx in 0..n {
    label_counts.clear();
    for &neighbor_idx in csr.neighbors(node_idx) {
        label_counts[labels[neighbor_idx]] += 1;
    }
    labels[node_idx] = most_frequent_label(&label_counts);
}
```

---

## ⚠️ Risks & Mitigations

### Risk 1: Girvan-Newman Still Too Slow
**Risk**: O(m²n) complexity might not improve enough  
**Mitigation**: Set clear expectations (2-3x is realistic, not 10x)  
**Fallback**: Document that it's inherently slow, recommend for small graphs only

### Risk 2: Infomap Convergence Issues
**Risk**: Random walks might behave differently with CSR  
**Mitigation**: Extensive testing, ensure random walk logic unchanged  
**Fallback**: Add toggle for CSR vs trait-based

### Risk 3: Time Overrun
**Risk**: 2-3 days might not be enough  
**Mitigation**: Prioritize Girvan-Newman (biggest user pain), others optional  
**Fallback**: Ship partial Batch 2 if needed

---

## ✅ Success Criteria

- [ ] All 3 algorithms optimized with CSR
- [ ] 2-3x aggregate speedup achieved
- [ ] STYLE_ALGO profiling added to all
- [ ] All tests passing (no regressions)
- [ ] Comprehensive documentation
- [ ] Benchmark validation complete

---

## 🚀 After Batch 2

### Batch 3: Validation & Documentation (1 day)
- Run full benchmark suite
- Update CHANGELOG for v0.6
- Create Performance Tuning Guide
- Update roadmap with new baselines

### Future Work
- New algorithms using established patterns
- Step primitives optimization (if needed)
- User-requested features based on v0.6 feedback

---

**Ready to start?** Let's begin with Girvan-Newman optimization!
