# Batch 2 Revised Plan: Skip Girvan-Newman

**Decision**: Skip Girvan-Newman optimization in Batch 2

---

## 🔍 Why Skip Girvan-Newman?

### Complexity Analysis

**Problem**: Girvan-Newman uses **dynamic edge removal**:
```rust
loop {
    // 1. Compute edge betweenness on CURRENT graph state
    let betweenness = compute_edge_betweenness(active_edges);
    
    // 2. Remove edge with max betweenness
    active_edges.remove(max_edge);
    
    // 3. Repeat until stopping criterion
}
```

**Why CSR doesn't fit**:
- CSR requires **fixed graph topology**
- Girvan-Newman **mutates topology every iteration**
- Would need to rebuild CSR ~100-1000 times
- Rebuilding CSR (10ms) × 1000 iterations = **10+ seconds overhead**

### Alternative Optimizations (Complex)

Could optimize individual BFS calls, but:
1. **Array-based storage** instead of HashMap (2x speedup)
2. **Pre-allocated buffers** for distance/sigma (1.5x speedup)
3. **Total**: ~3x speedup, but **3-5 days of work**

**ROI**: Not worth it for Batch 2

---

## 🎯 Revised Batch 2: Focus on LPA & Infomap

### New Priority Order

1. **Label Propagation (LPA)** - Day 1 (4 hours)
   - Already partially optimized
   - Easy CSR conversion
   - **Expected**: 50ms → 20ms (2.5x speedup)
   
2. **Infomap** - Day 2 (8 hours)
   - Random walks can use CSR
   - More complex but valuable
   - **Expected**: 300ms → 150ms (2x speedup)

3. **Girvan-Newman** - Future (if needed)
   - Defer to Batch 4 or later
   - Or accept current performance (it's O(m²n) anyway)

---

## 📅 Revised Execution Plan

### Day 1: Label Propagation (4 hours)
- **Morning** (2h): Add CSR caching + neighbor iteration
- **Late morning** (1.5h): Pre-allocate buffers + profiling
- **Afternoon** (0.5h): Test and validate
- **Deliverable**: 50ms → 20ms (2.5x speedup)

### Day 1 Afternoon: Infomap Setup (4 hours)
- **Afternoon** (2h): Analyze Infomap structure
- **Late afternoon** (2h): Start CSR integration
- **Deliverable**: Foundation for Day 2

### Day 2: Infomap Completion (8 hours)
- **Morning** (4h): Finish CSR random walks + pre-allocate
- **Afternoon** (3h): STYLE_ALGO profiling + testing
- **End of day** (1h): Benchmark and validate
- **Deliverable**: 300ms → 150ms (2x speedup)

### Day 2 Evening: Documentation (2 hours)
- BATCH_2_COMPLETE_SUMMARY.md
- LPA_OPTIMIZATION_COMPLETE.md
- INFOMAP_OPTIMIZATION_COMPLETE.md
- GIRVAN_NEWMAN_DEFERRED.md (explaining why)

---

## 📊 Revised Expected Outcomes

### Performance Improvements

| Algorithm | Before | After | Speedup | Time Saved |
|-----------|--------|-------|---------|------------|
| LPA | 50ms | 20ms | 2.5x | ~30ms |
| Infomap | 300ms | 150ms | 2x | ~150ms |
| **TOTAL** | **~350ms** | **~170ms** | **2.05x** | **~180ms** |

**Note**: Girvan-Newman deferred (2-5s stays 2-5s for now)

### Adjusted Success Metrics

- ✅ **2 of 3** algorithms optimized (LPA, Infomap)
- ✅ **2x aggregate speedup** (vs 2.5x originally)
- ✅ **~180ms saved** per run
- ✅ **2 days** instead of 3 (faster delivery)
- ⚠️ Girvan-Newman deferred to future work

---

## 🎯 Start with LPA Now?

**Recommended**: Yes - LPA is quick win (4 hours), then tackle Infomap.

**Alternative**: Document Girvan-Newman complexity, ship Batch 1, and defer all Batch 2 to later.

What would you like to do?
