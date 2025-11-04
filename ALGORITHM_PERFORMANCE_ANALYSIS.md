# Algorithm Performance Analysis & Optimization Plan

## Summary

Benchmark testing reveals that while groggy's PageRank is competitive, several algorithms need significant performance optimization to match industry-leading libraries like igraph and NetworKit.

## Benchmark Results (500 nodes, 2,500 edges)

### ✅ Algorithms That Are Competitive

**PageRank**
- groggy: 0.0035s
- networkx: 0.0027s (1.3x faster)
- **Status**: ✅ Competitive performance, within 30% of NetworkX

### ⚠️ Algorithms That Need Optimization

**Connected Components** 
- NetworKit: 0.0000s (baseline - extremely fast)
- igraph: 0.0001s (2.8x slower)
- networkx: 0.0002s (7.4x slower)  
- **groggy: 0.0016s (74.8x slower than NetworKit)**
- **Issue**: Union-find or BFS implementation likely not optimized

**Betweenness Centrality**
- igraph: 0.0130s (baseline)
- NetworKit: 0.0192s (1.5x slower)
- **groggy: 0.2536s (19.5x slower than igraph)**
- networkx: 0.4993s (38.5x slower)
- **Issue**: Brandes' algorithm implementation needs optimization

**Label Propagation (LPA)**
- NetworKit: 0.0004s (baseline)
- igraph: 0.0007s (1.8x slower)
- **groggy: 0.0026s (7x slower than NetworKit)**
- networkx: 0.0079s (21x slower)
- **Issue**: Label propagation iteration not efficient

## Performance Gaps to Close

| Algorithm | Current (groggy) | Target (best competitor) | Speedup Needed |
|-----------|-----------------|--------------------------|----------------|
| Connected Components | 0.0016s | 0.0000s (NetworKit) | **74.8x** |
| Betweenness | 0.2536s | 0.0130s (igraph) | **19.5x** |
| Label Propagation | 0.0026s | 0.0004s (NetworKit) | **7.0x** |
| PageRank | 0.0035s | 0.0027s (networkx) | ✅ 1.3x (acceptable) |

## Root Cause Analysis

### Connected Components - Needs 75x Speedup
**Likely issues:**
1. Not using optimized union-find data structure
2. Possibly creating intermediate subgraphs/views unnecessarily
3. May be doing extra attribute allocations during traversal

**Best practice (NetworKit/igraph):**
- Union-find with path compression and union-by-rank
- Single-pass over edges
- No intermediate allocations

### Betweenness Centrality - Needs 20x Speedup  
**Likely issues:**
1. Brandes' algorithm not fully optimized
2. Shortest path computations may be redundant
3. Dependency accumulation phase may have overhead

**Best practice (igraph):**
- Optimized Brandes' algorithm with efficient queue
- Vectorized dependency accumulation
- Minimal allocations during BFS

### Label Propagation - Needs 7x Speedup
**Likely issues:**
1. Label update phase doing unnecessary copies
2. Neighbor iteration not optimized
3. Random selection of neighbors may be inefficient

**Best practice (NetworKit):**
- In-place label updates
- Pre-allocated arrays for label tracking
- Fast neighbor access

## Optimization Strategy

### Phase 1: Profile & Identify Bottlenecks (Priority 1)
1. Run Rust profiler on each slow algorithm
2. Identify hot spots in the code
3. Measure memory allocations during execution

### Phase 2: Algorithm-Specific Optimizations

#### Connected Components (Highest Priority - 75x gap)
- [ ] Implement union-find with path compression
- [ ] Eliminate any intermediate subgraph creations
- [ ] Single-pass edge iteration
- [ ] Pre-allocate component ID array

#### Betweenness Centrality (High Priority - 20x gap)
- [ ] Review Brandes' algorithm implementation
- [ ] Optimize BFS phase with efficient queue
- [ ] Vectorize dependency accumulation
- [ ] Reduce allocations in shortest path tracking

#### Label Propagation (Medium Priority - 7x gap)
- [ ] In-place label updates
- [ ] Optimize neighbor iteration
- [ ] Pre-allocate label arrays
- [ ] Fast random neighbor selection

### Phase 3: Verify Improvements
- [ ] Re-run benchmarks after each optimization
- [ ] Target: Within 2-3x of best competitor
- [ ] Document any algorithmic trade-offs

## Next Steps

1. **Immediate**: Fix API compatibility issues in benchmark (PageRank params) ✅ DONE
2. **Next**: Profile Connected Components to find the 75x bottleneck
3. **Then**: Implement union-find optimization
4. **Then**: Move to Betweenness and LPA

## Notes

- PageRank is already competitive, showing that groggy's core iteration framework works well
- The gaps are in specific algorithm implementations, not the overall architecture
- NetworKit and igraph are heavily optimized C++ libraries - getting within 2-3x would be excellent
- Some algorithms may trade off features for performance (we should document this)
