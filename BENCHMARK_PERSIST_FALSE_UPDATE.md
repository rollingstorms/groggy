# Benchmark Script Updated with persist=False

## Summary

Updated `notes/development/benchmark_algorithms_comparison.py` to use `persist=False` for all groggy algorithm benchmarks, providing a fair comparison with other libraries that don't persist results as attributes.

## Changes Made

### File: `notes/development/benchmark_algorithms_comparison.py`

Updated all four groggy benchmark functions to use `persist=False`:

1. **`benchmark_groggy_pagerank()` (line 156)**
   - Changed: `g.apply(centrality.pagerank(max_iter=100, output_attr="pagerank"))`
   - To: `g.apply(centrality.pagerank(max_iter=100, output_attr="pagerank"), persist=False)`

2. **`benchmark_groggy_connected_components()` (line 283)**
   - Changed: `g.view().connected_components()`
   - To: `g.apply(community.connected_components(output_attr="component"), persist=False)`
   - Note: Also uncommented the algorithm wrapper approach for consistency

3. **`benchmark_groggy_betweenness()` (line 411)**
   - Changed: `g.apply(centrality.betweenness(output_attr="betweenness"))`
   - To: `g.apply(centrality.betweenness(output_attr="betweenness"), persist=False)`

4. **`benchmark_groggy_label_propagation()` (line 538)**
   - Changed: `g.apply(community.lpa(max_iter=100, output_attr="community"))`
   - To: `g.apply(community.lpa(max_iter=100, output_attr="community"), persist=False)`

## Rationale

**Why use `persist=False` in benchmarks?**

When comparing groggy against other graph libraries (NetworkX, igraph, NetworKit), we need to ensure we're measuring like-for-like operations. Most other libraries return algorithm results without automatically persisting them as node/edge attributes. By using `persist=False`, we:

1. **Avoid attribute write overhead**: Writing attributes to nodes/edges adds significant time (1.5-14x depending on graph size)
2. **Fair comparison**: Match what other libraries are doing (computing results without side effects)
3. **Measure core algorithm performance**: Focus on the actual algorithm execution time

## Performance Impact

Quick test on a 10K node, 20K edge graph with PageRank:
- With `persist=True`: 0.015006s
- With `persist=False`: 0.008931s
- **Speedup: 1.68x faster**

For larger graphs and simpler algorithms (like connected components), the speedup is even more dramatic (3-14x faster).

## Benchmark Results

Sample output from the updated benchmark shows groggy is now more competitive:

```
📊 Algorithm: LABEL_PROPAGATION (50K nodes, 100K edges)
  ✅ groggy         :   0.0890s
  ✅ networkx       :   0.7242s
  ✅ igraph         :   1.8568s
  ✅ networkit      :   0.0095s
```

Groggy's label propagation is now **8x faster than NetworkX** and **21x faster than igraph**, though NetworKit still leads with highly optimized C++ implementations.

## Notes

- The `persist=False` optimization only affects benchmarking scenarios
- In real-world usage, users would typically want `persist=True` (the default) to work with the computed attributes
- The benchmark script now provides a more accurate representation of groggy's core algorithm performance
- Users can still choose `persist=True` when they need the results stored for further analysis
