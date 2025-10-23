# Algorithms API Reference

Groggy's algorithm registry is exposed through the `groggy.algorithms` package. Algorithms are
represented as lightweight handles that describe the Rust implementation to run.

## Structure

```python
import groggy.algorithms as alg

pagerank = alg.centrality.pagerank(max_iter=50, output_attr="pr")
bfs = alg.pathfinding.bfs(start_attr="is_start", output_attr="dist")
lpa = alg.community.lpa(output_attr="community")
```

Each handle implements `AlgorithmHandle` and can be:

- Passed to `Subgraph.apply(handle)`
- Included in the `apply(subgraph, [...])` helper
- Added to a `Pipeline` (or builder-generated pipeline)

## Modules

- `groggy.algorithms.centrality`
- `groggy.algorithms.community`
- `groggy.algorithms.pathfinding`

Every function returns a configured `RustAlgorithmHandle`. See the module docstrings for supported
parameters and defaults.

## Metadata & Discovery

```python
import groggy.algorithms as alg

for algo_id in alg.list():
    print(algo_id)

info = alg.info("centrality.pagerank")
print(info["description"])

results = alg.search("community")
print(results)
```

The discovery APIs surface metadata sourced directly from the Rust registry.

## Related References

- [Pipeline API](pipeline.md)
- [Builder DSL](builder.md)
- [Algorithm Guide](../guide/algorithms.md)
