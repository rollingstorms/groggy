# Pipeline API Reference

The pipeline module lives at `groggy.pipeline` and exposes two primary entry points:

## `apply(subgraph, algorithm_or_pipeline)`

Convenience helper that accepts a `Subgraph`, plus one of the following:

- An `AlgorithmHandle` (runs the algorithm once)
- A list/tuple of handles (runs them sequentially)
- A `Pipeline` object (reuses an existing compiled pipeline)

Returns a new `Subgraph` with all algorithm results applied. Internally this wraps the same Rust
executor used by `Subgraph.apply()`.

## `Pipeline`

```python
from groggy import pipeline, algorithms

pipe = pipeline([
    algorithms.centrality.pagerank(output_attr="pr"),
    algorithms.pathfinding.bfs(start_attr="is_start", output_attr="dist"),
])

result = pipe(subgraph)
print(len(pipe))        # Number of steps
print(pipe)             # String summary
```

Methods:

- `__call__(subgraph)` / `run(subgraph)` — Execute the pipeline
- `__len__()` — Number of algorithm steps
- `__repr__()` — Developer-friendly summary

Pipeline objects automatically free their native handles when the Python object is collected.

### Related Modules

- [Builder DSL](builder.md)
- [Pipeline User Guide](../guide/pipeline.md)
- [Algorithm Catalogue](../guide/algorithms.md)
