# Builder DSL Tutorials

Hands-on walkthroughs for Groggy’s Builder DSL. You’ll build real algorithms using the current builder API (`gr.builder(...)`, `builder.iterate`, `map_nodes`, `attach_as`) and run them with `subgraph.apply(algo)`. Every tutorial uses the same Rust-backed execution path and the Batch Executor where applicable.

## Tutorial Series

### [1. Hello World - Your First Algorithm](01_hello_world.md)
**Time:** 15 minutes · **Difficulty:** Beginner  
Build a normalized degree score using `init_nodes`, `node_degrees`, `normalize`, and `attach_as`.

### [2. PageRank - Iterative Algorithms](02_pagerank.md)
**Time:** 30 minutes · **Difficulty:** Intermediate  
Use `builder.iterate()` plus neighbor aggregation to express iterative PageRank and let the Batch Executor accelerate it.

### [3. Label Propagation - Asynchronous Updates](03_lpa.md)
**Time:** 25 minutes · **Difficulty:** Intermediate  
Implement LPA with `map_nodes(async_update=True)` to model in-place label changes per iteration.

### [4. Custom Metrics - Advanced Compositions](04_custom_metrics.md)
**Time:** 45 minutes · **Difficulty:** Advanced  
Combine steps into multi-part metrics, wrap builder factories for reuse, and mix pre/post-processing around native algorithms.

## Learning Path

```
Start → Hello World
   → PageRank (iterative) ↘
   → Label Propagation      ↘ either order
   → Custom Metrics → API / your own pipelines
```

## Quick Reference (Builder API)

- Create builder: `b = gr.builder("name")`
- Initialize: `vals = b.init_nodes(default=1.0, unique=False)`
- Load attrs: `weights = b.load_attr("weight", default=1.0)`
- Neighbor map: `sums = b.map_nodes("sum(ranks[neighbors(node)])", inputs={"ranks": ranks})`
- Async map: `labels = b.map_nodes("mode(labels[neighbors(node)])", inputs={"labels": labels}, async_update=True)`
- Arithmetic: `b.core.add/mul/sub/div(vals, scalar_or_var)`
- Normalize: `b.normalize(vals, method="sum"|"max"|"minmax")`
- Attach: `b.attach_as("attr_name", vals)`
- Loops: `with b.iterate(k): ...` (enables Batch Executor when compatible)
- Run: `algo = b.build(); result = graph.view().apply(algo)`

## Prerequisites

- Python 3.8+
- Groggy installed (`maturin develop --release` locally)
- Basic graph concepts and Python familiarity

## Getting Help

- Builder guide: `docs/guide/builder.md`
- Algorithms guide: `docs/guide/algorithms.md`
- API reference: `docs/builder/api/`
- Issues: GitHub tracker

Ready? Start with [Tutorial 1: Hello World](01_hello_world.md).
