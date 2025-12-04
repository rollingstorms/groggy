# Builder DSL and Batch Executor

Groggy's Builder DSL lets you compose Rust-backed steps from Python, ship custom algorithms without writing Rust, and unlock the Batch Executor for fast iterative workloads.

- Compose pipelines with step primitives (load, map, normalize, attach).
- Use `builder.iterate()` to express structured loops; the Batch Executor compacts these loops for 10–100x speedups when compatible.
- Algorithms run on subgraphs and write results back as node attributes.

## When to Reach for the Builder

- You need a bespoke algorithm that combines existing primitives.
- You want fast, Rust-side execution while staying in Python.
- You are iterating on a pipeline before moving logic into the native layer.
If a single native algorithm fits, prefer `groggy.algorithms` directly; the builder is for composition and iteration-heavy flows.

## How Execution Works

1) You record steps in Python using the builder API.  
2) `build()` serializes the plan and sends it through the FFI to Rust.  
3) Rust validates the plan, releases the GIL for long runs, executes steps, and returns a subgraph with new attributes.  
4) If the plan contains `builder.iterate()` loops that are batchable, the Batch Executor runs; otherwise it falls back to step-by-step execution automatically.

## Batch Executor at a Glance {#batch-executor-at-a-glance}

- What it does: collapses structured loops into batched kernels for iterative algorithms.
- When it kicks in: loop bodies expressed via `builder.iterate()` that use supported steps (attr load/store, neighbor maps, arithmetic, normalize, etc.).
- Fallback: unsupported ops or validation failures drop to the regular step interpreter; behavior stays correct.
- Performance (illustrative):
  - PageRank: ~100x faster (1000 nodes, 100 iterations)
  - LPA: ~40x faster (10000 nodes)
  - Any `builder.iterate()` loop benefits when compatible
- Drift note: PageRank can show ~5–6% drift after 100+ iterations; typical 10–20 iterations are stable.

## Quick Start (non-iterative)

```python
import groggy as gr

b = gr.builder("degree_score")

nodes = b.init_nodes(default=0.0)
degrees = b.node_degrees(nodes)
normalized = b.normalize(degrees, method="max")
b.attach_as("degree_score", normalized)

algo = b.build()
result = gr.generators.karate_club().view().apply(algo)
print(result.nodes["degree_score"][:5])
```

Behind the scenes, the plan runs in Rust via `builder.step_pipeline`, returning a subgraph so you can keep chaining.

## Iterative Algorithms with `builder.iterate()`

`builder.iterate(count)` marks a loop; the Batch Executor uses this structure to batch iterations when the body is compatible.

### PageRank-style Loop

```python
import groggy as gr

b = gr.builder("pagerank_builder")
ranks = b.init_nodes(default=1.0)

with b.iterate(20):
    neighbor_sum = b.map_nodes(
        "sum(ranks[neighbors(node)])",
        inputs={"ranks": ranks},
    )
    ranks = b.core.add(
        b.core.mul(neighbor_sum, 0.85),  # damping
        0.15,                            # teleport
    )

b.attach_as("pagerank", ranks)
pagerank_algo = b.build()
result = graph.view().apply(pagerank_algo)
scores = result.nodes["pagerank"]
```

### Label Propagation (async updates)

```python
b = gr.builder("lpa_builder")
labels = b.init_nodes(unique=True)

with b.iterate(10):
    labels = b.map_nodes(
        "mode(labels[neighbors(node)])",
        inputs={"labels": labels},
        async_update=True,  # nodes can see earlier updates in the same iteration
    )

b.attach_as("community", labels)
lpa_algo = b.build()
communities = graph.view().apply(lpa_algo).nodes["community"]
```

## Patterns You’ll Use Often

- Pre/post-process around a native algorithm: run a native handle, then normalize or relabel with a small builder pipeline.
- Attribute pipelines: load attrs, transform with `core.add/mul/sub/div`, attach back with `attach_as`.
- Iterative refinement: express update rules in `builder.iterate()` to unlock batching (PageRank, LPA, custom relaxations).
- Masked workflows: load attributes, build boolean masks, and attach them for downstream filtering.

## Compatibility and Pitfalls

- Batch Executor only activates for loops created via `builder.iterate()` and supported steps; incompatible steps run correctly but without batching.
- Keep loop bodies deterministic and attribute-centric (no Python-side state mutation inside the loop).
- Default values matter: set `default` when loading attrs to avoid missing-data surprises.
- Prefer scalar literals in `core.*` ops; they’re converted to constants inside the plan.
- Async updates (`async_update=True`) are for algorithms that require in-place iteration semantics (e.g., LPA).

## See Also

- `docs/guide/algorithms.md` for native algorithms you can combine.
- `docs/guide/performance.md` for tuning and benchmarking tips.
