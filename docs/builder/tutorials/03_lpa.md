# Tutorial 3: Label Propagation — Async Updates with `map_nodes`

Implement LPA using the builder’s async neighbor updates. We’ll lean on `builder.iterate()` and `map_nodes(..., async_update=True)` so the Batch Executor can optimize the loop.

## What You’ll Learn

- Initializing unique labels with `init_nodes(unique=True)`
- Using `map_nodes` with `async_update=True` for in-place iteration
- Running iterative community detection with `subgraph.apply(algo)`

## Prerequisites

- [Tutorial 2: PageRank](02_pagerank.md)
- Basic sense of community detection

## Step 1: Builder Setup

```python
import groggy as gr

b = gr.builder("lpa_builder")
max_iter = 10
```

## Step 2: Unique Starting Labels

```python
labels = b.init_nodes(unique=True)  # each node gets a distinct ID
```

## Step 3: Async Label Updates

```python
with b.iterate(max_iter):
    labels = b.map_nodes(
        "mode(labels[neighbors(node)])",
        inputs={"labels": labels},
        async_update=True,  # nodes see earlier updates in the same pass
    )
```

- `mode(...)` picks the most common neighbor label.
- `async_update=True` applies updates in-place during the iteration—closer to classic LPA behavior and batchable when supported.

## Step 4: Attach and Build

```python
b.attach_as("community", labels)
lpa_algo = b.build()
```

## Step 5: Run on a Graph

```python
G = gr.generators.karate_club()
result = G.view().apply(lpa_algo)

communities = result.nodes["community"]
print(communities[:10])
```

## Variations

- **Synchronous update:** set `async_update=False` to update all nodes simultaneously each iteration.
- **Iteration budget:** increase `max_iter` for tougher graphs; LPA often stabilizes quickly.
- **Post-processing:** attach sizes per community by running a follow-up builder that counts labels and writes a size attribute.

## Tested Reference Snippet (from tests)

```python
def build_lpa(builder, iterations=10):
    labels = builder.init_nodes(unique=True)
    with builder.iterate(iterations):
        labels = builder.map_nodes(
            "mode(labels[neighbors(node)])",
            inputs={"labels": labels},
            async_update=True,
        )
    builder.attach_as("community", labels)
    return builder
```

## Performance Note

Because this uses `builder.iterate()` with supported steps, the **Batch Executor** will batch iterations when possible, giving substantial speedups on larger graphs.

## Recap

- Use `init_nodes(unique=True)` for seed labels.
- `map_nodes(..., async_update=True)` delivers async semantics and Batch Executor acceleration when compatible.
- Attach with `attach_as` and run with `subgraph.apply`.

Next: [Tutorial 4: Custom Metrics](04_custom_metrics.md) to compose multi-step pipelines and wrap them in reusable factories.
