# Tutorial 5: Sampler Pipelines

Sampler pipelines compile Builder steps into a sampler that returns a `SubgraphArray`.
They let you prepare dataset-ready subgraphs without leaving Rust execution.

This tutorial shows how to:
- generate a subgraph per node
- expand each to a neighborhood
- map a sampling sub-pipeline over each neighborhood
- emit the final `SubgraphArray`

## What You’ll Build

- 1-hop neighborhood for every node
- Sample 100 nodes within each neighborhood
- Emit one subgraph per neighborhood

## Step 1: Create the Builder

```python
import groggy as gr

b = gr.builder("random_100_1hop")
```

## Step 2: Expand Neighborhoods

```python
nbh = b.neighbors(b.iterate_nodes(), hops=1)
```

This produces a `SubgraphArray` — one neighborhood per node. Each element is a full `Subgraph` view.

If you want fewer seeds, you can sample first:

```python
nodes = b.sample_nodes(count=100, seed=42)
nbh = b.neighbors(nodes, hops=1)
```

## Step 3: Map a Sampler Over Each Neighborhood

```python
sampled = nbh.map(lambda b: b.sample_nodes(count=100, seed=42))
```

The `map(...)` call runs a sub-pipeline for each neighborhood.
The lambda receives a builder instance scoped to that one subgraph.
Return a node/edge selection and it will be emitted as a subgraph automatically.

You can also sample edges instead:

```python
sampled = nbh.map(lambda b: b.sample_edges(count=100, seed=42))
```

## Step 4: Emit Output Subgraphs

```python
b.emit_subgraphs(sampled, mode="per_item")
```

Modes:
- `per_item`: one output per input subgraph
- `unified`: a single subgraph from all outputs

## Step 5: Run the Sampler

```python
graph = gr.generators.karate_club()

samples = graph.view().sample(b.build_sampler())
print(len(samples))
```

Each element in `samples` is a `Subgraph` representing a sampled neighborhood.

## Key Takeaways

- Samplers return `SubgraphArray`.
- `iterate_nodes()` + `neighbors()` creates per-node neighborhoods.
- `map(...)` runs a sub-pipeline per subgraph.
- `emit_subgraphs(..., mode="per_item")` keeps one output per input.

## Common Patterns

- Uniform sample → neighborhood → per-item sample
- Per-item sampling with `map(...)` to avoid Python loops
- Switch `mode="unified"` to collapse outputs into a single subgraph

Next: Explore the Builder API reference in `docs/builder/api/`.
