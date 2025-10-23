# Builder DSL for Custom Algorithms

The Builder DSL lets you compose Groggy’s Rust step primitives directly from Python and execute
that pipeline without writing any Rust. This is ideal when you need a bespoke algorithm that
combines existing primitives like initialisation, node degree, normalisation, and attribute
attachment.

---

## When to Use the Builder

Use the Builder DSL when:

- You want to prototype a custom analysis without leaving Python
- You need a lightweight transformation that can run entirely in Rust
- You want to ship a repeatable pipeline to teammates who may not write Rust

If you just need a single built-in algorithm, keep using `groggy.algorithms` and `apply()` – the
builder is meant for higher-order composition.

---

## Quick Example

```python
import groggy as gr

# Build a simple "degree_score" algorithm
b = gr.builder("degree_score")

nodes = b.init_nodes(default=0.0)
degrees = b.node_degrees(nodes)
normalized = b.normalize(degrees, method="max")
b.attach_as("degree_score", normalized)

algo = b.build()

# Run against any subgraph
G = gr.generators.path_graph(6)
result = G.view().apply(algo)

for node in result.nodes[:3]:
    print(node.id, node.degree_score)
```

Behind the scenes, `builder.step_pipeline` executes the generated pipeline inside Rust. The
resulting subgraph is returned immediately so you can continue chaining (`sg.apply(algo).table()`).

---

## Step Primitives

The builder currently exposes the following primitives (with more on the way):

| Step ID                   | Builder Call         | Purpose                                  |
|--------------------------|----------------------|------------------------------------------|
| `core.init_nodes`        | `init_nodes()`       | Initialise a node map with a constant    |
| `core.node_degree`       | `node_degrees()`     | Compute node degrees                     |
| `core.normalize_node_values` | `normalize()`  | Normalise node map (`sum`, `max`, `minmax`)|
| `core.attach_node_attr`  | `attach_as()`        | Persist a node map as an attribute       |

Each builder method records an internal step specification. When you call `build()`, the spec is
serialised to JSON and handed to the Rust interpreter for validation and execution.

---

## Validation & Errors

- Missing parameters raise `ValueError` when `build()` is called
- Unknown step types raise a descriptive error during execution
- Normalisation with zero magnitude triggers a friendly `RuntimeError`

Use the `steps` list on the builder for debugging:

```python
for step in b.steps:
    print(step)
```

---

## Best Practices

- Prefer descriptive algorithm names – they become part of the registry ID (`custom.NAME`)
- Keep pipelines short; reuse existing algorithms where possible
- Chain `sg.apply(algo)` with `.table()`/`.viz` to stay in fluent mode

---

## Related Documentation

- [Algorithm Guide](algorithms.md)
- [Architecture Deep Dive](../concepts/architecture.md)
- [`apply()` Convenience Function](../quickstart.md#running-algorithms)


## Example: Label Propagation Wrapper

```python
b = gr.builder("lpa_wrapper")
base = b.init_nodes(default=0)
communities = b.normalize(base, method="sum")
b.attach_as("community_score", communities)

lpa_wrapper = b.build()
result = sg.apply(lpa_wrapper)
print(result.nodes.table()[["community_score"]].head())
```

Use this pattern when you want to add pre/post-processing around an existing algorithm (call the
handle separately) or when prototyping a custom scoring function before dropping into Rust.
