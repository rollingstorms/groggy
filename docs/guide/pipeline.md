# Pipeline API

Groggy's pipeline tooling lets you compose multiple algorithms and run them as a single unit.
Pipelines are executed entirely in Rust, so chaining additional work does not sacrifice
performance.

---

## Why Pipelines?

- Batch several algorithms together and run them in one pass
- Reuse the same sequence across different subgraphs
- Keep Python code declarative while the heavy lifting happens in Rust

Use the pipeline API when you need more than one algorithm result at a time, or when you want to
package an analysis for teammates to reuse.

---

## Three Ways to Run Algorithms

| Usage                    | When to choose it                             |
|-------------------------|-----------------------------------------------|
| `sg.apply(algo)`        | Quick runs with a single algorithm            |
| `sg.apply([...])`       | A short list of algorithms in order           |
| `pipeline([...])(sg)`   | Reuse the same pipeline across many subgraphs |

All three options call into the same Rust pipeline engine, so pick whichever feels most natural in
Python.

---

## Basic Example

```python
import groggy as gr

# Build a reusable pipeline that runs PageRank followed by BFS
pipe = gr.pipeline([
    gr.algorithms.centrality.pagerank(max_iter=40, output_attr="pr"),
    gr.algorithms.pathfinding.bfs(start_attr="is_start", output_attr="dist"),
])

G = gr.generators.karate_club()
G.nodes.set_attrs({0: {"is_start": True}})

result = pipe(G.view())
for node in list(result.nodes)[:3]:
    print(node.id, node.pr, node.dist)
```

You can achieve the same outcome with:

```python
result = gr.apply(G.view(), [
    gr.algorithms.centrality.pagerank(max_iter=40, output_attr="pr"),
    gr.algorithms.pathfinding.bfs(start_attr="is_start", output_attr="dist"),
])
```

And for quick one-offs:

```python
result = G.view().apply(gr.algorithms.community.lpa(output_attr="community"))
```

---

## Inspecting Pipelines

```python
pipe = gr.pipeline([
    gr.algorithms.centrality.pagerank(output_attr="score"),
    gr.algorithms.community.lpa(output_attr="community"),
])

print(pipe)          # Human-readable summary
print(len(pipe))     # Step count
```

To clean up resources, pipelines are dropped automatically, but you can call
`gr._groggy.pipeline.drop_pipeline(handle)` while debugging.

---

## Interop with the Builder DSL

The Builder DSL compiles custom step pipelines and returns an object that behaves exactly like the
examples above:

```python
b = gr.builder("degree_score")
nodes = b.init_nodes(default=0.0)
degrees = b.node_degrees(nodes)
b.attach_as("degree_score", degrees)
custom_algo = b.build()

# Works with apply() or within a larger pipeline
result = sg.apply(custom_algo)
```

---

## Further Reading

- [Algorithm Catalogue](algorithms.md)
- [Builder DSL](builder.md)
- [Architecture Deep Dive](../concepts/architecture.md)
- [Quickstart Guide](../quickstart.md)

