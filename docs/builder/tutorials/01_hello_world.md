# Tutorial 1: Hello World — Your First Builder Algorithm

Build a simple, normalized degree score using the current builder API. No decorators—just create a builder, add steps, build, and apply.

## What You’ll Build

`popularity = degree / max_degree` → a 0–1 score where 1.0 is the highest degree in the graph.

## Step 1: Create the Builder

```python
import groggy as gr

b = gr.builder("node_popularity")
```

`b` records steps and later produces an algorithm handle you can apply to any subgraph.

## Step 2: Compute Degrees

```python
nodes = b.init_nodes(default=0.0)
degrees = b.node_degrees(nodes)
```

- `init_nodes` seeds a node vector.
- `node_degrees` writes degrees into a new variable.

## Step 3: Normalize and Attach

```python
normalized = b.normalize(degrees, method="max")  # divide by max degree
b.attach_as("node_popularity", normalized)

popularity_algo = b.build()
```

`attach_as` stores the result as a node attribute; `build()` returns the algorithm handle.

## Step 4: Run It

```python
G = gr.generators.karate_club()
result = G.view().apply(popularity_algo)

scores = result.nodes["node_popularity"]
print(scores[:5])
```

- `view()` creates a subgraph for execution.
- `.apply()` runs the Rust-backed plan; the returned subgraph carries the new attribute.

## Complete Example

```python
import groggy as gr

b = gr.builder("node_popularity")
nodes = b.init_nodes(default=0.0)
degrees = b.node_degrees(nodes)
normalized = b.normalize(degrees, method="max")
b.attach_as("node_popularity", normalized)
popularity_algo = b.build()

G = gr.generators.karate_club()
result = G.view().apply(popularity_algo)

for nid, score in zip(result.nodes.ids(), result.nodes["node_popularity"]):
    print(f"{nid}: {score:.3f}")
```

## Try This Next

1) Weight in/out-degree differently: compute in/out separately (if directed) and combine with `core.add/mul`.  
2) Log scale: `log_deg = b.core.log(b.core.add(degrees, 1.0))`; then normalize.  
3) Threshold mask: attach a boolean `is_popular = normalized > 0.5` with `b.core.gt`.

## Key Takeaways

- Use `gr.builder(name)` to define a pipeline.
- `init_nodes` → `node_degrees` → `normalize` → `attach_as` covers many scoring tasks.
- `build()` returns a handle you can run with `subgraph.apply(...)`.

Next: [Tutorial 2: PageRank](02_pagerank.md) to learn loops and the Batch Executor.

❌ **Forgetting the decorator**
```python
# Wrong - no decorator
def compute_popularity(sG):
    return G.nodes().degrees()
```

✅ **Correct**
```python
@algorithm("node_popularity")
def compute_popularity(sG):
    return G.nodes().degrees()
```

❌ **Not returning the result**
```python
@algorithm("popularity")
def compute_popularity(sG):
    degrees = G.nodes().degrees()
    # Forgot to return!
```

✅ **Correct**
```python
@algorithm("popularity")
def compute_popularity(sG):
    degrees = G.nodes().degrees()
    return degrees  # or manually save with G.builder.attr.save()
```

❌ **Using Python sum() instead of .reduce()**
```python
# Wrong - trying to use Python's sum on VarHandle
max_degree = sum(degrees)  # This won't work!
```

✅ **Correct**
```python
# Use .reduce() to aggregate
max_degree = degrees.reduce("max")
```

---

**Ready for more?** Continue to [Tutorial 2: PageRank](02_pagerank.md) to learn about iterative algorithms with loops!
