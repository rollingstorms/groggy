# Tutorial 4: Custom Metrics — Composing Builder Pipelines

Combine builder steps into reusable metrics and factories. We’ll create a few node-level scores that mix attributes, degrees, neighbor signals, and iterative smoothing.

## What You’ll Learn

- Blending attributes and topology with `core` arithmetic + `normalize`
- Smoothing signals with `map_nodes` inside `builder.iterate()`
- Packaging builders as reusable factory functions

## Example 1: Degree-Weighted Engagement

Blend an attribute (e.g., activity) with node degree, then normalize to [0, 1].

```python
import groggy as gr

def engagement_score(attr: str = "activity", degree_weight: float = 0.5):
    b = gr.builder("engagement_score")

    nodes = b.init_nodes(default=0.0)
    degrees = b.node_degrees(nodes)
    activity = b.load_attr(attr, default=0.0)

    combined = b.core.add(
        b.core.mul(activity, 1.0 - degree_weight),
        b.core.mul(degrees, degree_weight),
    )

    score = b.normalize(combined, method="max")
    b.attach_as("engagement_score", score)
    return b.build()

# Usage
algo = engagement_score(attr="activity", degree_weight=0.3)
result = graph.view().apply(algo)
scores = result.nodes["engagement_score"]
```

## Example 2: Neighbor-Averaged Quality (1-pass)

Average a node attribute with its neighbors and normalize.

```python
def neighbor_quality(attr: str = "quality"):
    b = gr.builder("neighbor_quality")

    values = b.load_attr(attr, default=0.0)
    neighbor_avg = b.map_nodes(
        "mean(values[neighbors(node)])",
        inputs={"values": values},
    )

    blended = b.core.mul(
        b.core.add(values, neighbor_avg), 0.5
    )  # simple mean of self + neighbors

    score = b.normalize(blended, method="max")
    b.attach_as("neighbor_quality", score)
    return b.build()
```

## Example 3: Iterative Influence Smoothing

Spread a seed signal through the network for a few iterations (Batch Executor friendly).

```python
def influence_smoothing(attr: str = "seed", damping: float = 0.8, iterations: int = 10):
    b = gr.builder("influence_smoothing")
    signal = b.load_attr(attr, default=0.0)

    with b.iterate(iterations):
        neighbor_mean = b.map_nodes(
            "mean(signal[neighbors(node)])",
            inputs={"signal": signal},
        )
        signal = b.core.add(
            b.core.mul(signal, damping),          # keep some of the old signal
            b.core.mul(neighbor_mean, 1 - damping),  # mix in neighbors
        )

    smoothed = b.normalize(signal, method="max")
    b.attach_as("influence", smoothed)
    return b.build()
```

Run it:

```python
algo = influence_smoothing(attr="seed", damping=0.7, iterations=15)
result = graph.view().apply(algo)
influence = result.nodes["influence"]
```

## Tips for Your Own Metrics

- Normalize at the end (`method="sum"` for probabilities, `method="max"` for 0–1 scaling).
- Keep loops inside `builder.iterate()` to unlock the Batch Executor when compatible.
- Use `map_nodes` for neighbor aggregation; set `async_update=True` when you need in-place updates (see Tutorial 3).
- Attach intermediate signals if you want to inspect them (`b.attach_as("debug_attr", var)`).

You now have patterns for single-pass blends, neighbor aggregations, and iterative smoothing. Mix and match to fit your dataset. Next step: combine these builder algorithms with native ones from `groggy.algorithms` in your pipelines.
