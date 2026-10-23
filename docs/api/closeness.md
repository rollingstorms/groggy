# Closeness and harmonic centrality

The existing Rust constructor and first two Python positional arguments remain:

```python
from groggy import algorithms

algorithm = algorithms.centrality.closeness(
    weight_attr="length", output_attr="centrality",
    harmonic=False, direction="incoming", normalized=True,
)
result = selection.apply(algorithm)
```

Rust adds `with_direction(ClosenessDirection::Incoming)` and
`with_normalization(true)` to `ClosenessCentrality::new(harmonic, weight, output)`.
The registry exposes the same harmonic/direction/normalized options.

## Definitions

Defaults preserve Groggy's outgoing, raw harmonic definition. On directed graphs,
`outgoing` uses distances from u and `incoming` uses distances to u. Undirected
edges traverse both ways for both weighted and unweighted runs. Incoming directed
adjacency is private and cannot change the outgoing CSR cache used by other
algorithms. Parallel edges use their minimum length; loops do not add self scores.

For N selected nodes, let R be the number of distinct reachable other nodes and
S their distance sum, using the requested direction:

| Mode | Raw score | `normalized=True` |
| --- | --- | --- |
| `harmonic=True` | sum of reciprocal positive distances | raw score / (N-1) |
| `harmonic=False` | R / S | raw score * R / (N-1) |

Unreachable nodes contribute nothing. Isolates/singletons score zero, and empty
input returns no scores. Normalization does not imply a [0,1] range when arbitrary
positive lengths smaller than one are supplied.

Weights must be finite strictly positive numeric lengths (Float/Float64/Int/SmallInt);
missing values are one. Explicit zero, null, bool, negative, nonnumeric and
nonfinite lengths reject before computation, including unreachable selected
edges. This tightens the former zero-length behavior, which skipped reciprocal
terms or produced undefined classic scores. Dijkstra's separate zero-weight
support is unchanged.

The [NetworkX classic closeness reference](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.closeness_centrality.html)
documents incoming distances and reachable-fraction correction; the
[harmonic reference](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.harmonic_centrality.html)
documents incoming reciprocal sums. Compare explicit variants: Groggy's outgoing
harmonic default differs from the NetworkX classic default. These links establish
formula conventions, not an executed external-backend comparison.

## State, failure and output

Validate active selected node/edge IDs and endpoint closure before empty returns
or cache reads. Retained inactive IDs require historical inspection rather than
live computation. Cancellation is checked before work, within BFS/heap traversal,
between sources and before final persistence. No bounded cancellation latency is
promised for every validation, setup or high-degree adjacency scan.

Results use ordered original node IDs and f64 accumulation. Nonpersistent runs
return owned `(NodeId, f64)` rows in native/Python profile output
`centrality.closeness.scores`, without property writes. Persistent runs preflight
all scores for finite f64 representation and use one existing bulk property
setter. Overflow rejects before any score is written. Stored values retain the
same f64 precision as owned score output. This operation does not introduce a durable transaction
boundary or implement the unfinished state-pinned input-view contract.

Unweighted all-source traversal takes O(V(V+E)); weighted traversal uses the
native heap implementation, with up to O(V(V+E) log(V+E)) work. Buffers are reused
between sources. CSR, weight map, buffers and optional incoming transpose occupy
O(V+E) entries. Whole-call selection/order validation also sorts IDs. No kernel
speedup or production-scale capacity is claimed.

## Evidence and remaining scope

`tests/closeness_validation.rs` compares all 512 three-node edge sets in both
graph directions and weighted/unweighted modes, all harmonic/classic,
incoming/outgoing and normalized/raw options against an independent Floyd-Warshall
reference. Other fixtures cover empty/singleton/sparse/parallel/loop inputs,
missing/compact weights, stale/open selections, cancellation, cache direction
isolation and finite f64 scores beyond the f32 range in both output modes.
`tests/test_closeness_validation.py` checks rebuilt Python behavior and existing
positional arguments. Full item 28 remains open for external-backend, larger-scale,
remaining caller/option and state-pinned-input certification.
