# PageRank

Groggy retains its native CSR power-iteration implementation and existing Rust
constructor/Python signature. It now honors undirected reverse adjacency,
validates personalization and selected state, and returns nonpersistent scores.

```python
from groggy import algorithms
from groggy.pipeline import apply

result, profile = apply(
    selection,
    algorithms.centrality.pagerank(
        damping=0.85, max_iter=1000, tolerance=1e-12,
        personalization_attr="reset_weight",
    ),
    persist=False,
    return_profile=True,
)
scores = profile["outputs"]["centrality.pagerank.scores"]  # [(node_id, f64_score), ...]
converged = profile["stats"]["pr.converged"] == 1.0
```

## Model and validation

- Follow outgoing directed incidences; undirected edges contribute both
  orientations. Parallel instances contribute separate transition mass. An
  explicit self-loop contributes one transition incidence, matching shared CSR.
  Scores therefore measure incoming importance. No edge-weight option exists.
- Reset to the normalized personalization vector with probability `1-damping`.
  Dangling nodes distribute all their outgoing mass according to that same
  vector. Without a property, personalization is uniform.
- When a personalization property is selected, missing values retain Groggy's
  existing default weight of one. Float, Int and compact SmallInt weights are
  supported; explicit null, bool, nonnumeric, negative or nonfinite values reject.
  Zero individual weights are valid, but the selected total must be positive.
- Damping must be finite in [0, 1), the iteration budget positive, and tolerance
  finite and positive. Invalid registry budgets reject instead of being coerced.
- Every selected ID must be active; every selected edge must have both endpoints
  selected. Validate before empty returns or cache reads. A stale retained live
  selection rejects; it is not interpreted as a historical snapshot.
- Empty input has no scores; a singleton has score one. Original node IDs and
  topology are preserved, including sparse-ID selections. Checked index-span
  arithmetic falls back to sparse lookup instead of overflowing.

## Bounded computation and output

Start from uniform ranks. Stop when the maximum absolute coordinate change
between consecutive f64 iterates is at most tolerance, or at the iteration cap.
The existing bounded-approximation behavior is preserved: a capped run may return
scores with `pr.converged=0`. `pr.max_update` reports the last change, and
`pr.count.iteration` the number of completed iterations. This criterion is not a
certified distance to the exact stationary solution, particularly near damping
one. A caller needing convergence must inspect the profile and retry or reject.

Check finite/nonnegative scores during computation and normalize final f64 mass.
Persistent runs bulk-write f64 graph properties, retaining the computed values.
Nonpersistent runs return owned full-precision `(NodeId, f64)` rows
under `centrality.pagerank.scores`, using the typed native `NodeScores` output;
FFI only translates these rows. They do not write properties or publish history.
Cancellation is checked before computation, during iterations and before final
persistence. No bounded cancellation latency across every setup/high-degree loop
is promised. A final property write uses the existing bulk setter, not a new
durable transaction boundary.

Each power sweep uses O(V + E) work with O(V + E) CSR and vector storage. Ordered
collection and checked selection preflight add sorting and validation costs.
No speedup, production-scale capacity or external-library runtime dependency is
introduced.

## Compatibility and evidence

The [NetworkX PageRank reference](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html)
also documents reversing undirected edges and using personalization for dangling
mass. Groggy deliberately retains its existing missing-weight default of one,
maximum-coordinate stopping criterion, bounded approximation on exhausted
budget, and unweighted edge interface. These differ from NetworkX defaults;
comparing defaults is not a parity test. No NetworkX execution is claimed here.

`tests/pagerank_validation.rs` checks every three-node directed edge set under
both graph directions, uniform/nonuniform personalization and damping 0/0.85
against a separate pivoted linear solve of `(I-alpha*P^T)x=(1-alpha)*p`. Other
fixtures cover invalid controls/properties, all-zero personalization,
empty/singleton/sparse-ID/parallel/loop inputs, cancellation, stale selections,
bounded approximation and original-ID f64 outputs. `tests/test_pagerank_validation.py`
checks rebuilt Python behavior, f64 stored values and nonpersistent f64 rows.
Remaining work includes broader options, external-backend, near-one damping,
large-scale/cancellation and history-pinned input certification under item 28.
