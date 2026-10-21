# Label propagation

Label propagation operates on the selected node and edge IDs. Every selected ID
must be active and every selected edge must have both endpoints in the selection.
Validation precedes empty returns and cache reads. Inactive IDs remain retained
for history but cannot participate in a live algorithm run. No input topology is
modified. Results are written through one existing bulk property setter after
successful computation; this does not introduce a new durable transaction boundary.
Cancellation rejects before final property writes.

## Semantics

Python preserves the original first three positional arguments and adds options:

```python
from groggy import algorithms

labels = algorithms.community.lpa(
    max_iter=100, seed=42, output_attr="community",
    tolerance=0.0, seed_attr="initial_label",
)
result = selection.apply(labels)
```

Rust retains `LabelPropagation::new(max_iter, tolerance, output_attr, seed_attr)`;
`.with_seed(Some(seed))` enables locally seeded shuffling. The native registry's
omitted iteration budget remains 20; the Python factory explicitly supplies 100.

- Updates are asynchronous: later nodes in a sweep see earlier assignments.
- Without a random seed, visit ascending node IDs. With a seed, shuffle each
  sweep with a private RNG. Global random state is unaffected.
- Count outgoing edge incidences on directed graphs and both orientations on
  undirected graphs. Parallel edges vote separately; a self-loop votes once.
  There is no implicit self-vote. External edge weight attributes are not used.
- Initial labels are original node IDs unless `seed_attr` supplies a value.
  A missing property falls back to that node's ID. Equal logical property values
  identify one label. A dictionary of initial values remains immutable while
  node assignments change; output values are drawn from that dictionary.
- Isolates retain their initial label. A tied maximum retains the current label
  when it is one of the maxima; otherwise choose the label first encountered in
  ascending node-ID initialization order.
- `max_iter` must be positive; `tolerance` must be finite in [0, 1]; an explicit
  registry/Python random seed must be nonnegative. Stop when the changed fraction
  is at most tolerance or the sweep budget is exhausted. `lpa.converged` reports
  the tolerance criterion, not a global objective or a guarantee that no label
  can change. A zero tolerance requires a change-free sweep for this flag.
- Bounded runs can finish without convergence. Directed asynchronous LPA has no
  promised convergence or optimal-community guarantee.
- Nonpersistent execution returns `community.lpa.components` in the native
  context, grouping original node IDs by final label; it does not write properties.

Propagation takes expected O(V + E) per sweep with hash counting, plus initial
value hashing/copying. CSR, assignments, dictionary and scratch space require
O(V + E) entries plus property payloads. Ordered collection and checked selection
preflight add sorting and validation costs; this is not a whole-call linear-time
claim. No performance improvement or billion-node capacity is claimed.

## Evidence

Native `tests/lpa_validation.rs` checks all 512 three-node directed graphs
with unique and repeated initial labels against direct adjacency/value
propagation, plus failure, seed, sparse-ID and nonpersistent-output cases.
`tests/test_lpa_validation.py` checks the rebuilt Python surface.
NetworkX/Neo4j comparison and larger-scale verification remain open.
