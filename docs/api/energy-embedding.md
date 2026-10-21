# N-dimensional spring-electric embedding

The native `EnergyEmbedding`, `EnergyEmbeddingBuilder`, and
`ForceDirectedEmbedding` share one checked spring-electric solver. The existing
factory honors `EmbeddingConfig.seed` for both energy and force-directed engines.
This slice does not add a Python entry point.

```rust
use groggy::viz::embeddings::energy::EnergyEmbeddingBuilder;
let positions = EnergyEmbeddingBuilder::new()
    .iterations(100)
    .learning_rate(0.01)
    .with_spring_electric(1.0, 0.1, 1.0)
    .seed(42)
    .compute(&graph, 3)?;
```

## Objective and topology

For each distinct pair of nodes, let `r = sqrt(||p_i - p_j||² + 1e-6)`.
Minimize the **sum**, without edge or pair averaging, of:

- `0.5 * attraction_strength * (r - ideal_distance)²` for every active nonloop
  edge instance.
- `repulsion_strength / r` for every unordered pair of active nodes.

Every active edge participates; there is no ten-edge cutoff. Parallel and reverse
instances add separate springs. A directed edge has the same geometric spring as
its reverse, without claiming a direction-preserving representation. Loops have
zero energy and force. Checked endpoints must belong to the full active row set.
Edge properties/weights do not enter this objective.

Analytic derivatives of this objective drive updates. The implementation no
longer substitutes whole-matrix magnitude terms, a constant energy, or constant
gradients. It does not depend on the general autodiff library, which requires its
own audit. Pair smoothing prevents singular repulsion, but exactly coincident
points remain stationary; no random perturbation is added to supplied coordinates.

## Inputs, identity, and owned output

The graph must be nonempty and the requested dimension positive and representable.
A singleton is valid. The default initializer uses small random coordinates and a
local RNG when seeded. Repeated seeded calls are identical and leave the global
random stream unchanged.

`engine.optimize_positions(&positions, &graph)` offers explicit warm starts from
finite coordinates. Unlabeled rows mean ascending active node IDs; labeled rows
must contain exactly one canonical `node_<id>` for each active node. Reject wrong
row counts, malformed/duplicate/inactive IDs, and zero-width matrices. Preserve
labeled row order and return owned coordinates labeled by ID, with columns
`energy_0`, `energy_1`, etc. Input matrices, graph state, and temporal history are
not modified. Energy engines support explicit warm starts; the force-directed
wrapper reports no incremental interface. Neither supports streaming.

Learning rate must be finite and positive; attraction, repulsion, and ideal
distance must be finite and nonnegative. Zero iterations returns initialization
(or a copy of supplied coordinates), after validating parameters and evaluating
the objective. `Custom` and `StressMinimization` remain explicit unsupported errors,
including zero-iteration runs.

## Optimization and limits

Each iteration starts at the configured learning rate, optionally reduced by a
linear annealing schedule. Up to 64 halvings seek a finite candidate with energy
no greater than the previous accepted energy. Failure is an error. Zero gradients
or an unrepresentable coordinate update stop optimization; reaching the iteration
bound returns the current finite approximation. Debug history includes initial
energy and every accepted step, including the final energy.

This is a bounded local optimizer, with no global optimum or tolerance-based
convergence guarantee. Repulsion-only and disconnected layouts can spread without
a finite global minimum. There is no bounding disk or centering term. Overflow in
coordinates, pair distances, energy, or gradients rejects; arbitrary extreme-range
inputs are not promised. Budget exhaustion in backtracking rejects even if a much
smaller step could succeed.

An evaluation costs `O((E + N²) * D)` time with repulsion enabled and
`O(E * D + N * D)` without it; scratch memory is `O(E + N * D + T)`, including `T` accepted energy-history entries. Each iteration can
require 64 candidate evaluations. No acceleration, benchmarked throughput,
cancellation, GIL-release, or production-scale guarantee is added. Factory
pre/postprocessing and composite child-seed policy remain separate audit work.

## Evidence

Native unit tests compare the objective and analytic gradients with an independent
formula and central finite differences in 1, 2, and 5 dimensions, checking spring,
repulsion, and combined terms; translation invariance and balanced forces; and
monotonic actual energy under annealed/nonannealed backtracking. Integration tests
in `tests/energy_embedding.rs` cover sparse IDs, endpoint influence past the former
cutoff, loops/parallel edges, row permutations, ownership, invalid controls,
unsupported modes, singleton inputs, factory seeds, and unchanged global RNG.
