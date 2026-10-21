# Two-level map-equation community detection

`algorithms.community.infomap` now minimizes a two-level map-equation objective. The previous implementation voted on neighboring labels and ignored teleportation and trial count; that placeholder has been replaced. This is Groggy's in-house bounded greedy implementation, not a claim of parity with the external Infomap optimizer or its default flow model.

```python
from groggy import Graph, algorithms

graph = Graph(directed=True)
a, b = graph.add_node(), graph.add_node()
graph.add_edge(a, b)
graph.induced_subgraph([a, b]).apply(
    algorithms.community.infomap(
        teleportation=0.15, num_trials=10, max_iter=100,
        seed=42, flow_max_iter=10000,
    )
)
```

## Explicit flow model

Selected edges have unit weight; parallel instances affect transition probabilities. Directed edges supply outgoing transitions. Undirected edges supply both orientations, including two incidences for a self-loop. Stored weight attributes are not used.

Every step has a recorded uniform-node jump probability `teleportation` in `[0, 1]`. Non-jump steps follow outgoing edges proportionally to multiplicity. A dangling row jumps uniformly with probability one, even when teleportation is zero. Both jumps and edge steps contribute to module exits and visit rates. Recorded and unrecorded teleportation are different models; see the authors' [teleportation study](https://www.mapequation.org/assets/publications/LambiotteRosvall2012.pdf). This implementation chooses recorded uniform-node jumps explicitly.

The stationary solver starts uniformly and uses lazy iteration to avoid periodic oscillation without changing stationary solutions. It checks the original transition operator's L1 residual against `1e-12`, with the separately configured `flow_max_iter` budget. Failure to converge rejects before writing labels. At zero teleportation, reducible graphs need not have a unique stationary distribution; the uniform-start limiting mixture is the chosen convention. Residual tolerance is not an arbitrary-precision or ill-conditioning certificate.

Uniform recorded jumps connect states beyond stored edges. Consequently an edgeless nonempty selection has uniform flow and returns one codebook/community. These labels describe the selected flow model, rather than connected components of stored topology. Empty selections and single nodes are valid.

## Codebooks and search

The objective in bits per step is

```
L = q * H(index codebook) + sum_c (q_c + p_c) * H(module codebook c)
```

`p_c` is node visit mass inside a module and `q_c` is its exit rate. Each module codebook contains its nodes and an exit symbol. The index codebook uses module exit rates under the stationary-flow convention. See the original authors' [two-level map equation](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0018209).

The implementation computes teleportation exits analytically, avoiding a dense all-pairs transition matrix. Incremental moves update the affected module codebooks; each sweep rebuilds scores from actual flows to bound drift and rejects an increased score beyond floating tolerance. Improvements must exceed `1e-12` bits. Candidate moves include occupied communities and one empty singleton slot, including communities with no direct link because recorded jumps can connect them.

Every requested trial starts from singleton labels and uses an independently shuffled order from one local seeded RNG. The one-module solution is always retained as a baseline, and the lowest complete score wins. More trials with the same seed retain the earlier trial prefix, so the chosen score cannot worsen. Missing seed uses reproducible seed zero and does not alter process-wide RNG state. Ties retain the earlier candidate. Labels are canonical opaque community numbers in sorted original-node order.

`max_iter` bounds optimization sweeps per trial; `flow_max_iter` bounds stationary-flow iterations. Trial, sweep and flow budgets must be positive. Seed must be nonnegative. Rust preserves the existing constructor and adds `with_flow_budget`. Python adds a thin `community.infomap` factory. Metadata type validation accepts integer values for float parameters, matching the Rust parameter accessor; bool remains distinct.

## Publication, costs and remaining work

Active selected IDs/endpoints are validated before calculation. Cancellation, invalid parameters, flow nonconvergence or invalid numeric scores leave previous labels intact. Only final result labels use the existing bulk-column writer; source topology is unchanged. This does not add durable publication or immutable-view semantics to legacy algorithm execution.

Sparse walk/flow and module buffers use O(V+E) working memory; diagnostic records additionally grow with completed trials/sweeps. A sweep evaluates all occupied module candidates, with worst-case O(E + V²) work up to map/logarithm overhead. This implementation is a correctness-oriented small-graph kernel; large-scale search acceleration is not complete.

Still open: hierarchical/nested map-equation codebooks, coarse/fine optimization strategies, unrecorded or link-targeted teleportation, external strength-weight inputs, and external backend quality comparisons. Neither finite trials nor greedy node moves establish a global optimum.

## Evidence

Private `map_equation.rs` tests compare objective values and every single-node move against independently computed dense-transition/codebook entropy across all 81 small partitions, directed/undirected modes and teleportation 0/0.15/1. Analytic checks cover dangling rows, loop/parallel incidence, periodic paths, reducible zero-jump mixtures, forced nonconvergence and known one/two/singleton-codebook lengths.

`tests/infomap_validation.rs` and `tests/test_infomap_validation.py` exercise actual flow modules, teleportation dependence, trial execution, direction, local RNG behavior, invalid controls, stale IDs, cancellation, failed-flow preservation and empty/edgeless/gapped-ID selections. Bounded debug measurements and raw repetitions are recorded in `notes/benchmarks/infomap-2026-10-21.md`. No external graph package was installed or executed.
