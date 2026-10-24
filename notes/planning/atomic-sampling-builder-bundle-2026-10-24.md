# Checked mask sampling bundle — 2026-10-24

Closes **09.4b.2u / 09.4b.2v / 09.4b.2w**, extending **30.1 / AR7.8**.
Continues the requested larger cadence, grouping three related existing steps in
one implementation/verification commit. Broader execution and sampling-composer
parents remain partial; no new policy decision or approval gate is introduced.

## Node sampling (09.4b.2u)

Certify core.sample_nodes for property-atomic builders and nested loops. Select
from stable sorted active IDs and produce a sparse integer-1 mask. A local RNG
replaces shared thread reseeding: repeated supplied seeds are independent of
previous sampler calls and do not consume/reset the shared RNG. Calls without a
seed initialize a local RNG and do not promise repeatability. Exact sampled sets
are not a versioned portable contract across RNG/dependency changes.

Require exactly one count/fraction, with nonnegative exact count <= population or
finite fraction in [0,1]. Fraction rounds to nearest size, positive half upward.
Malformed supplied types, both controls together, and negative serialized seeds
reject instead of silently ignoring controls or wrapping signed seeds. Native
constructors still accept u64 seeds; serialized integer seeds fit nonnegative i64.
Cancellation covers validation/scanning/output installation. All preparation is
private, so errors and cancellation leave the target variable unchanged. Empty
masks alone create no version.

## Edge sampling (09.4b.2v)

Certify core.sample_edges with the same shared reservoir kernel and controls.
Validate active selected nodes, active selected edges and endpoint closure before
sampling. Keep the edge identity domain: parallel edges and stored self-loops are
distinct candidates and masks attach to edges. Directed/undirected orientation
does not alter edge-ID sampling probability. Persistent attachments share the
outer property batch; transient attachments remain owned edge columns.

## Map-key reservoirs (09.4b.2w)

Certify core.reservoir_sample over materialized node/edge maps. Sort source keys,
validate every key against the active selected population using a linear merge,
and run the same local Algorithm R. Source values are irrelevant, including zero
and null; this is key sampling rather than predicate filtering. Preserve up-to-k
semantics: k above source size returns all keys, while exact-count node/edge
sampling rejects an impossible count. Capacity is capped to min(k, source size),
including native usize::MAX requests. Source/target aliasing is safe. Validate even
for k=0, rather than silently accepting foreign identities. Supplied entity_type
must be nodes/edges text; malformed selectors and seeds reject.

## Compatibility, cost and remaining work

This repairs seeded nondeterminism caused by unordered candidate traversal and
shared RNG resets. Valid result cardinalities and sparse integer masks remain.
Invalid/ambiguous controls now error; historically sampled sets may change.
Property attachment writes selected rows and does not clear prior unsampled rows.
That existing sparse-write behavior is explicit in the documentation.

Three duplicated reservoir loops become one helper. Reservoir capacity/output
storage are O(min(k,N)); selected IDs and materialized source keys still cost O(N)
working storage, with sorting for map-key reservoirs. This is not an external
streaming iterator API with O(k) total memory. Validation adds selected identity
scans; allocation/sorting does not provide bounded cancellation latency. No owner
or graph-history clone and no FFI business logic is introduced.

Iterate/expand/emit/for-each subgraph steps and whole sampler-pipeline transactions
remain uncertified, as do community/temporal/path families. Unsupported-capability
fixtures now use sample.iterate_nodes so they still test a real unreviewed family.
API details/examples: `../../docs/api/builder-sampling.md`.

## Verification and bounded release receipt

**961 native tests passed, 52 ignored; 1,452 root Python API tests passed,
1 skipped; 56 executable documentation examples passed; strict all-target core
Clippy passed.** The focused sampling/sampler/atomic builder suite passed 175
cases. Builds used offline locked dependencies and in-tree artifacts. The root
Python gate excludes the owner benchmark test and separate legacy modules survey.

Release harness: `benches/sampling_builder_steps.rs`.
Six workloads (node/edge count, fraction and reservoir) at 1,000/10,000 nodes and
edges, five observations each, measure 10% mask selection, attachment and one
checked publication. Fixture/spec preparation is excluded. Validate every output
cell, cardinality and publication count. No before/after speedup, RSS/allocation,
FFI or production-scale claim.

Raw receipt: `../benchmarks/sampling-builder-bundle-2026-10-24.json`.
The validator accepts logical integer-1 cells in either native integer width and
null/absent unsampled cells, matching existing column storage.

| Workload | Nodes (= edges) | Median ms, five observations |
| --- | ---: | ---: |
| nodes_count | 1,000 | 0.199 |
| nodes_fraction | 1,000 | 0.172 |
| edges_count | 1,000 | 0.240 |
| edges_fraction | 1,000 | 0.213 |
| nodes_reservoir | 1,000 | 0.188 |
| edges_reservoir | 1,000 | 0.286 |
| nodes_count | 10,000 | 1.274 |
| nodes_fraction | 10,000 | 1.263 |
| edges_count | 10,000 | 1.766 |
| edges_fraction | 10,000 | 1.761 |
| nodes_reservoir | 10,000 | 1.628 |
| edges_reservoir | 10,000 | 2.457 |
