# Portable immutable collections and foreach — 2026-11-27

Finite component of existing 03.2, 09.4b.2, 28.16, 30.1 and AR7.8. No stable
checklist row is added; counts remain 217 checked / 51 open.

## Completed component checks

- [x] Share all four collection-producing sampling steps through sealed read kernels.
- [x] Prepare certified foreach bodies recursively without worker registry/callback calls.
- [x] Retain source pins, aliases, source authority, ordering and exact fixed child membership.
- [x] Refresh pending properties and provisional IDs, with nested loops, captures and sibling writes.
- [x] Preserve empty versus missing outputs, rollback, cancellation and legacy memory identity.
- [x] Qualify independent geometry, actual workers, memory parity, recovery and public examples.

Memory adapters retain `SubgraphArray` and unchanged pass-through subgraph identity,
source pins and caches. Workers use additive `StepValue::GraphViewArray` values
without writable owners. Rust exhaustive enum matches need the new arm; no Python
export is added. Selector authority requires the same accepted history identity
or a capture issued by this invocation. Numeric IDs alone do not grant authority.
Issued-capture identity checks are shared with table attachment, preserving weak
metadata reclamation.

Each child executes against the checked accepted root, projected onto its exact
node/edge membership. This admits selected provisional IDs without constructing a
fake owner. Pending value refresh and nested loops/captures preserve that frame.
Topology inside children still rejects, including a zero-iteration topology body.
A missing child result retains the original immutable selection; an explicit
empty result emits zero. Output types, owners and endpoint closure validate before
child work; overlapping writes retain deterministic last-item behavior. Failures
restore prior outputs and discard staged writes, allocations and journal changes.

Evidence: three native scope tests, 32 independent geometry cases across direction,
node/edge seeds, hop depth, emission mode and induction, plus execution rejection
cases in `tests/durable_subgraph_collections.rs`; 29 public cases in
`tests/test_durable_subgraph_collections.py`. Existing 19 sampling/foreach unit tests
remain passing. A review caught pass-through subgraph identity being regenerated;
the adapter now retains the original reference and checks entity identity directly.

Local gates pass **1,712 Rust tests** (54 ignored), **3,686 Python tests** (one skipped,
23 expected warnings), **305 documentation examples**, strict all-target Clippy,
formatting, Python 3.8 syntax and all 62 packaged Python/stub bytes. All 500
production sources match `372568a554b3e88676368cc9439a911ff2d6be94`.
[Local receipt](../benchmarks/read-collections-local-qualification-2026-11-27.json),
[wheel receipt](../benchmarks/read-collections-wheel-qualification-2026-11-27.json).

Selected membership sets and child value projections still allocate; no constant
cost, RSS or production-scale claim follows from portability. The matched benchmark
uses fresh memory owners and the prior qualified wheel; its evidence is recorded
separately. The standalone sampler facade, retained-history producers, remaining
path primitives/frontends/alternate plans, custom/streaming effects, topology
domains and existing A14 decisions remain explicit parent work.

## Bounded memory-adapter cost

- 100 nodes, per_node_passthrough: wall median +16.7%, process CPU +16.7%.
- 100 nodes, overlapping_edges_write: wall median -6.7%, process CPU -6.7%.
- 1000 nodes, per_node_passthrough: wall median -5.4%, process CPU -5.4%.
- 1000 nodes, overlapping_edges_write: wall median -3.1%, process CPU -3.1%.

The ABBA comparison uses 22 measured samples per case/runtime after warmups,
with graph construction outside timing. Both compared wheels passed full local
gates. This measures these memory-adapter paths only; it establishes no allocator,
RSS, durable journal-sync or general throughput guarantee. Raw samples and exact
wheel/source pins are in the [performance receipt](../benchmarks/read-collections-performance-2026-11-27.json).
