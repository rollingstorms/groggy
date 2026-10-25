# Checked path builder bundle — 2026-10-24

Closes **09.4b.2ag / 09.4b.2ah / 09.4b.2ai**, extending **30.1 / AR7.8**.
Continues the requested larger cadence across three existing native path helpers.
Temporal steps remain the next unreviewed family. These are implementations and
repairs under existing workbook decisions, not a new engine contract or approval
gate. Broader defaults/storage escapes, optimizer effects, pinned views and
standalone sampler transaction parents remain partial/open.

## Checked distance maps and shared topology (09.4b.2ag)

Certify shortest_path_map. Replace unchecked CSR warming and owner-only weight
collection with one shared selected topology used by all three helpers. Validate
active selected nodes/edges and endpoint closure, sort adjacency by neighbor/edge
ID and read weights through StepScope's property overlay. Undirected reverse
traversal works and selected edge omissions remain respected. Preserve every arc
for Dijkstra/walks rather than overwrite parallel weights in endpoint-pair maps.

Weighted inputs are finite nonnegative numeric scalars, missing=1 and explicit
null/Boolean/text/collection invalid. This applies to all selected edges, not only
reachable edges. Weight types include both integer widths and f32/f64. BFS outputs
exact checked i64 distances; weighted Dijkstra uses finite f64 distances and
ordinary floating rounding. Unreachable nodes stay absent. Source IDs are
nonnegative selected active integers or parseable unsigned decimal text. An
explored nonfinite accumulated path cost errors instead of returning infinity.
Failure/cancellation preserves the previous result map; persistent attachments
share one outer property acceptance and transient output remains owned.

## Loopless k-shortest node paths (09.4b.2ah)

Certify k_shortest_paths. Use the same checked shortest-search kernel for Yen spur
searches, excluding earlier root nodes as well as accepted next arcs sharing the
root. The previous implementation omitted root exclusions, quantized candidate
costs to integer micro-units and could let duplicate candidates consume requested
slots. Candidates now use full f64 cost plus node-sequence ordering and are
deduplicated before heap insertion. Result arrays are sorted by cost/sequence;
equal-cost cutoffs are deterministic but do not promise a global tie rank.

Preserve the Text scalar containing JSON [(node sequence, cost), ...]. For this
node-only interface, parallel arcs use the minimum traversal cost and do not
produce duplicate edge-distinct paths. k=0 gives an empty list, unreachable gives
an empty list, positive k with source=target gives one zero-cost singleton. Large
k does not preallocate k entries and stops when candidates exhaust. Controls,
endpoints and selected weights validate even for k=0. Body/kernel errors cannot
install a partial serialized result. Reference problem/source and exact contract:
`../../docs/api/builder-paths.md`.

## Stable selected walks (09.4b.2ai)

Certify random_walk. Sort node-map start IDs, validate every start even for zero
length, use a local seeded RNG and preserve edge incidence/multiplicity, including
undirected reverse traversal and self-loops. Seeded calls neither reset nor consume
shared RNG state. Unweighted choices are per stored arc. Weighted rows scale by
maximum weight before cumulative summation to avoid finite-weight sum overflow;
zero-probability arcs cannot be selected when positive mass exists. Tiny relative
probabilities still have f64 underflow limits. Preserve all-zero-row uniform
fallback. Restart is checked before dead-end stopping, preserving repeat-start
behavior at probability 1; zero length includes the start only.

Serialize the existing list-of-walks Text scalar after all walks succeed. Negative
length/k/IDs/seeds no longer wrap; supplied optional selectors/restart controls
must have their declared types/ranges. Walk allocation grows with actual emitted
nodes rather than preallocating an arbitrary requested length.

## Cost, compatibility and limits

The three helpers share checked adjacency, weight validation and shortest-search
logic rather than separate unchecked implementations. BFS/Dijkstra allocate
frontier/heap/visited maps; selected adjacency is rebuilt per helper invocation.
Yen repeats searches per spur and retains candidate paths; walk output and JSON
serialization scale with generated sequences. Old private CSR-warm timer keys
are not retained as fictitious work. No owner/history clone or FFI business logic
is introduced. These correctness allocations may change repeated-call costs;
there is no before/after speedup or whole-project regression claim.

Cancellation covers validation, edge/frontier/spur/walk loops and pre-installation;
row sorts, allocation and serialization have no bounded-latency promise. Changes
to formerly rounded/proxy/unchecked results and seeded walk sequences are deliberate
repairs. This bundle certifies the builder helpers, not every path consumer or
temporal reconstruction/query operation.

## Verification and bounded release receipt

**979 native tests passed, 52 ignored; 1,595 root Python API tests passed,
1 skipped; 72 executable documentation examples passed; strict all-target core
Clippy passed.** The final focused path Python run passed 63 cases and native
path module passed six tests. Builds used offline locked dependencies and in-tree
extension artifacts; root Python excludes the owner benchmark test and separate
legacy modules survey. Native independent DFS enumeration checks all 64 simple
undirected and 4,096 loop-free directed four-node graphs with zero/nonnegative
weights, comparing complete distinct simple paths and minimum costs. Other native
fixtures cover parallel minima, selected scope/reverse traversal, invalid IDs,
weights, cancellation preserving previous output, RNG isolation and huge weights.
Python tests cover staged weights for all three helpers, persistent/transient
outputs, exact tiny-cost order, simple path identity, invalid controls/weights,
empty/trivial/unreachable results, restart/dead ends, per-edge walk multiplicity,
map-start ordering, missing-versus-null weights and overflow errors.

Release harness: `benches/path_builder_steps.rs`. Six modes at 1,000/10,000 nodes
and N+1 edges, five observations each, use a directed ring plus shortcut for BFS,
Dijkstra/staged weights, Yen, uniform/weighted walks and many-start walks. Fixture/
spec preparation and validation excluded; calculation/serialization, bounded
property attachment and one checked publication included. Scalar results attach
to one reference row to avoid accidental N-fold serialization storage. Every
result/publication is validated, using existing typed property decoding for native
compact/compressed text. No matched speedup, RSS/FFI/allocation or production-scale
claim.

Raw receipt: `../benchmarks/path-builder-bundle-2026-10-24.json`.

| Workload | Nodes | Edges | Median ms, five observations |
| --- | ---: | ---: | ---: |
| shortest_unweighted | 1,000 | 1,001 | 2.073 |
| shortest_weighted | 1,000 | 1,001 | 3.106 |
| k_shortest | 1,000 | 1,001 | 0.295 |
| walk_uniform | 1,000 | 1,001 | 0.280 |
| walk_weighted | 1,000 | 1,001 | 1.501 |
| walk_many_starts | 1,000 | 1,001 | 0.732 |
| shortest_unweighted | 10,000 | 10,001 | 12.306 |
| shortest_weighted | 10,000 | 10,001 | 24.319 |
| k_shortest | 10,000 | 10,001 | 2.700 |
| walk_uniform | 10,000 | 10,001 | 2.693 |
| walk_weighted | 10,000 | 10,001 | 14.772 |
| walk_many_starts | 10,000 | 10,001 | 8.162 |
