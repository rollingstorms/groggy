# Native mutation defaults and complete accessor creation

Priority window began **2026-10-26 11:18:59 UTC**, with up to 90 minutes authorized.
This closes the native default/infallible-construction component of existing
09.4b.2, without adding checklist rows. Counts remain **201 checked / 67 open**.

## Delivered contract

- Ordinary native `Graph::new`, `Default`, directional and configured construction
  now use checked automatic publication. `new_publishing` is an alias.
- `add_node/add_nodes` return `GraphResult`; publication failures are not discarded
  or converted to panics/sentinel IDs. The infallible node allocator is private to
  prepared execution. Bulk insertion is one accepted version; count zero is a no-op.
- Custom configuration constructors return `GraphResult<Graph>` and validate before
  construction. Known-valid default/directional constructors stay infallible.
- Explicit `new_manual`, directional/type manual and `with_config_manual` factories
  preserve deferred assembly. Both policies have fallible insertion and checked ID/
  vector extents. An impossible count rejects before IDs/storage/history change.
- Legacy fixture, snapshot/import and private visualization assembly now names
  manual policy explicitly. Production callers propagate errors; unwraps are in
  tests/fixtures/bench harnesses. Intentional dirty-state tests change policy
  explicitly instead of relying on hidden infallible writes into publishing graphs.
- Python constructor validation delegates to native validated construction.
- Node accessor append/extend helpers were outside the pymethods block and could
  not be called, despite their docstrings. They are now exposed and marshal the
  complete input before borrowing/mutating the graph, then use one native bulk
  creation. Invalid later rows/keys/values/reserved fields/labels leave no new nodes,
  consumed IDs or accepted prefix. Explicit manual policy is retained.

This is an intentional source migration for native Rust consumers, authorized by
A03. Python's existing Graph constructor/insertion defaults remain as accepted.
No unsafe code, synthetic failure IDs or production panic wrappers were added.
API migration/examples: `../../docs/api/native-mutation-defaults.md`; its Rust
snippets are included in native rustdoc tests and Python examples execute in the
contract/docs gate. Existing manual materialized-selection contracts stay explicit.

## Verification and measurement

Five new native contract tests cover seven ordinary constructor forms, five manual
forms, scalar/bulk versions, checkpoints, dirty policy guards, invalid configuration,
impossible allocation and zero-count behavior. Eleven Python accessor cases cover
success, full-input rejection and manual acceptance. The existing native/Python
contract suites verify the migrated fixture/import/algorithm/table consumers.

Repeated and extended the existing `automatic_publication` release fixture to
exercise ordinary `Graph::new` and `add_node/add_nodes`: 10k-node bulk insertion
median **12.19 ms**, scalar insertion **24.10 ms**. These create **one versus 10,000
versions**, so they are not interchangeable history semantics or a matched speedup
claim. Receipt: `../benchmarks/native-default-publication-2026-10-26.json`.
There is no peak-RSS/whole-engine latency claim.

Full native **1,246 passed / 51 ignored**, including Rustdoc examples; strict
all-target Clippy passes. Root Python contract: **1,986 passed / 1 skipped**, plus **157 executable docs
examples**. The owner benchmark and legacy nested survey suites remain excluded. In-tree FFI rebuild retains the three pre-existing warnings;
Black/isort remain unavailable and no dependencies were installed.

## Fixed remaining 09.4b.2 inventory

1. Mutable raw pool/space escapes: restrict owner mutation access and retain internal
   corruption/recovery tests without exposing untracked writes to library callers.
2. Uncertified/topology multi-stage algorithm acceptance. Supported property pipelines
   already stage one acceptance; unsupported families still reject atomic requests.
   Effect-aware IR/default pipeline policy remains AR7.8/30.1.

The raw audit also found `GraphPool::set_subgraph_attribute` returning success
without storing anything, with a getter always returning None. Its generic entity
trait route must reject unsupported pooled attribute operations explicitly. This
is recorded under the existing mutation boundary scope, not a new roadmap row.
