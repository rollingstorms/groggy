# 09.4b.1 — Ordinary Python publication defaults and native coordinator policy

Implements the accepted switch for the ordinary Python graph surface. The parent
09.4/09.4b stays partial until the remaining native compatibility and compound
operation migrations are finished. This is not a new approval gate.

## Implemented

- Python `Graph()` defaults to checked automatic publication; `manual=True` is an
  explicit compatibility path. Rust exposes a publishing constructor/policy and
  fallible ordinary entry points while preserving existing infallible signatures.
- Native direct/bulk nodes, edge creation with all properties, setters/column
  setters, removals/cascades, kinds and managed identity/upserts route through the
  existing checked coordinator. Prepared execution suppresses nested dispatch
  and restores the policy after execution. Bulk paths remain columnar.
- Empty bulk operations are no-ops. Upsert no-op behavior is retained. Clock
  defaults, explicit writer/batch overrides and optimistic session guards remain
  canonical. Ordinary mutations invalidate older optimistic writer sessions.
- Base graph copying stages all topology/properties before allocation. Managed
  IDs and nonportable pooled/meta references reject with explicit diagnostics.
- Meta creation publishes required system fields together. A discovered logical
  type gap is fixed: compact text now works in base/meta classification/validation.
  One old collapse entry point now uses canonical meta creation instead of writing
  raw system fields around the coordinator.
- Default active imports and owned graph-producing factories publish one initial
  state with new acceptance clocks, then return clean publishing graphs. Explicit
  manual imports retain the old behavior. Imported active data is not journal
  recovery. No explicit checkpoint label is fabricated during import.

## Migration test changes

Dirty legacy fixtures explicitly select `manual=True`. A writer invalidated by an
ordinary write now expects a stale-head rejection; a per-call schema still does
not install a global policy, but its later validation rejects the actual invalid
proposed state rather than assuming ordinary writes are dirty. Numerical value
references and algorithm output assertions were not weakened.

## Remaining scope

Existing Rust default/infallible constructors, mutable pool/space escape hatches,
and higher-level collapse/meta, algorithm and table compound writeback need the
next audit. Core meta creation is one version, but an entire multi-stage collapse
plan is not thereby certified atomic. New portable pool recovery/references,
revert and effective-time reconstruction retain their existing roadmap items.

## Evidence

- Full native suites: **912 passed, 52 ignored**; targeted automatic-publication
  suite: 5 passed. Strict all-target Clippy passed with warnings denied.
- Rebuilt the in-tree extension; root Python API baseline excluding the owner's
  independent algorithm benchmark: **1002 passed, 1 existing skip**. This is not
  a certification of the older `tests/modules/` survey scripts.
- **31** selected documentation examples executed, including writer, property,
  checkpoint, temporal, automatic-publication and durable-storage examples.
- Release receipt: 6 workloads, five samples each, native sizes 1000/10000,
  `benches/automatic_publication_2026-10-23.json`. Bulk insertion uses one version;
  bulk setters use one additional version, scalar setters one per call. Fixture
  setup is excluded as recorded in the receipt. These are different retained
  histories, not a like-for-like speedup comparison, FFI budget or memory SLA.
- Rust formatting and whitespace checks passed for changed regions/new modules.
  Black is unavailable; no packages were installed. The three pre-existing
  unrelated FFI warnings remain; no new warnings were introduced.

No automatic pruning/storage schedule is introduced.

### Release observations

| Nodes | Workload | Median ms | Total state ID after workload |
| --- | --- | --- | --- |
| 1000 | bulk_insert | 1.002 | 1 |
| 1000 | bulk_set | 0.924 | 2 |
| 1000 | scalar_set | 2.291 | 1001 |
| 10000 | bulk_insert | 10.431 | 1 |
| 10000 | bulk_set | 9.506 | 2 |
| 10000 | scalar_set | 25.252 | 10001 |

Setter fixtures already contain one node-insertion state. These measurements
record this local native release run; scalar calls retain a different number of
versions and are not an equivalent-work speed comparison.
