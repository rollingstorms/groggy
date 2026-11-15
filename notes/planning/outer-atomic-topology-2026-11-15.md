# Outer atomic topology execution — 2026-11-15

Implemented the next chunk of the expanded 0.6.0 target. Existing rows 09.4b.2,
AR3, 30.1 and AR7.8 gain completed component evidence. The fixed inventory stays
215 checked / 53 unchecked; 44 unchecked rows are in release scope and nine remain
deferred. No additional rows or automatic broad parent closures.

## Implementation

- `Algorithm::atomic_mutation_effects` declares supported shared staging,
  pending-topology input safety and persistence requirements. Existing property
  certification retains its conservative default; unknown implementations reject.
  Builder capabilities derive from existing step effects without allocating a
  capability vector or creating another executor.
- The outer coordinator checks every algorithm's ordering and persistence before
  any prefix executes. Property calculations precede topology; certified topology,
  explicit pending captures and immutable-input computation can follow.
- Child builders borrow the outer batch and retain the original execution
  selection. Provisional IDs never pass through live-owner filtering. Creation
  positions are global to the shared batch; step variables remain builder-local.
- One `accept_selection` path now serves local and outer execution: validate and
  publish once, then project accepted additions and surviving selected originals.
  It preserves exact edge selection and excludes cascades without inducing edges.
- Cancellation, runtime/final validation errors and coordinate/owner mismatches
  discard staging and restore previous Context outputs. Context reuse is tested.
  Existing non-atomic execution continues publishing per algorithm.
- Public Python call shapes and the native `run_atomic_properties` entry point
  remain compatible. No FFI business logic or new numerical/graph kernels.

## Regression inventory

Four native integration tests cover shared creation/capture/cascade plans,
final timeline failures and late invalid targets, preflight rejection and exact
edge selection. Existing native coordinator tests add two cases for preflight
before externally observable prefixes and cancellation/foreign-owner rejection
after topology, including Context reuse.

Twenty-four Python cases cover directed/undirected and automatic/manual owners;
sparse lifetime IDs; parallel edges and loops; explicit effective clocks;
all 13 captured classic kernels observing earlier created entities; immutable
intermediates; archive/history round trips; invalid targets, timelines and duplicate
operations; persistence/order guards; empty plans; and ordinary outer publication.

The cascade case removes a node in one builder, captures the temporary missing
edge, then revises that edge in a later builder with an earlier effective time.
The final canonical plan rescues the edge. This verifies why original selected
membership must survive until outer acceptance.

Earlier tests that expected all outer topology/staged captures to reject now
exercise the supported behavior. Their nonpersistent and unadapted-operation
rejection checks remain. General topology callbacks are not silently certified.

## Verification

Executed on macOS arm64 with Rust 1.96.1 and CPython 3.9.6:

| Gate | Result |
| --- | --- |
| Full locked/offline native suite, including doctests | 1,579 passed; 54 ignored; zero failed; 223 suite results |
| Maintained root Python surface | 2,775 passed; one skipped; 14 established builder warnings |
| Curated executable documentation | 271 examples passed |
| Strict all-target Clippy | Passed |
| Rust formatting and diff whitespace checks | Passed |

Commands used the in-tree target directory (`CARGO_TARGET_DIR=python-groggy/target`)
and disabled incremental compilation. Core gates:

```sh
cargo test --offline --locked
cargo clippy --offline --locked --all-targets -- -D warnings
cargo fmt --all -- --check
```

The extension was rebuilt with `maturin develop --release --skip-install`, the
configured Python 3.9 interpreter and `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=0`.
Maintained Python/docs verification used `PYTHONPATH=python-groggy/python` and
`notes/scripts/verify_immutable_view_migration.py`. The focused Python suite also
passed 151 cases; strengthened cascade/lifetime cases are included in the final
full run. Existing binding compiler warnings were unchanged. No throughput or
new cross-platform qualification claim is made by this correctness chunk.

The rebuilt in-tree extension uses the current Rust implementation. Python wheels
were not installed. Existing unrelated review/benchmark work is untouched.
The earlier 19-platform candidate qualification remains historical evidence;
the expanded release will repeat final artifact/platform gates after implementation.

## Remaining boundaries and next chunk

The next component is supported property calculation/attachment after pending
topology, using the same canonical scope and rollback model. General topology
loops, durable pipeline dispatch, remaining facade/DSL lowering, naming/cache
policy, pipeline-history policy and unsupported custom effects remain open.

Explicit pending captures still materialize selected data and retain
`write_state_id=None` after success. Refresh an owner capture for accepted-state
writeback. Owner-thread/GIL behavior is unchanged. Custom capability declarations
are trusted contracts, not a sandbox for arbitrary callback side effects.

User-facing contract and executable example:
[outer atomic topology](../../docs/api/outer-atomic-topology.md).
