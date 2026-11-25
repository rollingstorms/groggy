# Portable message-passing blocks — 2026-11-25

Finite component of existing 03.2, 09.4b.2, 30.1, 30.3 and AR7.8. Supported
`core.execution_block` message-pass bodies now share memory/durable execution,
including the existing public builder facade. Stable flat counts remain 217
checked / 51 open; other frontends and composition parents remain partial.

## Completed component checks

- [x] Prepare and certify the existing message-pass body on the owner; expose its
  sealed native read kernel without creating a second block interpreter.
- [x] Preserve ordered/synchronous visibility, target row order, explicit
  directions, parallel incidences and the existing self-inclusion rule.
- [x] Preserve collection/mode, sum, checked arithmetic, comparisons, conditional
  selection, constants and apply, including exact large integer labels.
- [x] Compose blocks inside ordinary loops with selected pending creation and
  endpoint edits, one outer publication and exact timestamp recovery.
- [x] Keep streaming, invalid directions and unsupported bodies explicit in
  preflight, including in zero-iteration loops; preserve late-error rollback.
- [x] Remove the unused private owner-based CSR wrapper after its last caller
  migrates, leaving selected adjacency construction on the shared read scope.

## Contract and architecture

Owner adapters retain their existing unsupported-atomic-body check and delegate
execution to the sealed read implementation. Private body evaluation helpers now
use `ReadStepScope`; writable graph owners and registry dispatch do not cross
workers. Body parsing/certification is performed before detached execution.
Public APIs, supported operation definitions and result kinds are unchanged.

Ordered mode sees prior updates in the target's row order. Maps become ascending
ID columns; explicit columns retain their row order. Synchronous mode reads the
original input throughout one invocation. Default direction stays incoming;
explicit in/out/undirected mappings keep their prior behavior. Parallel neighbors
remain distinct and loops count once. `include_self` adds a node only if absent;
this is deliberately the established block rule, not the standalone collector's
additive self-inclusion rule. Inside-block neighbor aggregation remains an
unweighted sum; other variants use standalone aggregation steps.

## Evidence

`tests/durable_message_blocks.rs` compares 24 directedness/direction/ordering/self
cases against independent incidence arithmetic, memory execution, actual worker
execution and reopening at explicit property clocks. It includes noncontiguous
IDs, selected-outside isolation, provisional creation and endpoint revision,
parallel edges and self-loops. Negative cases reject unsupported/streaming bodies
even in zero loops and discard the whole staged publication on checked overflow.

`tests/test_durable_message_blocks.py` exercises the public Python step and
`AlgorithmBuilder.message_pass` surfaces, including the two visibility modes,
all six comparison operators/conditionals and collection/mode on integer labels
above f64's exact-integer range. Existing column-order and failure-preservation
unit tests remain in `execution_block.rs`.

Local qualification passes **1,678 native tests** (54 ignored), **3,513 packaged
Python tests** (one skipped), and **297 documentation examples**. The 23 expected
validation warnings comprise 14 from the earlier suite and nine additional
instances of the established message-target redefinition warning in the new
public-facade cases. Strict all-target Clippy, formatting, whitespace, Python
3.8 syntax and all 62 packaged Python/stub byte comparisons pass. All 500
production source files match runtime `233998982f1ee6d72d791e31fb6468e4b9d73612`.
See [local receipt](../benchmarks/portable-block-local-qualification-2026-11-25.json)
and [wheel receipt](../benchmarks/portable-block-wheel-qualification-2026-11-25.json). See the executable
[durable pipeline contract](../../docs/api/durable-pipelines.md). No new platform
qualification or performance improvement is claimed by this adapter migration.

## Remaining scope

Streaming is still unsupported. Other containers, retained history producers,
edge-aware registers, normalization options, fixed-child/provisional topology
and remaining alternate IR routes stay in the existing execution parents. A14
naming/cache/history and A15 production budgets remain unanswered.
