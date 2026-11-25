# Portable sampling masks — 2026-11-25

Finite component of existing 03.2, 09.4b.2, 30.1 and AR7.8. Node, edge and
reservoir mask steps now share the selected memory/durable read boundary.
Stable flat counts remain 217 checked / 51 open; owner-bound subgraph producers
and composition containers keep the broader execution parents partial.

## Completed component checks

- [x] Prepare sealed node/edge/reservoir mask kernels without worker owner access.
- [x] Preserve stable ID populations, local seeded RNG and distinct edge instances.
- [x] Preserve count rejection, fraction rounding and bounded reservoir requests.
- [x] Preserve full source-membership validation even with a zero reservoir size.
- [x] Exercise sparse selected IDs, pending creations, nested bodies, memory/worker
  parity, effective-time recovery and outside-selection isolation.
- [x] Verify late rejection discards provisional IDs and leaves the journal intact;
  unattached masks do not publish graph history.

## Contract and architecture

Thin owner adapters delegate to the sealed kernels. The shared sampler remains
in Rust and uses one invocation-local RNG over ascending selected IDs. Fixed
seeds reproduce results for the same population; no cross-call streaming state
or persisted RNG is introduced. Parallel edges and self-loops are individual
edge IDs. Results are owned ID-to-one maps, not membership changes. Explicit
attachment alone writes graph attributes.

Count sampling rejects requests larger than the selected population. Fraction
sampling accepts finite values in `[0, 1]` and rounds halfway upward. Reservoir
sampling caps k at its source size and rejects foreign source IDs even at zero k.
Source maps and previously bound aliases remain independent outputs.

Subgraph-returning samplers and foreach containers remain owner-bound because
the public subgraph-array value carries owner-connected selections. This slice
does not construct fake graph owners on workers or change that public value kind.

## Evidence

`tests/durable_sampling_steps.rs` compares four seeds across memory, actual worker
execution and reopening. Sparse selected nodes, pending node/edge creation,
parallel/self-loop edges and nested ordinary bodies preserve the same masks.
Count, containment and effective-time assertions are independent of RNG output.
Rejection and no-publication tests cover ID allocation and journal bytes.

`tests/test_durable_sampling_steps.py` covers node/edge counts and fractions,
halfway rounding, repeatability, bounded reservoir subsets, effective-time
recovery, rejection and unattached computation. Existing native sampling tests
retain cancellation, foreign-source zero-k rejection and output-preservation
coverage. The public example is in [durable pipelines](../../docs/api/durable-pipelines.md).

Local qualification passes **1,690 native tests** (54 ignored), **3,569 packaged
Python tests** (one skipped, 23 expected warnings) and **301 documentation
examples**. Strict all-target Clippy, formatting, whitespace, Python 3.8 syntax
and all 62 packaged Python/stub byte comparisons pass. All 500 production source
files match runtime `bd1c42ffdb9a333f0166b531894527a6f5bc4e1e`.
See [local receipt](../benchmarks/portable-sampling-local-qualification-2026-11-25.json)
and [wheel receipt](../benchmarks/portable-sampling-wheel-qualification-2026-11-25.json).
The local Python environment is the isolated recovered Python 3.9 environment,
whose full build string is recorded in the receipt. No new performance or
platform claim is made by this adapter migration.
