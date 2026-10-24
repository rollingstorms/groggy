# Atomic builder property loops — 2026-10-24

Closes flat checklist **09.4b.2h** as an extension of **30.1b**.

## Contract

- Native `iter.loop` recursively instantiates and certifies all descendants before
  any outer pipeline step executes, including zero-iteration loops.
- Certified loops share the outer Context's touched-column overlay. Property
  reads see earlier steps and iterations; only the outer successful run accepts
  one mixed property batch. A later iteration error discards every staged write.
- Check cancellation between iterations and between child steps, including empty
  bodies. The outer executor owns staging cleanup and final cancellation checks.
- Reject alternate `batch_plan` programs in atomic mode: their executor bypasses
  StepScope's property overlay. Malformed/unsupported descendants reject early.
- Legacy non-atomic loop/batch execution remains available. Transient loops read
  owner properties and return the final owned output for each output key.
- Nested definitions now have recursive preparation cost in atomic mode. Runtime
  memory remains the shared touched-column buffer plus existing loop variables.
  This is correctness work; no matched speedup or memory-budget claim is made.

## Remaining scope

Fused kernels, execution blocks, samplers, temporal and other uncertified families
remain explicit rejections. Atomic defaults, property-aware optimizer effects,
outer pipeline composition and durable run provenance remain parent work. No
Graphiti-specific behavior is introduced.

## Verification

See the matching status entry and commit for full gate results. Focused Python
coverage includes nested staged reads, a later-iteration division failure,
zero-iteration malformed/unsupported descendants, alternate batch plan rejection,
empty/transient execution and the public builder iteration API. Native coverage
checks recursive capability and cancellation of an empty loop body.

- Rebuilt the in-tree Python extension using the existing macOS dynamic-lookup
  linker settings; no package installation. Existing three FFI warnings remain.
- All **15 focused atomic-builder tests passed**; full root Python API gate:
  **1,091 passed, 1 skipped**, excluding the unrelated owner benchmark test.
- All **38 executable documentation examples passed**.
- Strict all-target native Clippy passed with warnings denied.
- Full offline/locked native test suite passed: **938 passed, 52 ignored**.
- `git diff --check` passed; owner development/benchmark changes remain separate.
