# Batch Plan v2 (Design Draft)

This document defines the expanded BatchPlan surface needed to support **all builder steps** in batch execution, with typed slots and scalar slots.

## Goals

- Cover all builder step types inside `builder.iterate()` loops.
- Support typed slots: float, int, bool, scalar (node-independent).
- Add conditionals (`compare`, `where`) and scalar reductions.
- Preserve deterministic behavior and in-place semantics.

## Slot Model

Slots are typed and can be either vector or scalar:

- `FloatVec`, `IntVec`, `BoolVec`: per-node vectors.
- `FloatScalar`, `IntScalar`, `BoolScalar`: scalars.

Rules:
- Vector ops expect vector slots.
- Scalar ops produce scalar slots.
- Broadcast converts scalar → vector.
- Arithmetic between scalar/vector yields vector (scalar broadcast).

## Proposed Instructions

### Loads/Stores

- `LoadNodeProp { dst, var_name }` → `*Vec`
- `StoreNodeProp { src, var_name }` → accepts `*Vec`
- `LoadScalar { dst, value }` → `FloatScalar`
- `BroadcastScalar { dst, scalar, reference }` → `FloatVec` (reference determines length)

### Unary Ops (vector)

- `Abs { dst, src }`
- `Clip { dst, src, min, max }`
- `Recip { dst, src, epsilon }`
- `Sqrt { dst, src }`
- `Exp { dst, src }`
- `Log { dst, src }`
- `Pow { dst, base, exp }` (vector or scalar exponent)

### Binary Ops (vector)

- `Add`, `Sub`, `Mul`, `Div`
- `Min`, `Max`

### Compare / Conditionals

- `Compare { dst, lhs, op, rhs }` → `BoolVec` (or `BoolScalar` if scalar inputs)
- `Where { dst, condition, if_true, if_false }` → vector/scalar

### Reductions (scalar)

- `ReduceScalar { dst, src, op }` → `*Scalar`
  - ops: sum, mean, min, max

### Graph Ops

- `NeighborAggregate { dst, src, operation, direction }` → `FloatVec`
- `NeighborMode { dst, src, tie_break, direction }` → `FloatVec`
- `CollectNeighborValues { dst, src, include_self }` → `List` (requires vector-of-vectors)
- `ModeList { dst, src, tie_break }` → `FloatVec`
- `NeighborModeUpdate` / `UpdateInPlace` (in-place semantics)

### Fused Ops

- `FusedMADD { dst, a, b, c }`
- `FusedAXPY { dst, alpha, x, y }`

## Lowering Strategy

1. Extend Python batch compiler to emit new instructions for:
   - `core.compare`, `core.where`
   - `core.reduce_scalar`, `core.broadcast_scalar`
   - `core.recip`, `core.sqrt`, `core.min`, `core.max`, `core.exp`, `core.log`, `core.pow`
2. Add validation + diagnostics for non-batchable steps.
3. For `map_nodes`, consider:
   - Lowering expression tree to BatchPlan (preferred)
   - Or keep as non-batchable until a separate evaluator is added.

## BatchExecutor Changes

- Extend slot storage to include scalar slots and bool vectors.
- Add per-instruction execution for new ops.
- Ensure scalar/vector broadcasting is explicit and deterministic.
- Preserve order for in-place updates.

## Open Questions

- How to represent `CollectNeighborValues` in BatchPlan? (likely a separate mode due to vector-of-vectors)
- Should `map_nodes` be lowered to BatchPlan or evaluated by a separate internal interpreter?
- Typed slots for `Text` or categorical values (needed by some builder ops)?
