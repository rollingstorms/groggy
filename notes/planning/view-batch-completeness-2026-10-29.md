# Complete scoped view publication — 2026-10-29

Priority 26.1/AR4 checked writes, with the existing 09.4b.2 compound boundary.
No new checklist row or broad default-ownership/transaction closure.

## Reproduced and repaired

The original Python extension failed six targeted regressions: node/edge property
unsets outside the selected scope were accepted; excluded edge revisions and
revisions to excluded endpoints were accepted; refreshed properties retained unset
values; and a rewired edge surviving a removed old endpoint disappeared from the
returned view. The graph coordinator itself accepted the correct graph write,
so the latter failures produced an inconsistent immutable result.

Scope validation now covers every prepared mutation target, including both unset
families, existing revised edge IDs and new revision endpoints. Identity bindings
remain covered through the property columns produced by canonical preparation.
Cascade checks still reject deletion of any excluded edge instance.

Projection now follows the coordinator's operation order: additions, endpoint
revisions, assignments, unsets and removals/cascades. Unset rows are removed rather
than represented by old values or null. The refreshed full payload and selected
membership are fully prepared before mutation and can authorize the next write.
Returned captures also retain the current immutable group catalog, so unchanged
meta-node references still resolve after a property batch or owner destruction.

Both scope validation and projection exhaustively destructure PreparedBatch.
Adding a future prepared field now fails compilation at both consumers until it
is explicitly considered. Clocks and identity bindings have documented reasons
for not adding separate targets; actual mutations are not hidden behind `..`.

The source consistency check uses the shared canonical exact retained-value
comparison, preserving distinctions in float width, signed zero and NaN bits.
Empty/missing property rows both represent absence. NaN remains supported; its
unchanged-source regression already passed before the fix and is retained here.

## Verification

Native cases exercise whole-request rejection with a valid staged prefix and
unused IDs, mixed node creation/edge creation/rewiring/property absence/identity/
cascade projection, exact effective-time states, immutable old input, next-write
authorization and retained meta-source worker lifetime. A private native unit
checks exact payload comparison and empty row equivalence. Python covers the same
API failures, reversal/self-loop endpoints, current state, and durable rejection
without journal-byte changes followed by successful mixed publication/reopen.

Full gates and timing receipts are recorded in the priority-window log. The
view_batch_publication release fixture measures the existing full projection cost
before optimizing source payload reuse; no O(1) view-write claim is made.

## Remaining scope

Actual immutable defaults/filter returns, effectful facade migration and general
multi-stage topology/custom effects remain the original open priorities. This
repair establishes complete single GraphBatch semantics through the existing
view-write boundary; it does not synthesize a new transaction API or policy.
