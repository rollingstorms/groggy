# Native three-way snapshot merge planning

The Rust state layer now has a pure merge planner. It does not mutate a Graph or
publish a version. Graph/DurableGraph publication and Python APIs are now available; see
[checked graph merges](graph-merges.md) for the complete acceptance contract.

## Choose a real base

`HistoryForest.merge_bases(left, right)` returns every best common ancestor, sorted
by state ID. An ancestor is excluded if another common ancestor descends from it.
This traverses each input ancestry and its common-parent edges once, rather than
running pairwise ancestry searches. Unknown/zero IDs reject. Unrelated retained
roots return an empty list. Criss-cross merge ancestry can have several valid bases.

`find_common_ancestor` returns the unique base or None for unrelated roots. Multiple
best bases reject instead of choosing by parent/traversal order. The previous
stack search could pick an older ancestor through a second parent before finding
a newer shared ancestor. Callers must explicitly select an ambiguous base or supply
a separately defined recursive-base strategy; none is inferred here.

## Plan and inspect

`state::plan_snapshot_merge(base, left, right, target_state)` accepts complete
logical GraphSnapshots from the same identity domain and an explicit base. It
validates structural inputs and equal directionality. It combines independent
entity/property changes, ordered endpoint revisions and distinct edge IDs. Separate
edge IDs stay separate, including parallel pairs and loops; matching endpoints do
not imply identical facts. Node order and conflict traversal are deterministic.

`SnapshotMergePlan.conflicts()` exposes typed inputs for node/edge delete-modify or
colliding-creation conflicts, divergent ordered endpoints, and individual property
conflicts. Property absence differs from Null. Whole logical types and exact float
bits, including NaN payloads and signed zero, remain distinguishable. Input property
records are borrowed during planning; owned output and conflict values survive the
inputs. Sorted entity/property unions add ordering cost; no constant-time or bulk
mutation performance guarantee is made for snapshot planning.

`into_snapshot()` succeeds only for a conflict-free candidate. The compatibility
`state::merge_snapshots` helper now delegates to that path instead of panicking.
There is no implicit winning side. A target_state labels the logical output; it
allocates no actual history version.

## Caller resolution

`resolve_with(callback)` consumes a plan and asks for explicit typed
`SnapshotMergeResolution` answers. A node/edge conflict accepts a complete entity
record or explicit absence. A property conflict accepts a whole typed value or
absence. An endpoint conflict accepts an ordered pair. Wrong answer kinds or
malformed node/edge record shapes reject. The executable Rust API example and
integration tests show choosing individual left/right values; callers can supply
custom records or abort with an error.

After direct answers, topology is recomputed. Provisional topology conflicts in
the initial inventory can disappear or new dependencies can arise. For each
remaining dangling edge, the caller must explicitly choose `DropEdge` or supply
exactly its missing nodes through `EndpointNodes`, including their chosen
properties. Extra/duplicate endpoints reject. No implicit cascade, endpoint-node
resurrection or entity-ID allocation occurs. All final structural invariants are
validated before returning a usable snapshot. Failure leaves every input and
owning graph unchanged.

## Ownership boundary and remaining acceptance work

`state::validate_snapshot` now implements the previously unfinished public
structural validator. Merge planning, retained integrity checking and explicit
restoration share it. Duplicate active nodes, inactive endpoints and visible
properties on inactive entities reject. Empty inactive legacy property maps are
harmless. This validator does not prove allocated IDs, external-name uniqueness,
valid pooled/group references, schema, write provenance or effective-time validity.

The owning coordinator now supplies head/branch/base and identity/schema/source
guards, a complete exact first-parent delta, explicit/default clocks, one two-parent
publication and durable synced delta replay. Python provides prepare/resolve/finish
and a checked writer convenience. Importing a logical output through an untracked
raw snapshot is not an accepted merge path. The planner itself has no wire format;
see the linked complete API for journal/archive compatibility and verified limits.

## Owner-bound capture and read-only preflight

Native `Graph.prepare_merge_read(right_state, base, expected_state)` now captures
retained recorded inputs with the graph's identity and selected branch/head.
`base=None` selects only a unique best common ancestor. An explicit nonzero base
must be an ancestor of both parents. Unrelated roots require the explicit empty
base `Some(0)`. Parents must be distinct retained published versions; dirty/stale,
unknown, identical and non-common selections reject before reconstruction.

`GraphMergeReadPlan.execute()` produces a `GraphMergePlan` off-owner. Inspect its
`parents()`, `base_state()` and `conflicts()`, then use `into_outcome()` for a
conflict-free plan or `resolve_with()` for explicit typed choices. Read plan,
conflict plan and resolved outcome are owned Send + Sync values. A cold capture
can reconstruct after owner changes/destruction because it owns immutable recorded
inputs. It does not permanently pin the owner's lookup or grant write authority.

`Graph.validate_merge_outcome(outcome, schema)` rechecks owner, current branch/head,
retained parents/base, structural/direction compatibility, allocated/unpurged IDs,
group references, external identities and an optional complete-target schema.
Intervening valid pool compaction is safe; collected parent/base lookups reject.
The right parent is an explicit version, not a subscription to a moving branch.
No versions, IDs, properties, branch heads, reservations or clocks change during
capture, resolution or validation. Resolved `snapshot().state_id` is zero as an
unpublished marker; it is not an accepted graph version.

Preflight alone is not durable/current-policy authority for later use. Publication
must recheck the live owner and validate the exact resolved first-parent delta,
effective timeline and installed durable schema immediately before acceptance.
Use the checked `Graph.finish_merge` or DurableGraph counterpart for publication;
they rebuild/recheck current policy and exact resolution clocks before acceptance.
Do not install the logical output through raw snapshot imports.
