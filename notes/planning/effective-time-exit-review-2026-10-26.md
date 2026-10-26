# 16.2 exit review — reconstructed effective-time graph

## Completed contract

- Branch/head selection is captured when the plan is prepared. Later mutations,
  branch changes, GC/cache invalidation and owner destruction do not change it.
- Typed topology/property events replay at signed-microsecond effective times;
  same-key ties use accepted version order. Explicit null and absent keys remain
  distinct, and lifecycle boundaries prevent old properties leaking into a new
  activation. All currently represented delta events are supported.
- Checked publication validates complete affected endpoint timelines atomically
  before allocation/mutation/durable append. Historical removed edges participate;
  the reader also rejects invalid raw intervals through T.
- Optional recorded cutoff remains explicit. Derived graph selections and table
  source captures retain clocks/head/referents, without implicit head writeback.
- Frozen group revisions retain selected source values, topology, direction and
  nested references independently of the current working graph. Revisions change
  when selected source state/values change; old revisions remain readable.
- Retained sources normalize to selected rows/direct dependencies; no mutable owner
  or full parent graph is kept. Canonical reads use GraphView.meta_subgraph.
- Durable typed group documents preserve nested dependencies, source clocks and
  explicit legacy aliases. Persisted references are checked; reopen/full-journal
  backup recover the same source payload. Duplicate retention adds no frame/state.
- Legacy versions with missing clocks accept caller-supplied effective/recorded
  times (native per-change overrides also supported). Membership-only references
  accept explicit matching source views. Known precise clocks/frozen references
  reject overrides; these inputs are read-local, not guessed or written back.
- Already resolved merges replay the first parent and the resolution delta,
  matching ordinary state reconstruction. Secondary-parent mutations are not
  applied twice. Native resolved merge publication normalizes precise clocks.
- A selected-head ancestry cache and lazy sorted event index support warm O(1)
  preparation and binary effective-prefix selection. Index construction/replay and
  frozen source preparation run outside the GIL where detached inputs permit it.
  Branch/head changes select a fresh context; GC releases the cache. Existing
  plans remain independently owned.

## Scope separation (existing checklist items)

11.2: add explicit property deletion/unset and finish coalescing/retained property
history. There is no unset mutation hidden or approximated as null in 16.2.
12.2: whole configuration restoration, owned temporal snapshot materialization and
repeated-switch allocation bounds. Canonical frozen reads are implemented here.
14.2: three-way conflict detection and caller merge policies. Temporal replay uses
an already accepted resolution; it does not invent a planner policy.
10.3/10.4: arbitrary raw-pool/existing-graph import, storage migrations, materialized
checkpoints, compaction and bounded recovery. Group records are an additive typed
journal feature; old binaries cannot read new group records, while existing records
remain readable. These were separate requirements before this item.
26.1: broad connected NodeView/Subgraph adapter migration; use canonical GraphView
for frozen source reads. No blanket claim that legacy expand adapters are frozen.
32/10.4/A02: larger workload/RSS/concurrency budgets and read/recovery checkpoint
optimization. Full graph reconstruction remains proportional to admitted content.

The closure does not redefine these scopes or add rows. 16.2 is complete against
its effective-state read/write contract, including explicit legacy/reference input
handling. Unrecoverable information is required from the caller rather than guessed.

## Benchmark receipt

`cargo bench --offline --locked --bench effective_time_reads`, release; same chain
and 101-version fixture as the first read receipt. Five warm repetitions per size.

| Nodes | Cold index + reconstruction | Warm reconstruction median | Warm empty prefix |
| --- | --- | --- | --- |
| 1,000 | 2.62 ms | 1.49 ms | 0.50 µs |
| 10,000 | 14.83 ms | 8.15 ms | 0.54 µs |
| 100,000 | 154.97 ms | 104.52 ms | 0.42 µs |

Machine receipt: `notes/benchmarks/effective-time-reads-indexed-2026-10-25.json`.
Cold and warm costs are distinguished; these are local bounded current costs, not
statistically controlled speedups, RSS/FFI/production capacity or O(1) full reads.
Group/source normalization copies selected payloads and is not an O(1) operation.

## Verification

Native completion cases exercise nested revisions, deactivation/owner destruction,
durable reopen/backup, read-local clocks/cutoffs, first-parent merge resolutions,
legacy source membership/alias persistence, frozen override rejection, and checked
unknown-reference rejection/ID reuse. Python completion cases cover the rebuilt
API, typed embeddings/bytes/direction, source-table pins, legacy input validation,
head/branch cache isolation and idempotent durable group retention.

Full gate results and the commit are recorded in the roadmap status log.

Final gate: 1,184 native passed / 51 ignored; 1,913 root Python passed / 1 skipped;
139 executable documentation examples; strict all-target Clippy passed. Scoped
rustfmt, stub AST parsing and git diff --check passed. The in-tree extension was
rebuilt without external installs. Owner benchmark work was preserved/excluded.
