# Effective-time read slice: 16.2 / 16.3

## Implemented

Native Graph and DurableGraph expose `prepare_read_at(T, recorded_by)` and
`read_view_at(T, recorded_by)`. The detached plan owns immutable Arc publication
records from the selected head's single-parent ancestry; replay does not borrow
the working pool and can run on another thread. Python captures the plan under
the GIL and releases the GIL for native replay. No checkpoint/delta is written.

Changes replay by signed-microsecond effective time and accepted ancestry order.
Same-key/same-effective-time corrections use later accepted order. Lifecycle
boundaries clear properties; writes outside an active lifetime do not leak into
reactivation. Direction, topology, parallel IDs, loops and ordinary typed values
remain exact. Null remains a value and a never-assigned key remains absent.

An explicit recorded cutoff filters publication acceptance clocks. The captured
head is fixed even if replay runs after subsequent publication/store destruction.
Selections and table source captures retain ReadTime (branch/head/effective time/
optional recorded cutoff). Temporal captures are read-only, preventing implicit
writeback against a head whose values they do not represent. Durable reopen
reconstructs the same effective-time values using retained publication metadata.

Replay checks edge/endpoint closure at every relevant effective-time boundary
through T, using incident-edge tracking rather than rescanning all edges for each
boundary. A bad earlier interval cannot disappear behind a valid final state.

## Explicit limits / remaining 16.2 work

- Accepting writes does not yet validate the entire effective topology timeline
  atomically. The read detects already-retained invalid timelines. Next priority
  is an incremental endpoint/lifetime validator in checked publication, including
  durable preparation before journal append; replaying the full history on every
  ordinary write would violate the engine's performance goals.
- Pooled subgraph-reference values and merge ancestry reject explicitly. Retained
  meta/group source reconstruction stays 12.2/16.2.
- Legacy manual versions lack precise per-change clocks and reject. Pending manual
  edits must be published first; they are never silently assigned guessed times.
- Property unset remains 11.2. This reader does not invent a deletion tombstone or
  conflate null with absence.
- Full lineage preparation/replay has no effective-time index/checkpoint reuse.

16.3's optional cutoff and derived-pin behavior is implemented for the supported
precisely timed lineage. 16.2 remains partial for the limits above. These are
existing checklist scopes; no new rows were added.

## Performance receipt

`cargo bench --bench effective_time_reads` (locked/offline, release), five runs
per case; chain created at time zero, 100 scalar corrections and read at time 50.
Checked node/edge counts and score against the expected historical state.

| Nodes | Edges | Versions | Prepare median | Replay median |
| --- | --- | --- | --- | --- |
| 1,000 | 999 | 101 | 16.8 µs | 2.26 ms |
| 10,000 | 9,999 | 101 | 15.6 µs | 12.84 ms |
| 100,000 | 99,999 | 101 | 34.7 µs | 147.10 ms |

Receipt: `notes/benchmarks/effective-time-reads-2026-10-25.json`. These are local
current costs, not a speedup, indexed-lookup, RSS, opening-time, concurrent-writer
or production-capacity claim. Prior journal-opening evidence remains separate.

## Verification

- Seven targeted native effective-read regressions: corrections, branch isolation,
  detached head capture, recorded cutoff, derived table clocks, null/absence,
  lifetime removal, invalid earlier endpoint intervals, negative/extreme clocks,
  clean/manual rejection, durable reopen and replay after store destruction.
- Eight Python regressions exercise the rebuilt Graph/DurableGraph APIs, temporal
  filtering/traversal/table export, immutable capture and invalid clock types.
- Full gate: 1,171 native passed / 51 ignored; 1,900 root Python passed /
  1 skipped; 136 executable documentation examples; strict all-target Clippy
  passed. Scoped rustfmt, stub AST parse and git diff --check passed.
