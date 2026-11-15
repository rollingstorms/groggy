# Graph allocation profile and validation repair — 2026-11-14

Existing priorities 10.4/32.2. This repairs measured work without claiming the
remaining memory/recovery budget or broader workload corpus is complete.

## Evidence

`core_workflow_scale` and the new `core_workflow_allocations` share one fixture in
`benches/support/core_workflow.rs`. The original benchmark uses a no-op observer;
the allocation companion adds process-wide System allocation/reallocation
counters. It reports per-phase live requested bytes, additional peak, allocation
calls and requested bytes. Counter overhead means its elapsed times must be
compared with that same instrumented configuration. Requested bytes exclude
allocator metadata/retained pages and are distinct from `/usr/bin/time -l` RSS.

Three fresh release processes each at 1k/100k, with local builds/tests stopped,
are retained in `notes/benchmarks/core-allocation-before-2026-11-14.json` and
`core-allocation-after-2026-11-14.json`. The before fixture has an additional empty
`queries` checkpoint immediately after input; it records no work. Compare the
later query checkpoint and identically named archive phases.

At 100k, the dominant additional requested peaks are archive save **377.63 MiB**
and recovery **370.21 MiB**. Cold captures add 68.02 MiB; the effective read adds
136.47 MiB. Releasing read holders lowers live requested storage from about
322.06 to 189.97 MiB. Releasing the graph drops it to about 0.07 MiB before
recovery. The original whole-process peak alone did not expose these boundaries.

## Repair

History archive restore previously constructed a complete pinned read object
only to validate and immediately discard it. The new validation-only path shares
the exact duplicate-node and endpoint-membership rules with capture construction,
without allocating adjacency/incident indexes. Real read captures keep their
existing indexes and semantics. Branch-head validation is retained.

For each 100k sample, archive save and recovery each avoid **400,009 allocation
calls**. Median instrumented save is 1.435 → 1.378 s; recovery 0.849 → 0.780 s.
These are observations from three before and three after runs, not a universal
speedup claim. Requested peaks remain essentially unchanged. RSS ranges are
822.61–823.94 MiB before and 771.73–821.34 MiB after; the variability and retained
allocator pages do not establish a large general memory reduction.

## Verification and remaining work

The focused native regression checks valid sparse IDs, parallel edges and a loop,
then confirms validation-only and real capture reject duplicate nodes and inactive
endpoints with the same errors. The empty snapshot remains valid. Full gates are
recorded in the combined priority-pass receipt.

The large transient archive representation/decoding remains a concrete memory
risk. No format rewrite, dropped validation, retained-data pruning, automatic
checkpoint schedule or universal latency/memory target was introduced. Broader
payload/history/degree distributions and large recovery budgets remain 32.1–32.2.
