# Prepared native meta-node creation — 2026-10-25

Existing scope: 09.4b.2 / 05.3. No new checkbox row.

Graph::create_meta_node now prepares a complete typed batch in manual mode as well
as publishing mode. Required meta identity/reference properties are injected only
by trusted native code after validation. The legacy allocate-first rollback path
and its no-op structural validator are removed. Public signatures stay stable.

Invalid retained references reject before ID allocation; a valid group may contain
inactive source entities. Publishing requires a clean head and accepts one state;
manual mode preserves pending edits. Stored group identity/count/membership do not
change during creation. Immutable snapshots retain exact required properties.

Reference-existence checks now use a borrowed HashMap membership lookup rather
than cloning the entire group's node/edge sets. Existing get_subgraph still returns
owned membership for callers that need it. Global get_meta_nodes still scans all
active nodes; this change does not implement a meta index or portable read pins.

## Verification

Two new native regressions cover both publication policies, inactive membership,
identity/reference properties, exact acceptance clocks, retained snapshots, invalid
references with unconsumed IDs and dirty publishing/manual compatibility. Full
gates: 1,017 native passed / 52 ignored; 1,730 Python passed / 1 skipped;
87 documentation examples; in-tree FFI build; strict all-target Clippy.

## Bounded optimized observations

`benches/meta_creation.rs`: five creations referencing one group per fixture.
Fixture creation excluded; creation includes preparation/storage and optional
publication. Separate global get_meta_nodes timing includes its active-node scan.
Median milliseconds:

| Group members | Manual creation | Publishing creation | Manual global selection | Publishing global selection |
| ---: | ---: | ---: | ---: | ---: |
| 1,000 | 0.0049 | 0.0071 | 0.1516 | 0.1547 |
| 10,000 | 0.0069 | 0.0090 | 1.6567 | 1.6699 |
| 100,000 | 0.0128 | 0.0269 | 21.4000 | 24.2195 |

Receipt: `../benchmarks/meta-creation-2026-10-25.json`. Creation does not enumerate
group membership; these local short timings are not asymptotic/latency guarantees.
No matched speedup, allocation/RSS, FFI, complete meta-index or portable-pool claim.
Remaining native defaults, mutable storage escapes, compound pipelines and pooled
retention retain their existing checklist scope.
