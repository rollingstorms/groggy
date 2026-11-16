# Pending value copying — 2026-11-16

Completed bounded performance repair within 30.1, AR7.8 and 32.3. The fixed
checklist inventory and broad parent requirements remain unchanged.

## Change

Selected pending captures previously cloned each accepted attribute row, then
cloned staged values over those copies. Large accepted vectors/text/bytes that
were being replaced were allocated and discarded inside that same capture.

The shared node/edge materializer now copies an accepted value only if that exact
entity/property has no staged assignment. The existing column-first overlay copies
the final assignment once. Explicit Null remains a replacement, not absence;
staging a column for one row does not hide its other accepted rows. Empty staged
column maps retain the original direct-clone fast path. There is no shared mutable
value, new cache, history/pool clone or publication-policy change.

## Matched allocation evidence

Native Rust 1.96.1 release, three fresh processes per revision, five observations
per case. Baseline production source: 6523a66a, with the same measurement workload.
Receipts record the actual production-file and measurement-source SHA256 values.
Two redundant String identity conversions were removed from the committed harness
after measurement for strict Clippy; the receipts preserve the measured source
hash rather than relabeling it as the final cosmetic source. Fixture and workload
behavior are unchanged.

Each case captures two f64 vector columns after zero, one or two scalar
replacements, with 100/1,000 node or edge entities and vector widths 8/1,024.
All values, selected membership, unchanged prior input and zero/one publication
are asserted. Execution/staging/capture/emission/publication are timed; fixture,
initial accepted pin, construction, assertions, destruction and compilation are not.
Requested allocation volume counts allocation/reallocation requests, not RSS,
allocator overhead or total retained memory. Peak extra requested bytes are also
recorded, and show much smaller changes than total allocation volume.

Medians of all fifteen observations, 1,000 entities / width 1,024:

| Entity | Replaced columns | Before requested bytes | After requested bytes | Before ms | After ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Node | 0 | 17,066,320 | 17,066,320 | 2.033 | 1.601 |
| Node | 1 | 27,580,967 | 18,788,967 | 8.792 | 8.266 |
| Node | 2 | 37,450,650 | 20,458,650 | 17.139 | 15.329 |
| Edge | 0 | 18,290,599 | 18,290,599 | 2.913 | 2.360 |
| Edge | 1 | 28,280,942 | 20,080,942 | 9.689 | 9.156 |
| Edge | 2 | 38,150,461 | 21,750,461 | 17.716 | 16.152 |

Two-column requested allocation volume falls about 45.4% for nodes and 43.0% for
edges in this fixture. The unchanged controls show substantial timing variation
with identical allocation counts: do not attribute the timing ratios entirely to
the implementation or infer a general speedup/latency budget. Wide captures still
copy every selected final logical value and prepare validated adjacency; this
repair avoids redundant copies rather than claiming zero-copy snapshots. When
staged columns exist, each accepted key additionally looks up its column/row in
the BTree overlay; very wide scalar-column workloads are not measured here. The
empty-overlay fast path avoids those lookups.

Raw receipts: `../benchmarks/pending-overwrites-{before,after}-2026-11-16-{1,2,3}.json`.
Reproduce with `cargo bench --offline --locked --bench pending_frame_allocations
-- --overwritten-values` and GROGGY_ATOMIC_BUILDER_PROFILE selecting a new path.
The original child-frame workload remains the default for this executable.

## Qualification

The targeted native PropertyBatch suite passes **19 tests**, including the new
mixed-type node/edge replacement regression. It covers explicit null, f64 bit
semantics, bytes, JSON/vectors, unrelated rows, new keys, multiple pending revisions,
publication and old immutable inputs. Strict all-target Clippy and Rust formatting
pass after removing the two redundant conversions. Final full native and fresh
Python/artifact gates follow the next implementation slice; no artifact is claimed
qualified by this measurement alone.
