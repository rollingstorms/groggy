# Matrix module consolidation measurements — 2026-10-22

AR7.2 moves implementations into one canonical matrix namespace. Legacy paths
remain aliases; no numerical algorithm or precision policy is intentionally
changed. The reference is `numerical_completion_current_2026-10-22.json` from
`b19aa9b7`. The post-move receipt is
`benches/numerical_matrix_consolidation_2026-10-22.json`.

Rebuild the existing 25-case optimized corpus, then rerun the already-built binary
with no Cargo/rustc jobs running. All seven checksums in every workload agree
with the reference within 1e-13 relative/absolute tolerance. Allocator instrumentation
and seven-sample medians are unchanged. Timings are regression observations,
not evidence of an optimization; normal machine noise remains possible.

Full native suites, import/type compatibility fixtures and 455 rebuilt Python
regressions passed. The broader numerical size/fill/dispatch and strict final
review gates remain separate from this module migration.
