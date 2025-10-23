# Architecture Maintenance Notes

This appendix tracks structural updates that affect the overall architecture documentation.

## October 2024

### Builder + Pipeline Integration
- Added `builder.step_pipeline` algorithm registration so Python-built specs execute in Rust.
- Extended step primitives (e.g., `core.normalize_node_values` now supports `sum`, `max`, `minmax`).
- `Subgraph.apply()` accepts single algorithms, lists, or pipeline objects, all funneled through the Rust executor.
- Python `builder.py` serialises step specs to JSON and the FFI now passes JSON payloads into the Rust pipeline builder.

### Documentation Touch Points
- New guides: `docs/guide/pipeline.md` and `docs/guide/builder.md`.
- Updated homepage quick links, navigation, and algorithm guide to surface the three execution paths (`apply`, list, pipeline).
- Concept and roadmap documents refreshed to mark Phase 5.1 and related Phase 6 tasks as complete.

Keep this section updated when major architectural surfaces evolve (e.g., new step primitives, FFI changes, or execution flows).
