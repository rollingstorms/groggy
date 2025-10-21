# Trait Delegation System Implementation Report

## Background
We originally sketched a Rust trait delegation layer intended to expose shared behaviors across Graph, Subgraph, Table, and Array entities. The concept was never integrated; instead, multiple Python-facing classes rely on dynamic `__getattr__` fallbacks to tunnel through to other objects. That indirection hides significant functionality from users and tooling, carries non-trivial runtime cost, and bypasses the Rust trait guarantees we already maintain in `src/traits/`.

## Current State Findings
- `PyGraph.__getattr__` iterates node/edge attributes and for arbitrary names constructs a full `Subgraph` before consulting Python attributes, making delegated APIs opaque and expensive (`python-groggy/src/ffi/api/graph.rs`).
- Neighborhood wrappers simply forward every unknown attribute to a materialized subgraph (`python-groggy/src/ffi/subgraphs/neighborhood.rs`), retaining the opacity.
- Table and array types defer missing members to their base counterparts or per-element application through generic getattr (`python-groggy/src/ffi/storage/table.rs`, `python-groggy/src/ffi/storage/array.rs`).
- Documentation already calls out the discoverability gap—delegated methods are flagged as “🟡” because static inspection cannot locate them (`documentation/planning/SYSTEMATIC_METHOD_REPORT_WITH_DELEGATION.md`).
- The experimental delegation prototype under `python-groggy/src/ffi/delegation/` defines traits but leaves every method as `PyNotImplementedError`, so nothing currently uses it.
- Core Rust traits in `src/traits/` remain comprehensive and production hardened, ready to be surfaced once the FFI bridge is wired up.

## Proposed Architecture
1. **First-class Delegation Module**: Finish `python-groggy/src/ffi/delegation/` so PyO3 wrappers implement `SubgraphOps`, `TableOps`, `GraphOps`, and array traits by borrowing existing core handles, performing error translation, and releasing the GIL for long-running operations.
2. **Explicit PyO3 Methods**: Expose trait-backed behaviors as concrete `#[pymethods]` on Graph/Subgraph/Table/Array classes. Python users and type stubs then discover the API without relying on magic attribute lookup.
3. **Targeted `__getattr__`**: Retain dynamic lookup only where necessary (e.g., column dictionaries, truly dynamic attribute access). All algorithmic delegation should route through explicit methods to improve safety and clarity.
4. **Tooling Integration**: Update `scripts/generate_stubs.py` so `.pyi` files list the trait-provided methods, keeping editors and documentation aligned.
5. **Trait Coverage Review**: Audit core trait definitions versus the dynamic methods currently exposed, filling any gaps or providing shims before cutting over.

## Migration Plan & Estimated Effort
| Phase | Scope | Estimated Effort | Notes |
| --- | --- | --- | --- |
| 0 | Trait surface audit, success criteria, dynamic access inventory | 3–4 days (1 engineer) | Produce mapping document and flag gaps. |
| 1 | Implement Rust delegation adapters with shared error/GIL handling | 1.5–2 weeks (2 engineers) | Cover `SubgraphOps`, `TableOps`, array traits; add tests. |
| 2 | Expose methods via PyO3 and stage deprecation hooks | 1 week (2 engineers) | Add explicit methods, feature flags, warning paths. |
| 3 | Tooling & documentation refresh (stubs, guides, persona notes) | 4–5 days (1 engineer + doc partner) | Regenerate `.pyi`, update docs, highlight new pattern. |
| 4 | Validation, performance checks, rollout | 3–4 days (shared) | Run cargo/pytest suites, benchmark hot paths, flip default. |

Total timeline: roughly four calendar weeks for a three-person squad (mix of Rust + Python engineers), assuming no major trait coverage surprises.

## Risks and Mitigations
- **Coverage gaps**: Some dynamically exposed behaviors may lack core trait equivalents. Address during Phase 0 and either extend traits or provide compatibility shims before cutover.
- **Performance regressions**: New adapters must avoid extra cloning and use `py.allow_threads()` around heavy calls to stay within the 100ns FFI budget. Add targeted benchmarks and microtests.
- **Backwards compatibility**: Removing magic delegation could break notebooks. Ship transitional warnings, a temporary feature flag, and clear migration docs.
- **Tooling drift**: Failing to update stub generation and docs would leave users unaware of the new explicit API. Treat stub/doco updates as gating tasks.
- **Security/robustness**: Explicit bindings reduce accidental surface area, but ensure new error paths do not leak internal data and remain consistent with existing exceptions.

## Immediate Next Steps
1. Approve Phase 0 scope and assign owner(s).
2. Build an end-to-end spike for one trait-backed method (e.g., `PySubgraph.table`) to validate adapter ergonomics before scaling the effort.
3. Draft developer comms outlining the deprecation of magic getattr delegation once the spike succeeds.
