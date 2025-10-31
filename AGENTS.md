# Repository Guidelines

## Project Structure & Architecture
- Three-tier layout: Rust core in `src/`, FFI translation only in `python-groggy/src/ffi/`, and user-facing Python API in `python-groggy/python/groggy/`.
- Keep algorithms and data structures in Rust; FFI should marshal types, handle errors, and release the GIL for long calls via `py.allow_threads()`.
- Python integration tests reside in `tests/`; Rust unit and integration tests live alongside their modules. Planning notes, personas, and experiments belong in `notes/` or `archive/`.
- Shared assets and demos live in `web/`, `docs/`, and `img/`; update existing themes instead of cloning new variants.

## Build, Test, and Development Commands
- `maturin develop --release` builds and installs the extension for local Python work; run it after changing Rust or FFI code.
- Local workflows run against the in-tree build artifacts—avoid re-installing the wheel with `pip`, and skip external package installs unless explicitly required.
- `cargo build --all-targets` (or `cargo check --all-features`) validates the core; use `cargo test viz::realtime` and similar module filters when iterating.
- `pytest tests -q` covers the documented Python surface; narrow to files like `pytest tests/test_viz_accessor.py -k snapshot` for focused debugging.
- Formatting and linting: `cargo fmt --all`, `cargo clippy --all-targets -- -D warnings`, `black .`, `isort .`, and `pre-commit run --all-files` before shipping.

## Development Principles & Performance
- Preserve attribute-first, columnar operations; bulk paths beat per-item loops.
- Enforce no business logic in FFI; cross-language safety and error translation take priority.
- Benchmark optimizations in `benches/` (via `cargo bench`) and capture notes when performance trade-offs are introduced.
- Watch the 100ns per-call FFI budget and maintain O(1) amortized expectations for core mutations.

## Coding Style & Naming Conventions
- Rust: 4-space indent, snake_case modules, CamelCase types, `///` docs for public APIs, and explicit error handling (avoid `unwrap` in production paths).
- Python: PEP 8 with `black`/`isort`, type hints, builders as lowercase factory functions, and re-exports curated in `__init__.py`.
- Tests mirror their feature (`tests/test_viz_*` for viz work) and should use descriptive function names that describe behavior, not issue numbers.

## Testing & Review Flow
- Rust changes require `cargo test` plus targeted suites covering new logic; add fixtures under `tests/data/` for cross-language regressions.
- Validate documentation and notebook examples you touch to keep persona-driven artifacts trustworthy.
- Commits follow the historical style (`feat:`, `fix:`, or a crisp imperative sentence); PRs list executed test commands, call out API changes, and embed visual diffs for UI updates.
- Reference persona responsibilities from `notes/planning/personas/` when requesting specialized reviews (e.g., Bridge for FFI, Rusty for performance).