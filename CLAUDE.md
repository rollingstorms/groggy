# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Groggy is a high-performance graph analytics library for Python with a Rust core, designed as a foundational graph library for years of development. The project combines graph topology with tabular data operations using a three-tier architecture:

- **Core (Rust)**: High-performance data structures, algorithms, and storage (`src/`)
- **FFI Layer**: Python-Rust bindings using PyO3 (`python-groggy/src/ffi/`)  
- **API Layer**: User-facing Python interface (`python-groggy/python/groggy/`)

## Essential Build Commands

### Development Build
```bash
cd python-groggy
maturin develop
```

### Release Build
```bash
cd python-groggy  
maturin develop --release
```

### Testing
```bash
# Rust unit tests
cargo test

# Specific Rust test module
cargo test core::array

# Python integration tests
python tests/test_documentation_validation.py
python tests/simple_validation_test.py
python tests/validation_test_suite.py
```

### Code Quality
```bash
cargo fmt                    # Format Rust code
cargo clippy                 # Rust linting
python -m pytest tests/     # Run Python tests
```

### Git Behavior

DO NOT COAUTHOR EVER

## Architecture Overview

### Three-Tier Architecture

**Core Layer (`src/core/`)**:
- `space.rs`: GraphSpace - Active state tracking with HashSets and attribute indexing
- `pool.rs`: GraphPool - Pure append-only columnar storage with memory pooling
- `history.rs`: HistoryForest - Git-like version control with content-addressed deltas
- `array.rs`, `table.rs`, `matrix.rs`: Unified storage views
- `traversal.rs`, `subgraph.rs`, `query.rs`: Graph algorithms and operations

**FFI Layer (`python-groggy/src/ffi/`)**:
- Pure translation layer with zero business logic
- All algorithms delegate to core Rust implementations
- PyO3 bindings with GIL management and memory safety
- Pattern: `py.allow_threads(|| { self.inner.some_operation().map_err(PyErr::from) })`

**API Layer (`python-groggy/python/groggy/`)**:
- Pythonic interface focused on ergonomic, streamlined APIs
- Table-style data access (pandas-like)
- Graph analytics and visualization integration
- NetworkX compatibility layer

### Core Data Structures

The architecture separates responsibilities:
- **GraphSpace**: Active state (which nodes/edges are currently active)
- **GraphPool**: Pure storage (all data that has ever existed) 
- **HistoryForest**: Version control (git-like branching and history)
- **Storage Views**: Array, Table, Matrix - unified interfaces to the same underlying data

### Performance Characteristics

- **Columnar storage**: Cache-friendly bulk operations
- **Append-only architecture**: Immutable, growing data structures
- **Memory pooling**: AttributeMemoryPool for string/vector reuse
- **Ultra-optimized indexing**: Attribute-first lookup patterns (50x speedup for bulk queries)
- **Content-addressed history**: Automatic deduplication of changes

## Development Persona System

The project uses a persona-driven development model with specialized roles:

- **Dr. V (Visioneer)**: Systems architect, strategic leadership, long-term vision
- **Rusty**: Rust core performance, data structures, memory optimization  
- **Bridge**: FFI pure translation, cross-language safety
- **Zen**: Python API ergonomics, user experience
- **Worf**: Security, memory safety, error handling
- **Al**: Algorithm implementation, complexity analysis
- **Arty**: Code quality, documentation, style standards
- **YN**: Innovation, paradigm shifts, future thinking

See `documentation/planning/personas/` for detailed role definitions.

## Key Development Principles

### Core Architecture Rules
- **FFI contains no business logic**: All algorithms implemented in Rust core, Bridge just translates
- **Streamlined and hard core**: Performance-first implementation with columnar thinking
- **Three-tier separation**: Clear boundaries between Core/FFI/API layers
- **Attribute-first optimization**: Data structures optimized for bulk attribute operations

### Performance Standards
- Core operations must meet O(1) amortized complexity targets
- Memory usage must scale linearly with data size
- FFI overhead <100ns per call for simple operations
- All optimizations must be benchmarked and validated

### Testing Requirements
- Comprehensive Rust unit tests for all core functionality
- Python integration tests validating documented features
- Performance regression tests for critical paths
- Documentation examples must be tested and working

## Common Issues and Solutions

### Build Issues
- Ensure `maturin` is installed: `pip install maturin`
- Use `maturin develop --release` for performance testing
- Check Python version compatibility (3.8+)

### FFI Development
- Never implement algorithms in FFI layer - delegate to core
- Use `py.allow_threads()` for long-running operations
- Handle errors with proper PyErr conversion
- Maintain memory safety across language boundaries

### Performance Debugging
- Use `cargo test --release` for performance-sensitive tests
- Profile with the actual workload patterns
- Focus on bulk operations over single-item optimizations
- Validate cache-friendly data access patterns

## Key Files and Directories

### Core Implementation
- `src/core/space.rs`: Active state management, the most performance-critical component
- `src/core/pool.rs`: Storage backend, handles all data persistence
- `src/core/history.rs`: Version control system, enables time-travel queries
- `src/api/graph.rs`: High-level graph operations coordinator

### FFI Bindings
- `python-groggy/src/ffi/api/graph.rs`: Main graph interface bindings
- `python-groggy/src/ffi/core/`: Core structure wrappers
- `python-groggy/src/ffi/utils.rs`: Type conversion utilities

### Python Interface
- `python-groggy/python/groggy/__init__.py`: Public API exports
- `python-groggy/python/groggy/graph.py`: Main Graph class
- `python-groggy/python/groggy/display/`: Rich display formatting

### Documentation and Planning
- `documentation/planning/`: Architecture decisions and future plans
- `documentation/development/`: Implementation guides and references
- `docs/`: Sphinx documentation (RST format)

## Testing Philosophy

Testing focuses on validating documented functionality works as promised:
- Python tests validate that README examples actually work
- Rust tests ensure core data structure correctness
- Integration tests verify cross-language memory safety
- Performance tests prevent regressions in critical paths

The project maintains 95%+ documented feature validation rate.