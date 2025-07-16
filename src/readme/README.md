# Groggy Rust Core (src_new): Module Guide

This directory contains the new Rust core for Groggy, providing high-performance graph processing, storage, and algorithm implementations. The Python API binds to these modules via PyO3.

---

## Modules Overview

- **lib.rs**: Main crate entry point; exposes core API to Python and Rust consumers.
- **graph/**: Main graph structures, collection delegation, core logic, and algorithms.
- **storage/**: Content-addressable and columnar storage for nodes/edges and attributes.
- **utils/**: Utilities for graph operations, conversions, and benchmarking.

---

## Quick Start (Rust)
```rust
use groggy_new::FastGraph;
let mut g = FastGraph::new();
g.add_node("n1");
g.add_edge("n1", "n2");
```

---

For more details, see README in each submodule.
