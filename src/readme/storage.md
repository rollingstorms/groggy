# Groggy Rust: storage/

Content-addressable and columnar storage for efficient, scalable graph persistence.

- **content_pool.rs**: Content-addressable node/edge storage
- **columnar_store.rs**: SIMD-optimized attribute storage
- **graph_store.rs**: Versioned, branchable graph state

## Example
```rust
let mut pool = ContentPool::new();
pool.intern_node(node);
let store = ColumnarStore::new();
store.bulk_set(...);
```
